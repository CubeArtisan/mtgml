import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tensorflow as tf
import numpy as np
import yaml
from tqdm.auto import tqdm

from mtgml.config.hyper_config import HyperConfig
from mtgml.constants import set_debug
from mtgml.generators.combined_generator import CombinedGenerator
from mtgml.models.combined_model import CombinedModel
from mtgml_native.generators.adj_mtx_generator import CubeAdjMtxGenerator, DeckAdjMtxGenerator, PickAdjMtxGenerator
from mtgml_native.generators.draftbot_generator import DraftbotGenerator
from mtgml_native.generators.recommender_generator import RecommenderGenerator

def is_nested(structure, path):
    # Not reducing this to simple return to allow easier future expansion with predicates.
    if isinstance(structure, (tuple, list, dict, Mapping)):
        return True
    return False


def should_skip_prop(structure, prop):
    if prop == '__dict__':
        return True
    if isinstance(structure, tf.keras.layers.Layer) and (prop == 'state_updates' or prop == 'updates'):
        return True
    if (hasattr(structure, '__class__') and hasattr(structure.__class__, prop)
        and isinstance(getattr(structure.__class__, prop), property)):
        return True
    return False


def gen_children_with_keys(structure):
    if isinstance(structure, (tuple, list)):
        yield from enumerate(structure)
    elif isinstance(structure, (dict, Mapping)):
        yield from structure.items()
    else:
        for prop in dir(structure):
            if not should_skip_prop(structure, prop) and hasattr(structure, prop):
                yield prop, getattr(structure, prop)


def gen_leaves_with_path(structure, should_recurse=is_nested, prefix=(), visited=None):
    if visited is None:
        visited = set()
    struct_id = id(structure)
    if should_recurse(structure, prefix) and struct_id not in visited:
        visited.add(struct_id)
        empty = True
        for key, child in gen_children_with_keys(structure):
            empty = False
            yield from gen_leaves_with_path(child, should_recurse=should_recurse, prefix=(*prefix, key), visited=visited)
        if empty:
            yield (prefix, structure)
    else:
        yield (prefix, structure)


def get_accessor_method(structure, key):
    to_throw = None
    non_str_throw = None
    if hasattr(structure, '__getitem__'):
        try:
            structure[key]
            return '__getitem__'
        except (KeyError, IndexError):
            to_throw = sys.exc_info()
        except TypeError:
            non_str_throw = sys.exc_info()
    if not isinstance(key, str) and non_str_throw is not None and non_str_throw[0] is not None:
        raise non_str_throw[1].with_traceback(non_str_throw[2])
    try:
        getattr(structure, key)
        return 'getattr'
    except AttributeError:
        if to_throw is None or to_throw[0] is None:
            raise
        else:
            raise to_throw[1].with_traceback(to_throw[2])


def access_by_path(structure, path, prefix=()) -> Any:
    if len(path) == 0:
        return structure
    else:
        key = path[0]
        accessor = get_accessor_method(structure, key)
        if accessor == '__getitem__':
            value = structure[key]
        elif accessor == 'getattr':
            value = getattr(structure, key)
        else:
            raise NotImplementedError(f'Do not know how to deal with accessor {accessor} at path {(*prefix, key)}.')
        return access_by_path(value, path[1:])

def set_by_path(structure, path, value):
    if len(path) == 0:
        raise KeyError('Cannot set a value on the empty path.')
    if len(path) >= 2:
        grandparent = access_by_path(structure, path[:-2])
        parent = access_by_path(grandparent, path[-2:-1])
    else:
        grandparent = None
        parent = access_by_path(structure, path[:-1])
    accessor = get_accessor_method(parent, path[-1])
    if accessor == '__getitem__':
        if hasattr(parent, '__setitem__') or not grandparent or not isinstance(path[-1], int):
            parent[path[-1]] = value
        else:
            updated = list(parent)
            updated[path[-1]] = value
            return set_by_path(structure, path[:-1], type(parent)(updated))
    elif accessor == 'getattr':
        setattr(parent, path[-1], value)
    else:
        raise NotImplementedError(f'Do not know how to deal with accessor {accessor} at path {tuple(path)}.')
    return structure


set_debug(False)
MODEL_PATH = Path('ml_files/latest')
MODEL = None
with open(MODEL_PATH / 'int_to_card.json', 'rb') as map_file:
    int_to_card = json.load(map_file)
CARD_TO_INT = {v['oracle_id']: k for k, v in enumerate(int_to_card)}
INT_TO_CARD = {int(k): v for k, v in enumerate(int_to_card)}


no_no_names = frozenset(('non_trainable_variables', 'variables', 'trainable_variables', 'submodules',
                         '_self_tracked_trackables', '_trainable_weights', '_self_unconditional_dependency_names',
                         '_non_trainable_variables', '_unconditional_dependency_names', '_undeduplicated_weights',
                         '_non_trainable_weights', 'trainable_weights', 'weights', 'non_trainable_weights',
                         '_tracking_metadata'))

def is_nested_or_module(structure, path):
    return (len(path) == 0 or not isinstance(path[-1], str) or path[-1] not in no_no_names) and (is_nested(structure, path) or isinstance(structure, tf.Module))


def replace_variables_with_constants(module):
    var_tqdm = tqdm(unit='variable', unit_scale=1, smoothing=0.1)
    for path, leaf in tqdm(gen_leaves_with_path(module, is_nested_or_module), unit='path', unit_scale=1, smoothing=0.1):
        if isinstance(leaf, tf.Variable):
            constant = tf.constant(leaf.value(), dtype=leaf.dtype, name=f'{leaf.name}Constant')
            def do_nothing(*args, **kwargs):
                return constant
            constant.assign_add = do_nothing
            set_by_path(module, path, constant)
            var_tqdm.update()
    return module


def call_interpreter(interpreter, **kwargs):
    mapping = {p['name']: p['index'] for p in interpreter.get_input_details()}
    for name, value in kwargs.items():
        interpreter.set_tensor(mapping[name], value)
    interpreter.invoke()
    return {p['name']: interpreter.get_tensor(p['index']) for p in interpreter.get_output_details()}


pick_batch_size = 1
seed = 31
cube_batch_size = 1
adj_mtx_batch_size = 1
noise_mean = 0.5
noise_std = 0.2
draftbot_train_generator = DraftbotGenerator('data/train_picks.bin', pick_batch_size, seed)
recommender_train_generator = RecommenderGenerator('data/train_cubes.bin', len(int_to_card),
                                                   cube_batch_size, seed, noise_mean, noise_std)
# deck_adj_mtx_generator = DeckAdjMtxGenerator('data/train_decks.bin', len(int_to_card), adj_mtx_batch_size, seed)
# deck_adj_mtx_generator.on_epoch_end()
deck_adj_mtx_generator = PickAdjMtxGenerator('data/train_picks.bin', len(int_to_card), adj_mtx_batch_size, seed)
deck_adj_mtx_generator.on_epoch_end()
cube_adj_mtx_generator = CubeAdjMtxGenerator('data/train_cubes.bin', len(int_to_card), adj_mtx_batch_size, seed * 7 + 3)
cube_adj_mtx_generator.on_epoch_end()
with draftbot_train_generator, recommender_train_generator:
    generator = CombinedGenerator(draftbot_train_generator, recommender_train_generator,
                                                 cube_adj_mtx_generator, deck_adj_mtx_generator)
    pick_train_example, cube_train_example, *rest = generator[0][0]
    pick_train_example = pick_train_example[:5]
    cube_train_example = cube_train_example[:1]
example = (*pick_train_example, *cube_train_example, *rest[0], *rest[1])


def get_model():
    global MODEL
    if MODEL is None:
        tf.keras.utils.set_random_seed(1)
        # tf.config.experimental.enable_op_determinism()
        tf.config.optimizer.set_jit(True)
        with open(MODEL_PATH / 'hyper_config.yaml', 'r') as config_file:
            data = yaml.load(config_file, yaml.Loader)
        hyper_config = HyperConfig(layer_type=CombinedModel, data=data, fixed={
            'num_cards': len(INT_TO_CARD) + 1,
            'DraftBots': {'card_weights': np.ones((len(INT_TO_CARD) + 1,))},
        })
        MODEL = hyper_config.build(name='CombinedModel', dynamic=False)
        if MODEL is None:
            print('Failed to load model')
            sys.exit(1)
        MODEL(example, training=True)
        MODEL.load_weights(MODEL_PATH / 'model').expect_partial()
        print("Loaded model")
    return MODEL


model = get_model()
model = replace_variables_with_constants(model)
model(example, training=False)
model.call_draftbots(*pick_train_example)
model.call_recommender(cube_train_example[0].astype(np.int16))
print(model.summary())

tf.saved_model.save(
    tf.keras.Model(),
    'data/draftbots_tflite_full',
    signatures={
        'call_draftbots': model.call_draftbots.get_concrete_function(),
        'call_recommender': model.call_recommender.get_concrete_function(),
    },
)


def draftbot_gen():
    for i in range(256):
        example = generator[i][0]
        basics, pool, seen, seen_coords, seen_coord_weights = example[0][:5]
        yield dict(basics=basics, pool=pool, seen=seen, seen_coords=seen_coords, seen_coord_weights=seen_coord_weights)


def recommender_gen():
    for i in range(256):
        example = generator[i][0]
        yield { 'cube': example[1][0].astype(np.int16) }


def combined_gen():
    for _, draftbot_example, recommender_example in zip(range(128), draftbot_gen(), recommender_gen()):
        yield ('call_draftbots', draftbot_example)
        yield ('call_recommder', recommender_example)


converter = tf.lite.TFLiteConverter.from_saved_model('data/draftbots_tflite_full')
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
converter.optimizations = [tf.lite.Optimize.DEFAULT, tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
# converter.representative_dataset = combined_gen
tflite_model = converter.convert()
Path('ml_files/testing_tflite').mkdir(exist_ok=True, parents=True)
with open('ml_files/testing_tflite/combined_model.tflite', 'wb') as f:
    f.write(tflite_model)
# tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)

interpreter = tf.lite.Interpreter(model_path='ml_files/testing_tflite/combined_model.tflite')
interpreter.allocate_tensors()
call_draftbots = interpreter.get_signature_runner('call_draftbots')
basics, pool, seen, seen_coords, seen_coord_weights = pick_train_example
draftbot_result = call_draftbots(basics=basics, pool=pool, seen=seen, seen_coords=seen_coords,
                                 seen_coord_weights=seen_coord_weights)
call_recommender = interpreter.get_signature_runner('call_recommender')
cube = cube_train_example[0]
recommender_result = call_recommender(cube=cube.astype(np.int16))

# converter = tf.lite.TFLiteConverter.from_concrete_functions(
#     [model.call_draftbots.get_concrete_function()], None
# )
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS,
# ]
# converter.optimizations = [tf.lite.Optimize.DEFAULT, tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
# # converter.representative_dataset = draftbot_gen
# tflite_model = converter.convert()
# with open('ml_files/testing_tflite/draftbot_model.tflite', 'wb') as f:
#     f.write(tflite_model)
# tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)

# converter = tf.lite.TFLiteConverter.from_concrete_functions(
#     [model.call_recommender.get_concrete_function()], model
# )
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS,
# ]
# converter.optimizations = [tf.lite.Optimize.DEFAULT, tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
# # converter.representative_dataset = recommender_gen
# tflite_model = converter.convert()
# with open('ml_files/testing_tflite/recommender_model.tflite', 'wb') as f:
#     f.write(tflite_model)
# tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)

# draftbots_interpreter = tf.lite.Interpreter(model_path='ml_files/testing_tflite/draftbot_model.tflite')
# draftbots_interpreter.allocate_tensors()
# basics, pool, seen, seen_coords, seen_coord_weights = pick_train_example
# draftbot_result = call_interpreter(draftbots_interpreter, basics=basics, pool=pool, seen=seen, seen_coords=seen_coords,
#                                    seen_coord_weights=seen_coord_weights)

# recommender_interpreter = tf.lite.Interpreter(model_path='ml_files/testing_tflite/recommender_model.tflite')
# recommender_interpreter.allocate_tensors()
# cube = cube_train_example[0]
# recommender_result = call_interpreter(recommender_interpreter, cube=cube.astype(np.int16))

print(draftbot_result)
print()
print(recommender_result)
