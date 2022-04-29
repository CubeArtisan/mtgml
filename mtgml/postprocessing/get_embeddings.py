import argparse
import datetime
import json
import locale
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import yaml

from mtgml_native.generators.adj_mtx_generator import CubeAdjMtxGenerator, DeckAdjMtxGenerator
from mtgml.generators.split_generator import SplitGenerator
from mtgml.config.hyper_config import HyperConfig
from mtgml.models.card_embeddings import CombinedCardModel
from mtgml.preprocessing.tokenize_card import tokens, tokenize_card
from mtgml.tensorboard.callback import TensorBoardFix
from mtgml.utils.grid import pad
from mtgml.utils.tqdm_callback import TQDMProgressBar

BATCH_CHOICES = tuple(2 ** i for i in range(4, 18))
EMBED_DIMS_CHOICES = tuple(2 ** i + j for i in range(0, 10) for j in range(2))
ACTIVATION_CHOICES = ('relu', 'selu', 'swish', 'tanh', 'sigmoid', 'linear', 'gelu', 'elu')
OPTIMIZER_CHOICES = ('adam', 'adamax', 'lazyadam', 'rectadam', 'novograd', 'lamb', 'adadelta',
                     'nadam', 'rmsprop')

if __name__ == "__main__":
    locale.setlocale(locale.LC_ALL, '')
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', '-o', '-n', type=str, required=True, help="The name to save this model under.")
    parser.add_argument('--debug', action='store_true', help='Enable debug dumping of tensor stats.')
    parser.add_argument('--deterministic', action='store_true', help='Try to keep the run deterministic so results can be reproduced.')
    parser.set_defaults(float_type=tf.float32, use_xla=True)
    args = parser.parse_args()

    logging.info('Loading card data for seeding weights.')
    with open('data/maps/int_to_card.json', 'r') as cards_file:
        cards_json = json.load(cards_file)
    tokenized = [tokenize_card(c) for c in cards_json]
    max_length = max(len(c) for c in tokenized)
    tokenized = np.array([pad(c, max_length) for c in tokenized], dtype=np.int32)
    config_path = Path('ml_files')/args.name/'nlp_hyper_config.yaml'
    data = {}
    if config_path.exists():
        with open(config_path, 'r') as config_file:
            data = yaml.load(config_file, yaml.Loader)
    print('Initializing Generators')
    adj_mtx_batch_size = 200
    cube_adj_mtx_generator = CubeAdjMtxGenerator('data/train_cubes.bin', len(tokenized), 64, 7 + 3)
    cube_adj_mtx_generator.on_epoch_end()
    deck_adj_mtx_generator = DeckAdjMtxGenerator('data/train_decks.bin', len(tokenized), 256, 7)
    deck_adj_mtx_generator.on_epoch_end()
    hyper_config = HyperConfig(layer_type=CombinedCardModel, data=data, fixed={
        'num_tokens': len(tokens),
        'card_token_map': tokenized,
        'deck_adj_mtx': deck_adj_mtx_generator.get_adj_mtx(),
        'cube_adj_mtx': cube_adj_mtx_generator.get_adj_mtx(),
    })
    print(f'There are {len(deck_adj_mtx_generator)} adjacency matrix batches')
    logging.info(f"There are {len(cards_json):n} cards being trained on.")
    if args.debug:
        log_dir = "logs/debug/"
        logging.info('Enabling Debugging')
        tf.debugging.experimental.enable_dump_debug_info(
            log_dir,
            tensor_debug_mode='FULL_HEALTH',
            circular_buffer_size=-1,
            tensor_dtypes=[args.float_type],
            # op_regex="(?!^(Placeholder|Constant)$)"
        )
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    dtype = hyper_config.get_choice('dtype', choices=(16, 32, 64), default=32,
                                    help='The size of the floating point numbers to use for calculations in the model')
    print(dtype, dtype==16)
    if dtype == 16:
        dtype = 'mixed_float16'
    elif dtype == 32:
        dtype = 'float32'
    elif dtype == 64:
        dtype = 'float64'
    tf.keras.mixed_precision.set_global_policy(dtype)

    tf.config.optimizer.set_jit(bool(hyper_config.get_bool('use_xla', default=False, help='Whether to use xla to speed up calculations.')))
    if args.debug:
        tf.config.optimizer.set_experimental_options=({
            'layout_optimizer': True,
            'constant_folding': True,
            'shape_optimization': True,
            'remapping': True,
            'arithmetic_optimization': True,
            'dependency_optimization': True,
            'loop_optimization': True,
            'function_optimization': True,
            'debug_stripper': False,
            'disable_model_pruning': True,
            'scoped_allocator_optimization': True,
            'pin_to_host_optimization': True,
            'implementation_selector': True,
            'disable_meta_optimizer': True,
            'min_graph_nodes': 1,
        })
    else:
        tf.config.optimizer.set_experimental_options=({
            'layout_optimizer': True,
            'constant_folding': True,
            'shape_optimization': True,
            'remapping': True,
            'arithmetic_optimization': True,
            'dependency_optimization': True,
            'loop_optimization': True,
            'function_optimization': True,
            'debug_stripper': True,
            'disable_model_pruning': False,
            'scoped_allocator_optimization': True,
            'pin_to_host_optimization': True,
            'implementation_selector': True,
            'disable_meta_optimizer': False,
            'min_graph_nodes': 1,
        })
    tf.config.threading.set_intra_op_parallelism_threads(32)
    tf.config.threading.set_inter_op_parallelism_threads(32)

    color_name = {'W': 'White', 'U': 'Blue', 'B': 'Black', 'R': "Red", 'G': 'Green'}

    logging.info('Loading Combined model.')
    output_dir = f'ml_files/{args.name}/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # strategy = tf.distribute.experimental.CentralStorageStrategy()
    # strategy = tf.distribute.MirroredStrategy()
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model = hyper_config.build(name='CombinedCardModel', dynamic=False, )
    if model is None:
        sys.exit(1)
    model.compile()
    latest = output_dir + 'nlp_model'
    logging.info('Loading Checkpoint.')
    model.load_weights(latest)

    logging.info('Starting training')
    with strategy.scope():
        train_generator = ((np.arange(len(tokenized)), [0 for _ in tokenized]),)
        dataset = tf.data.Dataset.from_tensor_slices(train_generator).batch(128)
        predictions = model.predict(
            dataset,
        )
        np.save(output_dir + 'card_embeddings.npy', predictions)
        print(predictions)
