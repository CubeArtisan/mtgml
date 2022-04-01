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

from mtgml.constants import MAX_TOKENS
from mtgml_native.generators.adj_mtx_generator import CubeAdjMtxGenerator, DeckAdjMtxGenerator
from mtgml.generators.combined_generator import CombinedGenerator
from mtgml.generators.split_generator import SplitGenerator
from mtgml.config.hyper_config import HyperConfig
from mtgml.models.card_embeddings import CombinedCardModel
from mtgml.preprocessing.tokenize_card import tokens, tokenizeCard
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

    parser.add_argument('--epochs', '-e', type=int, required=True, help="The maximum number of epochs to train for")
    parser.add_argument('--time-limit', '-t', type=int, default=0, help="The maximum time to train for")
    parser.add_argument('--name', '-o', '-n', type=str, required=True, help="The name to save this model under.")
    parser.add_argument('--seed', type=int, default=37, help='The random seed to initialize things with to improve reproducibility.')
    parser.add_argument('--debug', action='store_true', help='Enable debug dumping of tensor stats.')
    parser.add_argument('--profile', action='store_true', help='Enable profiling a range of batches from the first epoch.')
    parser.add_argument('--deterministic', action='store_true', help='Try to keep the run deterministic so results can be reproduced.')
    parser.set_defaults(float_type=tf.float32, use_xla=True)
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    logging.info('Loading card data for seeding weights.')
    with open('data/maps/int_to_card.json', 'r') as cards_file:
        cards_json = json.load(cards_file)
    tokenized = [pad(tokenizeCard(c), MAX_TOKENS) for c in cards_json]
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    with open(Path(log_dir) / 'token_metadata.tsv', 'w') as fp:
        fp.write('Index\tToken\n')
        for i, token in enumerate(tokens):
            fp.write(f'{i}\t{token}\n')
    config_path = Path('ml_files')/args.name/'hyper_config.yaml'
    data = {}
    if config_path.exists():
        with open(config_path, 'r') as config_file:
            data = yaml.load(config_file, yaml.Loader)
    print('Initializing Generators')
    hyper_config = HyperConfig(layer_type=CombinedCardModel, data=data, fixed={
        'num_cards': len(cards_json) + 1,
        'num_tokens': len(tokens),
        'card_token_map': tokenized,
    })
    adj_mtx_batch_size = hyper_config.get_int('adj_mtx_batch_size', min=8, max=2048, step=8, logdist=True, default=8,
                                              help='The number of rows of the adjacency matrices to evaluate at a time.')
    deck_adj_mtx_generator = DeckAdjMtxGenerator('data/train_decks.bin', len(cards_json), adj_mtx_batch_size, args.seed)
    deck_adj_mtx_generator.on_epoch_end()
    cube_adj_mtx_generator = CubeAdjMtxGenerator('data/train_cubes.bin', len(cards_json), adj_mtx_batch_size, args.seed * 7 + 3)
    cube_adj_mtx_generator.on_epoch_end()
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
    metadata = os.path.join(log_dir, 'metadata.tsv')
    with open(metadata, "w") as f:
        f.write('Index\tName\tColors\tMana Value\tType\n')
        f.write('0\tN/A\tN/A\tN/A\tN/A\n')
        for i, card in enumerate(cards_json):
            f.write(f'{i+1}\t"{card["name"]}"\t{"".join(color_name[x] for x in sorted(card.get("color_identity")))}\t{card["cmc"]}\t{card.get("type")}\n')

    logging.info('Loading Combined model.')
    output_dir = f'././ml_files/{args.name}/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # strategy = tf.distribute.experimental.CentralStorageStrategy()
    # strategy = tf.distribute.MirroredStrategy()
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model = hyper_config.build(name='CombinedCardModel', dynamic=False, )
    if model is None:
        sys.exit(1)
    optimizer = hyper_config.get_choice('optimizer', choices=('adam', 'adamax', 'adadelta', 'nadam', 'sgd', 'lazyadam', 'rectadam', 'novograd'),
                                        default='adam', help='The optimizer type to use for optimization')
    learning_rate = hyper_config.get_float(f"{optimizer}_learning_rate", min=1e-04, max=1e-02, logdist=True,
                                           default=1e-03, help=f'The learning rate to use for {optimizer}')
    # if hyper_config.get_bool('linear_warmup', default=False,
    #                          help='Whether to linearly ramp up the learning rate from zero for the first epoch.'):
    #     learning_rate = PiecewiseConstantDecayWithLinearWarmup(0, len(pick_generator_train), [0], [learning_rate, learning_rate])
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'adamax':
        opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer == 'adadelta':
        opt = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer == 'nadam':
        opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        momentum = hyper_config.get_float('sgd_momentum', min=1e-05, max=1e-01, logdist=True,
                                          default=1e-04, help='The momentum for sgd optimization.')
        nesterov = hyper_config.get_bool('sgd_nesterov', default=False, help='Whether to use nesterov momentum for sgd.')
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer == 'lazyadam':
        opt = tfa.optimizers.LazyAdam(learning_rate=learning_rate)
    elif optimizer == 'rectadam':
        weight_decay = hyper_config.get_float('rectadam_weight_decay', min=1e-08, max=1e-01, default=1e-06, logdist=True,
                                              help='The weight decay for rectadam optimization per batch.')
        opt = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'novograd':
        weight_decay = hyper_config.get_float('novograd_weight_decay', min=1e-08, max=1e-01, default=1e-06, logdist=True,
                                              help='The weight decay for novograd optimization per batch.')
        opt = tfa.optimizers.NovoGrad(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        raise Exception('Need to specify a valid optimizer type')
    model.compile(optimizer=opt)
    latest = tf.train.latest_checkpoint(output_dir)
    if latest is not None:
        logging.info('Loading Checkpoint.')
        model.load_weights(latest)

    logging.info('Starting training')
    with strategy.scope():
        train_generator = \
            SplitGenerator(CombinedGenerator(cube_adj_mtx_generator, deck_adj_mtx_generator),
                           hyper_config.get_int('epochs_for_completion', min=1, default=1,
                               help='The number of epochs it should take to go through the entire dataset.'))
        validation_generator = SplitGenerator(CombinedGenerator(cube_adj_mtx_generator, deck_adj_mtx_generator), 1)
        with open(config_path, 'w') as config_file:
            yaml.dump(hyper_config.get_config(), config_file)
        with open(log_dir + '/hyper_config.yaml', 'w') as config_file:
            yaml.dump(hyper_config.get_config(), config_file)
        callbacks = []
        if not args.debug:
            mcp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=output_dir + 'model',
                verbose=True,
                save_weights_only=True,
                save_freq='epoch')
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=log_dir + '/model-{epoch:04d}.ckpt',
                verbose=True,
                save_weights_only=True,
                save_freq='epoch')
            callbacks.append(mcp_callback)
            callbacks.append(cp_callback)
        if args.time_limit > 0:
            to_callback = tfa.callbacks.TimeStopping(seconds=60 * args.time_limit, verbose=1)
            callbacks.append(to_callback)
        nan_callback = tf.keras.callbacks.TerminateOnNaN()
        num_batches = len(train_generator)
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy_top_1', patience=8, min_delta=2**-8,
                                                       mode='max', restore_best_weights=True, verbose=True)
        tb_callback = TensorBoardFix(log_dir=log_dir, histogram_freq=None, write_graph=True,
                                     update_freq=128, embeddings_freq=None,
                                     profile_batch=0 if args.debug or not args.profile else (128, 135))
        BAR_FORMAT = "{n_fmt}/{total_fmt}|{bar}|{elapsed}/{remaining}s - {rate_fmt} - {desc}"
        tqdm_callback = TQDMProgressBar(smoothing=0.001, epoch_bar_format=BAR_FORMAT)
        callbacks.append(nan_callback)
        # callbacks.append(es_callback)
        callbacks.append(tb_callback)
        callbacks.append(tqdm_callback)
        model.fit(
            train_generator,
            # validation_data=validation_generator,
            # validation_freq=1,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=0,
            max_queue_size=2**10,
        )
