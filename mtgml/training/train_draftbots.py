import argparse
import datetime
import io
import json
import locale
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import yaml
import zstandard as zstd
from tensorboard.plugins.hparams import api as hp

from mtgml.config.hyper_config import HyperConfig
from mtgml.draftbots.draftbots import DraftBot
from mtgml.generators.picks_generator import PickGenerator
from mtgml.schedules.warmup_piecewise_constant_decay import PiecewiseConstantDecayWithLinearWarmup
from mtgml.tensorboard.callback import TensorBoardFix
from mtgml.utils.tqdm_callback import TQDMProgressBar
from mtgml.utils.range import Range

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
    parser.add_argument('--name', '-o', '-n', type=str, required=True, help="The name to save this model under.")
    parser.add_argument('--seed', type=int, default=37, help='The random seed to initialize things with to improve reproducibility.')
    parser.add_argument('--debug', action='store_true', help='Enable debug dumping of tensor stats.')
    parser.add_argument('--profile', action='store_true', help='Enable profiling a range of batches from the first epoch.')
    parser.add_argument('--deterministic', action='store_true', help='Try to keep the run deterministic so results can be reproduced.')
    parser.add_argument('--dir', type=str, required=True, help='The soure directory where the training and validation data are stored.')
    parser.set_defaults(float_type=tf.float32, use_xla=True)
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    directory = Path(args.dir)

    logging.info('Loading card data for seeding weights.')
    with open(directory/'int_to_card.json', 'r') as cards_file:
        cards_json = json.load(cards_file)
        card_ratings = [(c.get('elo', 1200) / 1200) - 1  for c in cards_json]
        card_names = [c['name'] for c in cards_json]

    config_path = Path('ml_files')/args.name/'hyper_config.yaml'
    data = {}
    if config_path.exists():
        with open(config_path, 'r') as config_file:
            data = yaml.load(config_file, yaml.Loader)
    hyper_config = HyperConfig(layer_type=DraftBot, data=data, fixed={
        'num_cards': len(cards_json),
    })
    batch_size = hyper_config.get_int('batch_size', min=8, max=2048, step=8, logdist=True, default=512,
                                      help='The number of samples to evaluate at a time')

    logging.info('Creating the pick Datasets.')
    seen_context_ratings = hyper_config.get_bool('seen_context_ratings', default=True,
                                                 help='Whether to rate cards based on the packs seen so far.')
    train_epochs_per_cycle = hyper_config.get_int('epochs_per_cycle', min=1, max=256, default=1,
                                                  help='The number of epochs for a full cycle through the training data')
    # pick_generator_train = PickPairGenerator(args.batch_size, directory/'training_parsed_picks',
    #                                          train_epochs_per_cycle, args.seed)
    pick_generator_train = PickGenerator(batch_size=batch_size, folder=directory/'training_parsed_picks',
                                         epochs_per_completion=train_epochs_per_cycle, seed=args.seed,
                                         skip_seen=not seen_context_ratings)
    logging.info(f"There are {len(pick_generator_train):,} training batches.")
    # pick_generator_test = pick_generator_train
    pick_generator_test = PickGenerator(batch_size=batch_size * 8, folder=directory/'validation_parsed_picks',
                                        epochs_per_completion=1, seed=args.seed, skip_seen=not seen_context_ratings)
    logging.info(f"There are {len(pick_generator_test):n} validation batches.")
    logging.info(f"There are {len(cards_json):n} cards being trained on.")
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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

    Path(log_dir).mkdir(exist_ok=True, parents=True)
    dtype = hyper_config.get_choice('dtype', choices=(16, 32, 64), default=32,
                                    help='The size of the floating point numbers to use for calculations in the model')
    if dtype == 16:
        dtype = 'mixed_float16'
    elif dtype == 32:
        dtype = 'float32'
    elif dtype == 64:
        dtype = 'float64'
    tf.keras.mixed_precision.set_global_policy(dtype)

    tf.config.optimizer.set_jit(hyper_config.get_bool('use_xla', default=False, help='Whether to use xla to speed up calculations.'))
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
        for i, card in enumerate(cards_json):
            f.write(f'{i+1}\t"{card["name"]}"\t{"".join(color_name[x] for x in sorted(card.get("color_identity")))}\t{card["cmc"]}\t{card.get("type")}\n')

    logging.info('Loading DraftBot model.')
    output_dir = f'././ml_files/{args.name}/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_batches = len(pick_generator_train)
    tensorboard_period = len(pick_generator_test)
    draftbots = hyper_config.build(name='DraftBot', dynamic=False)
    optimizer = hyper_config.get_choice('optimizer', choices=('adam', 'adamax', 'adadelta', 'nadam', 'sgd', 'lazyadam', 'rectadam', 'novograd'),
                                        default='adam', help='The optimizer type to use for optimization')
    learning_rate = hyper_config.get_float(f"{optimizer}_learning_rate", min=1e-04, max=1e-02, logdist=True,
                                           default=1e-03, help=f'The learning rate to use for {optimizer}')
    if hyper_config.get_bool('linear_warmup', default=False,
                             help='Whether to linearly ramp up the learning rate from zero for the first epoch.'):
        learning_rate = PiecewiseConstantDecayWithLinearWarmup(0, len(pick_generator_train), [0], [learning_rate, learning_rate])
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if optimizer == 'adamax':
        opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    if optimizer == 'adadelta':
        opt = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    if optimizer == 'nadam':
        opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    if optimizer == 'sgd':
        momentum = hyper_config.get_float('sgd_momentum', min=1e-05, max=1e-01, logdist=True,
                                          default=1e-04, help='The momentum for sgd optimization.')
        nesterov = hyper_config.get_bool('sgd_nesterov', default=False, help='Whether to use nesterov momentum for sgd.')
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    if optimizer == 'lazyadam':
        opt = tfa.optimizers.LazyAdam(learning_rate=learning_rate)
    if optimizer == 'rectadam':
        weight_decay = hyper_config.get_float('rectadam_weight_decay', min=1e-08, max=1e-01, default=1e-06, logdist=True,
                                              help='The weight decay for rectadam optimization per batch.')
        opt = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate, weight_decay=weight_decay)
    if optimizer == 'novograd':
        weight_decay = hyper_config.get_float('novograd_weight_decay', min=1e-08, max=1e-01, default=1e-06, logdist=True,
                                              help='The weight decay for novograd optimization per batch.')
        opt = tfa.optimizers.NovoGrad(learning_rate=learning_rate, weight_decay=weight_decay)
    draftbots.compile(optimizer=opt, loss=lambda y_true, y_pred: 0.0)
    latest = tf.train.latest_checkpoint(output_dir)
    if latest is not None:
        logging.info('Loading Checkpoint.')
        draftbots.load_weights(latest)
    with open(config_path, 'w') as config_file:
        yaml.dump(hyper_config.get_config(), config_file)
    with open(log_dir + '/hyper_config.yaml', 'w') as config_file:
        yaml.dump(hyper_config.get_config(), config_file)

    logging.info('Starting training')
    callbacks = []
    if not args.debug:
        mcp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=output_dir + 'model',
            monitor='val_accuracy_top_1',
            verbose=False,
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            save_freq='epoch')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=log_dir + '/model-{epoch:04d}.ckpt',
            monitor='val_accuracy_top_1',
            verbose=False,
            save_best_only=False,
            save_weights_only=True,
            mode='max',
            save_freq='epoch')
        callbacks.append(mcp_callback)
        callbacks.append(cp_callback)
    nan_callback = tf.keras.callbacks.TerminateOnNaN()
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy_top_1', patience=8, min_delta=2**-8,
                                                   mode='max', restore_best_weights=True, verbose=True)
    tb_callback = TensorBoardFix(log_dir=log_dir, histogram_freq=1, write_graph=True,
                                 update_freq=512, embeddings_freq=None,
                                 profile_batch=0 if args.debug or not args.profile else (num_batches // 2 - 16, num_batches // 2 + 15))
    BAR_FORMAT = "{n_fmt}/{total_fmt}|{bar}|{elapsed}/{remaining}s - {rate_fmt} - {desc}"
    tqdm_callback = TQDMProgressBar(smoothing=0.01, epoch_bar_format=BAR_FORMAT)
    callbacks.append(nan_callback)
    # callbacks.append(es_callback)
    callbacks.append(tb_callback)
    callbacks.append(tqdm_callback)
    draftbots.fit(
        pick_generator_train,
        validation_data=pick_generator_test,
        validation_freq=1,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0,
        max_queue_size=2**10,
    )
    if not args.debug:
        logging.info('Saving final model.')
        Path(f'{output_dir}/final').mkdir(parents=True, exist_ok=True)
        draftbots.save(f'{output_dir}/final', save_format='tf')
