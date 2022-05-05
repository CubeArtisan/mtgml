import argparse
import datetime
import json
import locale
import logging
import os
import sys
import time
from pathlib import Path

import horovod.tensorflow.keras as hvd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import yaml

from mtgml_native.generators.adj_mtx_generator import DeckAdjMtxGenerator
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


class CardIndexGenerator(tf.keras.utils.Sequence):
    def __init__(self, num_cards, rows, columns, deck_adj_mtx, cube_adj_mtx, use_columns):
        self.num_cards = num_cards
        self.use_columns = use_columns
        self.rows = rows if use_columns else (rows + columns)
        self.columns = columns
        self.deck_adj_mtx = deck_adj_mtx / np.sum(deck_adj_mtx, axis=1, keepdims=True)
        self.cube_adj_mtx = cube_adj_mtx
        if use_columns:
            valid_cards = [i for i, x in enumerate(self.deck_adj_mtx) if x[i] < 1]
        else:
            valid_cards = np.arange(num_cards, dtype=np.int32)
        self.row_idxs = np.array(valid_cards, dtype=np.int32)
        self.column_idxs = np.array(valid_cards, dtype=np.int32)
        self.num_valid = len(valid_cards)
        logging.debug('NUM VALID', self.num_valid)
        self.on_epoch_end()

    def on_epoch_end(self):
        np.random.shuffle(self.row_idxs)
        self.row_batches = []
        self.column_batches = []
        for i in range(self.num_valid // self.rows):
            row = self.row_idxs[i * self.rows:(i+1) * self.rows]
            if self.use_columns:
                np.random.shuffle(self.column_idxs)
                for j in range(self.num_valid // self.columns):
                    self.row_batches.append(row)
                    self.column_batches.append(self.column_idxs[j * self.columns:(j+1) * self.columns])
            else:
                self.row_batches.append(row)
                self.column_batches.append(())
        self.row_batches = np.array(self.row_batches, dtype=np.int32)
        self.column_batches = np.array(self.column_batches, dtype=np.int32)
        self.batch_idxs = np.arange(len(self.row_batches))
        np.random.shuffle(self.batch_idxs)

    def __len__(self):
        return len(self.batch_idxs) // hvd.size()

    def __getitem__(self, idx):
        idx = hvd.size() * idx + hvd.rank()
        batch_idx = self.batch_idxs[idx]
        return ((self.row_batches[batch_idx], self.column_batches[batch_idx]),)



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
    parser.add_argument('--fine-tuning', action='store_true', help='Fine tune the embeddings.')
    parser.set_defaults(float_type=tf.float32, use_xla=True)
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    logging.info('Loading card data for seeding weights.')
    with open('data/maps/int_to_card.json', 'r') as cards_file:
        cards_json = json.load(cards_file)
    tokenized = [tokenize_card(c) for c in cards_json]
    max_length = max(len(c) for c in tokenized)
    tokenized = np.array([pad(c, max_length) for c in tokenized], dtype=np.int32)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    with open(Path(log_dir) / 'token_metadata.tsv', 'w') as fp:
        fp.write('Index\tToken\n')
        for i, token in enumerate(tokens):
            fp.write(f'{i}\t{token}\n')
    config_path = Path('ml_files')/args.name/'nlp_hyper_config.yaml'
    data = {}
    if config_path.exists():
        with open(config_path, 'r') as config_file:
            data = yaml.load(config_file, yaml.Loader)
    logging.info('Initializing Generators')
    adj_mtx_batch_size = 200
    cube_adj_mtx_generator = [] # CubeAdjMtxGenerator('data/train_cubes.bin', len(tokenized), 64, args.seed * 7 + 3)
    # cube_adj_mtx_generator.on_epoch_end()
    deck_adj_mtx_generator = DeckAdjMtxGenerator('data/train_decks.bin', len(tokenized), 256, args.seed)
    deck_adj_mtx_generator.on_epoch_end()
    hyper_config = HyperConfig(layer_type=CombinedCardModel, data=data, fixed={
        'num_tokens': len(tokens),
        'card_token_map': tokenized,
        'deck_adj_mtx': deck_adj_mtx_generator.get_adj_mtx(),
        'cube_adj_mtx': [], # cube_adj_mtx_generator.get_adj_mtx(),
        'fine_tuning': args.fine_tuning,
    })
    logging.info(f'There are {len(deck_adj_mtx_generator)} adjacency matrix batches')
    logging.info(f"There are {len(cards_json):n} cards being trained on.")
    # os.environ["HOROVOD_ENABLE_XLA_OPS"] = "1"
    hvd.init()
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(physical_devices[hvd.local_rank()], 'GPU')
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
    elif optimizer == 'lamb':
        weight_decay = hyper_config.get_float('lamb_weight_decay', min=1e-10, max=1e-01, default=1e-06, logdist=True,
                                              help='The weight decay for lamb optimization per batch.')
        opt = tfa.optimizers.LAMB(learning_rate=learning_rate, weight_decay_rate=weight_decay)
    else:
        raise Exception('Need to specify a valid optimizer type')
    opt = hvd.DistributedOptimizer(opt)
    model.compile(optimizer=opt)
    latest = tf.train.latest_checkpoint(output_dir)
    latest = output_dir + 'nlp_model'
    if Path(latest + '.index').exists():
        logging.info('Loading Checkpoint.')
        model.load_weights(latest)

    logging.info('Starting training')
    # train_generator = \
    #     SplitGenerator(CombinedGenerator(cube_adj_mtx_generator, deck_adj_mtx_generator),
    #                    hyper_config.get_int('epochs_for_completion', min=1, default=1,
    #                        help='The number of epochs it should take to go through the entire dataset.'))
    # validation_generator = SplitGenerator(CombinedGenerator(cube_adj_mtx_generator, deck_adj_mtx_generatorhyper_config))
    epochs_for_completion = hyper_config.get_int('epochs_for_completion', min=1, default=1,
                                                 help='The number of epochs it shoudl take to go through the entire dataset.')
    if not args.fine_tuning:
        epochs_for_completion = 1
    count = 60
    train_generator = SplitGenerator(CardIndexGenerator(len(tokenized), count, count, deck_adj_mtx_generator.get_adj_mtx(),
                                                        [], # cube_adj_mtx_generator.get_adj_mtx(),
                                                        use_columns=args.fine_tuning),
                                     epochs_for_completion=epochs_for_completion)
    with open(config_path, 'w') as config_file:
        yaml.dump(hyper_config.get_config(), config_file)
    with open(log_dir + '/nlp_hyper_config.yaml', 'w') as config_file:
        yaml.dump(hyper_config.get_config(), config_file)
    callbacks = []
    if not args.debug and hvd.rank() == 0:
        mcp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=output_dir + 'nlp_model',
            verbose=True,
            save_weights_only=True,
            save_freq=len(train_generator))
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=log_dir + '/model-{epoch:04d}.ckpt',
            verbose=True,
            save_weights_only=True,
            save_freq=5 * len(train_generator))
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
                                 update_freq=max(len(train_generator) // 25, 1), embeddings_freq=None,
                                 profile_batch=0 if args.debug or not args.profile else (128, 135))
    BAR_FORMAT = "{n_fmt}/{total_fmt}|{bar}|{elapsed}/{remaining}s - {rate_fmt} - {desc}"
    tqdm_callback = TQDMProgressBar(smoothing=0.001, epoch_bar_format=BAR_FORMAT)
    hvd_callback = hvd.callbacks.BroadcastGlobalVariablesCallback(0)
    callbacks.append(hvd_callback)
    callbacks.append(nan_callback)
    # callbacks.append(es_callback)
    if hvd.rank() == 0:
        callbacks.append(tb_callback)
    callbacks.append(tqdm_callback)
    # model((tf.cast((0,1), dtype=tf.int32), tf.cast((1,), dtype=tf.int32)), training=False)
    # model((tf.cast((0,1), dtype=tf.int32), tf.cast((1,), dtype=tf.int32)), training=True)
    # model.temperature.assign(5)
    # logging.info(model.summary())
    model.fit(
        train_generator,
        # validation_data=validation_generator,
        # validation_freq=1,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0,
        max_queue_size=2**10,
    )
