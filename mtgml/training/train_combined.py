import argparse
import datetime
import json
import locale
import logging
import math
import os
import sys
from pathlib import Path

import tensorflow as tf
import tensorflow_addons as tfa
import yaml

# from mtgml_native.generators.adj_mtx_generator import PickAdjMtxGenerator
from mtgml_native.generators.draftbot_generator import DraftbotGenerator
from mtgml_native.generators.recommender_generator import RecommenderGenerator

from mtgml.config.hyper_config import HyperConfig
from mtgml.constants import OPTIMIZER_CHOICES, set_debug
from mtgml.generators.combined_generator import CombinedGenerator
from mtgml.generators.deck_generator import DeckGenerator
from mtgml.generators.split_generator import SplitGenerator
from mtgml.models.combined_model import CombinedModel
from mtgml.tensorboard.callback import TensorBoardFix
from mtgml.utils.tqdm_callback import TQDMProgressBar

BATCH_CHOICES = tuple(2**i for i in range(4, 18))
EMBED_DIMS_CHOICES = tuple(2**i + j for i in range(0, 10) for j in range(2))
if __name__ == "__main__":
    set_debug(False)
    locale.setlocale(locale.LC_ALL, "")
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", "-e", type=int, required=True, help="The maximum number of epochs to train for")
    parser.add_argument("--time-limit", "-t", type=int, default=0, help="The maximum time to train for")
    parser.add_argument("--name", "-o", "-n", type=str, required=True, help="The name to save this model under.")
    parser.add_argument(
        "--seed", type=int, default=37, help="The random seed to initialize things with to improve reproducibility."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug dumping of tensor stats.")
    parser.add_argument(
        "--profile", action="store_true", help="Enable profiling a range of batches from the first epoch."
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Try to keep the run deterministic so results can be reproduced."
    )
    parser.set_defaults(float_type=tf.float32, use_xla=True)
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except:
            logging.exception(f"Could not set memory growth on {device}.")
    logging.info("Loading card data for seeding weights.")
    with open("data/maps/int_to_card.json", "r") as cards_file:
        cards_json = json.load(cards_file)
    original_to_new_path = Path("data/maps/original_to_new_index.json")
    if original_to_new_path.exists():
        with original_to_new_path.open("r") as fp:
            original_to_new_index = json.load(fp)
    else:
        original_to_new_index = tuple(range(len(cards_json) + 1))
    num_cards = max(original_to_new_index) + 1
    new_to_original = {v: i for i, v in enumerate(original_to_new_index)}
    new_to_original = [new_to_original[i] - 1 for i in range(1, num_cards)]
    new_cards_json = [cards_json[i] for i in new_to_original]

    config_path = Path("ml_files") / args.name / "hyper_config.yaml"
    data = {}
    if config_path.exists():
        with open(config_path, "r") as config_file:
            data = yaml.load(config_file, yaml.Loader)
    print("Initializing Generators")
    hyper_config = HyperConfig(
        layer_type=CombinedModel,
        data=data,
        fixed={
            "num_cards": num_cards,
        },
    )
    tf.config.threading.set_intra_op_parallelism_threads(32)
    tf.config.threading.set_inter_op_parallelism_threads(32)
    # strategy = tf.distribute.MirroredStrategy()
    strategy = tf.distribute.experimental.CentralStorageStrategy()
    use_draftbots = (
        hyper_config.get_float(
            "draftbots_weight", default=1, min=0, max=128, help="The weight to multiply the draftbot loss by."
        )
        > 0
    )
    use_recommender = (
        hyper_config.get_float(
            "recommender_weight", default=1, min=0, max=128, help="The weight to multiply the recommender loss by."
        )
        > 0
    )
    use_deck_builder = (
        hyper_config.get_float(
            "deck_builder_weight", default=1, min=0, max=128, help="The weight to multiply the deck builder loss by."
        )
        > 0
    )
    pick_batch_size = (
        hyper_config.get_int(
            "pick_batch_size",
            min=8,
            max=2048,
            step=8,
            logdist=True,
            default=64,
            help="The number of picks to evaluate at a time",
        )
        * strategy.num_replicas_in_sync
    )
    cube_batch_size = (
        hyper_config.get_int(
            "cube_batch_size",
            min=8,
            max=2048,
            step=8,
            logdist=True,
            default=8,
            help="The number of cube samples to evaluate at a time",
        )
        * strategy.num_replicas_in_sync
    )
    deck_batch_size = (
        hyper_config.get_int(
            "deck_batch_size",
            min=8,
            max=2048,
            step=8,
            logdist=True,
            default=64,
            help="The number of decks to evaluate at a time",
        )
        * strategy.num_replicas_in_sync
    )
    adj_mtx_batch_size = (
        hyper_config.get_int(
            "adj_mtx_batch_size",
            min=8,
            max=2048,
            step=8,
            logdist=True,
            default=8,
            help="The number of rows of the adjacency matrices to evaluate at a time.",
        )
        * strategy.num_replicas_in_sync
    )
    noise_mean = hyper_config.get_float(
        "cube_noise_mean", min=0, max=1, default=0.7, help="The median of the noise distribution for cubes."
    )
    noise_std = hyper_config.get_float(
        "cube_noise_std", min=0, max=1, default=0.15, help="The median of the noise distribution for cubes."
    )
    draftbot_train_generator = DraftbotGenerator("data/train_picks.bin", pick_batch_size, args.seed)
    draftbot_validation_generator = DraftbotGenerator("data/validation_picks.bin", pick_batch_size, args.seed * 31)
    recommender_train_generator = RecommenderGenerator(
        "data/train_cubes.bin", num_cards - 1, cube_batch_size, args.seed, noise_mean, noise_std
    )
    recommender_validation_generator = RecommenderGenerator(
        "data/validation_cubes.bin", num_cards - 1, cube_batch_size, args.seed + 13, 0, 0
    )
    deck_train_generator = DeckGenerator("data/train_decks.bin", deck_batch_size, args.seed * 37)
    deck_validation_generator = DeckGenerator("data/validation_decks.bin", deck_batch_size, args.seed * 37)
    # deck_adj_mtx_generator = DeckAdjMtxGenerator('data/train_decks.bin', len(cards_json), adj_mtx_batch_size, args.seed)
    # deck_adj_mtx_generator.on_epoch_end()
    deck_adj_mtx_generator = deck_validation_generator
    # deck_adj_mtx_generator = PickAdjMtxGenerator(
    #     "data/train_picks.bin", num_cards - 1, adj_mtx_batch_size, args.seed * 29
    # )
    # deck_adj_mtx_generator.on_epoch_end()
    # cube_adj_mtx_generator = PickAdjMtxGenerator(
    #     "data/train_picks.bin", num_cards - 1, 8 * adj_mtx_batch_size, args.seed * 23
    # )
    # cube_adj_mtx_generator.on_epoch_end()
    cube_adj_mtx_generator = deck_validation_generator
    print(f"There are {len(draftbot_train_generator)} training pick batches")
    print(f"There are {len(draftbot_validation_generator)} validation pick batches")
    print(f"There are {len(recommender_train_generator)} training recommender batches")
    print(f"There are {len(recommender_validation_generator)} validation recommender batches")
    print(f"There are {len(deck_train_generator)} training deck batches")
    print(f"There are {len(deck_validation_generator)} validation deck batches")
    print(f"There are {len(deck_adj_mtx_generator)} adjacency matrix batches")
    print(f"There are {len(cube_adj_mtx_generator)} validation adjacency matrix batches")
    logging.info(f"There are {num_cards:n} cards being trained on.")
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.debug:
        log_dir = "logs/debug/"
        logging.info("Enabling Debugging")
        tf.debugging.experimental.enable_dump_debug_info(
            log_dir,
            tensor_debug_mode="FULL_HEALTH",
            circular_buffer_size=-1,
            tensor_dtypes=[args.float_type],
            # op_regex="(?!^(Placeholder|Constant)$)"
        )
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    Path(log_dir).mkdir(exist_ok=True, parents=True)
    dtype = hyper_config.get_choice(
        "dtype",
        choices=(16, 32, 64),
        default=32,
        help="The size of the floating point numbers to use for calculations in the model",
    )
    if dtype == 16:
        dtype = "mixed_float16"
    elif dtype == 32:
        dtype = "float32"
    elif dtype == 64:
        dtype = "float64"
    tf.keras.mixed_precision.set_global_policy(dtype)

    use_xla = bool(hyper_config.get_bool("use_xla", default=False, help="Whether to use xla to speed up calculations."))
    tf.config.optimizer.set_jit(use_xla)
    if args.debug:
        tf.config.optimizer.set_experimental_options = {
            "layout_optimizer": True,
            "constant_folding": True,
            "shape_optimization": True,
            "remapping": True,
            "arithmetic_optimization": True,
            "dependency_optimization": True,
            "loop_optimization": True,
            "function_optimization": True,
            "debug_stripper": False,
            "disable_model_pruning": True,
            "scoped_allocator_optimization": True,
            "pin_to_host_optimization": True,
            "implementation_selector": True,
            "disable_meta_optimizer": True,
            "auto_mixed_precision": False,
            "min_graph_nodes": 1,
        }
    else:
        tf.config.optimizer.set_experimental_options = {
            "layout_optimizer": True,
            "constant_folding": True,
            "shape_optimization": True,
            "remapping": True,
            "arithmetic_optimization": True,
            "dependency_optimization": True,
            "loop_optimization": True,
            "function_optimization": True,
            "debug_stripper": True,
            "disable_model_pruning": False,
            "scoped_allocator_optimization": True,
            "pin_to_host_optimization": True,
            "implementation_selector": True,
            "auto_mixed_precision": False,
            "disable_meta_optimizer": False,
            "min_graph_nodes": 1,
        }
        if bool(
            hyper_config.get_bool(
                "use_async_execution", default=False, help="Whether to enable experimental asynchronous execution."
            )
        ):
            tf.config.experimental.set_synchronous_execution(False)
    print("Synchronous execution:", tf.config.experimental.get_synchronous_execution())
    print("Tensor float 32 execution:", tf.config.experimental.tensor_float_32_execution_enabled())

    color_name = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}
    metadata = os.path.join(log_dir, "metadata.tsv")
    with open(metadata, "w") as f:
        f.write("Index\tName\tColors\tMana Value\tRarity\tSet\tType\n")
        f.write("0\tPlaceholderForNull\tNA\tNA\tNA\n")
        for i, card in enumerate(new_cards_json):
            f.write(
                f'{i+1}\t"{card["name"]}"\t{"".join(color_name[x] for x in sorted(card.get("color_identity")))}\t{card["cmc"]}\t{card.get("rarity")}\t{card.get("set")}\t{card.get("type")}\n'
            )

    logging.info("Loading Combined model.")
    output_dir = f"././ml_files/{args.name}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    epochs_for_completion = hyper_config.get_int(
        "epochs_for_completion",
        min=1,
        default=32,
        help="The number of epochs it should take to go through the entire dataset.",
    )
    train_generator = SplitGenerator(
        CombinedGenerator(
            draftbot_train_generator,
            recommender_train_generator,
            deck_validation_generator,
            deck_adj_mtx_generator,
            deck_adj_mtx_generator,
        ),
        epochs_for_completion,
    )
    validation_generator = SplitGenerator(
        CombinedGenerator(
            draftbot_validation_generator,
            recommender_validation_generator,
            deck_validation_generator,
            cube_adj_mtx_generator,
            cube_adj_mtx_generator,
        ),
        1,
    )
    optimizer = hyper_config.get_choice(
        "optimizer", choices=OPTIMIZER_CHOICES, default="adam", help="The optimizer type to use for optimization"
    )
    learning_rate = hyper_config.get_float(
        f"{optimizer}_learning_rate",
        min=1e-05,
        max=1.0,
        logdist=True,
        default=1e-03,
        help=f"The learning rate to use for {optimizer}",
    ) * math.sqrt(strategy.num_replicas_in_sync)
    weight_decay = hyper_config.get_float(
        f"{optimizer}_weight_decay",
        min=1e-10,
        max=1e-01,
        default=1e-06,
        logdist=True,
        help=f"The weight decay per batch for optimization  with {optimizer}.",
    )
    ema_frequency = hyper_config.get_int(
        f"{optimizer}_ema_overwrite_frequency",
        default=0,
        min=0,
        max=2**16,
        help="How often to overwrie weights with their moving averages. Set to 0 to disable.",
    )
    with strategy.scope():
        model = hyper_config.build(name="CombinedModel", dynamic=False)
        if model is None:
            sys.exit(1)
        if optimizer == "adam":
            opt = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                ema_overwrite_frequency=ema_frequency,
                use_ema=ema_frequency > 0,
                jit_compile=use_xla,
            )
        elif optimizer == "adamax":
            opt = tf.keras.optimizers.Adamax(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                ema_overwrite_frequency=ema_frequency,
                use_ema=ema_frequency > 0,
                jit_compile=use_xla,
            )
        elif optimizer == "adadelta":
            opt = tf.keras.optimizers.Adadelta(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                ema_overwrite_frequency=ema_frequency,
                use_ema=ema_frequency > 0,
                jit_compile=use_xla,
            )
        elif optimizer == "nadam":
            opt = tf.keras.optimizers.Nadam(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                ema_overwrite_frequency=ema_frequency,
                use_ema=ema_frequency > 0,
                jit_compile=use_xla,
            )
        elif optimizer == "sgd":
            momentum = hyper_config.get_float(
                "sgd_momentum",
                min=1e-05,
                max=1e-01,
                logdist=True,
                default=1e-04,
                help="The momentum for sgd optimization.",
            )
            nesterov = hyper_config.get_bool(
                "sgd_nesterov", default=False, help="Whether to use nesterov momentum for sgd."
            )
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        elif optimizer == "lazyadam":
            opt = tfa.optimizers.LazyAdam(learning_rate=learning_rate)
        elif optimizer == "rectadam":
            weight_decay = hyper_config.get_float(
                "rectadam_weight_decay",
                min=1e-08,
                max=1e-01,
                default=1e-06,
                logdist=True,
                help="The weight decay for rectadam optimization per batch.",
            )
            opt = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate, weight_decay=weight_decay)
        elif optimizer == "novograd":
            weight_decay = hyper_config.get_float(
                "novograd_weight_decay",
                min=1e-08,
                max=1e-01,
                default=1e-06,
                logdist=True,
                help="The weight decay for novograd optimization per batch.",
            )
            opt = tfa.optimizers.NovoGrad(learning_rate=learning_rate, weight_decay=weight_decay)
        elif optimizer == "lamb":
            weight_decay = hyper_config.get_float(
                "lamb_weight_decay",
                min=1e-10,
                max=1e-01,
                default=1e-06,
                logdist=True,
                help="The weight decay for lamb optimization per batch.",
            )
            opt = tfa.optimizers.LAMB(learning_rate=learning_rate, weight_decay_rate=weight_decay)
        elif optimizer == "adafactor":
            relative_step = bool(
                hyper_config.get_bool(
                    "adafactor_relative_step",
                    default=True,
                    help="Whether to use the learning rate schedule builtin to adafactor. It makes learning rate the min of lr and 1/sqrt(iterations + 1).",
                )
            )
            opt = tf.keras.optimizers.experimental.Adafactor(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                use_ema=ema_frequency > 0,
                relative_step=relative_step,
                ema_overwrite_frequency=ema_frequency,
                jit_compile=use_xla,
            )
        else:
            raise Exception("Need to specify a valid optimizer type")
        model.compile(optimizer=opt)
    latest = tf.train.latest_checkpoint(output_dir)
    if latest is not None:
        logging.info("Loading Checkpoint.")
        model.load_weights(latest)

    logging.info("Starting training")
    with draftbot_train_generator, recommender_train_generator, draftbot_validation_generator, recommender_validation_generator:
        with open(config_path, "w") as config_file:
            yaml.dump(hyper_config.get_config(), config_file)
        with open(log_dir + "/hyper_config.yaml", "w") as config_file:
            yaml.dump(hyper_config.get_config(), config_file)
        callbacks = []
        if not args.debug:
            mcp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=output_dir + "model", verbose=False, save_weights_only=True, save_freq="epoch"
            )
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=log_dir + "/model-{epoch:04d}.ckpt", verbose=True, save_weights_only=True, save_freq="epoch"
            )
            callbacks.append(mcp_callback)
            callbacks.append(cp_callback)
        if args.time_limit > 0:
            to_callback = tfa.callbacks.TimeStopping(seconds=60 * args.time_limit, verbose=1)
            callbacks.append(to_callback)
        nan_callback = tf.keras.callbacks.TerminateOnNaN()
        num_batches = len(train_generator)
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy_top_1",
            patience=8,
            min_delta=2**-8,
            mode="max",
            restore_best_weights=True,
            verbose=True,
        )
        tb_callback = TensorBoardFix(
            log_dir=log_dir,
            histogram_freq=0,
            write_graph=True,
            update_freq=256,
            embeddings_freq=None,
            profile_batch=0 if args.debug or not args.profile else (128, 135),
        )
        BAR_FORMAT = "{n_fmt}/{total_fmt}|{bar}|{elapsed}/{remaining}s - {rate_fmt} - {desc}"
        tqdm_callback = TQDMProgressBar(smoothing=0.001, epoch_bar_format=BAR_FORMAT)
        callbacks.append(nan_callback)
        # callbacks.append(es_callback)
        callbacks.append(tb_callback)
        callbacks.append(tqdm_callback)
        if use_draftbots or use_recommender or use_deck_builder:
            pick_train_example, cube_train_example, deck_train_example, *rest = validation_generator[0][0]
        else:
            pick_train_example, cube_train_example, deck_train_example, *rest = train_generator[0][0]
        pick_train_example = pick_train_example[:5]
        cube_train_example = cube_train_example[:1]
        deck_train_example = deck_train_example[:2]
        # Make sure it compiles the correct setup for evaluation
        with strategy.scope():
            model((pick_train_example, cube_train_example, deck_train_example, *rest), training=False)
            print(model.summary())
            model.fit(
                train_generator,
                validation_data=validation_generator if use_draftbots or use_recommender or use_deck_builder else None,
                validation_freq=max(epochs_for_completion // 10, 1)
                if use_draftbots or use_recommender or use_deck_builder
                else 0,
                epochs=args.epochs,
                callbacks=callbacks,
                verbose=0,
                max_queue_size=2**10,
            )
