import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

import mtgml.preprocessing.utils as utils
from mtgml.recommender.recommender import CC_Recommender


def is_valid_cube(cube):
    # return True
    return cube["numDecks"] > 0 and len(set(cube["cards"])) >= 120


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, help="The number of epochs to train for.")
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        choices=[2**i for i in range(0, 16)],
        help="The number of cubes/cards to train on at a time.",
    )
    parser.add_argument("--name", "-n", "-o", type=str, help="The folder under ml_files to save the model in.")
    parser.add_argument(
        "--regularization",
        "--reg",
        "-r",
        default=1,
        type=float,
        help="The relative weight of regularization vs reproducing cubes.",
    )
    parser.add_argument(
        "--card-reconstruction",
        "--card-rec",
        "-cr",
        default=0.1,
        type=float,
        help="The relative weight of reconstructing individual cards vs reproducing cubes.",
    )
    parser.add_argument("--noise", type=float, help="The mean number of random swaps to make per cube.")
    parser.add_argument(
        "--noise-stddev", type=float, default=0.1, help="The standard deviation of the amount of noise to apply."
    )
    parser.add_argument("--learning-rate", type=float, default=1e-04, help="The initial learning rate.")
    parser.add_argument("--seed", type=int, default=None, help="A random seed to provide reproducible runs.")
    parser.add_argument("--xla", action="store_true", help="Use the XLA optimizer on the model.")
    parser.add_argument(
        "--profile", action="store_true", help="Run profiling on part of the second batch to analyze performance."
    )
    parser.add_argument("--debug", action="store_true", help="Enable dumping debug information to logs/debug.")
    parser.add_argument(
        "--num-workers", "-j", type=int, default=1, help="Number of simulataneous workers to run to generate the data."
    )
    args = parser.parse_args()

    output = Path("ml_files") / args.name
    data = Path("data")
    maps = data / "maps"
    int_to_card_filepath = maps / "int_to_card.json"
    cube_folder = data / "cubes"
    log_dir = Path("logs/fit/") / datetime.now().strftime("%Y%m%d-%H%M%S")

    def load_adj_mtx():
        print("Loading Adjacency Matrix . . .\n")
        adj_mtx = np.load("data/adj_mtx.npy")
        return adj_mtx

    print("Loading card data and cube counts.")
    with open(int_to_card_filepath, "rb") as int_to_card_file:
        int_to_card = json.load(int_to_card_file)
    card_to_int = {v: i for i, v in enumerate(int_to_card)}
    num_cards = len(int_to_card)
    num_cubes = utils.get_num_objs(cube_folder, validation_func=is_valid_cube)
    print(f"There are {num_cubes} valid cubes.")

    def load_cubes():
        print("Loading Cube Data")
        cubes = utils.build_cubes(cube_folder, num_cubes, num_cards, card_to_int, validation_func=is_valid_cube)
        return cubes

    class TensorBoardFix(tf.keras.callbacks.TensorBoard):
        """
        This fixes incorrect step values when using the TensorBoard callback with custom summary ops
        """

        def on_train_begin(self, *args, **kwargs):
            super(TensorBoardFix, self).on_train_begin(*args, **kwargs)
            tf.summary.experimental.set_step(self._train_step)

        def on_test_begin(self, *args, **kwargs):
            super(TensorBoardFix, self).on_test_begin(*args, **kwargs)
            tf.summary.experimental.set_step(self._val_step)

    def reset_random_seeds(seed):
        # currently not used
        os.environ["PYTHONHASHSEED"] = str(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print("Reset random seeds")

    if args.seed is not None:
        reset_random_seeds(args.seed)

    if args.debug:
        log_dir = "logs/debug/"
        print("Enabling Debugging")
        tf.debugging.experimental.enable_dump_debug_info(
            log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1, op_regex="(?!^(Placeholder|Constant)$)"
        )

    print("Setting jit flag")
    tf.config.optimizer.set_jit(args.xla)

    print("Setting Up Model . . . \n")
    checkpoint_dir = output / "model"
    autoencoder = CC_Recommender(num_cards)
    latest = tf.train.latest_checkpoint(str(output))
    if latest is not None:
        print("Loading Checkpoint. Saved values are:")
        autoencoder.load_weights(latest)
    THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 0.9]
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=[
            tf.keras.losses.BinaryCrossentropy(name="cube_loss"),
            # tf.keras.losses.CategoricalCrossentropy(name='card_loss'),
            tf.keras.losses.KLDivergence(name="adj_mtx_loss"),
        ],
        # loss_weights=[1.0, args.card_reconstruction, args.regularization],
        loss_weights=[1.0, args.regularization],
        metrics=[
            (
                *[tf.keras.metrics.Recall(t, name=f"cube_recall_{t}") for t in THRESHOLDS],
                tf.keras.metrics.PrecisionAtRecall(0.5, name="cube_prec_at_recall"),
                *[tf.keras.metrics.Precision(t, name=f"cube_precision_{t}") for t in THRESHOLDS],
            ),
            # (
            #     tf.keras.metrics.Recall(THRESHOLDS, name='card_recall'),
            #     tf.keras.metrics.PrecisionAtRecall(0.99, name='card_prec_at_recall'),
            #     tf.keras.metrics.CategoricalAccuracy(name='card_accuracy'),
            # ),
            (
                tf.keras.metrics.Accuracy(name="adj_mtx_accuracy"),
                tf.keras.metrics.MeanAbsoluteError(name="adj_mtx_error"),
            ),
        ],
        from_serialized=True,
    )

    output.mkdir(exist_ok=True, parents=True)
    callbacks = []
    mcp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir, monitor="loss", verbose=False, save_best_only=False, mode="min", save_freq="epoch"
    )
    nan_callback = tf.keras.callbacks.TerminateOnNaN()
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=64, mode="min", restore_best_weights=True, verbose=True
    )
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.5,
        mode="min",
        patience=32,
        min_delta=1 / (2**14),
        cooldown=16,
        min_lr=1 / (2**20),
        verbose=True,
    )
    tb_callback = TensorBoardFix(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq=num_cubes // 10 // args.batch_size,
        profile_batch=0
        if not args.profile
        else (int(1.4 * num_cubes / args.batch_size), int(1.6 * num_cubes / args.batch_size)),
    )
    callbacks.append(mcp_callback)
    callbacks.append(nan_callback)
    callbacks.append(tb_callback)
    callbacks.append(lr_callback)
    # callbacks.append(es_callback)

    print(f"Creating a pool with {args.num_workers} different workers.")
    generator = Generator(
        load_adj_mtx(),
        load_cubes(),
        args.num_workers,
        args.batch_size,
        args.seed,
        args.noise,
        args.noise_stddev,
    )

    class GeneratorWrapper(tf.keras.utils.Sequence):
        def __getitem__(self, item):
            res = generator.__getitem__(item)
            return res

        def __len__(self):
            return generator.__len__()

    with generator:
        print("Starting training")
        wrapped = GeneratorWrapper()
        autoencoder.fit(
            wrapped,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=2,
        )
        print("Saving final model")
        autoencoder.save(output, save_format="tf")
