import struct
from pathlib import Path

import numpy as np
import tensorflow as tf

from mtgml.constants import MAX_DECK_SIZE

PREFIX = struct.Struct(f"{MAX_DECK_SIZE}H{MAX_DECK_SIZE}H{MAX_DECK_SIZE}H")


class DeckGenerator(tf.keras.utils.Sequence):
    def __init__(self, filename: str | Path, batch_size: int, seed: int):
        with open(filename, "rb") as fp:
            pool, copy_counts, targets = zip(*PREFIX.iter_unpack(fp.read()))
        self.pool = np.array(pool, dtype=np.int16)
        self.copy_counts = np.array(copy_counts, dtype=np.int16)
        self.targets = np.array(targets, dtype=np.float32)
        self.indices = np.arange(self.pool.shape[0])
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self):
        return (self.pool.shape[0] + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        self.rng.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        return (
            (self.pool[indices], self.copy_counts[indices], self.targets[indices]),
            np.zeros_like(self.targets[indices]),
        )
