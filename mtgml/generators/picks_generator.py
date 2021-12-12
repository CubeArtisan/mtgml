import json
import math

import numpy as np
import tensorflow as tf

from mtgml.generators.utils import load_npy_to_tensor

FIELDS = ('cards_in_pack', 'picked', 'seen', 'seen_coords', 'seen_coord_weights', 'coords',
          'coord_weights', 'y_idx')

__ALL__ = ('PickGenerator',)

class PickGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, folder, epochs_per_completion, seed=29):
        with open(folder / 'counts.json') as count_file:
            counts = json.load(count_file)
            self.context_count = counts['contexts']
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.Generator(np.random.PCG64(self.seed))
        self.shuffled_indices = np.arange(self.context_count)
        # We call on_epoch_end immediately so this'll become 0 and be an accurate count.
        self.epoch_count = -1
        self.epochs_per_completion = epochs_per_completion
        self.on_epoch_end()
        for FIELD in FIELDS:
            setattr(self, FIELD, load_npy_to_tensor(folder/f'{FIELD}.npy.zstd'))

    def reset_rng(self):
        self.rng = np.random.Generator(np.random.PCG64(self.seed))
        self.epoch_count = -1
        self.on_epoch_end()

    def __len__(self):
        idx_base, idx_max = self.get_epoch_context_counts()
        return math.ceil((idx_max - idx_base) / self.batch_size)

    def on_epoch_end(self):
        self.epoch_count += 1
        if self.epoch_count % self.epochs_per_completion == 0:
            self.rng.shuffle(self.shuffled_indices)

    def get_epoch_context_counts(self):
        pos_in_cycle = self.epoch_count % self.epochs_per_completion
        contexts_per_epoch_f = self.context_count / self.epochs_per_completion
        idx_base = math.ceil(pos_in_cycle * contexts_per_epoch_f)
        idx_max = min(math.ceil((pos_in_cycle + 1) * contexts_per_epoch_f), self.context_count)
        return idx_base, idx_max

    def __getitem__(self, idx):
        idx = min(idx, len(self) - 1)
        idx_base, idx_max = self.get_epoch_context_counts()
        min_idx_offset = idx * self.batch_size + idx_base
        max_idx_offset = min(min_idx_offset + self.batch_size, idx_max)
        context_idxs = self.shuffled_indices[min_idx_offset:max_idx_offset]
        result = tuple(getattr(self, FIELD)[context_idxs] for FIELD in FIELDS)
        return (result, result[-1])
