import tensorflow as tf


class SplitGenerator(tf.keras.utils.Sequence):
    def __init__(self, generator, epochs_for_completion):
        self.generator = generator
        self.epochs_for_completion = epochs_for_completion
        self.epoch_count = 0

    def __len__(self):
        return len(self.generator) // self.epochs_for_completion

    def on_epoch_end(self):
        self.epoch_count += 1
        if self.epoch_count % self.epochs_for_completion == 0:
            self.generator.on_epoch_end()

    def __getitem__(self, index):
        index += (self.epoch_count % self.epochs_for_completion) * len(self)
        return self.generator[index]
