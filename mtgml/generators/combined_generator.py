import tensorflow as tf


class CombinedGenerator(tf.keras.utils.Sequence):
    def __init__(self, *generators, training=False):
        self.generators = generators
        self.generator_indices = [0 for _ in generators]
        self.training = training

    def __len__(self):
        return max(len(gen) for gen in self.generators)

    def on_epoch_end(self):
        for gen in self.generators:
            gen.on_epoch_end()
        self.generator_indices = [0 for _ in self.generators]

    def __getitem__(self, index):
        result = []
        for gen in self.generators:
            if index % len(gen) == 0 and index > 0:
                gen.on_epoch_end()
            result.append(gen[index % len(gen)])
        return (result,)
