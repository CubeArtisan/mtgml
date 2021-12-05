import tensorflow as tf


class ZeroMasked(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(self, RemoveMasked).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask:
            return tf.expand_dims(tf.cast(mask, self.compute_dtype, name='float_mask'), -1,
                                  name='expanded_mask') * inputs
        else:
            return inputs
