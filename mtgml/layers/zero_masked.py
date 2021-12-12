import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer

class ZeroMasked(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            'supports_masking': True,
        }

    def call(self, inputs, mask=None):
        if mask:
            return tf.expand_dims(tf.cast(mask, self.compute_dtype, name='float_mask'), -1,
                                  name='expanded_mask') * inputs
        else:
            return inputs
