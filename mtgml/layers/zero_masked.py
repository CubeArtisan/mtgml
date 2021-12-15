import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer

class ZeroMasked(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            'supports_masking': True,
        }

    def call(self, inputs, mask=None):
        if mask is not None:
            while len(mask.shape) < len(inputs.shape):
                mask = tf.expand_dims(mask, -1)
            float_mask = tf.cast(mask, self.compute_dtype, name='float_mask')
            return float_mask * inputs
        else:
            print('No mask')
            return inputs
