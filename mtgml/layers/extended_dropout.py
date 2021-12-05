import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer


class ExtendedDropout(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            'rate': hyper_config.get_float('rate', min=0, max=0.99, step=0.01, default=0,
                                           help='The percent of values that get replaced with zero.'),
            'noise_shape': hyper_config.get_array('noise_shape', default=None,
                                                  help='The shape of the generated noise which will be broadcast as needed.'),
            'return_mask': hyper_config.get_bool('return_mask', default=False,
                                                 help='Whether to return both the output and the mask for the noise.'),
            'blank_last_dim': hyper_config.get_bool('blank_last_dim', default=False,
                                                    help='Apply dropout to the entire last dimension vs choosing for each element of the last dimension.')

        }

    def __init__(self, *args, **kwargs):
        super(self, ExtendedDropout).__init__(*args, **kwargs)
        self.supports_masking = True

    def call(self, inputs, training=False, mask=None):
        if 0 >= rate or not training:
            return inputs, tf.ones_like(inputs)
        elif rate >= 1:
            return tf.zeros_like(inputs), tf.zeros_like(inputs)
        else:
            noise_shape = self.noise_shape or tf.shape(inputs)
            if self.blank_last_dim:
                noise_shape[-1] = 1
            noise = tf.random.uniform(noise_shape, minval=0, maxval=1, dtype=tf.float32,
                                      seed=self.seed, name='noise')
            noise_mult = tf.where((noise + tf.zeros_like(inputs)) >= rate, tf.ones_like(inputs),
                                  tf.zeros_like(inputs), name='noise_mult')
            if scale:
                dropped_inputs = tf.multiply(inputs, noise_mult, name='dropped_inputs')
                if scale == 'sum':
                    result = tf.math.multiply(tf.math.divide_no_nan(tf.reduce_sum(inputs, name='sum_pre_drop'),
                                                                    tf.reduce_sum(dropped_inputs, name='sum_post_drop') + 1e-04,
                                                                    name='scaling_factor'),
                                              dropped_inputs)
                else:
                    result = tf.math.multiply(tf.math.divide_no_nan(tf.norm(inputs, axis=-1, keepdims=True, ord=scale, name='norm_pre_drop'),
                                                                    tf.norm(dropped_inputs, axis=-1, ord=scale, keepdims=True, name='norm_post_drop') + 1e-04,
                                                                    name='scaling_factor'),
                                              dropped_inputs)
            else:
                result = tf.math.multiply(inputs, noise_mult)
            if self.return_mask:
                noise_mask = noise >= rate
                if mask:
                    noise_mask = tf.math.logical_and(noise_mask, mask, name='combined_mask')
                return result, noise_mask
            else:
                return result
