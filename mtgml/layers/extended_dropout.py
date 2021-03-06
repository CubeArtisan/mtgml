import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer


class ExtendedDropout(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            'rate': tf.constant(hyper_config.get_float('rate', min=0, max=0.99, step=0.01, default=0.2,
                                                       help='The percent of values that get replaced with zero.'), dtype=tf.float32),
            'noise_shape': hyper_config.get_list('noise_shape', default=None,
                                                  help='The shape of the generated noise which will be broadcast as needed.'),
            'return_mask': hyper_config.get_bool('return_mask', default=False,
                                                 help='Whether to return both the output and the mask for the noise.'),
            'blank_last_dim': hyper_config.get_bool('blank_last_dim', default=False,
                                                    help='Apply dropout to the entire last dimension vs choosing for each element of the last dimension.'),

            'supports_masking': True,
        }

    def build(self, input_shapes):
        super(ExtendedDropout, self).build(input_shapes)
        self.mask_shape = list(input_shapes)
        self.mask_shape[0] = None
        if self.blank_last_dim:
            self.noise_shape = list(input_shapes)
            self.noise_shape[-1] = 1
            del self.mask_shape[-1]
        elif self.noise_shape is None:
            self.noise_shape = list(input_shapes)
        self.noise_shape[0] = None
        self.input_shapes = (None, *input_shapes[1:])

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
          return None
        concrete_inputs_shape = tf.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
          noise_shape.append(concrete_inputs_shape[i] if value is None else value)
        return tf.convert_to_tensor(noise_shape)

    def call(self, inputs, training=False, mask=None):
        print(self.rate)
        if 0 >= self.rate or not training:
            result = inputs
            noise_mask = tf.cast(tf.ones_like(inputs), tf.bool)
        elif self.rate >= 1:
            result = tf.zeros_like(inputs)
            noise_mask = tf.cast(tf.zeros_like(inputs), tf.bool)
        else:
            noise_shape = self._get_noise_shape(inputs)
            noise = tf.random.uniform(noise_shape, minval=0, maxval=1, dtype=self.compute_dtype,
                                      seed=self.seed, name='noise')
            noise_mult = tf.where(noise >= self.rate, tf.ones_like(inputs),
                                  tf.zeros_like(inputs), name='noise_mult')
            result = tf.math.multiply(inputs, noise_mult)
            noise_mask = noise >= self.rate
        result = tf.ensure_shape(result, self.input_shapes)
        if self.return_mask:
            if mask is not None:
                if tf.rank(mask) < tf.rank(noise_mask):
                    mask = tf.expand_dims(mask, -1)
                noise_mask = tf.math.logical_and(noise_mask, mask, name='combined_mask')
            noise_mask = tf.ensure_shape(noise_mask, self.mask_shape)
            return result, noise_mask
        else:
            return result
