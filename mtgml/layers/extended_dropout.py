import tensorflow as tf

from mtgml.constants import is_debug
from mtgml.layers.configurable_layer import ConfigurableLayer


class ExtendedDropout(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            "rate": hyper_config.get_float(
                "rate",
                min=0,
                max=0.99,
                step=0.01,
                default=0.2,
                help="The percent of values that get replaced with zero.",
            ),
            "noise_shape": hyper_config.get_list(
                "noise_shape", default=None, help="The shape of the generated noise which will be broadcast as needed."
            ),
            "return_mask": hyper_config.get_bool(
                "return_mask", default=False, help="Whether to return both the output and the mask for the noise."
            ),
            "blank_last_dim": hyper_config.get_bool(
                "blank_last_dim",
                default=False,
                help="Apply dropout to the entire last dimension vs choosing for each element of the last dimension.",
            ),
            "supports_masking": True,
        }

    def build(self, input_shapes):
        super(ExtendedDropout, self).build(input_shapes)
        self.mask_shape = list(input_shapes)
        self.mask_shape[0] = None
        if self.blank_last_dim:
            self.noise_shape: list[int | None] = [None for _ in input_shapes]
            self.noise_shape[-1] = 1
            del self.mask_shape[-1]
        elif self.noise_shape is None:
            self.noise_shape = [None for _ in input_shapes]
        self.noise_shape[0] = None
        self.input_shapes = (None, *input_shapes[1:])

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return None
        concrete_inputs_shape = tf.shape(inputs)
        if not tf.executing_eagerly():
            print("noise_shape", inputs.name, self.noise_shape)
        noise_shape = [concrete_inputs_shape[i] if value is None else value for i, value in enumerate(self.noise_shape)]
        return noise_shape
        # return tf.convert_to_tensor(noise_shape, name="noise_shape")

    def call(self, inputs, training=False, mask=None):
        noise_shape = self._get_noise_shape(inputs)
        if self.rate <= 0 or not training:
            result = inputs
            if self.return_mask:
                if mask is None:
                    return inputs, tf.reduce_any(tf.ones_like(inputs, dtype=tf.bool), axis=-1, keepdims=True)
                else:
                    return inputs, mask
            else:
                return inputs
        elif self.rate >= 1:
            result = tf.zeros_like(inputs)
            noise_mask = tf.zeros(noise_shape, dtype=tf.bool)
        else:
            noise = tf.random.uniform(
                noise_shape, minval=0, maxval=1, dtype=self.compute_dtype, seed=self.seed, name="noise"
            )
            noise_mask = noise >= self.rate
            result = tf.cast(noise_mask, dtype=inputs.dtype) * inputs
        if is_debug():
            result = tf.ensure_shape(result, self.input_shapes)
        if self.return_mask:
            if self.blank_last_dim:
                noise_mask = tf.squeeze(noise_mask, -1)
            if mask is not None:
                if tf.rank(mask) < tf.rank(noise_mask):
                    mask = tf.expand_dims(mask, -1)
                noise_mask = tf.math.logical_and(noise_mask, mask, name="combined_mask")
            if is_debug():
                noise_mask = tf.ensure_shape(noise_mask, self.mask_shape)
            else:
                noise_mask = tf.reshape(noise_mask, [tf.shape(noise_mask)[i] for i in range(len(self.mask_shape))])
            return result, noise_mask
        else:
            return result
