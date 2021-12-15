import tensorflow as tf

from mtgml.constants import ACTIVATION_CHOICES
from mtgml.layers.configurable_layer import ConfigurableLayer


class WrappedLayer(ConfigurableLayer):
    def call(self, *args, **kwargs):
        if 'mask' in kwargs:
            del kwargs['mask']
        return self.wrapped_layer(*args, **kwargs)


class WMultiHeadAttention(WrappedLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            'num_heads': hyper_config.get_int('num_heads', min=1, max=64, default=8,
                                              help='The number of separate heads of attention to use.'),
            'key_dims': hyper_config.get_int('key_dims', min=1, max=64, default=16,
                                             help='Size of the attention head for query and key.'),
            'value_dims': hyper_config.get_int('value_dims', min=1, max=64, default=16,
                                               help='Size of the attention head for value.'),
            'dropout': hyper_config.get_float('dropout', min=0, max=0.99, step=0.01, default=0.25,
                                              help='The percent of values to get dropped out'),
            'use_bias': hyper_config.get_bool('use_bias', default=True, help='Use bias in the dense layers'),
            'output_dims': hyper_config.get_int('output_dims', min=8, max=512, default=128,
                                                help='The number of output dimensions from this layer.'),
        }

    def build(self, input_shapes):
        super(WMultiHeadAttention, self).build(input_shapes)
        self.wrapped_layer = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dims,
                                                                value_dim=self.value_dims, dropout=self.dropout,
                                                                use_bias=self.use_bias, output_shape=(self.output_dims,),
                                                                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=self.seed),
                                                                name=self.name)


class WDense(WrappedLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            'dims': hyper_config.get_int('dims', min=8, max=512, default=128,
                                          help='The number of dimensions in the output of this layer.'),
            'activation': hyper_config.get_choice('activation', choices=ACTIVATION_CHOICES, default='selu',
                                                  help='The activation function on the output of the layer.'),
            'use_bias': hyper_config.get_bool('use_bias', default=True, help='Whether to use bias on the output.'),
        }

    def build(self, input_shapes):
        super(WDense, self).build(input_shapes)
        self.wrapped_layer = tf.keras.layers.Dense(self.dims, activation=self.activation, use_bias=self.use_bias,
                                                   kernel_initializer=tf.keras.initializers.GlorotNormal(seed=self.seed),
                                                   name=self.name)


class WDropout(WrappedLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            'rate': hyper_config.get_float('rate', min=0, max=0.99, step=0.01, default=0.5,
                                           help='The percent of values that get replaced with zero.'),
            'noise_shape': hyper_config.get_list('noise_shape', default=None,
                                                  help='The shape of the generated noise which will be broadcast as needed.'),
        }

    def build(self, input_shapes):
        super(WDropout, self).build(input_shapes)
        self.wrapped_layer = tf.keras.layers.Dropout(rate=self.rate, noise_shape=self.noise_shape,
                                                     seed=self.seed, name=self.name)
