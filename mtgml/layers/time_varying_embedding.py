import numpy as np
import tensorflow as tf
from mtgml.config.hyper_config import HyperConfig

from mtgml.constants import ACTIVATION_CHOICES, is_debug
from mtgml.layers.configurable_layer import ConfigurableLayer

class TimeVaryingEmbedding(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config: HyperConfig, input_shapes=None):
        time_shape = hyper_config.get_list('time_shape', default=None,
                                            help='The dimensions of the time space.')
        if not time_shape:
            raise NotImplementedError('You must supply a time shape.')
        return {
            'time_shape': time_shape,
            'dims': hyper_config.get_int('dims', min=8, max=256, default=32,
                                         help='The number of dimensions the cards should be embedded into.'),
            'activation': tf.keras.layers.Activation(hyper_config.get_choice('activation', default='linear',
                                                                             choices=ACTIVATION_CHOICES,
                                                                             help='The activation to apply before combining the weights.')),
            'supports_masking': True,
        }

    def build(self, input_shapes):
        super(TimeVaryingEmbedding, self).build(input_shapes)
        shape = (*self.time_shape, self.dims)
        if self.dims == 3:
            initializer = tf.constant_initializer(np.zeros(shape, dtype=np.float32) + np.log(np.exp([1.0, 0.01, 10.0]) - 1.0))
        else:
            initializer = tf.constant_initializer(np.log(np.exp(np.ones(shape, dtype=np.float32)) - 1.0))
        self.embeddings = self.add_weight('embeddings', shape=shape, initializer=initializer, trainable=True)

    def call(self, inputs, training=False, mask=None):
        if isinstance(inputs, (list, tuple)):
            coords, coord_weights = inputs
            component_embedding_values = self.activation(tf.gather_nd(self.embeddings, coords, name='component_embedding_values'),
                                                         training=training)
            result = tf.einsum('...xe,...x->...e', component_embedding_values, coord_weights,
                               name='embedding_values')
            if is_debug():
                result = tf.ensure_shape(result, (None, *coords.shape[1:-2], self.dims))
            return result
        else:
            return tf.gather_nd(self.embeddings, inputs, name='embedding_values')
