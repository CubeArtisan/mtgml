import tensorflow as tf


class TimeVaryingEmbedding(tf.keras.layers.Layer):
    @classmethod
    def get_properties(cls, hyper_config, input_shape):
        time_shape = hyper_config.get_array('time_shape', default=None,
                                            help='The dimensions of the time space.')
        if not time_shape and input_shape:
            raise NotImplementedError('You must supply a time shape.')
        return {
            'time_shape': time_shape,
            'dims': hyper_config.get_int('dims', min=8, max=256, default=32,
                                         help='The number of dimensions the cards should be embedded into.'),
        }

    def build(self, input_shape):
        super(self, TimeVaryingEmbedding).build(input_shapes)
        self.embeddings = self.add_weight('embeddings', shape=(*self.time_shape, self.dims),
                                          initializer=tf.keras.initializers.GlorotNormal(seed=self.seed),
                                          trainable=True)

    def call(self, inputs, training=False):
        if isinstance(inputs, (list, tuple)):
            coords, coord_weights = inputs
            component_embedding_values = tf.gather_nd(self.embeddings, coords, name='component_embedding_values')
            return tf.einsum('...xe,...x->...e', component_embedding_values, coord_weights,
                             name='embedding_values')
        else:
            return tf.gather_nd(self.embeddings, inputs, name='embedding_values')
