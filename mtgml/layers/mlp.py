import tensorflow as tf

from mtgml.constants import ACTIVATION_CHOICES
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.wrapped import WDense, WDropout


class MLP(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_hidden = hyper_config.get_int('num_hidden', min=0, max=8, default=1,
                                          help='The number of hidden layers in the MLP.')
        dense_props = {
            'dims': hyper_config.get_int('dims', min=8, max=1024, default=512, help='The number of dimensions for the output.'),
            'use_bias': hyper_config.get_bool('use_bias', default=True, help='Whether to add on a bias at each layer.'),
            'activation': hyper_config.get_choice('activation', choices=ACTIVATION_CHOICES, default='selu',
                                                  help='The activation function on the output of the layer.'),
        }
        props = {
            'layers': tuple(hyper_config.get_sublayer(f'Dense_{i}', sub_layer_type=WDense, seed_mod=i + 1,
                                                      fixed=dense_props, 
                                                      help=f'The {i+1}{"th" if i > 1 else "nd" if i == 1 else "st"} dense layer.')
                        for i in range(num_hidden + 1)),
            'layer_norms': [tf.keras.layers.LayerNormalization(name=f'LayerNorm{i}') for i in range(num_hidden + 1)]
                          if hyper_config.get_bool('use_layer_norm', default=False, help='Use layer normalization between layers')
                          else None,
            'batch_norms': [tf.keras.layers.BatchNormalization(name=f'BatchNorm{i}') for i in range(num_hidden + 1)]
                          if hyper_config.get_bool('use_batch_norm', default=False, help='Use batch normalization between layers')
                          else None,
            'dropout': hyper_config.get_sublayer(f'Dropout', sub_layer_type=WDropout, seed_mod=11,
                                                 fixed={'noise_shape': None},
                                                 help='The dropout applied after each hidden layer.')
        }
        return props

    def call(self, inputs, training=False):
        for i, hidden in enumerate(self.layers):
            inputs = hidden(inputs, training=training)
            inputs = self.dropout(inputs, training=training)
            if self.layer_norms:
                inputs = self.layer_norms[i](inputs, training=training)
            if self.batch_norms:
                inputs = self.batch_norm[i](inputs, training=training)
        return inputs
