import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.extended_dropout import ExtendedDropout
from mtgml.layers.wrapped import WDense, WDropout


class MLP(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_hidden = hyper_config.get_int('num_hidden', min=0, max=8, default=1,
                                          help='The number of hidden layers in the MLP.')
        props = {
            'hiddens': tuple(hyper_config.get_sublayer(f'Hidden_{i}', sub_layer_type=WDense, seed_mod=i + 1,
                                                       help=f'The {i+1}{"th" if i > 1 else "nd" if i == 1 else "st"} hidden layer.')
                        for i in range(num_hidden)),
            'dropouts': tuple(hyper_config.get_sublayer(f'Dropout_{i}', sub_layer_type=ExtendedDropout, seed_mod=11+3*i,
                                                        fixed={'noise_shape': None, 'return_mask': False, 'blank_last_dim': False},
                                                        help=f'The dropout applied after the {i+1}{"th" if i > 1 else "nd" if i == 1 else "st"} hidden layer.')
                              for i in range(num_hidden)),
            'final': hyper_config.get_sublayer('Final', sub_layer_type=WDense, seed_mod=19,
                                               help='The last dense layer in the MLP.'),
            'supports_masking': True,
        }
        return props

    def call(self, inputs, training=False, mask=None):
        for hidden, dropout in zip(self.hiddens, self.dropouts):
            inputs = dropout(hidden(inputs, training=training), training=training)
        return self.final(inputs, training=training)
