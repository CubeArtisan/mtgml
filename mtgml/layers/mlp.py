import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.wrapped import Dense, Dropout


class MLP(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_hidden = hyper_config.get_int('num_hidden', min=0, max=8, default=1,
                                          help='The number of hidden layers in the MLP.')
        props = {
            'hiddens': [hyper_config.get_sublayer(f'Hidden_{i}', sub_layer_type=Dense, seed_mod=i + 1,
                                                  help=f'The {i+1}{"th" if i > 1 else "nd" if i == 1 else "st"} hidden layer.')
                        for i in range(num_hidden)],
            'final': hyper_config.get_sublayer('Final', sub_layer_type=Dense, seed_mod=19,
                                               help='The last dense layer in the MLP.'),
        }
        if num_hidden > 0:
            props['dropout'] = hyper_config.get_sublayer(f'Dropout', sub_layer_type=Dropout, seed_mod=11,
                                                         help='The dropout applied after each hidden layer.')
        return props

    def call(self, inputs, training=False):
        for hidden in self.hiddens:
            inputs = hidden(inputs, training=training)
            inputs = self.dropout(inputs, training=training)
        return self.final_layer(inputs, training=training)
