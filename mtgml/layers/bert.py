import tensorflow as tf

from mtgml.constants import ACTIVATION_CHOICES
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.mlp import MLP
from mtgml.layers.wrapped import WMultiHeadAttention


class Transformer(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            'attention': hyper_config.get_sublayer(f'Attention', sub_layer_type=WMultiHeadAttention, seed_mod=39,
                                                   help=f'The initial attention layer'),
            'final_mlp': hyper_config.get_sublayer(f'FinalMLP', sub_layer_type=MLP, seed_mod = 47,
                                                   fixed={'use_layer_norm': False, 'use_batch_norm': False},
                                                   help=f'The final transformation.'),
            'layer_norm': tf.keras.layers.LayerNormalization(),
            'supports_masking': True,
        }

    def call(self, inputs, training=False):
        if len(inputs) == 1:
            tokens = inputs
            attended = self.attention(tokens, tokens, training=training)
        else:
            tokens, attention_mask = inputs
            attended = self.attention(tokens, tokens, attention_mask=attention_mask, training=training)
        transformed = self.final_mlp(attended, training=training)
        return self.layer_norm(transformed + tokens)


class BERT(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_layers = hyper_config.get_int('num_hidden_layers', min=0, max=16, default=2,
                                          help='Number of transformer blocks.') + 1

        attention_props = {
            'dropout': hyper_config.get_float('attention dropout_rate', min=0, max=1, default=0.25,
                                                   help='The dropout rate for the attention layers of the transformer blocks.'),
            'num_heads': hyper_config.get_int('num_heads', min=1, max=64, default=4,
                                              help='The number of separate heads of attention to use.'),
            'key_dims': hyper_config.get_int('key_dims', min=1, max=64, default=32,
                                             help='Size of the attention head for query and key.'),
            'value_dims': hyper_config.get_int('value_dims', min=1, max=64, default=32,
                                               help='Size of the attention head for value.'),
            'use_bias': hyper_config.get_bool('use_bias', default=True, help='Use bias in the dense layers'),
            'output_dims': hyper_config.get_int('output_dims', min=8, max=512, default=128,
                                                help='The number of output dimensions from this layer.'),
        }
        dense_props = {
            'num_hidden': hyper_config.get_int('num_hidden_dense', min=0, max=12, default=2, help='The number of hidden dense layers'),
            'dims': hyper_config.get_int('dims', min=8, max=1024, default=512, help='The number of dimensions for the output.'),
            'use_bias': hyper_config.get_bool('use_bias', default=True, help='Whether to add on a bias at each layer.'),
            'activation': hyper_config.get_choice('activation', choices=ACTIVATION_CHOICES, default='selu',
                                                  help='The activation function on the output of the layer.'),
            'Dropout': {'rate': hyper_config.get_float('dense dropout_rate', min=0, max=1, default=0.25,
                                                       help='The dropout rate for the dense layers of the transformer blocks.')},
        }
        return {
            'seq_length': input_shapes[1] if input_shapes is not None else 1,
            'layers': tuple(hyper_config.get_sublayer(f'Transformer_{i}', sub_layer_type=Transformer, seed_mod=23,
                                                      fixed={'FinalMLP': dense_props,
                                                             'Attention': attention_props},
                                                      help=f'The {i}th transformer layer.')
                            for i in range(num_layers)),
            'supports_masking': True,
        }

    def call(self, inputs, mask=None, training=False):
        token_embeds = inputs
        if mask is None:
            mask = tf.ones(tf.shape(token_embeds)[:-1], dtype=tf.bool)
        embeddings = tf.expand_dims(tf.cast(mask, dtype=self.compute_dtype), -1) * (token_embeds)
        attention_mask = tf.logical_and(tf.expand_dims(mask, -1), tf.expand_dims(mask, -2), name='attention_mask')
        for layer in self.layers:
            embeddings = layer((embeddings, attention_mask), training=training)
        return embeddings
