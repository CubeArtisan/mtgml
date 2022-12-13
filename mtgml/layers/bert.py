import tensorflow as tf

from mtgml.constants import ACTIVATION_CHOICES
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.extended_dropout import ExtendedDropout
from mtgml.layers.mlp import MLP
from mtgml.layers.wrapped import WDense, WMultiHeadAttention


class Transformer(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            'attention': hyper_config.get_sublayer(f'Attention', sub_layer_type=WMultiHeadAttention, seed_mod=39,
                                                   fixed={
                                                       'use_causal_mask': hyper_config.get_bool('use_causal_mask', default=False,
                                                                                                help='Ensure items only attend to items that came before them.'),
                                                   },
                                                   help=f'The initial attention layer'),
            'final_mlp': hyper_config.get_sublayer(f'FinalMLP', sub_layer_type=MLP, seed_mod=47,
                                                   fixed={'use_layer_norm': False, 'use_batch_norm': False},
                                                   help=f'The final transformation.'),
            'layer_norm': tf.keras.layers.LayerNormalization(),
            'use_causal_mask': hyper_config.get_bool('use_causal_mask', default=False,
                                                     help='Ensure items only attend to items that came before them.'),
            'final_dropout': hyper_config.get_sublayer('FinalDropout', sub_layer_type=ExtendedDropout, seed_mod=93,
                                                       fixed={'noise_shape': None, 'return_mask': False,
                                                              'blank_last_dim': False},
                                                       help='The dropout layer after applying the FinalMLP.'),
            'supports_masking': True,
        }

    def call(self, inputs, training=False, mask=None):
        if len(inputs) == 1:
            tokens = inputs
            attended = self.attention(tokens, tokens, training=training, mask=mask, use_causal_mask=self.use_causal_mask)
        else:
            tokens, attention_mask = inputs
            attended = self.attention(tokens, tokens, attention_mask=attention_mask, use_causal_mask=self.use_causal_mask,
                                      training=training, mask=mask)
        transformed = self.final_mlp(attended, training=training, mask=mask)
        transformed = self.final_dropout(transformed, training=training, mask=mask)
        return self.layer_norm(transformed + tokens, training=training)


class BERT(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_layers = hyper_config.get_int('num_hidden_layers', min=0, max=16, default=2,
                                          help='Number of transformer blocks.') + 1
        use_causal_mask = hyper_config.get_bool('use_causal_mask', default=False,
                                                help='Ensure items only attend to items that came before them.')
        original_token_dims = hyper_config.get_int('token_stream_dims', default=128, min=8, max=1024,
                                                   help='The size of the token embeddings passed between layers.')
        use_bias = hyper_config.get_bool('use_bias', default=True, help='Use bias in the dense layers')
        attention_output_dims = hyper_config.get_int('output_dims', min=8, max=1024, default=128,
                                                     help='The number of output dimensions from this layer.')
        attention_props = {
            'dropout': hyper_config.get_float('attention dropout_rate', min=0, max=1, default=0.25,
                                              help='The dropout rate for the attention layers of the transformer blocks.'),
            'num_heads': hyper_config.get_int('num_heads', min=1, max=64, default=4,
                                              help='The number of separate heads of attention to use.'),
            'key_dims': hyper_config.get_int('key_dims', min=1, max=64, default=32,
                                             help='Size of the attention head for query and key.'),
            'value_dims': hyper_config.get_int('value_dims', min=1, max=64, default=32,
                                               help='Size of the attention head for value.'),
            'use_bias': use_bias,
            'output_dims': attention_output_dims,
        }
        num_hidden_dense = hyper_config.get_int('num_hidden_dense', min=0, max=12, default=1, help='The number of hidden dense layers')
        dense_dropout_rate = hyper_config.get_float('dense dropout_rate', min=0, max=1, default=0.25,
                                                    help='The dropout rate for the dense layers of the transformer blocks.')

        activation= hyper_config.get_choice('dense_activation', choices=ACTIVATION_CHOICES, default='selu',
                                            help='The activation function on the output of the layer.')
        dense_props = {
            'num_hidden': num_hidden_dense,
            'Final': {
                'dims': original_token_dims,
                'use_bias': use_bias, 'activation': activation,
            },
        } | {
            f'Dropout_{i}': {'rate': dense_dropout_rate} for i in range(num_hidden_dense)
        } | {
            f'Hidden_{i}': {'dims': attention_output_dims * 2 ** (i + 1), 'use_bias': use_bias, 'activation': activation}
            for i in range(num_hidden_dense)
        }
        return {
            'initial_dropout': hyper_config.get_sublayer('InitialDropout', sub_layer_type=ExtendedDropout, seed_mod=29,
                                                         fixed={'return_mask': True, 'noise_shape': None},
                                                         help='The dropout to apply to the tokens before any other operations.'),
            'seq_length': input_shapes[1] if input_shapes is not None else 1,
            'transform_initial_tokens': hyper_config.get_sublayer('TransformInitialTokens', sub_layer_type=WDense,
                                                                  fixed={'dims': original_token_dims, 'use_bias': use_bias,
                                                                         'activation': 'linear'}, seed_mod=37,
                                                                  help='The layer to upscale or downscale the token embeddings.'),
            'layers': tuple(hyper_config.get_sublayer(f'Transformer_{i}', sub_layer_type=Transformer, seed_mod=23,
                                                      fixed={'FinalMLP': dense_props,
                                                             'Attention': attention_props,
                                                             'FinalDropout': {'rate': dense_dropout_rate},
                                                             'use_causal_mask': use_causal_mask,
                                                             'dense_dropout': dense_dropout_rate},
                                                      help=f'The {i}th transformer layer.')
                            for i in range(num_layers)),
            'supports_masking': True,
        }

    def call(self, inputs, mask=None, training=False):
        token_embeds = inputs
        if mask is None:
            mask = tf.ones(tf.shape(token_embeds)[:-1], dtype=tf.bool)
        token_embeds, dropout_mask = self.initial_dropout(token_embeds, training=training, mask=mask)
        if len(dropout_mask.shape) > len(mask.shape):
            mask = tf.math.reduce_any(dropout_mask, axis=-1)
        else:
            mask = dropout_mask
        token_embeds = self.transform_initial_tokens(token_embeds, training=training, mask=mask)
        embeddings = tf.expand_dims(tf.cast(mask, dtype=self.compute_dtype), -1) * token_embeds
        attention_mask = tf.logical_and(tf.expand_dims(mask, -1), tf.expand_dims(mask, -2), name='attention_mask')
        for layer in self.layers:
            embeddings = layer((embeddings, attention_mask), training=training, mask=mask)
        return embeddings
