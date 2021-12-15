import tensorflow as tf

from mtgml.config.hyper_config import HyperConfig
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.extended_dropout import ExtendedDropout
from mtgml.layers.mlp import MLP
from mtgml.layers.wrapped import WDropout, WMultiHeadAttention
from mtgml.layers.zero_masked import ZeroMasked
from mtgml.tensorboard.plot_attention_scores import plot_attention_scores

SET_EMBEDDING_CHOICES = ('additive', 'attentive')


class AdditiveSetEmbedding(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        decoding_dropout = hyper_config.get_float('decoding_dropout_rate', min=0, max=0.99, step=0.01, default=0.25,
                                                  help='The percent of values to dropout from the result of dense layers in the decoding step.')
        return {
            'encoder': hyper_config.get_sublayer('Encoder', sub_layer_type=MLP, seed_mod=13,
                                                 help='The mapping from the item embeddings to the embeddings to add.'),
            'decoder': hyper_config.get_sublayer('Decoder', sub_layer_type=MLP, seed_mod=17,
                                                 fixed={"dropout": decoding_dropout},
                                                 help='The mapping from the added item embeddings to the embeddings to return.'),
            'zero_masked': ZeroMasked(HyperConfig(seed=hyper_config.seed * 47)),
            'item_dropout': hyper_config.get_sublayer('ItemDropout', sub_layer_type=ExtendedDropout,
                                                      seed_mod=53, fixed={'all_last_dim': True, 'return_mask': False},
                                                      help='Drops out entire items from the set.'),
            'decoder_dropout': hyper_config.get_sublayer('DecodingDropout', sub_layer_type=WDropout,
                                                         seed_mod=43, fixed={'rate': decoding_dropout,
                                                                             'blank_last_dim': True,
                                                                             'noise_shape': None},
                                                      help='Drops out values from the decoding layers to improve generalizability.'),
            'normalize_sum': hyper_config.get_bool('normalize_sum', default=False,
                                                   help='Average the sum of embeddings by the number of non-masked items.'),
        }

    def call(self, inputs, training=False, mask=None):
        encoded_items = self.encoder(inputs, training=training)
        dropped_inputs = self.zero_masked(self.item_dropout(encoded_items, training=training))
        summed_embeds = tf.math.reduce_sum(dropped_inputs, -2, name='summed_embeds')
        if self.normalize_sum:
            num_valid = tf.math.reduce_sum(tf.cast(tf.keras.layers.Masking().compute_mask(dropped_inputs),
                                                   dtype=self.compute_dtype, name='mask'),
                                           axis=-1, keepdims=True, name='num_valid')
            summed_embeds = tf.math.divide(summed_embeds, num_valid + 1e-09, name='normalized_embeds')
        summed_embeds = self.decoder_dropout(summed_embeds, training=training)
        return self.decoder(summed_embeds, training=training)
        hidden = self.dropout(self.hidden(summed_embeds), training=training)
        return self.output_layer(hidden)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return tf.math.reduce_any(mask, axis=-1)


class AttentiveSetEmbedding(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        decoding_dropout = hyper_config.get_float('decoding_dropout_rate', min=0, max=0.99, step=0.01, default=0.25,
                                                  help='The percent of values to dropout from the result of dense layers in the decoding step.')
        return {
            'encoder': hyper_config.get_sublayer('Encoder', sub_layer_type=MLP, seed_mod=13,
                                                 help='The mapping from the item embeddings to the embeddings to add.'),
            'attention': hyper_config.get_sublayer('Attention', sub_layer_type=WMultiHeadAttention, seed_mod=37,
                                                 help='The mapping from the item embeddings to the embeddings to add.'),
            'decoder': hyper_config.get_sublayer('Decoder', sub_layer_type=MLP, seed_mod=17,
                                                 fixed={"dropout": decoding_dropout},
                                                 help='The mapping from the added item embeddings to the embeddings to return.'),
            'zero_masked': ZeroMasked(HyperConfig(seed=hyper_config.seed * 47)),
            'item_dropout': hyper_config.get_sublayer('ItemDropout', sub_layer_type=ExtendedDropout,
                                                      seed_mod=53, fixed={'all_last_dim': True, 'return_mask': True,
                                                                          'blank_last_dim': False, 'noise_shape': False},
                                                      help='Drops out entire items from the set.'),
            'decoder_dropout': hyper_config.get_sublayer('DecodingDropout', sub_layer_type=WDropout,
                                                         seed_mod=43, fixed={'rate': decoding_dropout,
                                                                             'blank_last_dim': True,
                                                                             'noise_shape': None},
                                                      help='Drops out values from the decoding layers to improve generalizability.'),
            'normalize_sum': hyper_config.get_bool('normalize_sum', default=False,
                                                   help='Average the sum of embeddings by the number of non-masked items.'),
            'log_scores': hyper_config.get_bool('log_scores', default=True,
                                                help='Whether to log an image of the attention scores.')
        }

    def call(self, inputs, training=False, mask=None):
        dropped, dropout_mask = self.item_dropout(inputs, training=training)
        dropout_mask = tf.cast(dropout_mask, tf.bool)
        dropout_mask = tf.math.reduce_any(dropout_mask, axis=-1)
        product_mask = tf.logical_and(tf.expand_dims(dropout_mask, -1), tf.expand_dims(dropout_mask, -2), name='product_mask')
        encoded_items = self.encoder(dropped, training=training)
        encoded_items, attention_scores = self.attention(encoded_items, encoded_items, training=training,
                                                         attention_mask=product_mask,
                                                         return_attention_scores=True)
        dropped_inputs = self.zero_masked(encoded_items, mask=dropout_mask)
        summed_embeds = tf.math.reduce_sum(dropped_inputs, -2, name='summed_embeds')
        if self.normalize_sum:
            num_valid = tf.math.reduce_sum(tf.cast(dropout_mask, dtype=self.compute_dtype, name='mask'),
                                           axis=-1, keepdims=True, name='num_valid')
            summed_embeds = tf.math.divide(summed_embeds, num_valid + 1e-09, name='normalized_embeds')
        summed_embeds = self.decoder_dropout(summed_embeds, training=training)
        # Tensorboard logging
        if tf.summary.should_record_summaries() and self.log_scores:
            images = tf.py_function(plot_attention_scores, inp=(attention_scores, True, self.name), Tout=[tf.uint8 for _ in range(self.attention.num_heads)])
            tf.summary.image(f'{self.name} AttentionScoresHead', images)
        return self.decoder(summed_embeds, training=training)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return tf.math.reduce_any(mask, axis=-1)
