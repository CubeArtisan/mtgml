import tensorflow as tf

from mtgml.constants import LARGE_INT
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
                                                 fixed={"Dropout": {'rate': decoding_dropout}},
                                                 help='The mapping from the added item embeddings to the embeddings to return.'),
            'zero_masked': ZeroMasked(HyperConfig(seed=hyper_config.seed * 47)),
            'item_dropout': hyper_config.get_sublayer('ItemDropout', sub_layer_type=ExtendedDropout,
                                                      seed_mod=53, fixed={'all_last_dim': True, 'return_mask': False,
                                                                          'blank_last_dim': True},
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
        positional_reduction = hyper_config.get_bool('positional_reduction', default=True,
                                                     help='Whether to use a positional reduction instead of sum.')
        attention_output_dims = hyper_config.get_int('atten_output_dims', min=8, max=512, step=8, default=64,
                                                     help='The number of dimensions in the output of the attention layer.')
        return {
            'encoder': hyper_config.get_sublayer('Encoder', sub_layer_type=MLP, seed_mod=13,
                                                 help='The mapping from the item embeddings to the embeddings to add.'),
            'attention': hyper_config.get_sublayer('Attention', sub_layer_type=WMultiHeadAttention, seed_mod=37,
                                                   fixed={'output_dims': attention_output_dims},
                                                   help='The mapping from the item embeddings to the embeddings to add.'),
            'decoder': hyper_config.get_sublayer('Decoder', sub_layer_type=MLP, seed_mod=17,
                                                 fixed={"Dropout": {'rate': decoding_dropout}},
                                                 help='The mapping from the added item embeddings to the embeddings to return.'),
            'zero_masked': ZeroMasked(HyperConfig(seed=hyper_config.seed * 47)),
            'item_dropout': hyper_config.get_sublayer('ItemDropout', sub_layer_type=ExtendedDropout,
                                                      seed_mod=53, fixed={'all_last_dim': True, 'return_mask': True,
                                                                          'blank_last_dim': False, 'noise_shape': None},
                                                      help='Drops out entire items from the set.'),
            'decoder_dropout': hyper_config.get_sublayer('DecodingDropout', sub_layer_type=WDropout,
                                                         seed_mod=43, fixed={'rate': decoding_dropout,
                                                                             'blank_last_dim': True,
                                                                             'noise_shape': None},
                                                      help='Drops out values from the decoding layers to improve generalizability.'),
            'normalize_sum': hyper_config.get_bool('normalize_sum', default=False,
                                                   help='Average the sum of embeddings by the number of non-masked items.'),
            'log_scores': hyper_config.get_bool('log_scores', default=True,
                                                help='Whether to log an image of the attention scores.'),
            'positional_reduction': positional_reduction,
            'atten_output_dims': attention_output_dims,
        }

    def build(self, input_shapes):
        super(AttentiveSetEmbedding, self).build(input_shapes)
        # if self.positional_reduction:
        #     self.position_weights = self.add_weight('positional_weights', initializer=tf.ones_initializer(),
        #                                             shape=(input_shapes[-2],input_shapes[-2]), trainable=True)
        self.final_atten_shape = (-1, *input_shapes[1:-1], self.atten_output_dims)
        self.flattened_shape = (-1, *input_shapes[-2:])
        self.mask_shape = (-1, *self.flattened_shape[1:-1])
        self.original_mask = (-1, *input_shapes[1:-1])

    def call(self, inputs, training=False, mask=None):
        inputs = tf.reshape(inputs, self.flattened_shape)
        if mask is not None:
            mask = tf.reshape(mask, self.mask_shape)
        dropped, dropout_mask = self.item_dropout(inputs, training=training, mask=mask)
        dropout_mask = tf.cast(dropout_mask, tf.bool)
        dropout_mask = tf.math.reduce_any(dropout_mask, axis=-1)
        counting_range = tf.range(tf.shape(dropout_mask)[-1])
        diffs = tf.expand_dims(counting_range, 1) - tf.expand_dims(counting_range, 0)
        product_mask = tf.logical_and(tf.expand_dims(dropout_mask, -1), tf.expand_dims(dropout_mask, -2), name='product_mask')
        product_mask = tf.logical_and(product_mask, diffs >= 0)
        encoded_items = self.encoder(dropped, training=training)
        encoded_items, attention_scores = self.attention(encoded_items, encoded_items, training=training,
                                                         attention_mask=product_mask,
                                                         return_attention_scores=True)
        # if self.positional_reduction:
        #     position_weights = tf.gather(self.position_weights, tf.reduce_sum(tf.cast(dropout_mask, tf.int32), axis=-1))
        #     position_weights = position_weights + tf.constant(LARGE_INT, dtype=self.compute_dtype) * tf.cast(dropout_mask, dtype=self.compute_dtype)
        #     position_weights = tf.nn.softmax(position_weights, axis=-1)
        #     encoded_items = tf.expand_dims(position_weights, -1) * encoded_items
        #     attention_scores = tf.expand_dims(tf.expand_dims(position_weights, -2), -1) * attention_scores
        encoded_items = tf.reshape(encoded_items, self.final_atten_shape)
        dropped_inputs = self.zero_masked(encoded_items, mask=tf.reshape(dropout_mask, self.original_mask))
        summed_embeds = tf.math.reduce_sum(dropped_inputs, -2, name='summed_embeds')
        if self.normalize_sum:
            num_valid = tf.math.reduce_sum(tf.cast(dropout_mask, dtype=self.compute_dtype, name='mask'),
                                           axis=-1, keepdims=True, name='num_valid')
            summed_embeds = tf.math.divide(summed_embeds, num_valid + 1e-09, name='normalized_embeds')
        summed_embeds = self.decoder_dropout(summed_embeds, training=training)
        # Tensorboard logging
        # if tf.summary.should_record_summaries() and self.log_scores:
        #     images = tf.py_function(plot_attention_scores, inp=(attention_scores, dropout_mask, True, self.name), Tout=[tf.uint8 for _ in range(self.attention.num_heads)])
        #     tf.summary.image(f'{self.name} AttentionScoresHead', images)
        return self.decoder(summed_embeds, training=training)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return tf.math.reduce_any(mask, axis=-1)
