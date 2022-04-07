from mtgml.layers.item_embedding import ItemEmbedding
import tensorflow as tf

from mtgml.constants import ACTIVATION_CHOICES
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.mlp import MLP
from mtgml.layers.wrapped import WDense, WMultiHeadAttention
from mtgml.models.adj_mtx import AdjMtxReconstructor


class Transformer(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            'attention': hyper_config.get_sublayer(f'Attention', sub_layer_type=WMultiHeadAttention, seed_mod=39,
                                                   help=f'The initial attention layer'),
            'final_mlp': hyper_config.get_sublayer(f'FinalMLP', sub_layer_type=MLP, seed_mod = 47,
                                                   fixed={'use_layer_norm': True, 'use_batch_norm': False},
                                                   help=f'The final transformation.'),
            'supports_masking': True,
        }

    def call(self, inputs, training=False):
        if len(inputs) == 1:
            tokens = inputs
            attended = self.attention(tokens, tokens, training=training)
        else:
            tokens, attention_mask = inputs
            attended = self.attention(tokens, tokens, attention_mask=attention_mask, training=training)
        return self.final_mlp(attended, training=training)


class DocumentEncoder(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_layers = hyper_config.get_int('num_hidden_layers', min=0, max=16, default=2,
                                          help='Number of transformer blocks.') + 1

        attention_props = {
            'dropout': hyper_config.get_float('attention dropout_rate', min=0, max=1, default=0.1,
                                                   help='The dropout rate for the attention layers of the transformer blocks.'),
            'num_heads': hyper_config.get_int('num_heads', min=1, max=64, default=4,
                                              help='The number of separate heads of attention to use.'),
            'key_dims': hyper_config.get_int('key_dims', min=1, max=64, default=64,
                                             help='Size of the attention head for query and key.'),
            'value_dims': hyper_config.get_int('value_dims', min=1, max=64, default=64,
                                               help='Size of the attention head for value.'),
            'use_bias': hyper_config.get_bool('use_bias', default=True, help='Use bias in the dense layers'),
            'output_dims': hyper_config.get_int('output_dims', min=8, max=512, default=256,
                                                help='The number of output dimensions from this layer.'),
        }
        dense_props = {
            'num_hidden': hyper_config.get_int('num_hidden_dense', min=0, max=12, default=1, help='The number of hidden dense layers'),
            'dims': hyper_config.get_int('dims', min=8, max=1024, default=1024, help='The number of dimensions for the output.'),
            'use_bias': hyper_config.get_bool('use_bias', default=True, help='Whether to add on a bias at each layer.'),
            'activation': hyper_config.get_choice('activation', choices=ACTIVATION_CHOICES, default='selu',
                                                  help='The activation function on the output of the layer.'),
            'Dropout': {'rate': hyper_config.get_float('dense dropout_rate', min=0, max=1, default=0.1,
                                                       help='The dropout rate for the dense layers of the transformer blocks.')},
        }
        return {
            'seq_length': input_shapes[0][1] if input_shapes is not None else 1,
            'layers': tuple(hyper_config.get_sublayer(f'Transformer_{i}', sub_layer_type=Transformer, seed_mod=23,
                                                      fixed={'FinalMLP': dense_props,
                                                             'Attention': attention_props},
                                                      help=f'The {i}th transformer layer.')
                            for i in range(num_layers)),
            'supports_masking': True,
        }

    def call(self, inputs, mask=None, training=False):
        tokens, token_embeddings, positional_embeddings = inputs
        embedded_tokens = tf.gather(token_embeddings, tokens, name='embeded_tokens')
        tokens_mask = tokens > 0
        if mask is not None:
            tokens_mask = tf.math.logical_and(tokens_mask, mask)
        embeddings = tf.expand_dims(tf.cast(tokens_mask, dtype=self.compute_dtype), -1) * (embedded_tokens + positional_embeddings)
        attention_mask = tf.logical_and(tf.expand_dims(tokens_mask, -1), tf.expand_dims(tokens_mask, -2), name='attention_mask')
        causal_mask = tf.linalg.band_part(tf.ones((self.seq_length, self.seq_length), dtype=self.compute_dtype), -1, 0)
        attention_mask = tf.logical_and(causal_mask > 0, attention_mask)
        for layer in self.layers:
            embeddings = layer((embeddings, attention_mask), training=training)
        return embeddings


class MaskTokens(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            'rate': hyper_config.get_float('rate', min=0, max=1, default=0.15,
                                           help='The mean proportion of tokens to replace with the MASK token.'),
            'supports_masking': True,
        }

    def call(self, inputs, training=False, mask=None):
        noise = tf.random.uniform(tf.shape(inputs), minval=0, maxval=1, dtype=self.compute_dtype,
                                  seed=self.seed, name='noise')
        to_mask = noise <= self.rate
        if mask is not None:
            to_mask = tf.math.logical_and(mask, to_mask, name='to_mask')
        return tf.where(to_mask, tf.ones_like(inputs), inputs), to_mask


class ReconstructToken(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        if input_shapes is None:
            embedding_dims = 1
        else:
            embedding_dims = input_shapes[1][-1]
        return {
            'dense': hyper_config.get_sublayer('ToEmbeddingDims', sub_layer_type=WDense, seed_mod=53,
                                               fixed={'dims': embedding_dims}, help='The dense layer to get the dims the same as the embeddings'),
            'layer_norm': tf.keras.layers.LayerNormalization(name='PreEmbeddingNorm'),
            'supports_masking': True,
        }

    def build(self, input_shapes):
        super(ReconstructToken, self).build(input_shapes)
        self.bias = self.add_weight('bias', shape=(input_shapes[1][0],), initializer=tf.zeros_initializer(), trainable=True)

    def call(self, inputs, training=False):
        tokens, embeddings = inputs
        transformed = self.dense(tokens, training=training)
        multiplied = tf.linalg.matmul(transformed, embeddings, transpose_b=True, name='multiplied')
        return tf.nn.bias_add(multiplied, self.bias, name='biased')


class MaskedModel(ConfigurableLayer, tf.keras.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            'mask_tokens': hyper_config.get_sublayer('MaskTokens', sub_layer_type=MaskTokens, seed_mod=3,
                                                     help='The amount of tokens to replace with the MASK token.'),
            'encode_tokens': hyper_config.get_sublayer('EncodeTokens', sub_layer_type=DocumentEncoder, seed_mod=5,
                                                       help='Process the tokens to figure out their meaning.'),
            'reconstruct_tokens': hyper_config.get_sublayer('ReconstructTokens', sub_layer_type=ReconstructToken, seed_mod=7,
                                                            help='Try to figure out the original identity of each token.'),
        }

    def __init__(self, *args, **kwargs):
        kwargs.update(trainable=False)
        super(MaskedModel, self).__init__(*args, **kwargs)

    def call(self, inputs, training=False, mask=None):
        tokens, token_embeddings, positional_embeddings = inputs
        masked_tokens, masked = self.mask_tokens(tokens, training=training, mask=tokens > 0)
        encoded_tokens = self.encode_tokens((masked_tokens, token_embeddings, positional_embeddings), training=training, mask=tokens > 0)
        reconstructed = self.reconstruct_tokens((encoded_tokens, token_embeddings), training=training)
        reconstruction_token_losses = tf.keras.metrics.sparse_categorical_crossentropy(tokens, tf.nn.softmax(reconstructed) + 1e-10)
        mask = tf.cast(tokens > 0, dtype=self.compute_dtype)
        reconstruction_example_losses = tf.reduce_sum(mask * reconstruction_token_losses, -1)
        reconstruction_loss = tf.nn.compute_average_loss(reconstruction_example_losses)
        self.add_loss(reconstruction_loss)

        tf.summary.scalar('reconstruction_loss', reconstruction_loss)
        self.add_metric(reconstruction_example_losses, 'reconstruction_loss')
        chosen = tf.math.argmax(reconstructed, axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_sum(mask * tf.cast(tokens == chosen, dtype=self.compute_dtype)) / tf.reduce_sum(mask)
        tf.summary.scalar('accuracy', accuracy)
        float_masked = tf.cast(masked, dtype=self.compute_dtype)
        masked_accuracy = tf.reduce_sum(float_masked * tf.cast(tokens == chosen, dtype=self.compute_dtype)) / tf.reduce_sum(float_masked)
        tf.summary.scalar('masked_accuracy', masked_accuracy)
        prob_correct = tf.reduce_sum(mask * tf.gather(tf.nn.softmax(reconstructed), tokens, batch_dims=2)) / tf.reduce_sum(mask)
        tf.summary.scalar('prob_correct', prob_correct)
        masked_prob_correct = tf.reduce_sum(float_masked * tf.gather(tf.nn.softmax(reconstructed), tokens, batch_dims=2)) / tf.reduce_sum(float_masked)
        tf.summary.scalar('masked_prob_correct', masked_prob_correct)
        return reconstructed, masked, reconstruction_loss


class Electra(ConfigurableLayer, tf.keras.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        embed_dims = hyper_config.get_int('embed_dims', min=8, max=512, default=64,
                                          help='The number of dimensions for the token embeddings')
        num_tokens = hyper_config.get_int('num_tokens', default=None, help='The number of tokens in the vocab including null and mask')
        return {
            'num_tokens': num_tokens,
            'token_embeds': hyper_config.get_sublayer('TokenEmbeddings', sub_layer_type=ItemEmbedding, seed_mod=11,
                                                      fixed={'dims': embed_dims, 'num_items': num_tokens},
                                                      help='The embeddings for the tokens.'),
            'position_embeds': hyper_config.get_sublayer('PositionalEmbeddings', sub_layer_type=ItemEmbedding, seed_mod=11,
                                                         fixed={'dims': embed_dims, 'num_items': input_shapes[1] if input_shapes is not None else 1},
                                                         help='The embeddings for the positions.'),
            'generator': hyper_config.get_sublayer('Generator', sub_layer_type=MaskedModel, seed_mod=61,
                                                   help='The generator that replaces some tokens with likely replacements.'),
            'encoder': hyper_config.get_sublayer('CardEncoder', sub_layer_type=DocumentEncoder, seed_mod=73,
                                                 help='Encodes the sampled card tokens.'),
            'hidden': hyper_config.get_sublayer('HiddenDense', sub_layer_type=WDense, seed_mod=83,
                                                      help='Hidden layer for determining if a token was replaced or not.'),
            'is_replaced': hyper_config.get_sublayer('IsReplaced', sub_layer_type=WDense, seed_mod=87,
                                                     fixed={'dims': 1, 'activation': 'sigmoid'},
                                                     help='Determine whether the token has been replaced or not.')
        }

    def build(self, input_shapes):
        super(Electra, self).build(input_shapes)
        self.token_embeds.build(input_shapes=(None, None))
        self.position_embeds.build(input_shapes=(None, None))

    def call(self, inputs, training=False):
        generated, masked_tokens, reconstruction_loss = self.generator((inputs, self.token_embeds.embeddings,
                                                                        self.position_embeds.embeddings), training=training)
        flat_generated = tf.reshape(generated, (-1, self.num_tokens))
        flat_sampled = tf.random.categorical(tf.math.log(tf.nn.softmax(flat_generated)),
                                             1, dtype=inputs.dtype, seed=self.seed, name='flat_sampled')
        sampled = tf.where(tf.logical_and(masked_tokens, inputs > 0),
                           tf.reshape(flat_sampled, tf.shape(inputs)), inputs)
        mask = tf.cast(inputs > 0, dtype=self.compute_dtype)
        sampled_encoded =  self.encoder((inputs, self.token_embeds.embeddings,
                                        self.position_embeds.embeddings), training=training, mask=inputs > 0)
        hidden = self.hidden(sampled_encoded, training=training)
        pred_replaced = self.is_replaced(hidden, training=training)
        pred_replaced = tf.concat([1 - pred_replaced, pred_replaced], axis=-1)
        was_replaced = inputs != sampled
        was_replaced = tf.cast(tf.stack([inputs == sampled, inputs != sampled], axis=-1), dtype=self.compute_dtype)
        pred_correct = tf.reduce_sum(was_replaced * pred_replaced, axis=-1) + 1e-05
        token_losses = -(mask + tf.cast(masked_tokens, dtype=self.compute_dtype)) * tf.math.log(tf.where(inputs > 0, pred_correct, tf.ones_like(pred_correct)))
        sample_losses = tf.reduce_sum(token_losses, axis=-1) / tf.reduce_sum(mask + tf.cast(masked_tokens, dtype=self.compute_dtype), axis=-1)
        sample_loss = tf.nn.compute_average_loss(sample_losses)
        self.add_loss(sample_loss)

        separation = (tf.reduce_sum(pred_replaced[:, :, 1] * was_replaced[:, :, 1]) / tf.reduce_sum(was_replaced[:, :, 1])
                        - tf.reduce_sum(pred_replaced[:, :, 1] * was_replaced[:, :, 0] * mask) / tf.reduce_sum(was_replaced[:, :, 0] * mask))
        tf.summary.scalar('sample_loss', sample_loss)
        self.add_metric(sample_loss, 'sample_loss')
        tf.summary.histogram('probs', tf.gather(tf.nn.softmax(generated), inputs, batch_dims=2))
        tf.summary.histogram('pred_replaced', pred_replaced[:, :, 1])
        tf.summary.scalar('separation', separation)
        pred_replaced_masked = mask * pred_replaced[:, :, 0]
        mean_pred_replaced = tf.reduce_sum(pred_replaced_masked) / tf.reduce_sum(mask)
        diff_from_mean_pred_replaced = mask * (pred_replaced_masked - mean_pred_replaced)
        var_pred_replaced = tf.reduce_sum(diff_from_mean_pred_replaced * diff_from_mean_pred_replaced) / tf.reduce_sum(mask)
        # self.add_loss(-100 * var_pred_replaced)
        tf.summary.scalar('var_pred_replaced', var_pred_replaced)
        accuracy = tf.reduce_sum(tf.expand_dims(mask, -1) * pred_replaced * was_replaced) / tf.reduce_sum(mask)
        tf.summary.scalar('prob_correct', accuracy)
        error_classes = tf.reduce_sum(mask * was_replaced[:, :, 1]) / tf.reduce_sum(mask)
        tf.summary.scalar('prob_replaced', error_classes)
        prob_mask_correct = tf.reduce_sum(pred_replaced[:, :, 1] * was_replaced[:, :, 1]) / tf.reduce_sum(was_replaced[:, :, 1])
        tf.summary.scalar('masked_prob_correct', prob_mask_correct)
        mask_accuracy = tf.reduce_sum(tf.cast(pred_replaced[:, :, 1] > 0.5, dtype=self.compute_dtype) * was_replaced[:, :, 1]) \
                        / tf.reduce_sum(was_replaced[:, :, 1])
        tf.summary.scalar('replaced_accuracy', mask_accuracy)
        return sample_loss + reconstruction_loss


class CombinedCardModel(ConfigurableLayer, tf.keras.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_cards = hyper_config.get_int('num_cards', default=None, help='The number of items that must be embedded. Should be 1 + the max index expected to see.')
        num_tokens = hyper_config.get_int('num_tokens', default=None, help='The number of tokens in the vocab including null and mask')
        embed_dims = hyper_config.get_int('embed_dims', min=8, max=512, default=64,
                                          help='The number of dimensions for the token embeddings')
        card_token_map = tf.constant(hyper_config.get_list('card_token_map', default=[], help='The map from cards to token sequences.'), dtype=tf.int32)
        return {
            'token_embeds': hyper_config.get_sublayer('TokenEmbeddings', sub_layer_type=ItemEmbedding, seed_mod=11,
                                                      fixed={'dims': embed_dims, 'num_items': num_tokens},
                                                      help='The embeddings for the tokens.'),
            'position_embeds': hyper_config.get_sublayer('PositionalEmbeddings', sub_layer_type=ItemEmbedding, seed_mod=11,
                                                         fixed={'dims': embed_dims, 'num_items':card_token_map.shape[1]},
                                                         help='The embeddings for the positions.'),
            'card_token_map': card_token_map,
            'card_text': hyper_config.get_sublayer('CardTextModel', sub_layer_type=MaskedModel, seed_mod=91,
                                                   fixed={'num_tokens': num_tokens},
                                                   help='The model for learning card representations.'),
            'cube_adj_mtx_reconstructor': hyper_config.get_sublayer('CubeAdjMtx', sub_layer_type=AdjMtxReconstructor,
                                                                    fixed={'num_cards': num_cards}, seed_mod=43,
                                                                    help='The model to reconstruct the cube adjacency matrix'),
            'deck_adj_mtx_reconstructor': hyper_config.get_sublayer('DeckAdjMtx', sub_layer_type=AdjMtxReconstructor,
                                                                    fixed={'num_cards': num_cards}, seed_mod=47,
                                                                    help='The model to reconstruct the deck adjacency matrix'),
        }
    def build(self, input_shapes):
        super(CombinedCardModel, self).build(input_shapes)
        self.token_embeds.build(input_shapes=(None, None))
        self.position_embeds.build(input_shapes=(None, None))

    def call(self, inputs, training=False):
        cube_card_indices, cube_adj_row = inputs[0]
        deck_card_indices, deck_adj_row = inputs[1]
        cube_card_indices = tf.cast(cube_card_indices, dtype=tf.int32)
        deck_card_indices = tf.cast(deck_card_indices, dtype=tf.int32)
        cube_tokens = tf.gather(self.card_token_map, cube_card_indices, name='cube_tokens')
        deck_tokens = tf.gather(self.card_token_map, deck_card_indices, name='deck_tokens')
        card_tokens = tf.concat([cube_tokens, deck_tokens], axis=0)
        generated, masked_tokens, card_text_loss = self.card_text((card_tokens, self.token_embeds.embeddings,
                                                                   self.position_embeds.embeddings), training=training)
        cube_card_embeds =  self.card_text.encode_tokens((cube_tokens, self.token_embeds.embeddings,
                                                          self.position_embeds.embeddings),
                                                   training=training, mask=cube_tokens > 0)[:, 0, :]
        deck_card_embeds =  self.card_text.encode_tokens((deck_tokens, self.token_embeds.embeddings,
                                                          self.position_embeds.embeddings),
                                                   training=training, mask=deck_tokens > 0)[:, 0, :]
        cube_loss = self.cube_adj_mtx_reconstructor((tf.range(tf.shape(cube_tokens)[0], dtype=tf.int32),
                                                     cube_adj_row, cube_card_embeds), training=training)
        deck_loss = self.deck_adj_mtx_reconstructor((tf.range(tf.shape(deck_tokens)[0], dtype=tf.int32),
                                                     deck_adj_row, deck_card_embeds), training=training)
        loss = cube_loss + deck_loss + card_text_loss
        tf.summary.scalar('loss', loss)
        return loss
        tf.summary.scalar('loss', card_text_loss)
        return card_text_loss
