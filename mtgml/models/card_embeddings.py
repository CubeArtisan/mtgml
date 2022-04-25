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


class DocumentEncoder(ConfigurableLayer):
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


class MaskTokens(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            'rate': hyper_config.get_float('rate', min=0, max=1, default=0.25,
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

    def call(self, inputs, training=False):
        tokens, token_embeddings, positional_embeddings = inputs
        masked_tokens, masked = self.mask_tokens(tokens, training=training, mask=tokens > 0)
        token_embeds = tf.gather(token_embeddings, masked_tokens) + tf.expand_dims(positional_embeddings, -3)
        encoded_tokens = self.encode_tokens(token_embeds, training=training, mask=tokens > 0)
        reconstructed = self.reconstruct_tokens((encoded_tokens, token_embeddings), training=training)
        reconstruction_token_losses = tf.keras.metrics.sparse_categorical_crossentropy(tokens, tf.nn.softmax(reconstructed) + 1e-10)
        mask = tf.cast(tokens > 0, dtype=self.compute_dtype)
        reconstruction_example_losses = tf.math.divide_no_nan(tf.reduce_sum(mask * reconstruction_token_losses, keepdims=True), tf.reduce_sum(mask, keepdims=True), name='reconstruction_sample_losses')
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
        num_tokens = hyper_config.get_int('num_tokens', default=None, help='The number of tokens in the vocab including null and mask')
        return {
            'num_tokens': num_tokens,
            'generator': hyper_config.get_sublayer('Generator', sub_layer_type=MaskedModel, seed_mod=61,
                                                   help='The generator that replaces some tokens with likely replacements.'),
            'encoder': hyper_config.get_sublayer('CardEncoder', sub_layer_type=DocumentEncoder, seed_mod=73,
                                                 help='Encodes the sampled card tokens.'),
            'hidden': hyper_config.get_sublayer('HiddenDense', sub_layer_type=WDense, seed_mod=83,
                                                      help='Hidden layer for determining if a token was replaced or not.'),
            'is_replaced': hyper_config.get_sublayer('IsReplaced', sub_layer_type=WDense, seed_mod=87,
                                                     fixed={'dims': 1, 'activation': 'sigmoid'},
                                                     help='Determine whether the token has been replaced or not.'),
            'down_sample_embeddings': hyper_config.get_sublayer('DownSampler', sub_layer_type=WDense, seed_mod=97,
                                                                help='The dense layer to down sample the embeddings for the MLM.'),
        }

    def call(self, inputs, training=False):
        tokens, token_embeddings, position_embeddings = inputs
        down_token_embeddings = self.down_sample_embeddings(token_embeddings, training=training)
        down_position_embeddings = self.down_sample_embeddings(position_embeddings, training=training)
        generated, masked_tokens, reconstruction_loss = self.generator((tokens, down_token_embeddings,
                                                                        down_position_embeddings), training=training)
        flat_generated = tf.reshape(generated, (-1, self.num_tokens))
        flat_sampled = tf.random.categorical(tf.math.log(tf.nn.softmax(flat_generated)),
                                             1, dtype=tokens.dtype, seed=self.seed, name='flat_sampled')
        sampled = tf.where(tf.logical_and(masked_tokens, tokens > 0),
                           tf.reshape(flat_sampled, tf.shape(tokens)), tokens)
        mask = tf.cast(tokens > 0, dtype=self.compute_dtype)
        sampled_encoded =  self.encoder((tokens, token_embeddings,
                                         position_embeddings), training=training, mask=tokens > 0)
        hidden = self.hidden(sampled_encoded, training=training)
        pred_replaced = tf.squeeze(self.is_replaced(hidden, training=training), -1)
        pred_correct = tf.where(tokens == sampled, 1 - pred_replaced, pred_replaced) + 1e-10
        token_losses = -mask * tf.math.log(tf.where(tokens > 0, pred_correct, tf.ones_like(pred_correct)))
        sample_losses = tf.math.divide_no_nan(tf.reduce_sum(token_losses, keepdims=True), tf.reduce_sum(mask, keepdims=True), name='sample_losses')
        sample_loss = tf.nn.compute_average_loss(sample_losses)
        self.add_loss(sample_loss)

        was_replaced = tf.cast(tokens != sampled, dtype=self.compute_dtype)
        tf.summary.scalar('sample_loss', sample_loss)
        self.add_metric(sample_loss, 'sample_loss')
        tf.summary.histogram('probs', tf.gather(tf.nn.softmax(generated), tokens, batch_dims=2))
        tf.summary.histogram('pred_replaced', pred_replaced)
        accuracy = tf.reduce_sum(mask * pred_replaced * was_replaced) / tf.reduce_sum(mask)
        tf.summary.scalar('prob_correct', accuracy)
        error_classes = tf.reduce_sum(mask * was_replaced) / tf.reduce_sum(mask)
        tf.summary.scalar('prob_replaced', error_classes)
        prob_mask_correct = tf.reduce_sum(pred_replaced * was_replaced) / tf.reduce_sum(was_replaced)
        tf.summary.scalar('masked_prob_correct', prob_mask_correct)
        mask_accuracy = tf.reduce_sum(tf.cast(pred_replaced > 0.5, dtype=self.compute_dtype) * was_replaced) \
                        / tf.reduce_sum(was_replaced)
        tf.summary.scalar('replaced_accuracy', mask_accuracy)
        return sample_loss + reconstruction_loss


class CombinedCardModel(ConfigurableLayer, tf.keras.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_tokens = hyper_config.get_int('num_tokens', default=None, help='The number of tokens in the vocab including null and mask')
        embed_dims = hyper_config.get_int('embed_dims', min=8, max=512, default=64,
                                          help='The number of dimensions for the token embeddings')
        card_token_map = tf.constant(hyper_config.get_list('card_token_map', default=[], help='The map from cards to token sequences.'), dtype=tf.int32)
        fine_tuning = hyper_config.get_bool('fine_tuning', default=False, help='Whether to run fine tuning on the model now.')
        print('fine_tuning', fine_tuning)
        return {
            'token_embeds': hyper_config.get_sublayer('TokenEmbeddings', sub_layer_type=ItemEmbedding, seed_mod=11,
                                                      fixed={'dims': embed_dims, 'num_items': num_tokens, 'trainable': not fine_tuning},
                                                      help='The embeddings for the tokens.'),
            'position_embeds': hyper_config.get_sublayer('PositionalEmbeddings', sub_layer_type=ItemEmbedding, seed_mod=11,
                                                         fixed={'dims': embed_dims, 'num_items': card_token_map.shape[1], 'trainable': not fine_tuning},
                                                         help='The embeddings for the positions.'),
            'card_token_map': card_token_map,
            'card_text': hyper_config.get_sublayer('CardTextModel', sub_layer_type=MaskedModel, seed_mod=91,
                                                   fixed={'num_tokens': num_tokens},
                                                   help='The model for learning card representations.'),
            'extra_layers': hyper_config.get_sublayer('ExtraLayers', sub_layer_type=DocumentEncoder, seed_mod=131,
                                                      help='The extra layers for learning card embeddings.')
                            if fine_tuning else None,
            'cube_adj_mtx_mlp': hyper_config.get_sublayer('CubeAdjMtxMLP', sub_layer_type=MLP,
                                                          help='The mlp to process the paired embeddings')
                                if fine_tuning else None,
            'cube_adj_mtx_dense': hyper_config.get_sublayer('CubeAdjMtxFinal', sub_layer_type=WDense,
                                                            fixed={'activation': 'sigmoid', 'dims': 1},
                                                            help='The layer to get the pairs of card adj_cells out.')
                                  if fine_tuning else None,
            'deck_adj_mtx_mlp': hyper_config.get_sublayer('DeckAdjMtxMLP', sub_layer_type=MLP,
                                                          help='The mlp to process the paired embeddings')
                                if fine_tuning else None,
            'deck_adj_mtx_dense': hyper_config.get_sublayer('DeckAdjMtxFinal', sub_layer_type=WDense,
                                                            fixed={'activation': 'sigmoid', 'dims': 1},
                                                            help='The layer to get the pairs of card adj_cells out.')
                                  if fine_tuning else None,
            'deck_adj_mtx': tf.constant(hyper_config.get_list('deck_adj_mtx', default=None, help=''), dtype=tf.float32)
                            if fine_tuning else None,
            'cube_adj_mtx': tf.constant(hyper_config.get_list('cube_adj_mtx', default=None, help=''), dtype=tf.float32)
                            if fine_tuning else None,
            'primary_layer': hyper_config.get_sublayer('TransformFirstCard', sub_layer_type=WDense,
                                                       fixed={'activation': 'linear', 'dims': embed_dims},
                                                       help='Transform the first card in the pair.')
                             if fine_tuning else None,
            'secondary_layer': hyper_config.get_sublayer('TransformSecondCard', sub_layer_type=WDense,
                                                         fixed={'activation': 'linear', 'dims': embed_dims},
                                                         help='Transform the second card in the pair.')
                               if fine_tuning else None,
            'fine_tuning': fine_tuning,
        }

    def build(self, input_shapes):
        super(CombinedCardModel, self).build(input_shapes)
        self.card_text.build(self.card_token_map.shape[1])
        self.token_embeds.build(input_shapes=(None, None))
        self.position_embeds.build(input_shapes=(None, None))

    def call(self, inputs, training=False):
        cube_card_indices, deck_card_indices = inputs
        cube_card_indices = tf.cast(cube_card_indices, dtype=tf.int32)
        deck_card_indices = tf.cast(deck_card_indices, dtype=tf.int32)
        cube_tokens = tf.gather(self.card_token_map, cube_card_indices, name='cube_tokens')
        deck_tokens = tf.gather(self.card_token_map, deck_card_indices, name='deck_tokens')
        if not self.fine_tuning:
            _, _, loss = self.card_text((cube_tokens, self.token_embeds.embeddings,
                                   self.position_embeds.embeddings), training=training)
        else:
            card_tokens = tf.concat([cube_tokens, deck_tokens], 0)
            token_embeds = tf.gather(self.token_embeds.embeddings, card_tokens) + tf.expand_dims(self.position_embeds.embeddings, -3)
            card_embeds =  self.card_text.encode_tokens(token_embeds, training=training, mask=card_tokens > 0)
            card_embeds = self.extra_layers(card_embeds, training=training, mask=card_tokens > 0)[:, 0]
            cube_card_embeds = card_embeds[:tf.shape(cube_card_indices)[0]]
            deck_card_embeds = card_embeds[tf.shape(cube_card_indices)[0]:]
            exp_cube_indices = tf.expand_dims(tf.zeros_like(deck_card_indices), -2) + tf.expand_dims(cube_card_indices, -1)
            exp_deck_indices = tf.expand_dims(tf.zeros_like(cube_card_indices), -1) + tf.expand_dims(deck_card_indices, -2)
            card_idx_mtx = tf.stack([exp_cube_indices, exp_deck_indices], -1)
            cube_prob_mtx = tf.cast(-tf.math.log(tf.gather_nd(self.cube_adj_mtx, card_idx_mtx) + 1e-06), dtype=self.compute_dtype)
            deck_prob_mtx = tf.cast(-tf.math.log(tf.gather_nd(self.deck_adj_mtx, card_idx_mtx) + 1e-06), dtype=self.compute_dtype)
            card_embed_mtx = tf.expand_dims(self.primary_layer(cube_card_embeds, training=training), -2) \
                             + tf.expand_dims(self.secondary_layer(deck_card_embeds, training=training), -3)
            cube_transformed = self.cube_adj_mtx_mlp(card_embed_mtx, training=training)
            deck_transformed = self.deck_adj_mtx_mlp(card_embed_mtx, training=training)
            cube_probs = tf.squeeze(self.cube_adj_mtx_dense(cube_transformed, training=training), -1)
            deck_probs = tf.squeeze(self.deck_adj_mtx_dense(deck_transformed, training=training), -1)
            cube_final = -tf.math.log(cube_probs + 1e-06)
            deck_final = -tf.math.log(deck_probs + 1e-06)
            cube_loss = tf.reduce_mean(tf.keras.metrics.mean_squared_error(cube_final, cube_prob_mtx))
            deck_loss = tf.reduce_mean(tf.keras.metrics.mean_squared_error(deck_final, deck_prob_mtx))
            self.add_loss(cube_loss)
            self.add_loss(deck_loss)

            loss = cube_loss + deck_loss
            tf.summary.histogram('cube_final', cube_final)
            tf.summary.histogram('deck_final', deck_final)
            # tf.summary.histogram('cube_prob_mtx', cube_prob_mtx)
            # tf.summary.histogram('deck_prob_mtx', deck_prob_mtx)
            # tf.summary.scalar('cube_loss', cube_loss)
            # tf.summary.scalar('deck_loss', deck_loss)
            self.add_metric(cube_loss, 'cube_loss')
            self.add_metric(deck_loss, 'deck_loss')
            return cube_card_embeds #, cube_probs, deck_probs
        tf.summary.scalar('loss', loss)
        return loss
