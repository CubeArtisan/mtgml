import horovod.tensorflow.keras as hvd
import tensorflow as tf

from mtgml.layers.item_embedding import ItemEmbedding
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.mlp import MLP
from mtgml.layers.wrapped import WDense
from mtgml.layers.bert import BERT


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
            'encode_tokens': hyper_config.get_sublayer('EncodeTokens', sub_layer_type=BERT, seed_mod=5,
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
        reconstruction_loss = tf.reduce_mean(reconstruction_example_losses) / 2.0
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
            'encoder': hyper_config.get_sublayer('CardEncoder', sub_layer_type=BERT, seed_mod=73,
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
        sample_loss = tf.reduce_mean(sample_losses) / 2.0
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
            'extra_layers': hyper_config.get_sublayer('ExtraLayers', sub_layer_type=BERT, seed_mod=131,
                                                      help='The extra layers for learning card embeddings.')
                            if fine_tuning else None,
            'deck_adj_mtx': tf.constant(hyper_config.get_list('deck_adj_mtx', default=None, help=''), dtype=tf.float32)
                            if fine_tuning else None,
            'cube_adj_mtx': tf.constant(hyper_config.get_list('cube_adj_mtx', default=None, help=''), dtype=tf.float32)
                            if fine_tuning else None,
            'cube_1_layer': hyper_config.get_sublayer('CubeTransformFirstCard', sub_layer_type=WDense,
                                                       fixed={'activation': 'linear', 'dims': embed_dims},
                                                       help='Transform the first card in the pair.')
                             if fine_tuning else None,
            'cube_1_layer': hyper_config.get_sublayer('CubeTransformFirstCard', sub_layer_type=WDense,
                                                       fixed={'activation': 'linear', 'dims': embed_dims},
                                                       help='Transform the first card in the pair.')
                             if fine_tuning else None,
            'cube_2_layer': hyper_config.get_sublayer('CubeTransformSecondCard', sub_layer_type=WDense,
                                                       fixed={'activation': 'linear', 'dims': embed_dims},
                                                       help='Transform the first card in the pair.')
                             if fine_tuning else None,
            # 'deck_row_layer': hyper_config.get_sublayer('DeckTransformFirstCard', sub_layer_type=WDense,
            #                                            fixed={'activation': 'linear', 'dims': embed_dims},
            #                                            help='Transform the first card in the pair.')
            #                  if fine_tuning else None,
            'deck_column_layer': hyper_config.get_sublayer('DeckTransformSecondCard', sub_layer_type=WDense,
                                                       fixed={'activation': 'linear', 'dims': embed_dims},
                                                       help='Transform the first card in the pair.')
                             if fine_tuning else None,
            'fine_tuning': fine_tuning,
        }

    def build(self, input_shapes):
        super(CombinedCardModel, self).build(input_shapes)
        self.card_text.build(self.card_token_map.shape[1])
        self.token_embeds.build(input_shapes=(None, None))
        self.position_embeds.build(input_shapes=(None, None))
        self.first_step = self.add_weight('first_step', shape=(), dtype=tf.bool, trainable=False, initializer=tf.constant_initializer(True))
        if self.fine_tuning:
            # self.cube_temperature = self.add_weight('cube_temperature', shape=(), initializer=tf.constant_initializer(0.5),
            #                                         trainable=True)
            self.deck_temperature = self.add_weight('deck_temperature', shape=(), initializer=tf.constant_initializer(0.5),
                                                    trainable=True)

    def compare_with_adj_mtx(self, row_embeds, column_embeds, card_idx_mtx, mtx_type, name=None):
        adj_mtx = getattr(self, f'{mtx_type}_adj_mtx')
        row_layer = getattr(self, f'{mtx_type}_row_layer', None) or (lambda x: x)
        column_layer = getattr(self, f'{mtx_type}_column_layer')
        temperature = getattr(self, f'{mtx_type}_temperature')
        with tf.name_scope(name or 'CompareWithAdjMtx'):
            unnormalized_true_probs = tf.cast(tf.gather_nd(adj_mtx, card_idx_mtx, name='adj_mtx_probs') + 1e-10, dtype=self.compute_dtype)
            true_probs = unnormalized_true_probs / tf.reduce_sum(unnormalized_true_probs, axis=-1, keepdims=True, name='true_probs_row_sum')
            row_embeds_transformed = row_layer(row_embeds)
            column_embeds_transformed = column_layer(column_embeds)
            exp_row_embeds = tf.expand_dims(row_embeds_transformed, -2) + tf.expand_dims(tf.zeros_like(column_embeds_transformed), -3)
            exp_column_embeds = tf.expand_dims(tf.zeros_like(row_embeds_transformed), -2) + tf.expand_dims(column_embeds_transformed, -3)
            similarities = -tf.keras.losses.cosine_similarity(exp_row_embeds, exp_column_embeds)
            unnormalized_pred_probs = tf.nn.softmax(similarities * temperature, name='pred_probs') + 1e-10 # We add this to guarantee none get rounded to zero and to match how we did it above so they give the same values.
            pred_probs = unnormalized_pred_probs / tf.reduce_sum(unnormalized_pred_probs, axis=-1, keepdims=True, name='pred_probs')
            # We divide by log(2) to normalize it to bits instead of nats (base e) for interpretability.
            divergence_loss = tf.reduce_mean(tf.math.abs(tf.keras.losses.kl_divergence(true_probs, pred_probs))) / tf.math.log(2.0)
            # By keeping the temperature low we give it the best chance of avoiding vanishing gradients.
            # It also encourages it to use the whole range of cosine similarity instead of bunching together.
            # L2 is better than L1 loss here because we want it small not zero.
            temperature_loss = temperature * temperature * 0.01
            self.add_loss(divergence_loss)
            self.add_loss(temperature_loss)

            # More losses worth considering but not using to optimize.
            l1_loss = tf.reduce_mean(tf.math.abs(true_probs - pred_probs), name='l1_loss')
            l2_loss = tf.reduce_mean(tf.math.pow(true_probs - pred_probs, 2), name='l2_loss')
            # Need to work out if this truly is a valid loss for float outcomes instead of binary. I believe it is.
            cross_entropy_loss = -tf.reduce_mean(true_probs * tf.math.log(pred_probs) + (1 - true_probs) * tf.math.log(1 - pred_probs))
            self.add_metric(divergence_loss, f'{mtx_type}_divergence_loss')
            self.add_metric(temperature, f'{mtx_type}_temperature')
            self.add_metric(l1_loss, f'{mtx_type}_l1_loss')
            self.add_metric(l2_loss, f'{mtx_type}_l2_loss')
            self.add_metric(cross_entropy_loss, f'{mtx_type}_cross_entropy_loss')

            # Some additional metrics worth logging.
            mean_similarity = 1 - tf.reduce_mean(similarities)
            # We again normalize to bits for interpretability
            pred_entropy = tf.reduce_mean(tf.reduce_sum(pred_probs * -tf.math.log(pred_probs), axis=-1)) / tf.math.log(2.0)
            true_entropy = tf.reduce_mean(tf.reduce_sum(true_probs * -tf.math.log(true_probs), axis=-1)) / tf.math.log(2.0)
            tf.summary.scalar('mean_similarity', mean_similarity)
            tf.summary.scalar('entropy_pred', pred_entropy)
            tf.summary.scalar('entropy_true', true_entropy)
            tf.summary.histogram('probs_true', true_probs)
            tf.summary.histogram('probs_pred', pred_probs)

            return temperature_loss + divergence_loss



    def call(self, inputs, training=False):
        row_card_indices, column_card_indices = inputs
        row_card_indices = tf.cast(row_card_indices, dtype=tf.int32)
        column_card_indices = tf.cast(column_card_indices, dtype=tf.int32)
        card_indices = tf.concat([row_card_indices, column_card_indices], axis=0, name='card_indices')
        if not self.fine_tuning:
            card_tokens = tf.gather(self.card_token_map, card_indices, name='cube_tokens')
            _, _, loss = self.card_text((card_tokens, self.token_embeds.embeddings,
                                   self.position_embeds.embeddings), training=training)
            return loss
        else:
            card_tokens = tf.gather(self.card_token_map, card_indices, name='card_tokens')
            token_embeds = tf.gather(self.token_embeds.embeddings, card_tokens) + tf.expand_dims(self.position_embeds.embeddings, -3)
            card_embeds =  self.card_text.encode_tokens(token_embeds, training=training, mask=card_tokens > 0)
            card_embeds = self.extra_layers(card_embeds, training=training, mask=card_tokens > 0)[:, 0]
            if training:
                # full_card_indices = hvd.allgather(card_indices, name='full_card_indices')
                full_card_indices = card_indices
                exp_row_indices = tf.expand_dims(tf.zeros_like(full_card_indices), -2) + tf.expand_dims(card_indices, -1)
                exp_column_indices = tf.expand_dims(tf.zeros_like(card_indices), -1) + tf.expand_dims(full_card_indices, -2)
                card_idx_mtx = tf.stack([exp_row_indices, exp_column_indices], -1, name='card_idx_mtx')
                full_card_embeds = card_embeds
                # full_card_embeds = hvd.allgather(card_embeds, name='full_card_embeds')
                deck_loss = self.compare_with_adj_mtx(card_embeds, full_card_embeds, card_idx_mtx, 'deck', name='CompareWithDeckAdjMtx')
                loss = deck_loss
                return loss
            else:
                row_card_embeds = card_embeds[:tf.shape(row_card_indices)[0]]
                return row_card_embeds
