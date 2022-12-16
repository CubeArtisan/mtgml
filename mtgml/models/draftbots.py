import math

import tensorflow as tf

from mtgml.constants import LARGE_INT, MAX_CARDS_IN_PACK, MAX_PICKED, MAX_SEEN_PACKS, is_debug
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.contextual_rating import ContextualRating
from mtgml.layers.extended_dropout import ExtendedDropout
from mtgml.layers.mlp import MLP
from mtgml.layers.set_embedding import AttentiveSetEmbedding
from mtgml.layers.time_varying_embedding import TimeVaryingEmbedding
from mtgml.utils.masked import reduce_mean_masked, reduce_variance_masked

POOL_ORACLE_METADATA = {
    'title': 'Pool Synergy',
    'tooltip': 'How well the card goes with the cards that have already been picked.',
}
SEEN_ORACLE_METADATA = {
    'title': 'Seen Synergy',
    'tooltip': 'How well the card goes with the cards that have already been seen, looking for things like openness and combos.',
}
RATING_ORACLE_METADATA = {
    'title': 'Card Rating',
    'tooltip': 'How good the card is in a vacuum.',
}


class DraftBot(ConfigurableLayer, tf.keras.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        pool_context_ratings = hyper_config.get_bool('pool_context_ratings', default=True,
                                                     help='Whether to rate cards based on how the go with the other cards in the pool so far.')
        seen_context_ratings = hyper_config.get_bool('seen_context_ratings', default=True,
                                                     help='Whether to rate cards based on the packs seen so far.')
        item_ratings = hyper_config.get_bool('item_ratings', default=True, help='Whether to give each card a rating independent of context.')
        sublayer_metadatas = []
        if pool_context_ratings:
            sublayer_metadatas.append(POOL_ORACLE_METADATA)
        if seen_context_ratings:
            sublayer_metadatas.append(SEEN_ORACLE_METADATA)
        if item_ratings:
            sublayer_metadatas.append(RATING_ORACLE_METADATA)
        seen_pack_dims = hyper_config.get_int('seen_pack_dims', min=8, max=512, step=8, default=32,
                                              help='The number of dimensions to embed seen packs into.')\
            if seen_context_ratings else 0
        num_cards = hyper_config.get_int('num_cards', min=1, max=None, default=None,
                                         help='The number of items that must be embedded. Should be 1 + the max index expected to see.')
        pool_embed_dropout_rate = hyper_config.get_float('pool_embed_dropout_rate', min=0, max=1, default=0.25,
                                         help='The percent of entries of the card embeddings for the pool that should be dropped out.')\
                                    if pool_context_ratings else None
        sublayer_count = len([x for x in (pool_context_ratings, seen_context_ratings, item_ratings) if x])
        return {
            'seen_pack_dims': seen_pack_dims,
            'num_cards': num_cards - 1,
            'rate_off_pool': hyper_config.get_sublayer('RatingFromPool', sub_layer_type=ContextualRating,
                                                       fixed={'use_causal_mask': True},
                                                       seed_mod=31, help='The layer that rates based on the other cards that have been picked.')
                             if pool_context_ratings else None,
            'dropout_pool_embeds': hyper_config.get_sublayer('PoolDenseDropout', sub_layer_type=ExtendedDropout, seed_mod=71,
                                                             fixed={'rate': pool_embed_dropout_rate, 'noise_shape': None,
                                                                    'blank_last_dim': False},
                                                            help='The layer that drops out part of the card embeddings for the cards in the pool.')
                                  if pool_context_ratings else None,
            'seen_pack_dims': seen_pack_dims,
            'embed_pack': hyper_config.get_sublayer('EmbedPack', sub_layer_type=AttentiveSetEmbedding,
                                                    fixed={'use_causal_mask': False, 'Decoder': {'Final': {'dims': seen_pack_dims}}},
                                                    seed_mod=37, help='The layer that embeds the packs that have been seen so far.')
                          if seen_context_ratings else None,
            'embed_pack_position': hyper_config.get_sublayer('EmbedPackPosition',
                                                             sub_layer_type=TimeVaryingEmbedding,
                                                             fixed={'dims': seen_pack_dims,
                                                                    'time_shape': (3, 15)},
                                                             seed_mod=23, help='The embedding for the position in the draft')
                                  if seen_context_ratings else None,
            'rate_off_seen': hyper_config.get_sublayer('RatingFromSeen', sub_layer_type=ContextualRating,
                                                       fixed={'use_causal_mask': True, 'token_stream_dims': seen_pack_dims},
                                                       seed_mod=31, help='The layer that rates based on the embeddings of the packs that have been seen.')
                             if seen_context_ratings else None,
            'rate_card': hyper_config.get_sublayer('CardRating', sub_layer_type=MLP, seed_mod=13,
                                                   fixed={'Final': {'dims': 1, 'activation': 'linear'}},
                                                   help='Translates embeddings into linear ratings.')
                         if item_ratings else None,
            'log_loss_weight': hyper_config.get_float('log_loss_weight', min=0, max=1, step=0.01,
                                                      default=0.5, help='The weight given to log_loss. Triplet loss weight is 1 - log_loss_weight'),
            'score_variance_weight': hyper_config.get_float('score_variance_weight', min=-1, max=1,
                                                      default=1e-03, help='The weight given to the variance of the combined scores.')
                                    if item_ratings else 0,
            'rating_variance_weight': hyper_config.get_float('rating_variance_weight', min=-1, max=1,
                                                      default=1e-03, help='The weight given to the variance of the card ratings.')
                                    if item_ratings else 0,
            'seen_variance_weight': hyper_config.get_float('seen_variance_weight', min=-1, max=1,
                                                      default=1e-02, help='The weight given to the variance of the seen contextual ratings.')
                                    if seen_context_ratings else 0,
            'pool_variance_weight': hyper_config.get_float('pool_variance_weight', min=-1, max=1,
                                                      default=1e-02, help='The weight given to the variance of the pool contextual ratings.')
                                    if pool_context_ratings else 0,
            'margin': hyper_config.get_float('margin', min=0, max=10, step=0.1, default=2,
                                             help='The margin by which we want the correct choice to beat the incorrect choices.'),
            'sublayer_weights': hyper_config.get_sublayer('SubLayerWeights', sub_layer_type=TimeVaryingEmbedding,
                                                          fixed={'dims': sublayer_count, 'time_shape': (3, 15),
                                                                 'activation': 'softplus'},
                                                          help='The weights for each of the sublayers that get combined together linearly.'),
            'sublayer_metadata': sublayer_metadatas
        }

    def call(self, inputs, training=False):
        if is_debug():
            # basics = tf.ensure_shape(tf.cast(inputs[0], dtype=tf.int32, name='basics'), (None, MAX_BASICS))
            pool = tf.ensure_shape(tf.cast(inputs[1], dtype=tf.int32, name='pool'), (None, MAX_PICKED))
            seen_packs = tf.ensure_shape(tf.cast(inputs[2], dtype=tf.int32, name='seen_packs'), (None, MAX_SEEN_PACKS, MAX_CARDS_IN_PACK))
            seen_coords = tf.ensure_shape(tf.cast(inputs[3], dtype=tf.int32, name='seen_coords'), (None, MAX_SEEN_PACKS, 4, 2))
            seen_coord_weights = tf.ensure_shape(tf.cast(inputs[4], dtype=self.compute_dtype, name='seen_coord_weights'), (None, MAX_SEEN_PACKS, 4))
            card_embeddings = tf.ensure_shape(tf.cast(inputs[5], dtype=self.compute_dtype, name='card_embeddings'), (self.num_cards + 1, None))
        else:
            batch_size = tf.shape(inputs[1])[0]
            max_picked = tf.shape(inputs[1])[1]
            max_seen_packs = tf.shape(inputs[2])[1]
            max_cards_in_pack = tf.shape(inputs[2])[2]
            # basics = tf.cast(inputs[0], dtype=tf.int32, name='basics')
            pool = tf.cast(tf.reshape(inputs[1], (batch_size, max_picked)), dtype=tf.int32, name='pool')
            seen_packs = tf.cast(tf.reshape(inputs[2], (batch_size, max_seen_packs, max_cards_in_pack)), dtype=tf.int32, name='seen_packs')
            seen_coords = tf.cast(tf.reshape(inputs[3], (batch_size, max_seen_packs, 4, 2)), dtype=tf.int32, name='seen_coords')
            seen_coord_weights = tf.cast(tf.reshape(inputs[4], (batch_size, max_seen_packs, 4)), dtype=self.compute_dtype, name='seen_coord_weights')
            card_embeddings = tf.cast(inputs[5], dtype=self.compute_dtype, name='card_embeddings')
        loss_dtype = tf.float32
        sublayer_scores = []
        mask = tf.cast(seen_packs > 0, dtype=self.compute_dtype, name='pack_mask')
        # Shift pool right by 1
        pool = tf.concat([tf.zeros((tf.shape(pool)[0], 1), dtype=pool.dtype), pool[:, :-1]], axis=1)
        pool_embeds = tf.gather(card_embeddings, pool, name='pool_embeds')
        seen_pack_embeds = tf.gather(card_embeddings, seen_packs, name='seen_pack_embeds')
        if self.rate_off_pool or self.rate_off_seen:
            if self.rate_off_pool:
                pool_embeds_dropped = self.dropout_pool_embeds(pool_embeds, training=training)
                mask_pair = (tf.cast(mask, tf.bool), pool > 0)
                sublayer_scores.append(self.rate_off_pool((seen_pack_embeds, pool_embeds_dropped), training=training, mask=mask_pair))
            if self.rate_off_seen:
                mask_pair = (tf.cast(mask, tf.bool), tf.reduce_any(tf.cast(mask, tf.bool), -1))
                flat_seen_pack_embeds = tf.reshape(seen_pack_embeds, (-1, tf.shape(seen_pack_embeds)[-2], tf.shape(seen_pack_embeds)[-1]))
                flat_pack_embeds = self.embed_pack(flat_seen_pack_embeds, training=training)
                pack_embeds = tf.reshape(flat_pack_embeds, (-1, tf.shape(seen_pack_embeds)[1], tf.shape(flat_pack_embeds)[-1]))
                position_embeds = self.embed_pack_position((seen_coords, seen_coord_weights), training=training)
                pack_embeds = pack_embeds * tf.constant(math.sqrt(self.seen_pack_dims), dtype=self.compute_dtype) + position_embeds
                pack_embeds._keras_mask = tf.math.reduce_any(seen_packs > 0, axis=-1)
                sublayer_scores.append(self.rate_off_seen((seen_pack_embeds, pack_embeds), training=training, mask=mask_pair))
        if self.rate_card:
            card_ratings = tf.squeeze(self.rate_card(seen_pack_embeds, training=training), axis=-1) * mask
            sublayer_scores.append(card_ratings)
            if is_debug():
                tf.summary.histogram('card_ratings', card_ratings)
        sublayer_scores = tf.stack(sublayer_scores, axis=-1, name='stacked_sublayer_scores')
        sublayer_weights = tf.math.softplus(self.sublayer_weights((seen_coords, seen_coord_weights), training=training))
        sublayer_scores = tf.cast(sublayer_scores, dtype=self.compute_dtype, name='sublayer_scores')
        sublayer_weights = tf.cast(sublayer_weights, dtype=self.compute_dtype, name='sublayer_weights')
        scores = tf.einsum('...ps,...s->...p', sublayer_scores, sublayer_weights, name='scores')
        if len(inputs) > 6:
            mask = tf.cast(mask, dtype=loss_dtype)
            if is_debug():
                y_idx = tf.ensure_shape(tf.cast(inputs[6], dtype=tf.int32, name='y_idx'), (None, None))
                riskiness = tf.ensure_shape(tf.cast(inputs[7], dtype=tf.float32, name='riskiness'), (None, None, MAX_CARDS_IN_PACK))
            else:
                y_idx = tf.cast(inputs[6], dtype=tf.int32, name='y_idx')
                riskiness = tf.cast(inputs[7], dtype=tf.float32, name='riskiness')
            num_in_pack = tf.math.reduce_sum(mask, axis=-1, name='num_in_pack')
            mask_freebies_1 = num_in_pack > 1.0
            pos_scores = (tf.cast(scores, dtype=loss_dtype, name='cast_scores') + tf.constant(LARGE_INT, dtype=loss_dtype)) * mask
            neg_scores = (tf.constant(LARGE_INT, dtype=loss_dtype) - tf.cast(scores, dtype=loss_dtype, name='cast_scores')) * mask
            pos_mask = tf.cast(tf.expand_dims(y_idx == 0, -1), dtype=pos_scores.dtype)
            scores = tf.add(pos_scores * pos_mask, neg_scores * (1 - pos_mask), name='scores_after_y_idx')
            probs = tf.nn.softmax(scores, axis=-1, name='probs')
            probs_correct = probs[:, :, 0]
            probs_for_loss = tf.concat([probs[:, :, :1], tf.constant(1, dtype=self.compute_dtype) - probs[:, :, 1:]], axis=-1)
            log_losses = tf.negative(tf.math.log((1 - 2e-10) * probs_for_loss + 1e-10, name='log_probs'), name='log_loss')
            loss_mask = tf.cast(tf.expand_dims(mask_freebies_1, -1), dtype=loss_dtype) * mask
            log_losses = riskiness * log_losses
            log_loss = reduce_mean_masked(log_losses, mask=loss_mask, axis=[0,1,2], name='log_loss')
            tf.summary.scalar('log_loss', log_loss)
            log_loss_weighted = tf.math.multiply(log_loss,
                                           tf.constant(self.log_loss_weight, dtype=loss_dtype, name='log_loss_weight'),
                                           name='log_loss_weighted')
            self.add_metric(log_losses, 'pick_log_loss')
            score_diffs = tf.subtract(tf.add(tf.constant(self.margin, dtype=scores.dtype),
                                             scores[:, :, 1:]), scores[:, :, :1],
                                      name='score_diffs') * loss_mask[:, :, 1:]
            clipped_diffs = tf.math.maximum(tf.constant(0, dtype=loss_dtype), score_diffs, name='clipped_score_diffs') * riskiness[:, :, 1:]
            triplet_losses = reduce_mean_masked(clipped_diffs, loss_mask[:, :, 1:], axis=[0,1,2], name='triplet_loss')
            tf.summary.scalar('triplet_loss', triplet_losses)
            triplet_loss_weighted = tf.math.multiply(triplet_losses,
                                                     tf.constant(1 - self.log_loss_weight, dtype=loss_dtype,
                                                                 name='triplet_loss_weight'),
                                                     name='triplet_loss_weighted')
            self.add_metric(triplet_losses, 'triplet_loss')
            mask = tf.cast(mask, dtype=self.compute_dtype)
            variances = reduce_variance_masked(sublayer_scores * tf.expand_dims(sublayer_weights, -2),
                                               mask=tf.expand_dims(loss_mask, -1), axis=-2, name='oracle_pack_variances')
            score_variance = reduce_variance_masked(scores, mask=loss_mask, axis=-1,
                                                    name='score_variances')
            i = 0
            variance_mask = tf.reduce_any(loss_mask > 0, axis=-1)
            if self.rate_off_pool:
                pool_variance = variances[:, :, i]
                i += 1
            else:
                pool_variance = tf.zeros_like(score_variance)
            if self.rate_off_seen:
                seen_variance = variances[:, :, i]
                i += 1
            else:
                seen_variance = tf.zeros_like(score_variance)
            if self.rate_card:
                rating_variance = variances[:, :, i]
                i += 1
            else:
                rating_variance = tf.zeros_like(score_variance)
            score_variance_loss = reduce_mean_masked(score_variance, mask=variance_mask, axis=[0,1], name='score_variance_loss')
            score_variance_loss_weighted = tf.math.multiply(score_variance_loss, tf.constant(self.score_variance_weight, dtype=loss_dtype, name='score_variance_weight'),
                                                           name='score_variance_loss_weighted')
            self.add_metric(score_variance_loss, 'score_variance_loss')
            tf.summary.scalar('score_variance_loss', score_variance_loss)
            pool_variance_loss =reduce_mean_masked(pool_variance, mask=variance_mask, axis=[0, 1], name='pool_variance_loss')
            pool_variance_loss_weighted = tf.math.multiply(pool_variance_loss, tf.constant(self.pool_variance_weight, dtype=loss_dtype, name='pool_variance_weight'),
                                                           name='pool_variance_loss_weighted')
            if self.rate_off_pool:
                self.add_metric(pool_variance_loss, 'pool_variance_loss')
                tf.summary.scalar('pool_variance_loss', pool_variance_loss)
            seen_variance_loss = reduce_mean_masked(seen_variance, mask=variance_mask, axis=[0,1], name='seen_variance_loss')
            seen_variance_loss_weighted = tf.math.multiply(seen_variance_loss, tf.constant(self.seen_variance_weight, dtype=loss_dtype, name='seen_variance_weight'),
                                                           name='seen_variance_loss_weighted')
            if self.rate_off_seen:
                self.add_metric(seen_variance_loss, 'seen_variance_loss')
                tf.summary.scalar('seen_variance_loss', seen_variance_loss)
            rating_variance_loss = reduce_mean_masked(rating_variance, mask=variance_mask, axis=[0,1], name='rating_variance_loss')
            rating_variance_loss_weighted = tf.math.multiply(rating_variance_loss, tf.constant(self.rating_variance_weight, dtype=loss_dtype, name='rating_variance_weight'),
                                                           name='rating_variance_loss_weighted')
            if self.rate_card:
                self.add_metric(rating_variance_loss, 'rating_variance_loss')
                tf.summary.scalar('rating_variance_loss', rating_variance_loss)
            max_scores = tf.reduce_logsumexp(scores - tf.constant(LARGE_INT, dtype=loss_dtype), axis=-1)
            max_scores = max_scores + tf.stop_gradient(tf.reduce_max(scores - LARGE_INT, axis=-1) - max_scores)
            min_scores = -tf.reduce_logsumexp(-scores + mask * tf.constant(LARGE_INT, dtype=loss_dtype), axis=-1)
            min_scores = min_scores + tf.stop_gradient(tf.reduce_min(scores + (1 - 2 * mask) * LARGE_INT, axis=-1)
                                                       - min_scores)
            chosen_idx = tf.zeros_like(y_idx)
            scores = tf.cast(scores, dtype=tf.float32)
            chosen_idx = tf.reshape(chosen_idx, (-1,))
            scores = tf.reshape(scores, (-1, tf.shape(scores)[-1]))
            mask_freebies_1 = tf.reshape(mask_freebies_1, (-1,))
            mask_freebies_2 = tf.reshape(num_in_pack > 2.0, (-1,))
            mask_freebies_3 = tf.reshape(num_in_pack > 3.0, (-1,))
            top_1_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(chosen_idx, scores, 1)
            top_2_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(chosen_idx, scores, 2)
            top_3_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(chosen_idx, scores, 3)
            accuracy = reduce_mean_masked(top_1_accuracy, mask=mask_freebies_1, name='accuracy_top_1')
            self.add_metric(accuracy, 'accuracy')
            scores = tf.cast(scores, dtype=self.compute_dtype)
            # Logging for Tensorboard
            tf.summary.scalar('accuracy_top_1', accuracy)
            tf.summary.scalar('accuracy_top_2', reduce_mean_masked(top_2_accuracy, mask=mask_freebies_2, name='accuracy_top_2'))
            tf.summary.scalar('accuracy_top_3', reduce_mean_masked(top_3_accuracy, mask=mask_freebies_3, name='accuracy_top_3'))
            tf.summary.scalar('prob_correct', reduce_mean_masked(tf.reshape(probs_correct, (-1,)), mask=mask_freebies_1,
                                                                 name='mean_prob_correct'))
            if is_debug():
                tf.summary.histogram('max_diff_scores', max_scores - min_scores)
                mask = tf.reshape(mask, (-1, tf.shape(mask)[-1]))
                tf.summary.histogram('scores', (scores - tf.constant(LARGE_INT, dtype=self.compute_dtype)) * mask)
                tf.summary.histogram('scores_0', ((scores - tf.constant(LARGE_INT, dtype=self.compute_dtype)) * mask)[:, :1])
                tf.summary.histogram('score_diffs', score_diffs)
                tf.summary.histogram('probs', probs)
                tf.summary.histogram('log_losses', log_losses)
                tf.summary.histogram('triplet_losses', clipped_diffs)
            # tf.summary.histogram('score_variance', score_variance)
            # tf.summary.histogram('pool_variance', pool_variance)
            # tf.summary.histogram('seen_variance', seen_variance)
            # tf.summary.histogram('rating_variance', rating_variance)
            # tf.summary.histogram('prob_chosen', probs_correct)
            loss = tf.cast(triplet_loss_weighted + log_loss_weighted + pool_variance_loss_weighted
                           + seen_variance_loss_weighted + rating_variance_loss_weighted
                           + score_variance_loss_weighted, dtype=self.compute_dtype)
            return loss
        return sublayer_scores, sublayer_weights
