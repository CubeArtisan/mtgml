import math

import numpy as np
import tensorflow as tf

from mtgml.constants import LARGE_INT, MAX_CARDS_IN_PACK, MAX_PICKED, MAX_SEEN_PACKS
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.contextual_rating import ContextualRating
from mtgml.layers.mlp import MLP
from mtgml.layers.set_embedding import AttentiveSetEmbedding
from mtgml.layers.time_varying_embedding import TimeVaryingEmbedding
from mtgml.layers.wrapped import WDense

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
        seen_pack_dims = hyper_config.get_int('seen_pack_dims', min=8, max=2048, step=8, default=512,
                                              help='The number of dimensions to embed seen packs into.')\
            if seen_context_ratings else 0
        num_cards = hyper_config.get_int('num_cards', min=1, max=None, default=None,
                                         help='The number of items that must be embedded. Should be 1 + the max index expected to see.')
        sublayer_count = len([x for x in (pool_context_ratings, seen_context_ratings, item_ratings) if x])
        # dims = hyper_config.get_int('dims', min=8, max=1024, default=512,
        #                             help='The number of dims for the transformer stream.')
        return {
            'card_embeddings': tf.constant(hyper_config.get_list('card_embeddings', default=None, help='The card embeddings.')),
            'num_cards': num_cards - 1,
            'rate_off_pool': hyper_config.get_sublayer('RatingFromPool', sub_layer_type=ContextualRating,
                                                       seed_mod=31, help='The layer that rates based on the other cards that have been picked.')
                             if pool_context_ratings else None,
            'seen_pack_dims': seen_pack_dims,
            'embed_pack': hyper_config.get_sublayer('EmbedPack', sub_layer_type=AttentiveSetEmbedding,
                                                    fixed={'Final': {'dims': seen_pack_dims, 'activation': 'linear'}},
                                                    seed_mod=37, help='The layer that embeds the packs that have been seen so far.')
                          if seen_context_ratings else None,
            'embed_pack_position': hyper_config.get_sublayer('EmbedPackPosition',
                                                             sub_layer_type=TimeVaryingEmbedding,
                                                             fixed={'dims': seen_pack_dims,
                                                                    'time_shape': (3, 15)},
                                                             seed_mod=23, help='The embedding for the position in the draft')
                                  if seen_context_ratings else None,
            'rate_off_seen': hyper_config.get_sublayer('RatingFromSeen', sub_layer_type=ContextualRating,
                                                       seed_mod=31, help='The layer that rates based on the embeddings of the packs that have been seen.')
                             if seen_context_ratings else None,
            'rate_card': hyper_config.get_sublayer('CardRating', sub_layer_type=WDense, seed_mod=13,
                                                   fixed={'dims': 1, 'activation': 'linear'},
                                                   help='Translates embeddings into linear ratings.')
                         if item_ratings else None,
            'log_loss_weight': hyper_config.get_float('log_loss_weight', min=0, max=1, step=0.01,
                                                      default=0.5, help='The weight given to log_loss. Triplet loss weight is 1 - log_loss_weight'),
            'margin': hyper_config.get_float('margin', min=0, max=10, step=0.01, default=2.5,
                                             help='The margin by which we want the correct choice to beat the incorrect choices.'),
            'sublayer_weights': hyper_config.get_sublayer('SubLayerWeights', sub_layer_type=TimeVaryingEmbedding,
                                                          fixed={'dims': sublayer_count, 'time_shape': (3, 15)},
                                                          help='The weights for each of the sublayers that get combined together linearly.'),
            'sublayer_metadata': sublayer_metadatas,
            # 'downcast_embeddings': hyper_config.get_sublayer('DowncastEmbeddings', sub_layer_type=WDense,
            #                                                  fixed={ 'dims': dims },
            #                                                  help='Downcast the size of the card embeddings to make it fit in memory.'),
        }

    def call(self, inputs, training=False):
        card_choices = tf.ensure_shape(tf.cast(inputs[0], dtype=tf.int32, name='card_choices'), (None, MAX_CARDS_IN_PACK))
        # basics = tf.ensure_shape(tf.cast(inputs[1], dtype=tf.int32, name='basics'), (None, MAX_BASICS))
        pool = tf.ensure_shape(tf.cast(inputs[2], dtype=tf.int32, name='pool'), (None, MAX_PICKED))
        seen_packs = tf.ensure_shape(tf.cast(inputs[3], dtype=tf.int32, name='seen_packs'), (None, MAX_SEEN_PACKS, MAX_CARDS_IN_PACK))
        seen_coords = tf.ensure_shape(tf.cast(inputs[4], dtype=tf.int32, name='seen_coords'), (None, MAX_SEEN_PACKS, 4, 2))
        seen_coord_weights = tf.ensure_shape(tf.cast(inputs[5], dtype=self.compute_dtype, name='seen_coord_weights'), (None, MAX_SEEN_PACKS, 4))
        coords = tf.ensure_shape(tf.cast(inputs[6], dtype=tf.int32, name='coords'), (None, 4, 2))
        coord_weights = tf.ensure_shape(tf.cast(inputs[7], dtype=self.compute_dtype, name='coord_weights'), (None, 4))
        loss_dtype = tf.float32
        sublayer_scores = []
        card_choice_embeds = tf.gather(self.card_embeddings, card_choices)
        mask = tf.cast(card_choices > 0, dtype=self.compute_dtype, name='pack_mask')
        print(self.card_embeddings.shape)
        if self.rate_off_pool:
            pool_embeds = tf.gather(self.card_embeddings, pool)
            sublayer_scores.append(self.rate_off_pool((card_choice_embeds, pool_embeds), training=training, mask=(card_choices > 0, pool > 0)))
        if self.rate_off_seen:
            flat_seen_packs = tf.reshape(seen_packs, (-1, tf.shape(seen_packs)[-1]))
            seen_pack_embeds = tf.gather(self.card_embeddings, flat_seen_packs)
            flat_pack_embeds = self.embed_pack(seen_pack_embeds, training=training, mask=flat_seen_packs > 0)
            pack_embeds = tf.reshape(flat_pack_embeds, tf.concat((tf.shape(seen_packs)[:-1], [-1]), axis=0))
            position_embeds = self.embed_pack_position((seen_coords, seen_coord_weights), training=training)
            pack_embeds =  pack_embeds + position_embeds
            seen_mask = tf.math.reduce_any(seen_packs > 0, axis=-1)
            sublayer_scores.append(self.rate_off_seen((card_choice_embeds, pack_embeds), training=training, mask=(card_choices > 0, seen_mask)))
        if self.rate_card:
            card_ratings = tf.squeeze(self.rate_card(card_choice_embeds, training=training), axis=-1) * mask
            sublayer_scores.append(card_ratings)
            tf.summary.histogram('card_ratings', card_ratings)
        sublayer_scores = tf.stack(sublayer_scores, axis=-1, name='stacked_sublayer_scores')
        sublayer_weights = tf.math.softplus(self.sublayer_weights((coords, coord_weights), training=training))
        sublayer_scores = tf.cast(sublayer_scores, dtype=self.compute_dtype)
        sublayer_weights = tf.cast(sublayer_weights, dtype=self.compute_dtype)
        scores = tf.einsum('...ps,...s->...p', sublayer_scores, sublayer_weights, name='scores')
        # scores = tf.reduce_sum(sublayer_scores, axis=-1)
        if len(inputs) > 9:
            mask = tf.cast(mask, dtype=loss_dtype)
            y_idx = tf.ensure_shape(tf.cast(inputs[8], dtype=tf.int32, name='y_idx'), (None,))
            riskiness = tf.ensure_shape(tf.cast(inputs[9], dtype=tf.float32, name='riskiness'), (None, MAX_CARDS_IN_PACK))
            pos_scores = (tf.cast(scores, dtype=loss_dtype, name='cast_scores') + tf.constant(LARGE_INT, dtype=loss_dtype)) * mask
            neg_scores = (tf.constant( LARGE_INT, dtype=loss_dtype) - tf.cast(scores, dtype=loss_dtype, name='cast_scores')) * mask
            scores = tf.where(tf.expand_dims(y_idx == 0, -1), pos_scores, neg_scores, name='scores_after_y_idx')
            probs = tf.nn.softmax(scores, axis=-1, name='probs')
            probs_correct = tf.concat([probs[:, :1], tf.constant(1, dtype=loss_dtype) - probs[:, 1:]], axis=1)
            prob_chosen = tf.gather(probs, 0, axis=1, name='prob_chosen')
            num_in_packs = tf.reduce_sum(mask, name='num_in_packs')
            log_losses = tf.negative(tf.math.log((1 - 2e-10) * probs_correct + 1e-10, name='log_probs'), name='log_loss')
            sample_weights = tf.where(scores > tf.expand_dims(scores[:, 0], -1), riskiness, tf.ones_like(riskiness))
            log_losses = sample_weights * log_losses
            log_loss = tf.reduce_sum(log_losses) / num_in_packs
            tf.summary.scalar('log_loss', log_loss)
            log_loss_weighted = tf.math.multiply(log_loss,
                                           tf.constant(self.log_loss_weight, dtype=loss_dtype, name='log_loss_weight'),
                                           name='log_loss_weighted')
            self.add_metric(log_losses, 'pick_log_loss')
            score_diffs = tf.subtract(tf.add(tf.constant(self.margin, dtype=scores.dtype),
                                             scores[:, 1:]), scores[:, :1],
                                      name='score_diffs') * mask[:, 1:]
            clipped_diffs = tf.math.maximum(tf.constant(0, dtype=loss_dtype), score_diffs, name='clipped_score_diffs') * sample_weights[:, 1:] - (sample_weights[:, 1:] - 1) * self.margin
            triplet_losses = tf.math.reduce_sum(clipped_diffs) / num_in_packs
            tf.summary.scalar('triplet_loss', tf.reduce_mean(triplet_losses))
            triplet_loss_weighted = tf.math.multiply(triplet_losses,
                                           tf.constant(1 - self.log_loss_weight, dtype=loss_dtype, name='triplet_loss_weight'),
                                           name='triplet_loss_weighted')
            self.add_metric(triplet_losses, 'triplet_loss')
            # max_scores = tf.reduce_logsumexp(scores - tf.constant(LARGE_INT, dtype=loss_dtype), axis=-1)
            # max_scores = max_scores + tf.stop_gradient(tf.reduce_max(scores - LARGE_INT, axis=-1) - max_scores)
            # max_scores = tf.reduce_max(scores - tf.constant(LARGE_INT), axis=-1)
            # min_scores = -tf.reduce_logsumexp(-scores + mask * tf.constant(LARGE_INT, dtype=loss_dtype), axis=-1)
            # min_scores = min_scores + tf.stop_gradient(tf.reduce_min(scores + (1 - 2 * mask) * LARGE_INT, axis=-1) - min_scores)
            # tf.summary.histogram('max_diff_scores', max_scores - min_scores)
            #tf.summary.histogram('min_scores', min_scores)
            #tf.summary.histogram('max_scores', max_scores)
            # low_score_loss = tf.reduce_mean(tf.maximum(min_scores - max_scores, -0.1)) + 0.1
            #high_score_loss = tf.reduce_mean(tf.maximum(max_scores - min_scores, 10)) - 10
            # self.add_metric(low_score_loss, 'low_score_loss')
            # self.add_metric(high_score_loss, 'high_score_loss')
            # score_loss = low_score_loss + high_score_loss
            score_loss = 0
            chosen_idx = tf.zeros_like(y_idx)
            scores = tf.cast(scores, dtype=tf.float32)
            top_1_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(chosen_idx, scores, 1)
            top_2_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(chosen_idx, scores, 2)
            top_3_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(chosen_idx, scores, 3)
            self.add_metric(top_1_accuracy, 'accuracy')
            scores = tf.cast(scores, dtype=self.compute_dtype)
            mask = tf.cast(mask, dtype=self.compute_dtype)
            #Logging for Tensorboard
            tf.summary.scalar('accuracy_top_1', tf.reduce_mean(top_1_accuracy))
            tf.summary.scalar('accuracy_top_2', tf.reduce_mean(top_2_accuracy))
            tf.summary.scalar('accuracy_top_3', tf.reduce_mean(top_3_accuracy))
            tf.summary.scalar('prob_correct', tf.reduce_mean(prob_chosen))
            tf.summary.histogram('scores', (scores - tf.constant(LARGE_INT, dtype=self.compute_dtype)) * mask)
            tf.summary.histogram('scores_0', ((scores - tf.constant(LARGE_INT, dtype=self.compute_dtype)) * mask)[:, :1])
            # tf.summary.histogram('score_diffs', score_diffs)
            tf.summary.histogram('probs', probs)
            tf.summary.histogram('prob_chosen', prob_chosen)
            # tf.summary.histogram('log_losses', log_losses)
            # tf.summary.histogram('triplet_losses', clipped_diffs)
            loss = tf.cast(triplet_loss_weighted + log_loss_weighted + score_loss, dtype=self.compute_dtype)
            self.add_loss(loss)
            return loss
        return sublayer_scores, sublayer_weights
