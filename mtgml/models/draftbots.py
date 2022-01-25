import math

import tensorflow as tf

from mtgml.constants import LARGE_INT, MAX_BASICS, MAX_CARDS_IN_PACK, MAX_PICKED, MAX_SEEN_PACKS
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.contextual_rating import ContextualRating
from mtgml.layers.item_embedding import ItemEmbedding
from mtgml.layers.mlp import MLP
from mtgml.layers.set_embedding import AttentiveSetEmbedding
from mtgml.layers.time_varying_embedding import TimeVaryingEmbedding


class DraftBot(ConfigurableLayer, tf.keras.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        print(input_shapes)
        pool_context_ratings = hyper_config.get_bool('pool_context_ratings', default=True,
                                                     help='Whether to rate cards based on how the go with the other cards in the pool so far.')
        seen_context_ratings = hyper_config.get_bool('seen_context_ratings', default=True,
                                                     help='Whether to rate cards based on the packs seen so far.')
        item_ratings = hyper_config.get_bool('item_ratings', default=True, help='Whether to give each card a rating independent of context.')
        seen_pack_dims = hyper_config.get_int('seen_pack_dims', min=8, max=512, step=8, default=32,
                                              help='The number of dimensions to embed seen packs into.')\
            if seen_context_ratings else 0
        num_cards = hyper_config.get_int('num_cards', min=1, max=None, default=None,
                                         help='The number of items that must be embedded. Should be 1 + the max index expected to see.')
        sublayer_count = len([x for x in (pool_context_ratings, seen_context_ratings, item_ratings) if x])
        return {
            'num_cards': num_cards - 1,
            'embed_cards': hyper_config.get_sublayer('EmbedCards', sub_layer_type=ItemEmbedding,
                                                    fixed={'num_items': num_cards},
                                                    seed_mod=29, help='The embeddings for the card objects.'),
            'rate_off_pool': hyper_config.get_sublayer('RatingFromPool', sub_layer_type=ContextualRating,
                                                       seed_mod=31, help='The layer that rates based on the other cards that have been picked.')
                             if pool_context_ratings else None,
            'seen_pack_dims': seen_pack_dims,
            'embed_pack': hyper_config.get_sublayer('EmbedPack', sub_layer_type=AttentiveSetEmbedding,
                                                    fixed={'Decoder': {'Final': {'dims': seen_pack_dims, 'activation': 'linear'}}},
                                                    seed_mod=37, help='The layer that embeds the packs that have been seen so far.')
                          if seen_context_ratings else None,
            'pack_position_count': input_shapes[3][1] if input_shapes else 45,
            'embed_pack_position': hyper_config.get_sublayer('EmbedPackPosition',
                                                             sub_layer_type=TimeVaryingEmbedding,
                                                             fixed={'dims': seen_pack_dims,
                                                                    'time_shape': (3, 15)},
                                                             seed_mod=23, help='The embedding for the position in the draft')
                                  if seen_context_ratings else None,
            'rate_off_seen': hyper_config.get_sublayer('RatingFromSeen', sub_layer_type=ContextualRating,
                                                       seed_mod=31, help='The layer that rates based on the embeddings of the packs that have been seen.')
                             if seen_context_ratings else None,
            'rate_card': hyper_config.get_sublayer('CardRating', sub_layer_type=MLP, seed_mod=13,
                                                   fixed={'Final': {'dims': 1, 'activation': 'relu'}},
                                                   help='Translates embeddings into linear ratings.')
            # 'rate_card': hyper_config.get_sublayer('CardRating', sub_layer_type=ItemRating, seed_mod=13,
            #                                        fixed={'num_items': num_cards},
            #                                        help='The linear ordering of cards by value.')
                         if item_ratings else None,
            'log_loss_weight': hyper_config.get_float('log_loss_weight', min=0, max=1, step=0.01,
                                                      default=0.5, help='The weight given to log_loss vs triplet_loss. Triplet loss weight is 1 - log_loss_weight'),
            'margin': hyper_config.get_float('margin', min=0, max=10, step=0.01, default=2.5,
                                             help='The margin by which we want the correct choice to beat the incorrect choices.'),
            'sublayer_weights': hyper_config.get_sublayer('SubLayerWeights', sub_layer_type=TimeVaryingEmbedding,
                                                          fixed={'dims': sublayer_count, 'time_shape': (3, 15)},
                                                          help='The weights for each of the sublayers that get combined together linearly.'),
        }

    def call(self, inputs, training=False):
        card_choices = tf.ensure_shape(tf.cast(inputs[0], dtype=tf.int32, name='card_choices'), (None, MAX_CARDS_IN_PACK))
        basics = tf.ensure_shape(tf.cast(inputs[1], dtype=tf.int32, name='basics'), (None, MAX_BASICS))
        pool = tf.ensure_shape(tf.cast(inputs[2], dtype=tf.int32, name='pool'), (None, MAX_PICKED))
        seen_packs = tf.ensure_shape(tf.cast(inputs[3], dtype=tf.int32, name='seen_packs'), (None, MAX_SEEN_PACKS, MAX_CARDS_IN_PACK))
        seen_coords = tf.ensure_shape(tf.cast(inputs[4], dtype=tf.int32, name='seen_coords'), (None, MAX_SEEN_PACKS, 4, 2))
        seen_coord_weights = tf.ensure_shape(tf.cast(inputs[5], dtype=self.compute_dtype, name='seen_coord_weights'), (None, MAX_SEEN_PACKS, 4))
        coords = tf.ensure_shape(tf.cast(inputs[6], dtype=tf.int32, name='coords'), (None, 4, 2))
        coord_weights = tf.ensure_shape(tf.cast(inputs[7], dtype=self.compute_dtype, name='coord_weights'), (None, 4))
        y_idx = tf.ensure_shape(tf.cast(inputs[8], dtype=tf.int32, name='y_idx'), (None,))
        loss_dtype = tf.float32
        sublayer_scores = []
        card_choice_embeds = self.embed_cards(card_choices, training=training)
        shifts = []
        mask = tf.cast(card_choices > 0, dtype=self.compute_dtype, name='pack_mask')
        if self.rate_off_pool:
            pool_embeds = self.embed_cards(pool, training=training)
            sublayer_scores.append(self.rate_off_pool((card_choice_embeds, pool_embeds), training=training))
            shifts.append(LARGE_INT)
        if self.rate_off_seen:
            seen_pack_embeds = self.embed_cards(seen_packs, training=training)
            pack_embeds = self.embed_pack(seen_pack_embeds, training=training)
            position_embeds = self.embed_pack_position((seen_coords, seen_coord_weights), training=training)
            pack_embeds = tf.constant(math.sqrt(self.seen_pack_dims), self.compute_dtype) * pack_embeds + position_embeds
            pack_embeds._keras_mask = tf.math.reduce_any(seen_packs > 0, axis=-1)
            sublayer_scores.append(self.rate_off_seen((card_choice_embeds, pack_embeds), training=training))
            shifts.append(LARGE_INT)
        if self.rate_card:
            card_ratings = tf.squeeze(self.rate_card(card_choice_embeds, training=training), axis=-1) * mask
            sublayer_scores.append(card_ratings)
            shifts.append(0.0)
            tf.summary.histogram('card_ratings', card_ratings)
        sublayer_scores = tf.stack(sublayer_scores, axis=-1, name='stacked_sublayer_scores')
        sublayer_weights = tf.math.softplus(self.sublayer_weights((coords, coord_weights), training=training))
        sublayer_scores = tf.cast(sublayer_scores, dtype=self.compute_dtype)
        sublayer_weights = tf.cast(sublayer_weights, dtype=self.compute_dtype)
        scores = tf.einsum('...ps,...s->...p', sublayer_scores, sublayer_weights, name='scores')
        shifts = tf.cast(tf.stack(shifts, axis=-1, name='shifts'), dtype=self.compute_dtype)
        shifts = tf.einsum('...s,s->...', sublayer_weights, shifts, name='shifts_weighted')
        scores = tf.cast(scores, dtype=loss_dtype, name='cast_scores') * mask
        neg_scores = tf.math.negative(scores) + tf.constant(2 * LARGE_INT, dtype=loss_dtype) * mask
        pos_scores = scores
        scores = tf.where(tf.expand_dims(y_idx == 0, -1), pos_scores, neg_scores, name='scores_after_y_idx')
        probs = tf.nn.softmax(scores, axis=-1, name='probs')
        prob_chosen = tf.gather(probs, 0, axis=1, name='prob_chosen')
        num_in_packs = tf.reduce_sum(mask, name='num_in_packs')
        log_losses = tf.constant(32, dtype=loss_dtype) * tf.negative(tf.math.log(prob_chosen + 1e-09, name='log_probs'), name='log_loss')
        tf.summary.scalar('log_loss', tf.reduce_mean(log_losses))
        self.add_loss(tf.math.multiply(tf.math.reduce_mean(log_losses, name='log_loss'),
                                       tf.constant(self.log_loss_weight, dtype=loss_dtype, name='log_loss_weight'),
                                       name='log_loss_weighted'))
        self.add_metric(log_losses, 'pick_log_loss')
        score_diffs = tf.subtract(tf.add(tf.constant(self.margin, dtype=scores.dtype),
                                         scores[:, 1:]), scores[:, :1],
                                  name='score_diffs') * mask[:, 1:]
        clipped_diffs = tf.math.maximum(tf.constant(0, dtype=loss_dtype), score_diffs, name='clipped_score_diffs'),
        triplet_losses = tf.constant(32, dtype=loss_dtype) * tf.math.reduce_sum(clipped_diffs) / num_in_packs
        tf.summary.scalar('triplet_loss', tf.reduce_mean(triplet_losses))
        self.add_loss(tf.math.multiply(triplet_losses,
                                       tf.constant(1 - self.log_loss_weight, dtype=loss_dtype, name='triplet_loss_weight'),
                                       name='triplet_loss_weighted'))
        self.add_metric(triplet_losses, 'triplet_loss')
        chosen_idx = tf.zeros_like(y_idx)
        scores = tf.cast(scores, dtype=tf.float32)
        top_1_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(chosen_idx, scores, 1)
        top_2_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(chosen_idx, scores, 2)
        top_3_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(chosen_idx, scores, 3)
        self.add_metric(top_1_accuracy, 'accuracy_top_1')
        tf.summary.scalar('accuracy_top_1', tf.reduce_mean(top_1_accuracy))
        self.add_metric(top_2_accuracy, 'accuracy_top_2')
        tf.summary.scalar('accuracy_top_2', tf.reduce_mean(top_2_accuracy))
        self.add_metric(top_3_accuracy, 'accuracy_top_3')
        tf.summary.scalar('accuracy_top_3', tf.reduce_mean(top_3_accuracy))
        self.add_metric(prob_chosen, 'prob_correct')
        tf.summary.scalar('prob_correct', tf.reduce_mean(prob_chosen))
        scores = tf.cast(scores, dtype=self.compute_dtype)
        mask = tf.cast(mask, dtype=self.compute_dtype)
        #Logging for Tensorboard
        tf.summary.histogram('scores', (scores - tf.expand_dims(shifts, -1)) * mask)
        tf.summary.histogram('score_diffs', score_diffs)
        tf.summary.histogram('prob_correct', prob_chosen)
        tf.summary.histogram('log_losses', log_losses)
        tf.summary.histogram('triplet_losses', clipped_diffs)
        return scores