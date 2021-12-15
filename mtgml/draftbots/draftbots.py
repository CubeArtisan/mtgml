import math

import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.contextual_rating import ContextualRating
from mtgml.layers.item_embedding import ItemEmbedding
from mtgml.layers.item_rating import ItemRating
from mtgml.layers.set_embedding import AdditiveSetEmbedding
from mtgml.layers.time_varying_embedding import TimeVaryingEmbedding
from mtgml.tensorboard.timeseries import log_timeseries


class DraftBot(ConfigurableLayer, tf.keras.models.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
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
            'embed_cards': hyper_config.get_sublayer('EmbedCards', sub_layer_type=ItemEmbedding,
                                                    fixed={'num_items': num_cards},
                                                    seed_mod=29, help='The embeddings for the card objects.'),
            'rate_off_pool': hyper_config.get_sublayer('RatingFromPool', sub_layer_type=ContextualRating,
                                                       seed_mod=31, help='The layer that rates based on the other cards that have been picked.')
                             if pool_context_ratings else None,
            'seen_pack_dims': seen_pack_dims,
            'embed_pack': hyper_config.get_sublayer('EmbedPack', sub_layer_type=AdditiveSetEmbedding,
                                                    fixed={'Decoder': {'Final': {'dims': seen_pack_dims}}},
                                                    seed_mod=37, help='The layer that embeds the packs that have been seen so far.')
                          if seen_context_ratings else None,
            'pack_position_count': input_shapes[2][1] if input_shapes else 45,
            'embed_pack_position': hyper_config.get_sublayer('EmbedPackPosition',
                                                             sub_layer_type=TimeVaryingEmbedding,
                                                             fixed={'dims': seen_pack_dims,
                                                                    'time_shape': (3, 15)},
                                                             seed_mod=23, help='The embedding for the position in the draft')
                                  if seen_context_ratings else None,
            'rate_off_seen': hyper_config.get_sublayer('RatingFromSeen', sub_layer_type=ContextualRating,
                                                       seed_mod=31, help='The layer that rates based on the embeddings of the packs that have been seen.')
                             if seen_context_ratings else None,
            'rate_card': hyper_config.get_sublayer('CardRating', sub_layer_type=ItemRating, seed_mod=13,
                                                   fixed={'num_items': num_cards},
                                                   help='The linear ordering of cards by value.'),
            'log_loss_weight': hyper_config.get_float('log_loss_weight', min=0, max=1, step=0.01,
                                                      default=0.5, help='The weight given to log_loss vs triplet_loss. Triplet loss weight is 1 - log_loss_weight'),
            'margin': hyper_config.get_float('margin', min=0, max=10, step=0.01, default=2.5,
                                             help='The margin by which we want the correct choice to beat the incorrect choices.'),
            'sublayer_weights': hyper_config.get_sublayer('SubLayerWeights', sub_layer_type=TimeVaryingEmbedding,
                                                          fixed={'dims': sublayer_count, 'time_shape': (3, 15)},
                                                          help='The weights for each of the sublayers that get combined together linearly.'),
        }

    def call(self, inputs, training=False):
        with tf.summary.record_if(tf.summary.experimental.get_step() % 1024 == 0 if
                                  tf.summary.experimental.get_step() is not None else False):
            inputs = (
                tf.cast(inputs[0], dtype=tf.int32, name='card_choices'),
                tf.cast(inputs[1], dtype=tf.int32, name='pool'),
                tf.cast(inputs[2], dtype=tf.int32, name='seen_packs'),
                tf.cast(inputs[3], dtype=tf.int32, name='seen_coords'),
                tf.cast(inputs[4], dtype=tf.float32, name='seen_coord_weights'),
                tf.cast(inputs[5], dtype=tf.int32, name='coords'),
                tf.cast(inputs[6], dtype=tf.float32, name='coord_weights'),
                tf.cast(inputs[7], dtype=tf.int32, name='y_idx'),
            )
            loss_dtype = tf.float32
            sublayer_scores = []
            card_choice_embeds = self.embed_cards(inputs[0], training=training)
            pool_embeds = self.embed_cards(inputs[1], training=training)
            seen_pack_embeds = self.embed_cards(inputs[2], training=training)
            if self.rate_off_pool:
                sublayer_scores.append(self.rate_off_pool((card_choice_embeds, pool_embeds), training=training))
            if self.rate_off_seen:
                pack_embeds = self.embed_pack(seen_pack_embeds, training=training)
                pack_embeds_mask = pack_embeds._keras_mask
                position_embeds = tf.expand_dims(self.embed_pack_position(inputs[3:5], training=training), -2)
                pack_embeds = tf.constant(math.sqrt(self.seen_pack_dims), self.compute_dtype) * pack_embeds + position_embeds
                pack_embeds._keras_mask = pack_embeds_mask
                sublayer_scores.append(self.rate_off_seen((card_choice_embeds, pack_embeds), training=training))
            if self.rate_card:
                sublayer_scores.append(self.rate_card(inputs[0], training=training))
            sublayer_scores = tf.stack(sublayer_scores, axis=-1, name='stacked_sublayer_scores')
            sublayer_weights = tf.math.softplus(self.sublayer_weights(inputs[5:7], training=training))
            scores = tf.einsum('...ps,...s->...p', sublayer_scores, sublayer_weights)
            mask = tf.cast(inputs[0] > 0, dtype=loss_dtype, name='pack_mask')
            scores = tf.cast(scores, dtype=loss_dtype, name='cast_scores') * mask
            neg_scores = tf.math.negative(scores) + tf.constant(2e+09, dtype=loss_dtype) * mask
            pos_scores = scores
            scores = tf.where(tf.expand_dims(inputs[7] == 0, -1), pos_scores, neg_scores, name='scores_after_y_idx')
            probs = tf.nn.softmax(scores, axis=-1, name='probs')
            prob_chosen = tf.gather(probs, 0, axis=1, name='prob_chosen')
            num_in_packs = tf.reduce_sum(mask, name='num_in_packs')
            log_losses = tf.negative(tf.math.log(prob_chosen + 1e-09, name='log_probs'), name='log_losses')
            self.add_loss(tf.math.multiply(tf.math.reduce_mean(log_losses, name='log_loss'),
                                           tf.constant(self.log_loss_weight, dtype=loss_dtype, name='log_loss_weight'),
                                           name='log_loss_weighted'))
            self.add_metric(log_losses, 'pick_log_loss')
            score_diffs = tf.subtract(tf.add(tf.constant(self.margin, dtype=scores.dtype),
                                             scores[:, 1:]), scores[:, :1],
                                      name='score_diffs') * mask[:, 1:]
            clipped_diffs = tf.math.maximum(tf.constant(0, dtype=loss_dtype), score_diffs, name='clipped_score_diffs'),
            triplet_losses = tf.math.reduce_sum(clipped_diffs) / num_in_packs
            self.add_loss(tf.math.multiply(triplet_losses,
                                           tf.constant(1 - self.log_loss_weight, dtype=loss_dtype, name='triplet_loss_weight'),
                                           name='triplet_loss_weighted'))
            self.add_metric(triplet_losses, 'triplet_losses')
            chosen_idx = tf.zeros_like(inputs[7])
            top_1_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(chosen_idx, scores, 1)
            top_2_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(chosen_idx, scores, 2)
            top_3_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(chosen_idx, scores, 3)
            self.add_metric(top_1_accuracy, 'accuracy_top_1')
            self.add_metric(top_2_accuracy, 'accuracy_top_2')
            self.add_metric(top_3_accuracy, 'accuracy_top_3')
            self.add_metric(prob_chosen, 'prob_correct')
            #Logging for Tensorboard
            tf.summary.histogram('outputs/scores', (scores - tf.constant(2e+09, dtype=loss_dtype)) * mask)
            tf.summary.histogram('outputs/score_diffs', score_diffs)
            tf.summary.histogram('outputs/prob_correct', prob_chosen)
            tf.summary.histogram('outputs/log_losses', log_losses)
            tf.summary.histogram('outputs/triplet_losses', clipped_diffs)
            return scores
