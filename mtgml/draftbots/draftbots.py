import math

import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.contextual_rating import ContextualRating
from mtgml.layers.item_embedding import ItemEmbedding
from mtgml.layers.item_rating import ItemRating
from mtgml.layers.set_emedding import AdditiveSetEmbedding
from mtgml.layers.time_varying_embedding import TimeVaryingEmbedding
from mtgml.tensorboard.timeseries import log_timeseries


class DraftBot(tf.keras.models.Model, ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        pool_context_ratings = hyper_config.get_bool('pool_context_ratings', default=False,
                                                     help='Whether to rate cards based on how the go with the other cards in the pool so far.')
        seen_context_ratings = hyper_config.get_bool('seen_context_ratings', default=False,
                                                     help='Whether to rate cards based on the packs seen so far.')
        item_ratings = hyper_config.get_bool('item_ratings', default=False, help='Whether to give each card a rating independent of context.')
        seen_pack_dims = hyper_config.get_int('seen_pack_dims', min=8, max=512, step=8, default=32,
                                              help='The number of dimensions to embed seen packs into.')\
            if seen_context_ratings else 0
        props = {
            'embed_card': hyper_config.get_sublayer('EmbedCards', sub_layer_type=ItemEmbedding,
                                                    seed_mod=29, help='The embeddings for the card objects.')
            'rate_off_pool': hyper_config.get_sublayer('RatingFromPool', sub_layer_type=ContextualRating,
                                                       seed_mod=31, help='The layer that rates based on the other cards that have been picked.')
                             if pool_context_ratings else None,
            'seen_pack_dims': seen_pack_dims,
            'embed_pack': hyper_config.get_sublayer('EmbedPack', sub_layer_type=AdditiveSetEmbedding,
                                                    fixed={'Decoder': {'final': {'dims': seen_pack_dims}}},
                                                    seed_mod=37, help='The layer that embeds the packs that have been seen so far.')
                          if seen_context_ratings else None,
            'pack_position_count': input_shapes[2][1] if input_shapes else 45,
            'embed_pack_position': hyper_config.get_sublayer('EmbedPackPosition',
                                                             sub_layer_type=TimeVaryingEmbedding,
                                                             fixed={'dims': seen_pack_dims,
                                                                    'time_shape': (input_shapes[2][1] if input_shapes else 45,)},
                                                             seed_mod=23, help='The embedding for the position in the draft')
                                  if seen_context_ratings else None
            'rate_off_seen': hyper_config.get_sublayer('RatingFromSeen', sub_layer_type=ContextualRating,
                                                       seed_mod=31, help='The layer that rates based on the embeddings of the packs that have been seen.')
                             if seen_context_ratings else None,
            'rate_card': hyper_config.get_sublayer('CardRating', sub_layer_type=ItemRating, seed__mod=13),
        }

    def call(self, inputs, training=False):
        inputs = (
            tf.cast(inputs[0], dtype=tf.int32, name='card_choices'),
            tf.cast(inputs[1], dtype=tf.int32, name='pool'),
            tf.cast(inputs[2], dtype=tf.int32, name='seen_packs'),
            tf.cast(inputs[3], dtype=tf.int32, name='coords'),
            tf.cast(inputs[4], dtype=tf.float32, name='coord_weights'),
            tf.cast(inputs[5], dtype=tf.int32, name='y_idx'),
        )
        loss_dtype = tf.float32
        sublayer_scores = []
        card_choice_embeds = self.embed_card(inputs[0])
        pool_embeds = self.embed_card(inputs[1])
        seen_pack_embeds = self.embed_card(inputs[2])
        position_embed  
        if self.pool_context_ratings:
            sublayer_scores.append(self.pool_rating_layer((card_choice_embeds, pool_embeds)))
        if self.seen_context_ratings:
            pack_embeds = self.embed_pack(seen_pack_embeds)
            pack_positions = tf.reshape(tf.range(self.pack_position_count), (1, -1, 1, 1))
            position_embeds = self.embed_pack_position(pack_positions, tf.ones_like(pack_positions))
            pack_embeds = tf.math.sqrt(self.seen_pack_dims) * pack_embeds + position_embeds
            sublayer_scores.append(self.rate_off_seen((card_choice_embeds, pack_embeds)))
        if self.item_ratings:
            sublayer_scores.append(self.rate_card(inputs[0], training=training))
        if len(sublayer_scores) > 1:
            sublayer_scores = tf.stack(sublayer_scores, axis=-1, name='stacked_sublayer_scores')
            scores = tf.math.reduce_sum(sublayer_scores, axis=-1, name='scores')
        else:
            scores = sublayer_scores[0]
        scores = tf.cast(scores, dtype=loss_dtype, name='cast_scores')
        mask = tf.cast(inputs[0] > 0, dtype=loss_dtype, name='pack_mask')
        neg_scores = tf.stop_gradient(tf.math.reduce_max(scores, axis=-1, keepdims=True)) - scores
        pos_scores = scores - tf.stop_gradient(tf.math.reduce_min(scores, axis=-1, keepdims=True))
        scores = tf.where(tf.expand_dims(inputs[5] == 0, -1), pos_scores, neg_scores, name='scores_after_y_idx')
        scores = tf.math.multiply(scores, mask, name='masked_scores')
        probs_with_zeros = tf.nn.softmax(scores, axis=-1, name='probs_with_zeros')
        probs = tf.linalg.normalize(tf.math.multiply(probs_with_zeros, mask, name='masked_probs'),
                                    ord=1, axis=-1, name='probs')[0]
        prob_chosen = tf.gather(probs, 0, axis=1, name='prob_chosen')
        num_in_packs = tf.reduce_sum(mask, name='num_in_packs')
        log_losses = tf.negative(tf.math.log(prob_chosen + 1e-04, name='log_probs'), name='log_losses')
        self.add_loss(tf.math.multiply(tf.math.reduce_mean(log_losses, name='log_loss'),
                                       tf.constant(self.log_loss_weight, dtype=loss_dtype, name='log_loss_weight'),
                                       name='log_loss_weighted'))
        self.add_metric(log_losses, 'pick_log_loss')
        score_diffs = tf.subtract(tf.add(tf.constant(self.margin, dtype=scores.dtype),
                                         scores[:, 1:]), tf.expand_dims(scores[:, 0], -1),
                                  name='score_diffs')
        clipped_diffs = mask[:, 1:] * tf.math.maximum(tf.constant(0, dtype=loss_dtype), score_diffs, name='clipped_score_diffs'),
        triplet_losses = tf.math.reduce_sum(clipped_diffs) / num_in_packs
        self.add_loss(tf.math.multiply(triplet_losses,
                                       tf.constant(self.triplet_loss_weight, dtype=loss_dtype, name='triplet_loss_weight'),
                                       name='triplet_loss_weighted'))
        self.add_metric(triplet_losses, 'triplet_losses')
        chosen_idx = tf.zeros_like(inputs[5])
        top_1_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(chosen_idx, scores, 1)
        top_2_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(chosen_idx, scores, 2)
        top_3_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(chosen_idx, scores, 3)
        self.add_metric(top_1_accuracy, 'accuracy_top_1')
        self.add_metric(top_2_accuracy, 'accuracy_top_2')
        self.add_metric(top_3_accuracy, 'accuracy_top_3')
        self.add_metric(prob_chosen, 'prob_correct')
        #Logging for Tensorboard
        tf.summary.histogram('outputs/scores', scores)
        tf.summary.histogram('outputs/score_diffs', score_diffs)
        tf.summary.histogram('outputs/prob_correct', prob_chosen)
        tf.summary.histogram('outputs/log_losses', log_losses)
        tf.summary.histogram('outputs/triplet_losses', clipped_diffs)
        return scores
