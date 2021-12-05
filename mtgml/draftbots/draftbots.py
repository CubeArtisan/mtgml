import math

import tensorflow as tf

from mtgml.tensorboard.timeseries import log_timeseries


class DraftBot(tf.keras.models.Model):
    def __init__(self, num_items, embed_dims=64, seen_dims=16, pool_dims=32,
                 triplet_loss_weight=0.0, dropout_pool=0.0, dropout_seen=0.0,
                 margin=1, log_loss_weight=1.0, activation='selu',
                 pool_context_ratings=True, seen_context_ratings=True, item_ratings=True,
                 bounded_distance=True, final_activation='linear', normalize_sum=True,
                 pool_hidden_units=64, seen_hidden_units=32, pool_dropout_dense=0.0,
                 seen_dropout_dense=0.0, time_shape=(3, 15), **kwargs):
        kwargs.update({'dynamic': False})
        super(DraftBot, self).__init__(**kwargs)
        self.num_items = num_items
        self.embed_dims = embed_dims
        self.seen_dims = seen_dims
        self.pool_dims = pool_dims
        self.triplet_loss_weight = triplet_loss_weight
        self.dropout_pool = dropout_pool
        self.dropout_seen = dropout_seen
        self.margin = margin
        self.log_loss_weight = log_loss_weight
        self.activation = activation
        self.pool_context_ratings = pool_context_ratings
        self.seen_context_ratings = seen_context_ratings
        self.item_ratings = item_ratings
        self.bounded_distance = bounded_distance
        self.final_activation = final_activation
        self.normalize_sum = normalize_sum
        self.pool_hidden_units = pool_hidden_units
        self.seen_hidden_units = seen_hidden_units
        self.pool_dropout_dense = pool_dropout_dense
        self.seen_dropout_dense = seen_dropout_dense
        self.time_shape = time_shape
        if not pool_context_ratings and not seen_context_ratings and not item_ratings:
            raise ValueError('Must have at least one sublayer enabled')

    def get_config(self):
        config = super(DraftBot, self).get_config()
        config.update({
            "num_items": self.num_items,
            "embed_dims": self.embed_dims,
            "seen_dims": self.seen_dims,
            "pool_dims": self.pool_dims,
            "triplet_loss_weight": self.triplet_loss_weight,
            "dropout_pool": self.dropout_pool,
            "dropout_seen": self.dropout_seen,
            "margin": self.margin,
            "log_loss_weight": self.log_loss_weight,
            "activation": self.activation,
            "bounded_distance": self.bounded_distance,
            "final_activation": self.final_activation,
            "normalize_sum": self.normalize_sum,
            "pool_hidden_units": self.pool_hidden_units,
            "seen_hidden_units": self.seen_hidden_units,
            "pool_dropout_dense": self.pool_dropout_dense,
            "seen_dropout_dense": self.seen_dropout_dense,
            "time_shape": self.time_shape,
        })
        return config

    def build(self, input_shapes):
        sublayers = 0
        if self.pool_context_ratings:
            self.pool_rating_layer = ContextualRating(self.num_items, self.embed_dims, self.pool_dims,
                                                        item_dropout_rate=self.dropout_pool,
                                                        dense_dropout_rate=self.pool_dropout_dense,
                                                        activation=self.activation,
                                                        bounded_distance=self.bounded_distance,
                                                        final_activation=self.final_activation,
                                                        normalize_sum=self.normalize_sum,
                                                        hidden_units=self.pool_hidden_units,
                                                        time_shape=self.time_shape,
                                                        name='RatingFromPool')
            sublayers += 1
        if self.seen_context_ratings:
            self.seen_rating_layer = ContextualRating(self.num_items, self.embed_dims, self.seen_dims,
                                                      item_dropout_rate=self.dropout_seen,
                                                      dense_dropout_rate=self.seen_dropout_dense,
                                                      activation=self.activation,
                                                      bounded_distance=self.bounded_distance,
                                                      final_activation=self.final_activation,
                                                      normalize_sum=self.normalize_sum,
                                                      hidden_units=self.seen_hidden_units,
                                                      time_shape=self.time_shape,
                                                      name='RatingFromSeen')
            sublayers += 1
        if self.item_ratings:
            self.item_rating_layer = ItemRating(self.num_items, bounded=self.bounded_distance,
                                                name='CardRating')
            sublayers += 1
        if sublayers > 1:
            self.time_varying = TimeVaryingLinearEmbedding(self.time_shape, sublayers,
                                                  initializer=tf.constant_initializer(-4),
                                                  name='OracleCombination')

    def call(self, inputs, training=False):
        inputs = (
            tf.cast(inputs[0], dtype=tf.int32, name='card_choices'),
            tf.cast(inputs[1], dtype=tf.int32, name='pool'),
            tf.cast(inputs[2], dtype=tf.int32, name='seen'),
            tf.cast(inputs[3], dtype=tf.int32, name='coords'),
            tf.cast(inputs[4], dtype=tf.float32, name='coord_weights'),
            tf.cast(inputs[5], dtype=tf.int32, name='inputs[5]'),
        )
        loss_dtype = tf.float32
        sublayer_scores = []
        if self.pool_context_ratings:
            sublayer_scores.append(self.pool_rating_layer((inputs[0], inputs[1], (inputs[3], inputs[4]))))
        if self.seen_context_ratings:
            sublayer_scores.append(self.seen_rating_layer((inputs[0], inputs[2], (inputs[3], inputs[4]))))
        if self.item_ratings:
            sublayer_scores.append(self.item_rating_layer(inputs[0], training=training))
        if len(sublayer_scores) > 1:
            sublayer_weights = 8 * tf.math.softplus(self.time_varying((inputs[3], inputs[4]), training=training), name='sublayer_weights')
            sublayer_scores = tf.stack(sublayer_scores, axis=-1, name='stacked_sublayer_scores')
            scores = tf.einsum('...o,...co->...c', sublayer_weights, sublayer_scores, name='scores')
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
