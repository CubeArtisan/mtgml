import math

import tensorflow as tf

from mtgml.tensorboard.timeseries import log_timeseries
from mtgml.utils.dropout import dropout


class TimeVaryingLinearEmbedding(tf.keras.layers.Layer):
    def __init__(self, time_shape, embedding_dims,
                 initializer=tf.constant_initializer(0),
                 **kwargs):
        super(TimeVaryingLinearEmbedding, self).__init__(**kwargs)
        self.time_shape = time_shape
        self.embedding_dims = embedding_dims
        self.initializer = initializer

    def build(self, input_shape):
        self.embeddings = self.add_weight('embeddings', shape=(*self.time_shape, self.embedding_dims),
                                          initializer=self.initializer, trainable=True)

    def get_config(self):
        config = super(TimeVaryingLinearEmbedding, self).get_config()
        config.update({
            "time_shape": self.time_shape,
            "embedding_dims": self.embedding_dims,
            "initializer": self.initializer,
        })
        return config

    def call(self, inputs, training=False):
        coords, coord_weights = inputs
        component_embedding_values = tf.gather_nd(self.embeddings, coords, name='component_embedding_values')
        embedding_values = tf.einsum('...xe,...x->...e', component_embedding_values, coord_weights,
                                     name='embedding_values')
        return embedding_values


class SetEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_items, embed_dims, time_shape, final_dims=None, item_dropout_rate=0.0,
                 dense_dropout_rate=0.0, activation='selu', final_activation='linear',
                 normalize_sum=True, hidden_units=None, **kwargs):
        super(SetEmbedding, self).__init__(**kwargs)
        self.num_items = num_items
        self.embed_dims = embed_dims
        self.item_dropout_rate = item_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.final_dims = final_dims or embed_dims
        self.activation = activation
        self.final_activation = final_activation
        self.normalize_sum = normalize_sum
        self.hidden_units = hidden_units or (2 * self.embed_dims)
        self.time_shape = time_shape

    def get_config(self):
        config = super(SetEmbedding, self).get_config()
        config.update({
            "num_items": self.num_items,
            "embed_dims": self.embed_dims,
            "time_shape": self.time_shape,
            "item_dropout_rate": self.item_dropout_rate,
            "dense_dropout_rate": self.dense_dropout_rate,
            "final_dims": self.final_dims,
            "activation": self.activation,
            "final_activation": self.final_activation,
            "normalize_sum": self.normalize_sum,
            "hidden_units": self.hidden_units,
        })
        return config

    def build(self, input_shape):
        embed_stdev = 1 / math.sqrt(self.embed_dims)
        self.embeddings = self.add_weight('item_embeddings', shape=(self.num_items - 1, self.embed_dims),
                                          initializer=tf.random_normal_initializer(0, embed_stdev,
                                                                                   seed=241),
                                          trainable=True)
        self.hidden = tf.keras.layers.Dense(self.hidden_units, activation=self.activation,
                                            use_bias=True, name='hidden')
        self.output_layer = tf.keras.layers.Dense(self.final_dims, activation=self.final_activation,
                                                  use_bias=True, name='output_layer')
        self.dropout = tf.keras.layers.Dropout(self.dense_dropout_rate)
        self.activation_layer = tf.keras.layers.Activation(self.activation)
        self.zero_embed = tf.constant(0, shape=(1, self.embed_dims), dtype=self.compute_dtype)
        if self.time_shape:
            self.time_embedding = TimeVaryingLinearEmbedding(self.time_shape, self.embed_dims,
                                                             initializer=tf.random_normal_initializer(0, embed_stdev, seed=789),
                                                             name='TimeEmbedding')

    def call(self, inputs, training=False):
        dropped_inputs = dropout(inputs[0], self.item_dropout_rate, training=training, name='inputs_dropped')
        embeddings = tf.concat([self.zero_embed, self.embeddings], 0, name='embeddings')
        item_embeds = tf.gather(embeddings, dropped_inputs, name='item_embeds')
        summed_embeds = tf.math.reduce_sum(item_embeds, 1, name='summed_embeds')
        if self.time_shape:
            summed_embeds = tf.math.add(summed_embeds, self.time_embedding(inputs[1], training=training),
                                        name='summed_embeds_with_time')
        if self.normalize_sum:
            num_valid = tf.math.reduce_sum(tf.cast(inputs[0] > 0, dtype=self.compute_dtype, name='mask'),
                                           axis=-1, keepdims=True, name='num_valid') + 1
            summed_embeds = tf.math.divide(summed_embeds, num_valid, name='normalized_embeds')
        summed_embeds = self.dropout(self.activation_layer(summed_embeds), training=training)
        hidden = self.dropout(self.hidden(summed_embeds), training=training)
        return self.output_layer(hidden)


class ContextualRating(tf.keras.layers.Layer):
    def __init__(self, num_items, embed_dims, context_dims, time_shape
                 item_dropout_rate=0.0, dense_dropout_rate=0.0, activation='selu',
                 bounded_distance=True, final_activation='linear', normalize_sum=True,
                 hidden_units=None, **kwargs):
        super(ContextualRating, self).__init__(**kwargs)
        self.num_items = num_items
        self.embed_dims = embed_dims
        self.context_dims = context_dims
        self.item_dropout_rate = item_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.activation = activation
        self.bounded_distance = bounded_distance
        self.final_activation = final_activation
        self.normalize_sum = normalize_sum
        self.hidden_units = hidden_units
        self.time_shape = time_shape

    def get_config(self):
        config = super(ContextualRating, self).get_config()
        config.update({
            "num_items": self.num_items,
            "embed_dims": self.embed_dims,
            "context_dims": self.context_dims,
            "time_shape": self.time_shape,
            "item_dropout_rate": self.item_dropout_rate,
            "dense_dropout_rate": self.dense_dropout_rate,
            "activation": self.activation,
            "bounded_distance": self.bounded_distance,
            "final_activation": self.final_activation,
            "normalize_sum": self.normalize_sum,
            "hidden_units": self.hidden_units,
        })
        return config

    def build(self, input_shape):
        self.item_embeddings = self.add_weight('item_embeddings', shape=(self.num_items - 1, self.embed_dims),
                                               initializer=tf.random_normal_initializer(0, 1 / self.embed_dims / self.embed_dims),
                                               trainable=True)
        self.pool_embedding = SetEmbedding(self.num_items, self.context_dims, self.time_shape,
                                           final_dims=self.embed_dims,
                                           item_dropout_rate=self.item_dropout_rate,
                                           dense_dropout_rate=self.dense_dropout_rate,
                                           activation=self.activation,
                                           final_activation=self.final_activation,
                                           normalize_sum=self.normalize_sum,
                                           hidden_units=self.hidden_units, name='pool_set_embedding')
        self.zero_embed = tf.constant(0, shape=(1, self.embed_dims), dtype=self.compute_dtype)

    def call(self, inputs, training=False):
        item_indices, context_indices, time_coords = inputs
        embeddings = tf.concat([self.zero_embed, self.item_embeddings], 0, name='embeddings')
        item_embeds = tf.gather(embeddings, item_indices, name='item_embeddings')
        context_embeds = self.pool_embedding((context_indices, time_coords), training=training)
        distances = tf.reduce_sum(tf.math.square(tf.math.subtract(item_embeds,
                                                                  tf.expand_dims(context_embeds, 1, name='expanded_context_embeds'),
                                                                  name='embed_differences')),
                                  axis=-1, name='squared_distances')
        if self.bounded_distance:
            one = tf.constant(1, dtype=self.compute_dtype)
            nonlinear_distances = tf.math.divide(one, tf.math.add(one, distances,
                                                                  name='distances_incremented'),
                                                 name='nonlinear_distances')
        else:
            nonlinear_distances = tf.math.negative(distances, name='negative_distances')
        nonlinear_distances = tf.cast(item_indices > 0, dtype=self.compute_dtype) * nonlinear_distances
        # Logging for tensorboard
        tf.summary.histogram('outputs/distances', distances)
        tf.summary.histogram('outputs/nonlinear_distances', nonlinear_distances)
        return nonlinear_distances


class ItemRating(tf.keras.layers.Layer):
    def __init__(self, num_items, bounded=True, **kwargs):
        super(ItemRating, self).__init__(**kwargs)
        self.num_items = num_items
        self.bounded = bounded

    def get_config(self):
        config = super(ItemRating, self).get_config()
        config.update({
            "num_items": self.num_items,
            "bounded": self.bounded,
        })
        return config

    def build(self, input_shape):
        self.item_rating_logits = self.add_weight('item_rating_logits', shape=(self.num_items - 1,),
                                                  initializer=tf.random_uniform_initializer(-0.05, 0.05,
                                                                                           seed=743),
                                                  trainable=True)
        self.zero_rating = tf.constant(0, shape=(1,), dtype=self.compute_dtype)

    def call(self, inputs, training=False):
        if self.bounded:
            item_ratings = tf.concat([self.zero_rating, tf.nn.sigmoid(32 * self.item_rating_logits, name='item_ratings')],
                                     0, name='item_ratings')
        else:
            item_ratings = tf.concat([self.zero_rating, tf.nn.softplus(32 * self.item_rating_logits, name='item_ratings')],
                                     0, name='item_ratings')
        ratings = tf.gather(item_ratings, inputs, name='ratings')
        # Logging for Tensorboard
        tf.summary.histogram('weights/item_ratings', item_ratings)
        return ratings


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
