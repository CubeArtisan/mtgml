import math

import tensorflow as tf

from mtgml.constants import (
    DEFAULT_PACKS_PER_PLAYER,
    DEFAULT_PICKS_PER_PACK,
    EPSILON,
    LARGE_INT,
    MAX_CARDS_IN_PACK,
    MAX_PICKED,
    MAX_SEEN_PACKS,
    should_ensure_shape,
    should_log_histograms,
)
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.contextual_rating import ContextualRating
from mtgml.layers.extended_dropout import ExtendedDropout
from mtgml.layers.mlp import MLP
from mtgml.layers.set_embedding import AttentiveSetEmbedding
from mtgml.layers.time_varying_embedding import TimeVaryingEmbedding
from mtgml.utils.masked import (
    reduce_max_masked,
    reduce_mean_masked,
    reduce_min_masked,
    reduce_sum_masked,
    reduce_variance_masked,
)

POOL_ORACLE_METADATA = {
    "title": "Pool Synergy",
    "tooltip": "How well the card goes with the cards that have already been picked.",
    "name": "pool",
}
SEEN_ORACLE_METADATA = {
    "title": "Seen Synergy",
    "tooltip": "How well the card goes with the cards that have already been seen, looking for things like openness and combos.",
    "name": "seen",
}
RATING_ORACLE_METADATA = {
    "title": "Card Rating",
    "tooltip": "How good the card is in a vacuum.",
    "name": "rating",
}


class DraftBot(ConfigurableLayer, tf.keras.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        pool_context_ratings = hyper_config.get_bool(
            "pool_context_ratings",
            default=True,
            help="Whether to rate cards based on how the go with the other cards in the pool so far.",
        )
        seen_context_ratings = hyper_config.get_bool(
            "seen_context_ratings", default=True, help="Whether to rate cards based on the packs seen so far."
        )
        item_ratings = hyper_config.get_bool(
            "item_ratings", default=True, help="Whether to give each card a rating independent of context."
        )
        sublayer_metadatas = []
        if pool_context_ratings:
            sublayer_metadatas.append(POOL_ORACLE_METADATA)
        if seen_context_ratings:
            sublayer_metadatas.append(SEEN_ORACLE_METADATA)
        if item_ratings:
            sublayer_metadatas.append(RATING_ORACLE_METADATA)
        seen_pack_dims = (
            hyper_config.get_int(
                "seen_pack_dims",
                min=8,
                max=512,
                step=8,
                default=32,
                help="The number of dimensions to embed seen packs into.",
            )
            if seen_context_ratings
            else 0
        )
        num_cards = hyper_config.get_int(
            "num_cards",
            min=1,
            max=None,
            default=None,
            help="The number of items that must be embedded. Should be 1 + the max index expected to see.",
        )
        sublayer_count = len([x for x in (pool_context_ratings, seen_context_ratings, item_ratings) if x])
        return {
            "seen_pack_dims": seen_pack_dims,
            "num_cards": num_cards - 1,
            "rate_off_pool": hyper_config.get_sublayer(
                "RatingFromPool",
                sub_layer_type=ContextualRating,
                fixed={"use_causal_mask": True},
                seed_mod=31,
                help="The layer that rates based on the other cards that have been picked.",
            )
            if pool_context_ratings
            else None,
            "dropout_pool_embeds": hyper_config.get_sublayer(
                "PoolDenseDropout",
                sub_layer_type=ExtendedDropout,
                seed_mod=71,
                fixed={
                    "noise_shape": None,
                    "blank_last_dim": False,
                    "return_mask": False,
                },
                help="The layer that drops out part of the card embeddings for the cards in the pool.",
            )
            if pool_context_ratings
            else None,
            "seen_pack_dims": seen_pack_dims,
            "embed_pack": hyper_config.get_sublayer(
                "EmbedPack",
                sub_layer_type=AttentiveSetEmbedding,
                fixed={"use_causal_mask": False, "output_dims": seen_pack_dims},
                seed_mod=37,
                help="The layer that embeds the packs that have been seen so far.",
            )
            if seen_context_ratings
            else None,
            "embed_pack_position": hyper_config.get_sublayer(
                "EmbedPackPosition",
                sub_layer_type=TimeVaryingEmbedding,
                fixed={
                    "dims": seen_pack_dims,
                    "time_shape": (DEFAULT_PACKS_PER_PLAYER, DEFAULT_PICKS_PER_PACK),
                    "activation": "linear",
                },
                seed_mod=23,
                help="The embedding for the position in the draft",
            )
            if seen_context_ratings
            else None,
            "rate_off_seen": hyper_config.get_sublayer(
                "RatingFromSeen",
                sub_layer_type=ContextualRating,
                fixed={"use_causal_mask": True, "token_stream_dims": seen_pack_dims},
                seed_mod=31,
                help="The layer that rates based on the embeddings of the packs that have been seen.",
            )
            if seen_context_ratings
            else None,
            "rate_card": hyper_config.get_sublayer(
                "CardRating",
                sub_layer_type=MLP,
                seed_mod=13,
                fixed={"Final": {"dims": 1, "activation": "linear"}},
                help="Translates embeddings into linear ratings.",
            )
            if item_ratings
            else None,
            "lambdas": {
                "log": hyper_config.get_float(
                    "log_loss_weight",
                    min=0,
                    max=1,
                    step=0.01,
                    default=0.5,
                    help="The weight given to probability log loss.",
                ),
                "triplet": hyper_config.get_float(
                    "triplet_loss_weight",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    default=0.2,
                    help="The weight given to the triplet separation loss.",
                ),
                "score_variance": hyper_config.get_float(
                    "score_variance_weight",
                    min=-1,
                    max=1,
                    default=1e-03,
                    help="The weight given to the variance of the combined scores.",
                ),
                "rating_variance": hyper_config.get_float(
                    "rating_variance_weight",
                    min=-1,
                    max=1,
                    default=1e-03,
                    help="The weight given to the variance of the card ratings.",
                )
                if item_ratings
                else 0,
                "seen_variance": hyper_config.get_float(
                    "seen_variance_weight",
                    min=-1,
                    max=1,
                    default=1e-02,
                    help="The weight given to the variance of the seen contextual ratings.",
                )
                if seen_context_ratings
                else 0,
                "pool_variance": hyper_config.get_float(
                    "pool_variance_weight",
                    min=-1,
                    max=1,
                    default=1e-02,
                    help="The weight given to the variance of the pool contextual ratings.",
                )
                if pool_context_ratings
                else 0,
                "extremeness": hyper_config.get_float(
                    "extremeness_weight",
                    default=1.0,
                    min=0.0,
                    max=1000,
                    help="The multiplier to scale the probability margin loss by. Suggested is 1 / probability_margin.",
                ),
                "sublayer_weights_l2": hyper_config.get_float(
                    "sublayer_weights_l2_weight",
                    default=0.001,
                    min=0.0,
                    max=1.0,
                    help="The multiplier to scale the loss on the square of the sublayer weights.",
                ),
            },
            "margin": hyper_config.get_float(
                "margin",
                min=0,
                max=10,
                step=0.1,
                default=2,
                help="The margin by which we want the correct choice to beat the incorrect choices.",
            ),
            "extremeness_margin": hyper_config.get_float(
                "probability_margin",
                default=0.01,
                min=0.0,
                max=0.5,
                help="The distance from the endpoints (0, 1) at which to start pushing the predicted probability back towards 0.5.",
            ),
            "sublayer_weights": hyper_config.get_sublayer(
                "SubLayerWeights",
                sub_layer_type=TimeVaryingEmbedding,
                fixed={
                    "dims": sublayer_count,
                    "time_shape": (DEFAULT_PACKS_PER_PLAYER, DEFAULT_PICKS_PER_PACK),
                    "activation": "softplus",
                },
                help="The weights for each of the sublayers that get combined together linearly.",
            ),
            "sublayer_metadata": sublayer_metadatas,
        }

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs[1])[0]
        max_picked = tf.shape(inputs[1])[1]
        max_seen_packs = tf.shape(inputs[2])[1]
        max_cards_in_pack = tf.shape(inputs[2])[2]
        if should_ensure_shape():
            # basics = tf.ensure_shape(tf.cast(inputs[0], dtype=tf.int32, name='basics'), (None, MAX_BASICS))
            pool = tf.ensure_shape(tf.cast(inputs[1], dtype=tf.int32, name="pool"), (None, MAX_PICKED))
            seen_packs = tf.ensure_shape(
                tf.cast(inputs[2], dtype=tf.int32, name="seen_packs"), (None, MAX_SEEN_PACKS, MAX_CARDS_IN_PACK)
            )
            seen_coords = tf.ensure_shape(
                tf.cast(inputs[3], dtype=tf.int32, name="seen_coords"), (None, MAX_SEEN_PACKS, 4, 2)
            )
            seen_coord_weights = tf.ensure_shape(
                tf.cast(inputs[4], dtype=self.compute_dtype, name="seen_coord_weights"), (None, MAX_SEEN_PACKS, 4)
            )
            card_embeddings = tf.ensure_shape(
                tf.cast(inputs[5], dtype=self.compute_dtype, name="card_embeddings"), (self.num_cards + 1, None)
            )
        else:
            # basics = tf.cast(inputs[0], dtype=tf.int32, name='basics')
            pool = tf.cast(tf.reshape(inputs[1], (batch_size, max_picked)), dtype=tf.int32, name="pool")
            seen_packs = tf.cast(
                tf.reshape(inputs[2], (batch_size, max_seen_packs, max_cards_in_pack)),
                dtype=tf.int32,
                name="seen_packs",
            )
            seen_coords = tf.cast(
                tf.reshape(inputs[3], (batch_size, max_seen_packs, 4, 2)), dtype=tf.int32, name="seen_coords"
            )
            seen_coord_weights = tf.cast(
                tf.reshape(inputs[4], (batch_size, max_seen_packs, 4)),
                dtype=self.compute_dtype,
                name="seen_coord_weights",
            )
            card_embeddings = tf.cast(inputs[5], dtype=self.compute_dtype, name="card_embeddings")
        sublayer_scores_list = []
        bool_mask = seen_packs > 0
        pack_mask = tf.reduce_any(bool_mask, -1)
        mask = tf.cast(bool_mask, dtype=self.compute_dtype, name="pack_mask")
        # Shift pool right by 1
        pool = tf.concat([tf.zeros_like(pool[:, :1]), pool[:, :-1]], axis=1)
        pool_embeds = tf.gather(card_embeddings, pool, name="pool_embeds")
        seen_pack_embeds = tf.gather(card_embeddings, seen_packs, name="seen_pack_embeds")
        card_dims = tf.shape(seen_pack_embeds)[-1]
        if self.rate_off_pool:
            pool_embeds_dropped = self.dropout_pool_embeds(pool_embeds, training=training)
            pool_scores = self.rate_off_pool(
                (seen_pack_embeds, pool_embeds_dropped), training=training, mask=(tf.cast(mask, tf.bool), pool > 0)
            )
            sublayer_scores_list.append(pool_scores)
        if self.rate_off_seen:
            flat_seen_pack_embeds = tf.reshape(
                seen_pack_embeds, (-1, max_cards_in_pack, card_dims), name="flat_seen_pack_embeds"
            )
            flat_seen_pack_mask = tf.reshape(bool_mask, (-1, max_cards_in_pack), name="flat_seen_pack_embeds")
            flat_pack_embeds = self.embed_pack(flat_seen_pack_embeds, training=training, mask=flat_seen_pack_mask)
            pack_embeds = tf.reshape(
                flat_pack_embeds, (-1, max_seen_packs, self.seen_pack_dims), name="pack_embeds_pre"
            )
            position_embeds = self.embed_pack_position((seen_coords, seen_coord_weights), training=training)
            pack_embeds = pack_embeds + position_embeds / tf.constant(
                math.sqrt(self.seen_pack_dims), dtype=self.compute_dtype
            )
            pack_embeds._keras_mask = bool_mask
            seen_scores = self.rate_off_seen(
                (seen_pack_embeds, pack_embeds), training=training, mask=(bool_mask, pack_mask)
            )
            sublayer_scores_list.append(seen_scores)
        if self.rate_card:
            all_card_ratings = tf.squeeze(self.rate_card(card_embeddings[1:], training=training), axis=-1)
            card_ratings = tf.gather(all_card_ratings, seen_packs, name="pack_card_ratings")
            sublayer_scores_list.append(card_ratings)
            if should_log_histograms():
                tf.summary.histogram("card_ratings", all_card_ratings)
        sublayer_scores = tf.stack(sublayer_scores_list, axis=0, name="stacked_sublayer_scores")
        sublayer_weights = tf.math.softplus(self.sublayer_weights((seen_coords, seen_coord_weights), training=training))
        sublayer_scores = tf.cast(sublayer_scores, dtype=self.compute_dtype, name="sublayer_scores")
        sublayer_weights = tf.cast(sublayer_weights, dtype=self.compute_dtype, name="sublayer_weights")
        sublayers_weighted = tf.einsum(
            "s...p,...s->s...p", sublayer_scores, sublayer_weights, name="sublayers_weighted"
        )
        scores = tf.reduce_sum(sublayers_weighted, axis=0, name="scores")
        if len(inputs) > 6:
            loss_dtype = tf.float32
            mask = tf.cast(mask, dtype=loss_dtype)
            if should_ensure_shape():
                y_idx = tf.ensure_shape(tf.cast(inputs[6], dtype=tf.int32, name="y_idx"), (None, MAX_SEEN_PACKS))
                riskiness = tf.ensure_shape(
                    tf.cast(inputs[7], dtype=tf.float32, name="riskiness"), (None, MAX_SEEN_PACKS, MAX_CARDS_IN_PACK)
                )
            else:
                y_idx = tf.cast(tf.reshape(inputs[6], (batch_size, max_seen_packs)), dtype=tf.int32, name="y_idx")
                riskiness = tf.cast(
                    tf.reshape(inputs[7], (batch_size, max_seen_packs, max_cards_in_pack)),
                    dtype=tf.float32,
                    name="riskiness",
                )
            num_in_pack = tf.math.reduce_sum(mask, axis=-1, name="num_in_pack")
            pos_mask = tf.cast(tf.expand_dims(y_idx == 0, -1), dtype=loss_dtype) * tf.constant(
                2, dtype=loss_dtype
            ) - tf.constant(1.0, dtype=loss_dtype)
            scores = (
                pos_mask * tf.cast(scores, dtype=loss_dtype, name="cast_scores")
                + tf.constant(LARGE_INT, dtype=loss_dtype)
            ) * mask
            probs = tf.nn.softmax(scores, axis=-1, name="probs")
            probs_for_loss = tf.concat(
                [probs[:, :, :1], tf.constant(1, dtype=self.compute_dtype) - probs[:, :, 1:]], axis=2
            )
            score_diffs = tf.subtract(
                tf.add(tf.constant(self.margin, dtype=scores.dtype), scores[:, :, 1:]),
                scores[:, :, :1],
                name="score_diffs",
            )
            clipped_diffs = tf.concat(
                [
                    tf.zeros_like(score_diffs[:, :, :1]),
                    tf.math.maximum(tf.constant(0, dtype=loss_dtype), score_diffs, name="clipped_score_diffs"),
                ],
                axis=2,
            )
            card_losses = {
                "log": -tf.math.log((1 - 2 * EPSILON) * probs_for_loss + EPSILON, name="log_probs"),
                "triplet": clipped_diffs,
                "extremeness": tf.maximum(
                    tf.math.abs(0.5 - probs) - 0.5 + self.extremeness_margin, 0.0, name="extremeness_losses"
                ),
            }
            card_weights = riskiness * tf.cast(tf.expand_dims(num_in_pack > 1.0, -1), dtype=loss_dtype) * mask
            pack_losses = {
                f"{name}_variance": reduce_variance_masked(values, mask=mask, axis=-1, name=f"{name}_variance")
                for name, values in zip(
                    [meta["name"] for meta in self.sublayer_metadata] + ["score"],
                    tf.unstack(sublayers_weighted, num=len(sublayer_scores), axis=0) + [scores],
                )
            } | {"sublayer_weights_l2": tf.reduce_sum(tf.square(sublayer_weights), axis=-1, name="weights_l2")}
            pack_weights = tf.cast(pack_mask, loss_dtype)
            _, loss = self.collapse_losses(((card_losses, card_weights), (pack_losses, pack_weights), {}, {}))
            mask_freebies_1 = num_in_pack > 1.0
            max_score = reduce_max_masked(scores, mask=mask, axis=2, name="max_score")
            min_score = reduce_min_masked(scores, mask=mask, axis=2, name="min_score")
            position = reduce_sum_masked(
                tf.cast(scores[:, :, 1:] >= scores[:, :, :1], dtype=tf.float32), mask=mask[:, :, 1:], axis=2
            )
            int_metrics = {"position": position}
            float_metrics = {
                "top_1_accuracy": reduce_mean_masked(
                    tf.cast(position < 1.0, dtype=tf.float32),
                    mask=mask_freebies_1,
                    axis=None,
                    name="accuracy_top_1",
                ),
                "top_2_accuracy": reduce_mean_masked(
                    tf.cast(position < 2.0, dtype=tf.float32),
                    mask=num_in_pack > 2.0,
                    axis=None,
                    name="accuracy_top_2",
                ),
                "top_3_accuracy": reduce_mean_masked(
                    tf.cast(position < 3.0, dtype=tf.float32),
                    mask=num_in_pack > 3.0,
                    axis=None,
                    name="accuracy_top_3",
                ),
                "max_score": max_score,
                "min_score": min_score,
                "max_score_diffs": max_score - min_score,
                "prob_correct": probs[:, :, 0],
                "score_0": scores[:, :, 0],
                "loss": loss,
            } | {f"{k}_loss": v for k, v in pack_losses.items()}
            ranges = {
                "position": (1, 15),
                "prob_correct": (0, 1),
                "probs": (0, 1),
                "extremeness_loss": (0.0, self.extremeness_margin),
            }
            saturates = {"position"}
            self.log_metrics(int_metrics, float_metrics, ranges, saturates, mask_freebies_1)
            card_float_metrics = {"scores": scores, "probs": probs} | {f"{k}_loss": v for k, v in card_losses.items()}
            self.log_metrics(
                {},
                card_float_metrics,
                ranges,
                saturates,
                mask * tf.cast(tf.expand_dims(mask_freebies_1, 2), mask.dtype),
            )
            return loss
        return sublayer_scores, sublayer_weights
