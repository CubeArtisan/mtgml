import math

import tensorflow as tf

from mtgml.constants import (
    EPSILON,
    LARGE_INT,
    MAX_COPIES,
    MAX_DECK_SIZE,
    should_log_histograms,
)
from mtgml.layers.bert import BERT
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.wrapped import WDense
from mtgml.tensorboard.timeseries import log_integer_histogram


def smooth_stepper(order: int = 1):
    coefficients = [
        (-(1**i)) * math.comb(order + i, i) * math.comb(2 * order + 1, order - i) for i in range(order + 1)
    ]

    @tf.function
    def smoothstep(x: tf.Tensor, name: str | None = None):
        with tf.name_scope(name or f"SmoothStep{order}") as scope:
            x = tf.clip_by_value(x, EPSILON, 1.0, name="clipped")
            powers = tf.math.pow(
                x,
                tf.constant(
                    [order + i + 1 for i in range(order + 1)],
                    shape=(order + 1, *[1 for _ in range(len(x.shape))]),
                    dtype=x.dtype,
                    name="exponents",
                ),
                name="powers",
            )
            return tf.einsum(
                "n,n...->...", tf.constant(coefficients, dtype=x.dtype, name="coefficients"), x, name=scope
            )

    return smoothstep


@tf.function
def normalize_probabilities_sum(probabilities, mask, desired_sum: float = 40, name: str | None = None):
    with tf.name_scope(name or "NormalizeProbabilitiesSum") as scope:
        max_x = tf.math.reduce_logsumexp(
            probabilities
            - (1 - tf.cast(mask, dtype=probabilities.dtype)) * tf.constant(LARGE_INT, dtype=probabilities.dtype),
            axis=1,
            keepdims=True,
            name="max_value",
        )
        sum_x = tf.math.reduce_sum(
            probabilities * tf.cast(mask, dtype=probabilities.dtype), axis=1, keepdims=True, name="sum_x"
        )
        numerator = tf.nn.softplus(
            tf.constant(desired_sum, dtype=probabilities.dtype, name="desired_sum") * max_x - sum_x, name="unscaled"
        )
        offset = tf.math.divide(
            numerator,
            tf.math.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=1, keepdims=True) - desired_sum + EPSILON,
            name="offset",
        )
        shifted = probabilities + offset
        return tf.math.multiply(
            tf.constant(desired_sum, dtype=probabilities.dtype),
            shifted / tf.math.reduce_sum(shifted, axis=1, keepdims=True, name="total_shifted"),
            name=scope,
        )


@tf.function
def normalize_probabilities_sum_weak(probabilities, mask, desired_sum: float = 40, name: str | None = None):
    with tf.name_scope(name or "NormalizeProbabilitiesMaxSum") as scope:
        sum_x = (
            tf.math.reduce_sum(
                probabilities * tf.cast(mask, dtype=probabilities.dtype), axis=1, keepdims=True, name="sum_x"
            )
            + EPSILON
        )
        return tf.where(sum_x > desired_sum, desired_sum * probabilities / sum_x, probabilities, name=scope)


class DeckBuilder(ConfigurableLayer, tf.keras.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            "is_land_lookup": hyper_config.get_list(
                "is_land_lookup",
                default=None,
                help="A lookup table for what cards are lands to allow for regularization",
            ),
            "embed_pool": hyper_config.get_sublayer(
                "EmbedPool",
                sub_layer_type=BERT,
                fixed={
                    "use_causal_mask": False,
                    "use_position_embeds": True,
                    "num_positions": MAX_COPIES,
                    "InitialDropout": {"blank_last_dim": False},
                },
                help="Allow the card embeddings to interact with each other and give final embeddings.",
            ),
            "deck_metrics": {
                "deck_recall_at_75": tf.keras.metrics.RecallAtPrecision(0.75, name="deck_recall_at_75"),
                "deck_recall_at_90": tf.keras.metrics.RecallAtPrecision(0.90, name="deck_recall_at_90"),
                "deck_recall_at_95": tf.keras.metrics.RecallAtPrecision(0.95, name="deck_recall_at_95"),
                "deck_precis_at_75": tf.keras.metrics.PrecisionAtRecall(0.75, name="deck_precis_at_75"),
                "deck_precis_at_90": tf.keras.metrics.PrecisionAtRecall(0.90, name="deck_precis_at_90"),
                "deck_precis_at_95": tf.keras.metrics.PrecisionAtRecall(0.95, name="deck_precis_at_95"),
            },
            "get_inclusion_prob": hyper_config.get_sublayer(
                "InclusionProb",
                sub_layer_type=WDense,
                fixed={"dims": 1, "activation": "sigmoid"},
                seed_mod=97,
                help="The layer to convert the final embeddings of the cards to probabilities of inclusion.",
            ),
            "margin": hyper_config.get_float(
                "probability_margin",
                default=0.001,
                min=0.0,
                max=0.5,
                help="The distance from the endpoints (0, 1) at which to start pushing the predicted probability back towards 0.5.",
            ),
            "lambdas": {
                "extremeness": hyper_config.get_float(
                    "margin_weight",
                    default=1.0,
                    min=0.0,
                    max=1000,
                    help="The multiplier to scale the probability margin loss by. Suggested is 1 / probability_margin.",
                ),
                "log": hyper_config.get_float(
                    "log_weight",
                    default=1.0,
                    min=0.0,
                    max=20.0,
                    help="How much weight to apply to the cross entropy loss.",
                ),
                "mse": hyper_config.get_float(
                    "mse_weight",
                    default=0.01,
                    min=0.0,
                    max=20.0,
                    help="How much weight to apply to the mean squared error between true predictions and our probability of inclusion.",
                ),
                "mae": hyper_config.get_float(
                    "mae_weight",
                    default=0.01,
                    min=0.0,
                    max=20.0,
                    help="How much weight to apply to the mean absolute error between true predictions and our probability of inclusion.",
                ),
                "land_count": hyper_config.get_float(
                    "land_count_weight",
                    default=0.001,
                    min=0.0,
                    max=20.0,
                    help="How much weight to apply to punish it for having land counts other than 17.",
                ),
                "total_prob": hyper_config.get_float(
                    "total_prob_weight",
                    default=1.0,
                    min=0.0,
                    max=20.0,
                    help="How much weight to apply to punish it for having total probability away from 40.",
                ),
            },
            "scale_correct": hyper_config.get_float(
                "scale_correct",
                default=5.0,
                min=0.0,
                max=100.0,
                step=1.0,
                help="How much to add to the scale factor for the penalty for leaving out a correct card.",
            ),
        }

    def call(self, inputs, training=False):
        pool_cards = tf.cast(inputs[0], dtype=tf.int32, name="pool_cards")
        instance_nums = tf.cast(inputs[1], dtype=tf.int32, name="instance_counts")
        card_embeddings = tf.cast(inputs[2], dtype=self.compute_dtype, name="card_embeddings")
        pool_card_embeddings = tf.gather(card_embeddings, pool_cards, name="pool_card_embeddings")
        pool_mask = pool_cards > 0
        pool_final_embeddings = self.embed_pool(
            (pool_card_embeddings, instance_nums), training=training, mask=pool_mask
        )
        inclusion_probs = self.get_inclusion_prob(pool_final_embeddings, training=training, mask=pool_mask)
        pool_mask_float = tf.cast(pool_mask, dtype=self.compute_dtype)
        standardized_inclusion_probs = tf.squeeze(inclusion_probs, axis=-1) * pool_mask_float
        for i in range(1, MAX_COPIES):
            standardized_inclusion_probs = standardized_inclusion_probs * tf.where(
                instance_nums < i,
                tf.ones_like(inclusion_probs[:, :, 0]),
                tf.concat([tf.ones_like(inclusion_probs[:, :i, 0]), inclusion_probs[:, :-i, 0]], axis=1),
            )
        if len(inputs) == 4:
            true_deck = tf.cast(inputs[3], dtype=self.compute_dtype, name="true_deck")
            probs = (1 - 2 * EPSILON) * standardized_inclusion_probs + EPSILON
            is_land = tf.cast(
                tf.gather(self.is_land_lookup, pool_cards, name="is_land_bool"),
                dtype=self.compute_dtype,
                name="is_land",
            )
            card_weights = tf.where(
                true_deck > 0,
                tf.cast(1 + instance_nums, dtype=self.compute_dtype) + self.scale_correct + 8.0 * is_land,
                pool_mask_float + is_land,
                name="card_weights",
            )
            total_sample_weights = tf.math.reduce_sum(card_weights, axis=1, name="total_sample_weights")
            card_losses = {
                "log": -(true_deck * tf.math.log(probs) + (1.0 - true_deck) * tf.math.log(1.0 - probs)),
                "mse": tf.math.squared_difference(true_deck, probs, name="mse_card_losses"),
                "mae": tf.math.abs(true_deck - probs, name="mae_card_losses"),
                "extremeness": tf.maximum(
                    tf.math.abs(0.5 - standardized_inclusion_probs) - 0.5 + self.margin, 0.0, name="extremeness_losses"
                ),
            }
            true_land_counts = tf.math.reduce_sum(is_land * true_deck, axis=1, name="true_land_counts")
            sample_losses = {
                k: tf.math.reduce_sum(card_weights * v, axis=1, name=f"sample_{k}_losses") / total_sample_weights
                for k, v in card_losses.items()
            } | {
                "land_count": tf.math.square(
                    tf.math.abs(tf.math.reduce_sum(is_land * standardized_inclusion_probs, axis=1) - true_land_counts),
                    name="land_count_losses",
                ),
                "total_prob": tf.math.squared_difference(
                    40.0, tf.math.reduce_sum(probs, axis=1, name="total_probs"), name="total_prob_losses"
                ),
            }
            losses = {k: tf.math.reduce_mean(v) for k, v in sample_losses.items()}
            loss = sum(self.lambdas[k] * v for k, v in losses.items())
            for k, v in sample_losses.items():
                self.add_metric(v, f"{k}_loss")
                tf.summary.scalar(f"{k}_loss", losses[k])

            # Tensorboard logging
            for name, metric in self.deck_metrics.items():
                metric.update_state(true_deck, standardized_inclusion_probs)
                tf.summary.scalar(name, metric.result())
            top_40 = tf.math.top_k(standardized_inclusion_probs, k=40, sorted=False, name="top_40")
            top_40_probs = top_40.values
            top_40_indices = top_40.indices
            included_lands = tf.math.reduce_sum(
                tf.gather(is_land, top_40_indices, batch_dims=1, name="included_is_land"),
                axis=1,
                name="included_lands",
            )
            int_metrics = {
                "land_counts": included_lands,
                "land_count_errors": tf.math.abs(included_lands - true_land_counts, name="land_count_errors"),
                "accuracy_counts": tf.math.reduce_sum(
                    tf.gather(true_deck, top_40_indices, batch_dims=1, name="included_is_true"),
                    axis=1,
                    name="accuracy_counts",
                ),
            }
            float_metrics = {
                "top_40_total_prob": tf.math.reduce_sum(top_40_probs, axis=1, name="top_40_total_probs"),
                "total_prob": tf.math.reduce_sum(standardized_inclusion_probs, axis=1, name="total_prob"),
                "total_land_prob": tf.math.reduce_sum(
                    standardized_inclusion_probs * is_land, axis=1, name="total_land_prob"
                ),
            }
            metrics = int_metrics | float_metrics
            for name, values in metrics.items():
                tf.summary.scalar(f"mean_{name}", tf.math.reduce_mean(values))
                self.add_metric(values, name)
            if should_log_histograms():
                tf.summary.histogram("probabilities", standardized_inclusion_probs)
                for name, values in float_metrics.items():
                    tf.summary.histogram(name, values)
                log_integer_histogram(
                    "land_counts", int_metrics["land_counts"], start_index=12, max_index=21, saturate=True
                )
                log_integer_histogram(
                    "land_count_errors", int_metrics["land_count_errors"], start_index=0, max_index=5, saturate=True
                )
                log_integer_histogram(
                    "accuracy_counts", int_metrics["accuracy_counts"], start_index=26, max_index=40, saturate=True
                )
                for name, values in sample_losses.items():
                    tf.summary.histogram(f"{name}_losses", values)
            return loss
        return standardized_inclusion_probs
