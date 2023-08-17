import tensorflow as tf

from mtgml.constants import EPSILON, MAX_COPIES
from mtgml.layers.bert import BERT
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.wrapped import WDense


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
                pool_mask_float,
                name="card_weights",
            )
            true_land_counts = tf.einsum("bc,bc->b", is_land, true_deck, name="true_land_counts")
            card_losses = {
                "log": tf.keras.losses.binary_crossentropy(
                    tf.expand_dims(true_deck, -1),
                    tf.constant(1 - 2 * EPSILON, dtype=self.compute_dtype) * tf.expand_dims(probs, -1)
                    + tf.constant(EPSILON, dtype=self.compute_dtype),
                ),
                "mse": tf.math.squared_difference(true_deck, probs, name="mse_card_losses"),
                "mae": tf.math.abs(true_deck - probs, name="mae_card_losses"),
                "extremeness": tf.maximum(
                    tf.math.abs(0.5 - standardized_inclusion_probs) - 0.5 + self.margin, 0.0, name="extremeness_losses"
                ),
            }
            sample_losses = {
                "land_count": tf.math.squared_difference(
                    tf.einsum("bc,bc->b", is_land, standardized_inclusion_probs),
                    true_land_counts,
                    name="land_count_losses",
                ),
                "total_prob": tf.math.squared_difference(
                    40.0, tf.math.reduce_sum(probs, axis=1, name="total_probs"), name="total_prob_losses"
                ),
            }
            _, loss = self.collapse_losses(((card_losses, card_weights), sample_losses, {}))
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
            ranges = {
                "extremeness_loss": (0, self.margin),
                "land_counts": (12, 21),
                "land_count_errors": (0, 5),
                "accuracy_counts": (23, 37),
                "probability": (0, 1),
            }
            saturate_keys = {"land_counts", "land_count_errors", "accuracy_counts"}
            float_metrics = {
                "top_40_total_prob": tf.math.reduce_sum(top_40_probs, axis=1, name="top_40_total_probs"),
                "total_prob": tf.math.reduce_sum(standardized_inclusion_probs, axis=1, name="total_prob"),
                "total_land_prob": tf.math.reduce_sum(
                    standardized_inclusion_probs * is_land, axis=1, name="total_land_prob"
                ),
                "probability": standardized_inclusion_probs,
                "loss": loss,
            } | {f"{k}_loss": v for k, v in (card_losses | sample_losses).items()}
            self.log_metrics(int_metrics, float_metrics, ranges, saturate_keys)
            return loss
        return standardized_inclusion_probs
