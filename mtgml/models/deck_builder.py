import math

import tensorflow as tf
import tensorflow_probability as tfp

from mtgml.constants import EPSILON, MAX_COPIES, MAX_DECK_SIZE, is_debug
from mtgml.layers.bert import BERT
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.wrapped import WDense


@tf.function
def scan_probs(acc, xs):
    return (tf.where(xs[1], acc[0], tf.ones_like(acc[0]), name="propagating_prob") * xs[0], tf.zeros_like(xs[1]))


class DeckBuilder(ConfigurableLayer, tf.keras.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_cards = hyper_config.get_int(
            "num_cards",
            min=1,
            max=None,
            default=None,
            help="The number of cards that must be embedded. Should be 1 + maximum index in the input.",
        )
        card_embed_dims = hyper_config.get_int(
            "embed_dims",
            default=256,
            min=8,
            max=1024,
            help="The number of dimensions for the tokens passed through the attenion layers.",
        )
        return {
            "num_cards": num_cards,
            "card_embed_dims": card_embed_dims,
            "embed_pool": hyper_config.get_sublayer(
                "EmbedPool",
                sub_layer_type=BERT,
                fixed={
                    "token_stream_dims": card_embed_dims,
                    "use_causal_mask": False,
                    "use_position_embeds": True,
                    "num_positions": MAX_COPIES,
                },
                help="Allow the card embeddings to interact with each other and give final embeddings.",
            ),
            "deck_metrics": {
                "deck_recall_at_25": tf.keras.metrics.RecallAtPrecision(0.25, name="deck_recall_at_25"),
                "deck_recall_at_50": tf.keras.metrics.RecallAtPrecision(0.5, name="deck_recall_at_50"),
                "deck_recall_at_75": tf.keras.metrics.RecallAtPrecision(0.75, name="deck_recall_at_75"),
                "deck_precis_at_25": tf.keras.metrics.PrecisionAtRecall(0.25, name="deck_recall_at_25"),
                "deck_precis_at_50": tf.keras.metrics.PrecisionAtRecall(0.5, name="deck_recall_at_50"),
                "deck_precis_at_75": tf.keras.metrics.PrecisionAtRecall(0.75, name="deck_recall_at_75"),
                "deck_abs_error": tf.keras.metrics.MeanAbsoluteError(name="deck_abs_error"),
            },
            "get_inclusion_prob": hyper_config.get_sublayer(
                "InclusionProb",
                sub_layer_type=WDense,
                fixed={"dims": 1, "activation": "sigmoid"},
                seed_mod=97,
                help="The layer to convert the final embeddings of the cards to probabilities of inclusion.",
            ),
        }

    def call(self, inputs, training=None):
        pool_cards = tf.reshape(tf.cast(inputs[0], dtype=tf.int32, name="pool_cards"), (-1, MAX_DECK_SIZE))
        instance_nums = tf.reshape(tf.cast(inputs[1], dtype=tf.int32, name="instance_counts"), (-1, MAX_DECK_SIZE))
        card_embeddings = tf.cast(inputs[2], dtype=self.compute_dtype, name="card_embeddings")
        pool_card_embeddings = tf.gather(card_embeddings, pool_cards, name="pool_card_embeddings")
        pool_mask = pool_cards > 0
        pool_final_embeddings = self.embed_pool(
            (pool_card_embeddings, instance_nums), training=training, mask=pool_mask
        )
        inclusion_probs = tf.squeeze(
            self.get_inclusion_prob(pool_final_embeddings, training=training, mask=pool_mask), axis=-1
        )
        pool_mask_float = tf.cast(pool_mask, dtype=self.compute_dtype)
        standardized_inclusion_probs = (
            tfp.math.scan_associative(
                scan_probs,
                elems=(inclusion_probs, instance_nums > 0),
                axis=1,
                max_num_levels=math.ceil(math.log2(MAX_DECK_SIZE + 1)),
                name="standardized_inclusion_probs",
            )[0]
            * pool_mask_float
        )
        if len(inputs) == 4:
            true_deck = tf.reshape(tf.cast(inputs[3], dtype=self.compute_dtype, name="true_deck"), (-1, MAX_DECK_SIZE))
            probs = tf.constant(1 - 2 * EPSILON, dtype=self.compute_dtype) * standardized_inclusion_probs + tf.constant(
                EPSILON, dtype=self.compute_dtype
            )
            deck_losses = (
                -(true_deck * tf.math.log(probs) + (1.0 - true_deck) * tf.math.log(1.0 - probs)) * pool_mask_float
            )
            deck_losses = tf.reduce_sum(deck_losses, axis=-1, name="deck_losses") / tf.math.reduce_sum(
                tf.cast(pool_mask, dtype=tf.float32), axis=-1, name="card_counts"
            )
            deck_loss = tf.math.reduce_mean(deck_losses)
            loss = deck_loss
            self.add_metric(deck_losses, "deck_loss")
            tf.summary.scalar("deck_loss", deck_loss)
            if is_debug():
                tf.summary.histogram("probabilities", standardized_inclusion_probs)
            for name, metric in self.deck_metrics.items():
                metric.update_state(true_deck, standardized_inclusion_probs)
                tf.summary.scalar(name, metric.result())
            return loss
        return standardized_inclusion_probs
