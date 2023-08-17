import tensorflow as tf

from mtgml.constants import should_log_histograms
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.mlp import MLP
from mtgml.layers.wrapped import WDense


class AdjMtxReconstructor(ConfigurableLayer, tf.keras.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_cards = hyper_config.get_int(
            "num_cards",
            min=1,
            max=None,
            default=None,
            help="The number of cards that must be embedded. Should be 1 + maximum index in the input.",
        )
        final_dims = input_shapes[2][-1] if input_shapes is not None else 8
        # final_dims = hyper_config.get_int('comparison_dims', default=256, min=8, max=1024, help='The number of dimensions for similarity comparisons')
        return {
            "num_cards": num_cards,
            "transform_single_card": hyper_config.get_sublayer(
                "TransformSingleCard",
                sub_layer_type=MLP,
                fixed={"Final": {"activation": "linear", "dims": final_dims}},
                help="The MLP layer that tries to reconstruct the adjacency matrix row for the single card cube",
            ),
            "transform_embeddings": hyper_config.get_sublayer(
                "TransformEmbeddings",
                sub_layer_type=WDense,
                fixed={"activation": "linear", "dims": final_dims, "use_bias": False},
                help="The Dense layer to transform all the other card embeddings for comparison.",
            ),
            "margin": hyper_config.get_float(
                "probability_margin",
                default=0.0001,
                min=0.0,
                max=0.5,
                help="The distance from the endpoints (0, 1) at which to start pushing the predicted probability back towards 0.5.",
            ),
            "lambdas": {
                "extremeness": hyper_config.get_float(
                    "margin_weight",
                    default=100.0,
                    min=0.0,
                    max=1000.0,
                    help="The multiplier to scale the probability margin loss by. Suggested is 1 / probability_margin.",
                ),
                "bin_kl_divergence": hyper_config.get_float(
                    "kl_divergence_weight",
                    default=1.0,
                    min=0.0,
                    max=20.0,
                    help="How much weight to apply to the binary kl divergence loss.",
                ),
            },
        }

    def build(self, input_shapes):
        super(AdjMtxReconstructor, self).build(input_shapes)
        self.temperature = self.add_weight(
            "temperature", initializer=tf.constant_initializer(5), shape=(), trainable=True
        )

    def call(self, inputs, training=None):
        """
        input contains two things:
            input[0] = the binary vectors representing the collections
            input[1] = a diagonal matrix of size (self.N X self.N)

        We run the same encoder for each type of input, but with different
        decoders. This is because the goal is to make sure that the compression
        for collections still does a reasonable job compressing individual items.
        So a penalty term (regularization) is added to the model in the ability to
        reconstruct the probability distribution (adjacency matrix) on the item level
        from the encoding.

        The hope is that this regularization enforces this conditional probability to be
        embedded in the recommendations. As the individual items must pull towards items
        represented strongly within the graph.
        """
        single_card = tf.cast(inputs[0], dtype=tf.int32, name="single_card")
        adj_row = tf.cast(inputs[1], dtype=self.compute_dtype, name="adj_row")
        card_embeddings = tf.cast(inputs[2], dtype=self.compute_dtype, name="card_embeddings")
        transformed_card_embeddings = self.transform_embeddings(card_embeddings[1:], training=training)
        embed_single_card = self.transform_single_card(tf.gather(card_embeddings, single_card), training=training)
        embed_single_card_exp = tf.expand_dims(embed_single_card, -2)
        transformed_card_embeddings_exp = tf.expand_dims(transformed_card_embeddings, -3)
        similarities = -tf.keras.losses.cosine_similarity(
            embed_single_card_exp, transformed_card_embeddings_exp, axis=-1
        )
        pred_adj_row = tf.nn.softmax(similarities * self.temperature, axis=-1)
        # Normalize to bits not nats.
        adj_mtx_losses = tf.keras.losses.kl_divergence(adj_row, pred_adj_row) / tf.math.log(2.0)
        card_losses = {
            "bin_kl_divergence": adj_mtx_losses,
            "extremeness": tf.maximum(
                tf.math.abs(0.5 - pred_adj_row) - 0.5 + self.extremeness_margin, 0.0, name="extremeness_losses"
            ),
        }
        card_weights = tf.ones_like(adj_mtx_losses)
        sample_losses = {
            k: tf.einsum("bc,bc->b", per_card_loss, card_weights, name=f"sample_{k}_losses")
            / tf.cast(tf.shape(per_card_loss)[1], self.compute_dtype)
            for k, per_card_loss in card_losses.items()
        }
        losses = {
            k: tf.math.reduce_mean(per_sample_loss, name=f"{k}_loss") for k, per_sample_loss in sample_losses.items()
        }
        loss = tf.add_n([self.lambdas[k] * raw_loss for k, raw_loss in losses.items()], name="loss")
        int_metrics = {}
        float_metrics = (
            {
                "temperature": self.temperature,
                "pred_row_entropy": tf.reduce_sum(pred_adj_row * -tf.math.log(pred_adj_row), axis=1),
                "true_row_entropy": tf.reduce_sum(adj_row * -tf.math.log(adj_row + 1e-10), axis=1),
                "similarities": similarities,
            }
            | {f"{k}_loss": v for k, v in sample_losses.items()}
            | {"loss": loss}
        )
        ranges = {"extremeness": (0, self.margin)}
        saturates = set()
        self.log_metrics(int_metrics, float_metrics, ranges, saturates, "")
        return loss
