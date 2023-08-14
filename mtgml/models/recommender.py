import tensorflow as tf

from mtgml.constants import EPSILON, should_log_histograms
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.mlp import MLP
from mtgml.layers.set_embedding import AttentiveSetEmbedding
from mtgml.layers.wrapped import WDense


class CubeRecommender(ConfigurableLayer, tf.keras.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_cards = hyper_config.get_int(
            "num_cards",
            min=1,
            max=None,
            default=None,
            help="The number of cards that must be embedded. Should be 1 + maximum index in the input.",
        )
        embed_dims = hyper_config.get_int(
            "embed_dims", default=256, min=8, max=1024, help="The number of dimensions for similarity comparisons"
        )
        return {
            "num_cards": num_cards,
            "embed_cube": hyper_config.get_sublayer(
                "EmbedCube",
                sub_layer_type=AttentiveSetEmbedding,
                fixed={
                    "Decoder": {"Final": {"activation": "linear", "dims": embed_dims}},
                    "ItemDropout": {"rate": 0.0},
                    "use_causal_mask": False,
                },
                help="Combine the card embeddings to get an embedding for the cube.",
            ),
            "transform_cards": hyper_config.get_sublayer(
                "TransformCards",
                sub_layer_type=MLP,
                fixed={"Final": {"dims": embed_dims, "activation": "linear"}},
                help="Transform card embeddings to a different orientation",
            ),
            "cube_metrics": {
                "cube_recall_at_25": tf.keras.metrics.RecallAtPrecision(0.25, name="cube_recall_at_25"),
                "cube_recall_at_50": tf.keras.metrics.RecallAtPrecision(0.5, name="cube_recall_at_50"),
                "cube_recall_at_75": tf.keras.metrics.RecallAtPrecision(0.75, name="cube_recall_at_75"),
                "cube_precis_at_25": tf.keras.metrics.PrecisionAtRecall(0.25, name="cube_precis_at_25"),
                "cube_precis_at_50": tf.keras.metrics.PrecisionAtRecall(0.5, name="cube_precis_at_50"),
                "cube_precis_at_75": tf.keras.metrics.PrecisionAtRecall(0.75, name="cube_precis_at_75"),
            },
            "scale_relevant_cards": hyper_config.get_float(
                "scale_relevant_cards",
                min=0,
                max=100.0,
                default=5,
                help="The amount to scale the loss on the cards in the input cube and the true cube.",
            ),
            "margin": hyper_config.get_float(
                "probability_margin",
                default=0.001,
                min=0.0,
                max=0.5,
                help="The distance from the endpoints (0, 1) at which to start pushing the predicted probability back towards 0.5.",
            ),
            "lambdas": {
                "log": hyper_config.get_float(
                    "log_weight",
                    min=0.0,
                    max=10.0,
                    default=1.0,
                    help="The amount to scale the log/crossentropy loss for the cubes.",
                ),
                "mse": hyper_config.get_float(
                    "log_weight",
                    min=0.0,
                    max=10.0,
                    default=0.0,
                    help="The amount to scale the log/crossentropy loss for the cubes.",
                ),
                "mae": hyper_config.get_float(
                    "log_weight",
                    min=0.0,
                    max=10.0,
                    default=0.0,
                    help="The amount to scale the log/crossentropy loss for the cubes.",
                ),
                "temperature_reg": hyper_config.get_float(
                    "temperature_reg_weight",
                    min=0,
                    max=10,
                    default=0.01,
                    help="The amount to scale the squared temperature by for loss.",
                ),
                "extremeness": hyper_config.get_float(
                    "margin_weight",
                    default=1.0,
                    min=0.0,
                    max=1000,
                    help="The multiplier to scale the probability margin loss by. Suggested is 1 / probability_margin.",
                ),
            },
        }

    def build(self, input_shapes):
        super(CubeRecommender, self).build(input_shapes)
        self.temperature = self.add_weight(
            "temperature",
            initializer=tf.constant_initializer(5),
            shape=(),
            trainable=True,
        )
        self.similarity_offset = self.add_weight(
            "similarity_offset",
            initializer=tf.constant_initializer(0.5),
            shape=(),
            trainable=True,
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
        input_cube = tf.cast(inputs[0], dtype=tf.int32, name="input_cube")
        if len(inputs) == 3:
            card_embeddings = tf.cast(inputs[2], dtype=self.compute_dtype, name="card_embeddings")
        else:
            card_embeddings = tf.cast(inputs[1], dtype=self.compute_dtype, name="card_embeddings")
        embed_input_cube = tf.gather(card_embeddings, input_cube, name="embed_input_cube")
        encoded_input_cube = tf.identity(
            self.embed_cube(embed_input_cube, training=training), name="encoded_input_cube"
        )
        transformed_cards = self.transform_cards(card_embeddings[1:], training=training)
        encoded_input_cube_exp = tf.expand_dims(encoded_input_cube, -2)
        transformed_cards_exp = tf.expand_dims(transformed_cards, -3)
        similarities = -tf.keras.losses.cosine_similarity(encoded_input_cube_exp, transformed_cards_exp, axis=-1)
        decoded_input_cube = tf.nn.sigmoid(
            (similarities - self.similarity_offset) * self.temperature, name="decoded_input_cube"
        )
        if len(inputs) == 3:
            true_cube = tf.cast(inputs[1], dtype=tf.int32, name="true_cube_arr")
            true_cube = tf.reduce_max(
                tf.one_hot(true_cube, depth=self.num_cards, axis=-1, dtype=self.compute_dtype), axis=-2
            )[:, 1:]
            card_losses = {
                "log": tf.keras.losses.binary_crossentropy(
                    tf.expand_dims(true_cube, -1),
                    tf.constant(1 - 2 * EPSILON, dtype=self.compute_dtype) * tf.expand_dims(decoded_input_cube, -1)
                    + tf.constant(EPSILON, dtype=self.compute_dtype),
                ),
                "mse": tf.math.squared_difference(true_cube, decoded_input_cube, name="mse_card_losses"),
                "mae": tf.math.abs(true_cube - decoded_input_cube, name="mae_card_losses"),
                "extremeness": tf.maximum(
                    tf.math.abs(0.5 - decoded_input_cube) - 0.5 + self.margin, 0.0, name="extremeness_losses"
                ),
            }
            input_cube_spread = tf.reduce_max(
                tf.one_hot(input_cube, depth=self.num_cards - 1, axis=-1, dtype=self.compute_dtype), axis=-2
            )
            combined_cube = input_cube_spread + true_cube
            scaled_cubes = combined_cube * tf.constant(self.scale_relevant_cards - 1, dtype=self.compute_dtype)
            card_weights = tf.constant(1, dtype=self.compute_dtype) + scaled_cubes
            sample_losses = {
                k: tf.einsum("bc,bc->b", card_weights, per_card_loss, name=f"sample_{k}_losses")
                for k, per_card_loss in card_losses.items()
            }
            losses = {k: tf.math.reduce_mean(per_sample_loss) for k, per_sample_loss in sample_losses.items()} | {
                "temperature_reg": tf.square(self.temperature, name="l2_temp")
            }
            loss = tf.add_n([self.lambdas[k] * raw_loss for k, raw_loss in losses.items()], name="loss")
            for k, v in losses.items():
                if k in sample_losses:
                    self.add_metric(sample_losses[k], f"{k}_loss")
                tf.summary.scalar(f"{k}_loss", v)
            self.add_metric(self.temperature, "cube_temperature")
            if should_log_histograms():
                tf.summary.histogram("similarities", similarities)
                for name, values in sample_losses.items():
                    tf.summary.histogram(f"{name}_losses", values)
            for name, metric in self.cube_metrics.items():
                metric.update_state(true_cube, decoded_input_cube)
                tf.summary.scalar(name, metric.result())
            return loss
        return decoded_input_cube, encoded_input_cube
