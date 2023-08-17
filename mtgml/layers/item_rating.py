import tensorflow as tf

from mtgml.config.hyper_config import HyperConfig
from mtgml.constants import should_log_histograms
from mtgml.layers.configurable_layer import ConfigurableLayer


class ItemRating(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config: HyperConfig, input_shapes=None):
        num_items = hyper_config.get_int(
            "num_items", min=1, max=2**31 - 1, default=None, help="The number of items that should be given ratings"
        )
        if not num_items and input_shapes:
            raise NotImplementedError("You must supply the number of items.")
        return {
            "num_items": num_items,
            "activation": tf.keras.layers.Activation(
                hyper_config.get_choice(
                    "activation",
                    default="linear",
                    choices=("linear", "sigmoid", "softplus", "tanh"),
                    help="The activation function to restrict the range of the resultant rating.",
                ),
                name="RangeConstraint",
            ),
            "supports_masking": True,
        }

    def build(self, input_shape):
        super(ItemRating, self).build(input_shape)
        self.item_rating_logits = self.add_weight(
            "item_rating_logits",
            shape=(self.num_items,),
            initializer=tf.random_uniform_initializer(-0.05, 0.05, seed=self.seed),
            trainable=True,
        )

    def call(self, inputs, training=False):
        zero_rating = tf.constant(0, shape=(1,), dtype=self.compute_dtype)
        item_ratings = tf.concat([zero_rating, self.activation(self.item_rating_logits)], axis=0, name="item_ratings")
        ratings = tf.gather(item_ratings, inputs, name="ratings")
        if should_log_histograms():
            tf.summary.histogram("weights/item_ratings", item_ratings)
        return ratings
