import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer


class ItemEmbedding(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_items = hyper_config.get_int(
            "num_items",
            default=None,
            help="The number of items that must be embedded. Should be 1 + the max index expected to see.",
        )
        if not num_items and input_shapes:
            raise NotImplementedError("You must supply the number of items.")
        return {
            "num_items": num_items,
            "dims": hyper_config.get_int(
                "dims", min=8, max=512, default=512, help="The number of dimensions the items should be embedded into."
            ),
            "trainable": hyper_config.get_bool(
                "trainable", default=True, help="Whether the embeddings should be trained."
            ),
        }

    def build(self, input_shapes):
        super(ItemEmbedding, self).build(input_shapes)
        self.embeddings = self.add_weight(
            "embeddings",
            shape=(self.num_items, self.dims),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            trainable=self.trainable,
        )

    def compute_mask(self, inputs, mask=None):
        our_mask = inputs > 0
        if mask:
            return tf.math.logical_and(our_mask, tf.expand_dims(mask, -1), name="combined_mask")
        else:
            return our_mask

    def call(self, inputs):
        embeddings = tf.concat(
            (tf.constant(0, shape=(1, self.dims), dtype=self.compute_dtype), self.embeddings),
            axis=0,
            name="embeddings_with_zero",
        )
        return tf.gather(embeddings, inputs, "gathered_embeddings")
