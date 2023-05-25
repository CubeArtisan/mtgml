import tensorflow as tf

from mtgml.config.hyper_config import HyperConfig
from mtgml.constants import ACTIVATION_CHOICES, LARGE_INT, should_log_histograms
from mtgml.layers.bert import BERT
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.mlp import MLP
from mtgml.layers.wrapped import WDense
from mtgml.layers.zero_masked import ZeroMasked


class ContextualRating(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        measure_dims = hyper_config.get_int(
            "measure_dims", min=8, max=256, step=8, default=64, help="The number of dimensions to calculate distance in"
        )
        final_activation = hyper_config.get_choice(
            "final_activation",
            choices=ACTIVATION_CHOICES,
            default="linear",
            help="The final activation before calculating distance",
        )
        embed_item = hyper_config.get_sublayer(
            "EmbedItem",
            sub_layer_type=MLP,
            seed_mod=71,
            fixed={"Final": {"dims": measure_dims, "activation": final_activation}},
            help="Transforms the card embeddings to the embedding used to calculate distances.",
        )
        embed_context = hyper_config.get_sublayer(
            "EmbedContext",
            sub_layer_type=BERT,
            fixed={
                "use_causal_mask": True,
                "use_shifted_causal_mask": hyper_config.get_bool(
                    "use_shifted_causal_mask", default=False, help="Don't allow items to attend to themselves"
                ),
            },
            help="The Attentive set embedding layer to use if set_embed_type is 'attentive'",
        )
        project_context = hyper_config.get_sublayer(
            "ProjectContext",
            sub_layer_type=WDense,
            seed_mod=93,
            fixed={"dims": measure_dims, "activation": final_activation},
            help="Project the context embeddings to the space for measuring distance.",
        )
        # TODO: Add option to use cosine similarity (with temp) or inner product instead of l2 distance.
        return {
            "embed_item": embed_item,
            "embed_context": embed_context,
            "project_context": project_context,
            "bounded_distance": hyper_config.get_bool(
                "bounded_distance", default=False, help="Transform the distance to be in the range (0, 1)"
            ),
            "supports_masking": True,
        }

    def call(self, inputs, training=False, mask=None):
        items, context = inputs
        if mask is not None:
            items_mask, context_mask = mask
        else:
            items_mask = None
            context_mask = None
        item_embeds = self.embed_item(items, mask=items_mask, training=training)
        context_embeds_pre = self.embed_context(context, training=training, mask=context_mask)
        context_embeds = self.project_context(context_embeds_pre, training=training, mask=context_mask)
        distances = tf.reduce_sum(
            tf.math.squared_difference(item_embeds, tf.expand_dims(context_embeds, -2), name="squared_embed_diffs"),
            -1,
            name="squared_distances",
        )
        if self.bounded_distance:
            one = tf.constant(1, dtype=self.compute_dtype)
            large = tf.constant(LARGE_INT, dtype=self.compute_dtype)
            nonlinear_distances = tf.math.divide(
                large, tf.math.add(one, distances, name="distances_incremented"), name="nonlinear_distances"
            )
        else:
            nonlinear_distances = tf.math.negative(distances, name="reversed_distances")
        if should_log_histograms():
            tf.summary.histogram("distances", distances)
            if self.bounded_distance:
                tf.summary.histogram("nonlinear_distances", nonlinear_distances)
        return nonlinear_distances
