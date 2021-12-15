import tensorflow as tf

from mtgml.config.hyper_config import HyperConfig
from mtgml.constants import ACTIVATION_CHOICES
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.mlp import MLP
from mtgml.layers.set_embedding import AdditiveSetEmbedding, AttentiveSetEmbedding, SET_EMBEDDING_CHOICES
from mtgml.layers.zero_masked import ZeroMasked


class ContextualRating(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        measure_dims = hyper_config.get_int('measure_dims', min=8, max=256, step=8, default=128,
                                            help='The number of dimensions to calculate distance in')
        final_activation = hyper_config.get_choice('final_activation', choices=ACTIVATION_CHOICES,
                                                   default='linear',
                                                   help='The final activation before calculating distance')
        embed_item = hyper_config.get_sublayer('EmbedItem', sub_layer_type=MLP,
                                               fixed={'Final': {'dims': measure_dims,
                                                                'activation': final_activation}},
                                               help='Transforms the card embeddings to the embedding used to calculate distances.')
        set_embed_type = hyper_config.get_choice('set_embed_type', choices=SET_EMBEDDING_CHOICES,
                                                 default='attentive', help='The kind of set embedding to use to get the contexts embedding for distance calculation.')
        if set_embed_type == 'additive':
            embed_context = hyper_config.get_sublayer('EmbedContext', sub_layer_type=AdditiveSetEmbedding,
                                                      fixed={'Decoder':
                                                             {'Final': {'dims': measure_dims,
                                                                        'activation': final_activation}}},
                                                      help="The Additive set embedding layer to use if set_embed_type is 'additive'")
        elif set_embed_type == 'attentive':
            embed_context = hyper_config.get_sublayer('EmbedContext', sub_layer_type=AttentiveSetEmbedding,
                                                      fixed={'Decoder':
                                                             {'Final': {'dims': measure_dims,
                                                                        'activation': final_activation}}},
                                                      help="The Attentive set embedding layer to use if set_embed_type is 'attentive'")
        else:
            raise NotImplementedError('This form of set_embed_type is not supported for this layer')
        return {
            'embed_item': embed_item,
            'embed_context': embed_context,
            'bounded_distance': hyper_config.get_bool('bounded_distance', default=False,
                                                      help='Transform the distance to be in the range (0, 1)'),
            'zero_masked': ZeroMasked(HyperConfig(seed=hyper_config.seed * 47)),
        }

    def call(self, inputs, training=False, mask=None):
        items, contexts = inputs
        item_embeds = self.embed_item(items, training=training)
        context_embeds = self.embed_context(contexts, training=training)
        embed_diffs = tf.math.subtract(item_embeds, tf.expand_dims(context_embeds, 1, name='expanded_context_embeds'),
                                       name='embed_diffs')
        distances = tf.reduce_sum(tf.math.square(embed_diffs, name='squared_embed_diffs'), -1, name='distances')
        if self.bounded_distance:
            one = tf.constant(1, dtype=self.compute_dtype)
            nonlinear_distances = tf.math.divide(one, tf.math.add(one, distances,
                                                                  name='distances_incremented'),
                                                 name='nonlinear_distances')
        else:
            nonlinear_distances = tf.math.subtract(tf.constant(1e+09, dtype=self.compute_dtype),
                                                   distances, name='negative_distances')
        nonlinear_distances = self.zero_masked(nonlinear_distances, mask=mask[0])
        # Logging for tensorboard
        tf.summary.histogram('outputs/distances', distances)
        tf.summary.histogram('outputs/nonlinear_distances', nonlinear_distances)
        return nonlinear_distances

