import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.item_embedding import ItemEmbedding


class CombinedModel(ConfigurableLayer, tf.keras.models.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_cards = hyper_config.get_int('num_cards', min=1, max=None, default=None,
                                         help='The number of items that must be embedded. Should be 1 + the max index expected to see.')
        embed_cards = hyper_config.get_sublayer('EmbedCards', sub_layer_type=ItemEmbedding,
                                             fixed={'num_items': num_cards},
                                             seed_mod=29, help='The embeddings for the card objects.'),
        return {
            'embed_cards': embed_cards,
            'draftbots': DrafBots
        }
