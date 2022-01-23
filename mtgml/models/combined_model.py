import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.item_embedding import ItemEmbedding
from mtgml.models.draftbots import DraftBot
from mtgml.models.recommender import CubeRecommender


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
            'draftbots': hyper_config.get_sublayer('DraftBots', sub_layer_type=DraftBot,
                                                   fixed={'num_cards': num_cards, 'EmbedCards': {'dims': 1}},
                                                   help='The model for the draftbots'),
            'cube_recommender': hyper_config.get_sublayer('CubeRecommender', sub_layer_type=CubeRecommender,
                                                          fixed={'num_cards': num_cards, 'EmbedCards': {'dims': 1}},
                                                          help='The model for the draftbots'),
        }

    def build(self, input_shapes):
        super(CombinedModel, self).build(input_shapes)
        self.draftbots.embed_cards = self.embed_cards
        self.cube_recommender.embed_cards = self.embed_cards
        print(input_shapes)

    def call(self, inputs, training=False):
        return self.draftbots(inputs[0], training=training), self.cube_recommender(inputs[1], training=training)
