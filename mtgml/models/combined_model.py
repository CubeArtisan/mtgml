import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.item_embedding import ItemEmbedding
from mtgml.models.adj_mtx import AdjMtxReconstructor
from mtgml.models.draftbots import DraftBot
from mtgml.models.recommender import CubeRecommender


class CombinedModel(ConfigurableLayer, tf.keras.models.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_cards = hyper_config.get_int('num_cards', min=1, max=None, default=None,
                                         help='The number of items that must be embedded. Should be 1 + the max index expected to see.')
        result = {
            'embed_cards': hyper_config.get_sublayer('EmbedCards', sub_layer_type=ItemEmbedding,
                                             fixed={'num_items': num_cards},
                                             seed_mod=29, help='The embeddings for the card objects.'),
            'draftbots': hyper_config.get_sublayer('DraftBots', sub_layer_type=DraftBot,
                                                   fixed={'num_cards': num_cards}, seed_mod=13,
                                                   help='The model for the draftbots'),
            'just_draftbots': hyper_config.get_bool('just_draftbots', default=True,
                                                    help="Whether to just train the draftbots."),
        }
        if not result['just_draftbots']:
            result.update({
                'cube_recommender': hyper_config.get_sublayer('CubeRecommender', sub_layer_type=CubeRecommender,
                                                              fixed={'num_cards': num_cards}, seed_mod=17,
                                                              help='The model for the draftbots'),
                'cube_adj_mtx_reconstructor': hyper_config.get_sublayer('CubeAdjMtxReconstructor', sub_layer_type=AdjMtxReconstructor,
                                                                        fixed={'num_cards': num_cards}, seed_mod=19,
                                                                        help='The model to reconstruct the cube adjacency matrix'),
                'deck_adj_mtx_reconstructor': hyper_config.get_sublayer('DeckAdjMtxReconstructor', sub_layer_type=AdjMtxReconstructor,
                                                                        fixed={'num_cards': num_cards}, seed_mod=23,
                                                                        help='The model to reconstruct the deck adjacency matrix'),
                })
        return result

    def build(self, input_shapes):
        super(CombinedModel, self).build(input_shapes)
        self.embed_cards.build(input_shapes=(None, None))

    def call(self, inputs, training=False):
        with tf.experimental.async_scope():
            if len(inputs[0]) > 9:
                draftbot_loss = self.draftbots((*inputs[0][0:-2], self.embed_cards.embeddings, *inputs[0][-2:]), training=training)
            else:
                draftbot_loss = self.draftbots((*inputs[0], self.embed_cards.embeddings), training=training)
            if not self.just_draftbots:
                recommender_loss = self.cube_recommender((*inputs[1], self.embed_cards.embeddings), training=training)
        if len(inputs[0]) > 9:
            loss = draftbot_loss
            if not self.just_draftbots:
                loss += recommender_loss
            self.add_loss(loss)
            tf.summary.scalar('loss', loss)
        return ()
