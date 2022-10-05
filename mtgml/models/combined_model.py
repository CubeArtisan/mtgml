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
            'draftbots_weight': hyper_config.get_float('draftbots_weight', default=1, min=0, max=128,
                                                  help='The weight to multiply the draftbot loss by.'),
            'recommender_weight': hyper_config.get_float('recommender_weight', default=1, min=0, max=128,
                                                    help='The weight to multiply the recommender loss by.'),
            'deck_adj_mtx_weight': hyper_config.get_float('deck_adj_mtx_weight', default=1, min=0, max=128,
                                                     help='The weight to multiply the deck adjacency matrix loss by.'),
            'cube_adj_mtx_weight': hyper_config.get_float('cube_adj_mtx_weight', default=1, min=0, max=128,
                                                     help='The weight to multiply the cube adjacency matrix loss by.'),
        }
        if result['draftbots_weight'] > 0:
            result.update({
                'draftbots': hyper_config.get_sublayer('DraftBots', sub_layer_type=DraftBot,
                                                       fixed={'num_cards': num_cards}, seed_mod=13,
                                                       help='The model for the draftbots'),
            })
        if result['recommender_weight'] > 0:
            result.update({
                'cube_recommender': hyper_config.get_sublayer('CubeRecommender', sub_layer_type=CubeRecommender,
                                                              fixed={'num_cards': num_cards}, seed_mod=17,
                                                              help='The model for the draftbots'),
            })
        if result['deck_adj_mtx_weight'] > 0:
            result.update({
                'deck_adj_mtx_reconstructor': hyper_config.get_sublayer('DeckAdjMtxReconstructor', sub_layer_type=AdjMtxReconstructor,
                                                                        fixed={'num_cards': num_cards}, seed_mod=23,
                                                                        help='The model to reconstruct the deck adjacency matrix'),
            })
        if result['cube_adj_mtx_weight'] > 0:
            result.update({
                'cube_adj_mtx_reconstructor': hyper_config.get_sublayer('CubeAdjMtxReconstructor', sub_layer_type=AdjMtxReconstructor,
                                                                        fixed={'num_cards': num_cards}, seed_mod=23,
                                                                        help='The model to reconstruct the cube adjacency matrix'),
            })
        return result

    def build(self, input_shapes):
        super(CombinedModel, self).build(input_shapes)
        self.embed_cards.build(input_shapes=(None, None))

    def call(self, inputs, training=False):
        with tf.experimental.async_scope():
            loss = 0
            if self.draftbots_weight > 0:
                if len(inputs[0]) > 6:
                    draftbot_loss = self.draftbots((*inputs[0][0:-2], self.embed_cards.embeddings, *inputs[0][-2:]), training=training)
                    loss += self.draftbots_weight * draftbot_loss
                else:
                    draftbot_loss = self.draftbots((*inputs[0], self.embed_cards.embeddings), training=training)
            if self.recommender_weight:
                recommender_loss = self.cube_recommender((*inputs[1], self.embed_cards.embeddings), training=training)
                loss += self.recommender_weight * recommender_loss
            if self.deck_adj_mtx_weight > 0:
                deck_adj_mtx_loss = self.deck_adj_mtx_reconstructor((*inputs[3], self.embed_cards.embeddings), training=training)
                loss += self.deck_adj_mtx_weight * deck_adj_mtx_loss
            if self.cube_adj_mtx_weight > 0:
                cube_adj_mtx_loss = self.cube_adj_mtx_reconstructor((*inputs[2], self.embed_cards.embeddings), training=training)
                loss += self.cube_adj_mtx_weight * cube_adj_mtx_loss
            if len(inputs[0]) > 6:
                self.add_loss(loss)
                tf.summary.scalar('loss', loss)
        return loss
