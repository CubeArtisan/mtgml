import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.item_embedding import ItemEmbedding
from mtgml.layers.mlp import MLP
from mtgml.layers.set_embedding import AttentiveSetEmbedding

"""
    - adj_mtx is the adjacency matrix created by create_mtx.py
    and then updated such that each row sums to 1.
    - decoded_for_reg is an output of the model
"""
class CubeRecommender(ConfigurableLayer, tf.keras.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_cards = hyper_config.get_int('num_cards', min=1, max=None, default=None,
                                         help='The number of cards that must be embedded. Should be 1 + maximum index in the input.')
        return {
            'num_cards': num_cards,
            'embed_cards': hyper_config.get_sublayer('EmbedCards', sub_layer_type=ItemEmbedding,
                                                     fixed={'num_items': num_cards},
                                                     help='The card embeddings.'),
            'embed_cube': hyper_config.get_sublayer('EmbedCube', sub_layer_type=AttentiveSetEmbedding,
                                                    help='Combine the card embeddings to get an embedding for the cube.'),
            'recover_adj_mtx': hyper_config.get_sublayer('RecoverAdjMtx', sub_layer_type=MLP,
                                                         fixed={'Final': {'activation': 'softmax', 'dims': num_cards - 1}}),
            'recover_cube': hyper_config.get_sublayer('RecoverCube', sub_layer_type=MLP,
                                                      fixed={'Final': {'activation': 'softmax', 'dims': num_cards - 1}}),
        }

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
        cards = tf.range(1, self.num_cards)
        card_embeds = self.embed_cards(cards, training=training)
        decoded_for_reg = []
        if isinstance(inputs, tuple):
            x, identity = inputs
            identity_embeds = identity * card_embeds
            encode_for_reg = self.embed_cube(identity_embeds, training=training)
            decoded_for_reg = self.recover_adj_mtx(encode_for_reg, training=training)
        else:
            x = inputs
        cube_card_embeds = card_embeds * x
        encoded = self.embed_cube(cube_card_embeds, training=training)
        reconstruction = self.recover_cube(encoded, training=training)
        if isinstance(inputs, tuple):
            return reconstruction, decoded_for_reg
        else:
            return reconstruction

