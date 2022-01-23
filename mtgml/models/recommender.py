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
            "num_cards": num_cards,
            'embed_cards': hyper_config.get_sublayer('EmbedCards', sub_layer_type=ItemEmbedding,
                                                     fixed={'num_items': num_cards},
                                                     help='The card embeddings.'),
            'embed_cube': hyper_config.get_sublayer('EmbedCube', sub_layer_type=AttentiveSetEmbedding,
                                                    help='Combine the card embeddings to get an embedding for the cube.'),
            'recover_adj_mtx': hyper_config.get_sublayer('RecoverAdjMtx', sub_layer_type=MLP,
                                                         fixed={'Final': {'activation': 'softmax', 'dims': num_cards - 1}},
                                                         help='The MLP layer that tries to reconstruct the adjacency matrix row for the single card cube'),
            'recover_cube': hyper_config.get_sublayer('RecoverCube', sub_layer_type=MLP,
                                                      fixed={'Final': {'activation': 'softmax', 'dims': num_cards - 1}},
                                                      help='The MLP that tries to reconstruct the one hot encoding of the cube'),
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
        if len(inputs) == 2:
            noisy_cube = tf.cast(inputs[0], dtype=tf.int32, name='noisy_cube')
            true_cube = tf.cast(inputs[1], dtype=tf.int32, name='true_cube_arr')
            true_cube = tf.reduce_max(tf.one_hot(true_cube, depth=self.num_cards, axis=-1), axis=-2)
        else:
            noisy_cube = tf.cast(inputs[0], dtype=tf.int32, name='noisy_cube')
            single_card = tf.cast(inputs[1], dtype=tf.int32, name='single_card')
            true_cube = tf.cast(inputs[2], dtype=tf.int32, name='true_cube')
            true_cube = tf.reduce_max(tf.one_hot(true_cube, depth=self.num_cards, axis=-1), axis=-2)
            adj_row = tf.cast(inputs[3], dtype=self.compute_dtype, name='adj_row')
            embed_single_card = self.embed_cards(single_card, training=training)
            encoded_single_card = self.embed_cube(embed_single_card, training=training)
            decoded_single_card = self.recover_adj_mtx(encoded_single_card, training=training)
            card_losses = tf.keras.losses.kl_divergence(adj_row, decoded_single_card)
            self.add_loss(tf.math.reduce_mean(card_losses, axis=-1) * tf.constant(self.card_loss_weight, dtype=self.compute_dtype))
            self.add_metric(card_losses, 'card_losses')
        embed_noisy_cube = self.embed_cards(noisy_cube, training=training)
        encoded_noisy_cube = self.embed_cube(embed_noisy_cube, training=training)
        decoded_noisy_cube = self.recover_cube(encoded_noisy_cube, training=training)
        cube_losses = tf.keras.losses.binary_cross_entropy(true_cube, decoded_noisy_cube)
        self.add_loss(tf.math.reduce_mean(cube_losses, axis=-1) * tf.constant(self.cube_loss_weight, dtype=self.compute_dtype))
        self.add_metric(cube_losses, 'card_losses')
        return true_cube

