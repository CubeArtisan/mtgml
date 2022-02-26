import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer
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
            'embed_cube': hyper_config.get_sublayer('EmbedCube', sub_layer_type=AttentiveSetEmbedding,
                                                    help='Combine the card embeddings to get an embedding for the cube.'),
            'recover_adj_mtx': hyper_config.get_sublayer('RecoverAdjMtx', sub_layer_type=MLP,
                                                         fixed={'Final': {'activation': 'softmax', 'dims': num_cards - 1}},
                                                         help='The MLP layer that tries to reconstruct the adjacency matrix row for the single card cube'),
            'recover_cube': hyper_config.get_sublayer('RecoverCube', sub_layer_type=MLP,
                                                      fixed={'Final': {'activation': 'sigmoid', 'dims': num_cards - 1}},
                                                      help='The MLP that tries to reconstruct the one hot encoding of the cube'),
            'cube_metrics': {
                'cube_recall_at_25': tf.keras.metrics.RecallAtPrecision(0.25, name='cube_recall_at_25'),
                'cube_recall_at_50': tf.keras.metrics.RecallAtPrecision(0.5, name='cube_recall_at_50'),
                'cube_recall_at_75': tf.keras.metrics.RecallAtPrecision(0.75, name='cube_recall_at_75'),
                'cube_precis_at_25': tf.keras.metrics.PrecisionAtRecall(0.25, name='cube_recall_at_25'),
                'cube_precis_at_50': tf.keras.metrics.PrecisionAtRecall(0.5, name='cube_recall_at_50'),
                'cube_precis_at_75': tf.keras.metrics.PrecisionAtRecall(0.75, name='cube_recall_at_75'),
                'cube_abs_error': tf.keras.metrics.MeanAbsoluteError(name='cube_abs_error'),
            },
            'scale_relevant_cards': hyper_config.get_float('scale_relevant_cards', min=0, max=100.0, default=5,
                                                           help='The amount to scale the loss on the cards in the noisy cube and the true cube.')
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
        noisy_cube = tf.cast(inputs[0], dtype=tf.int32, name='noisy_cube')
        if len(inputs) == 3:
            card_embeddings = tf.cast(inputs[2], dtype=self.compute_dtype, name='card_embeddings')
        else:
            card_embeddings = tf.cast(inputs[1], dtype=self.compute_dtype, name='card_embeddings')
        embed_noisy_cube = tf.gather(card_embeddings, noisy_cube)
        # embed_noisy_cube = self.embed_cards(noisy_cube, training=training)
        encoded_noisy_cube = self.embed_cube(embed_noisy_cube, training=training)
        decoded_noisy_cube = self.recover_cube(encoded_noisy_cube, training=training)
        if len(inputs) == 3:
            true_cube = tf.cast(inputs[1], dtype=tf.int32, name='true_cube_arr')
            true_cube = tf.reduce_max(tf.one_hot(true_cube, depth=self.num_cards, axis=-1, dtype=self.compute_dtype), axis=-2)[:,1:]
            cube_losses = tf.keras.losses.binary_crossentropy(tf.expand_dims(true_cube, -1), tf.expand_dims(decoded_noisy_cube, -1))
            noisy_cube_spread = tf.reduce_max(tf.one_hot(noisy_cube, depth=self.num_cards - 1, axis=-1), axis=-2)
            scaled_cubes = (noisy_cube_spread + true_cube) * tf.constant(self.scale_relevant_cards, dtype=self.compute_dtype)
            true_cube_card_ratio = (tf.constant(1, dtype=self.compute_dtype) - true_cube - noisy_cube_spread) + scaled_cubes
            cube_losses = tf.reduce_mean(cube_losses * true_cube_card_ratio, axis=-1)
            loss = tf.math.reduce_mean(cube_losses, axis=-1)
            self.add_metric(cube_losses, 'cube_loss')
            tf.summary.scalar('cube_loss', tf.reduce_mean(cube_losses))
            for name, metric in self.cube_metrics.items():
                metric.update_state(true_cube, decoded_noisy_cube)
                tf.summary.scalar(name, metric.result())
            return loss
        return decoded_noisy_cube

