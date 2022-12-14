import tensorflow as tf
from mtgml.constants import is_debug

from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.set_embedding import AttentiveSetEmbedding
from mtgml.layers.wrapped import WDense
from mtgml.layers.mlp import MLP

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
        embed_dims = hyper_config.get_int('embed_dims', default=256, min=8, max=1024, help='The number of dimensions for similarity comparisons')
        return {
            'num_cards': num_cards,
            'embed_cube': hyper_config.get_sublayer('EmbedCube', sub_layer_type=AttentiveSetEmbedding,
                                                    fixed={'Decoder': {'Final': {'activation': 'linear', 'dims': embed_dims}},
                                                           'ItemDropout': {'rate': 0.0}},
                                                    help='Combine the card embeddings to get an embedding for the cube.'),
            'transform_cards': hyper_config.get_sublayer('TransformCards', sub_layer_type=MLP,
                                                         fixed={'Final': {'dims': embed_dims, 'activation': 'linear'}},
                                                         help='Transform card embeddings to a different orientation'),
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
                                                           help='The amount to scale the loss on the cards in the noisy cube and the true cube.'),
            'temperature_reg_weight': hyper_config.get_float('temperature_reg_weight', min=0, max=10, default=0.01,
                                                             help='The amount to scale the squared temperature by for loss.'),
        }

    def build(self, input_shapes):
        super(CubeRecommender, self).build(input_shapes)
        self.temperature = self.add_weight('temperature', initializer=tf.constant_initializer(2),
                                           shape=(), trainable=True)

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
        embed_noisy_cube = tf.gather(card_embeddings, noisy_cube, name='embed_noisy_cube')
        encoded_noisy_cube = tf.identity(self.embed_cube(embed_noisy_cube, training=training), name='encoded_noisy_cube')
        transformed_cards = self.transform_cards(card_embeddings[1:], training=training)
        encoded_noisy_cube_exp = tf.expand_dims(encoded_noisy_cube, -2)
        transformed_cards_exp = tf.expand_dims(transformed_cards, -3)
        similarities = -tf.keras.losses.cosine_similarity(encoded_noisy_cube_exp, transformed_cards_exp, axis=-1)
        decoded_noisy_cube = tf.nn.sigmoid((similarities - 0.5) * self.temperature, name='decoded_noisy_cube')
        if len(inputs) == 3:
            true_cube = tf.cast(inputs[1], dtype=tf.int32, name='true_cube_arr')
            true_cube = tf.reduce_max(tf.one_hot(true_cube, depth=self.num_cards, axis=-1, dtype=self.compute_dtype), axis=-2)[:,1:]
            cube_losses = tf.keras.losses.binary_crossentropy(tf.expand_dims(true_cube, -1), tf.constant(1 - 2e-10, dtype=self.compute_dtype) * tf.expand_dims(decoded_noisy_cube, -1) + tf.constant(1e-10, dtype=self.compute_dtype))
            noisy_cube_spread = tf.reduce_max(tf.one_hot(noisy_cube, depth=self.num_cards - 1, axis=-1, dtype=self.compute_dtype), axis=-2)
            scaled_cubes = (noisy_cube_spread + true_cube) * tf.constant(self.scale_relevant_cards, dtype=self.compute_dtype)
            true_cube_card_ratio = (tf.constant(1, dtype=self.compute_dtype) - true_cube - noisy_cube_spread) + scaled_cubes
            cube_losses = tf.reduce_mean(cube_losses * true_cube_card_ratio, axis=-1)
            loss = tf.math.reduce_mean(cube_losses) + self.temperature * self.temperature * self.temperature_reg_weight
            self.add_metric(cube_losses, 'cube_loss')
            self.add_metric(self.temperature, 'cube_temperature')
            tf.summary.scalar('cube_loss', tf.reduce_mean(cube_losses))
            if is_debug():
                tf.summary.histogram('similarities', similarities)
            for name, metric in self.cube_metrics.items():
                metric.update_state(true_cube, decoded_noisy_cube)
                tf.summary.scalar(name, metric.result())
            return loss
        return decoded_noisy_cube, encoded_noisy_cube
