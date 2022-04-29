import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.mlp import MLP
from mtgml.layers.contextual_rating import ContextualRating
from mtgml.layers.wrapped import WDense

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
        dims = hyper_config.get_int('dims', min=8, max=1024, default=512,
                                    help='The number of dims for the transformer stream.')
        return {
            'card_embeddings': tf.constant(hyper_config.get_list('card_embeddings', default=None, help='The card embeddings.')),
            "num_cards": num_cards,
            'downcast_embeddings': hyper_config.get_sublayer('DowncastEmbeddings', sub_layer_type=WDense,
                                                             fixed={ 'dims': dims },
                                                             help='Downcast the size of the card embeddings to make it fit in memory.'),
            'embed_cube': hyper_config.get_sublayer('EmbedCube', sub_layer_type=ContextualRating,
                                                    help='Combine the card embeddings to get an embedding for the cube.'),
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
        inputs = inputs[0]
        if len(inputs) == 1 and len(inputs[0]) == 2:
            inputs = inputs[0]
        noisy_cube = tf.cast(inputs[0], dtype=tf.int32, name='noisy_cube')
        embed_noisy_cube = tf.gather(self.card_embeddings, noisy_cube, name='embed_noisy_cube_pre')
        embed_noisy_cube = self.downcast_embeddings(embed_noisy_cube, training=training, mask=noisy_cube > 0)
        cube_distances = self.embed_cube((self.card_embeddings[1:], embed_noisy_cube), training=training,
                                         mask=(None, noisy_cube > 0))
        max_op = tf.grad_pass_through(lambda x: tf.maximum(tf.constant(0, dtype=self.compute_dtype), x))
        decoded_noisy_cube = max_op(tf.constant(1, dtype=self.compute_dtype) + tf.math.tanh(cube_distances))
        if len(inputs) == 2:
            true_cube = tf.cast(inputs[1], dtype=tf.int32, name='true_cube_arr')
            print('true_cube', true_cube.shape)
            true_cube = tf.reduce_max(tf.one_hot(true_cube, depth=self.num_cards, axis=-1, dtype=self.compute_dtype), axis=-2)[:,1:]
            cube_losses = tf.keras.losses.binary_crossentropy(tf.expand_dims(true_cube, -1), tf.constant(1 - 2e-10, dtype=self.compute_dtype) * tf.expand_dims(decoded_noisy_cube, -1) + tf.constant(1e-10, dtype=self.compute_dtype))
            noisy_cube_spread = tf.reduce_max(tf.one_hot(noisy_cube, depth=self.num_cards - 1, axis=-1, dtype=self.compute_dtype), axis=-2)
            scaled_cubes = (noisy_cube_spread + true_cube) * tf.constant(self.scale_relevant_cards, dtype=self.compute_dtype)
            true_cube_card_ratio = (tf.constant(1, dtype=self.compute_dtype) - true_cube - noisy_cube_spread) + scaled_cubes
            cube_losses = tf.reduce_mean(cube_losses * true_cube_card_ratio, axis=-1)
            loss = tf.math.reduce_mean(cube_losses, axis=-1)
            self.add_metric(cube_losses, 'cube_loss')
            tf.summary.scalar('cube_loss', tf.reduce_mean(cube_losses))
            for name, metric in self.cube_metrics.items():
                metric.update_state(true_cube, decoded_noisy_cube)
                tf.summary.scalar(name, metric.result())
            self.add_loss(loss)
            return loss
        return decoded_noisy_cube

