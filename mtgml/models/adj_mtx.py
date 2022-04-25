from mtgml.layers.wrapped import WDense
import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.mlp import MLP

"""
    - adj_mtx is the adjacency matrix created by create_mtx.py
    and then updated such that each row sums to 1.
    - decoded_for_reg is an output of the model
"""
class AdjMtxReconstructor(ConfigurableLayer, tf.keras.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_cards = hyper_config.get_int('num_cards', min=1, max=None, default=None,
                                         help='The number of cards that must be embedded. Should be 1 + maximum index in the input.')
        return {
            "num_cards": num_cards,
            'recover_adj_mtx': hyper_config.get_sublayer('RecoverAdjMtx', sub_layer_type=MLP,
                                                         help='The MLP layer that tries to reconstruct the adjacency matrix row for the single card cube'),
            'final_recovery': hyper_config.get_sublayer('FinalRecoverAdjMtx', sub_layer_type=WDense,
                                                        fixed={'activation': 'softmax', 'dims': num_cards - 1},
                                                        help='The last layer for reconstructing the adj_mtx.'),
            'adj_mtx_metrics': {
                'adj_mtx_abs_error': tf.keras.metrics.MeanAbsoluteError(name='adj_mtx_abs_error'),
            },
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
        card_embed = tf.cast(inputs[0], dtype=self.compute_dtype, name='single_card')
        adj_row = tf.cast(inputs[1], dtype=self.compute_dtype, name='adj_row')
        decoded_adj_row_pre = self.recover_adj_mtx(card_embed, training=training)
        decoded_adj_row = self.final_recovery(decoded_adj_row_pre, training=training)
        adj_mtx_losses = tf.keras.losses.kl_divergence(adj_row, decoded_adj_row)
        loss = tf.nn.compute_average_loss(adj_mtx_losses)
        self.add_metric(adj_mtx_losses, f'{self.name}_loss')
        tf.summary.scalar(f'{self.name}_loss', tf.reduce_mean(adj_mtx_losses))
        self.add_loss(loss)
        # for name, metric in self.adj_mtx_metrics.items():
        #     metric.update_state(adj_row, decoded_single_card)
        #     tf.summary.scalar(name, metric.result())
        return loss
