import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.item_embedding import ItemEmbedding
from mtgml.models.adj_mtx import AdjMtxReconstructor
from mtgml.models.deck_builder import DeckBuilder
from mtgml.models.draftbots import DraftBot
from mtgml.models.recommender import CubeRecommender


class CombinedModel(ConfigurableLayer, tf.keras.models.Model):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_cards = hyper_config.get_int(
            "num_cards",
            min=1,
            max=None,
            default=None,
            help="The number of items that must be embedded. Should be 1 + the max index expected to see.",
        )
        result = {
            "embed_cards": hyper_config.get_sublayer(
                "EmbedCards",
                sub_layer_type=ItemEmbedding,
                fixed={"num_items": num_cards},
                seed_mod=29,
                help="The embeddings for the card objects.",
            ),
            "draftbots_weight": hyper_config.get_float(
                "draftbots_weight", default=1, min=0, max=128, help="The weight to multiply the draftbot loss by."
            ),
            "recommender_weight": hyper_config.get_float(
                "recommender_weight", default=1, min=0, max=128, help="The weight to multiply the recommender loss by."
            ),
            "deck_builder_weight": hyper_config.get_float(
                "deck_builder_weight",
                default=1,
                min=0,
                max=128,
                help="The weight to multiply the deck builder loss by.",
            ),
            "deck_adj_mtx_weight": hyper_config.get_float(
                "deck_adj_mtx_weight",
                default=1,
                min=0,
                max=128,
                help="The weight to multiply the deck adjacency matrix loss by.",
            ),
            "cube_adj_mtx_weight": hyper_config.get_float(
                "cube_adj_mtx_weight",
                default=1,
                min=0,
                max=128,
                help="The weight to multiply the cube adjacency matrix loss by.",
            ),
        }
        if result["draftbots_weight"] > 0:
            result.update(
                {
                    "draftbots": hyper_config.get_sublayer(
                        "DraftBots",
                        sub_layer_type=DraftBot,
                        fixed={"num_cards": num_cards},
                        seed_mod=13,
                        help="The model for the draftbots",
                    ),
                }
            )
        if result["recommender_weight"] > 0:
            result.update(
                {
                    "cube_recommender": hyper_config.get_sublayer(
                        "CubeRecommender",
                        sub_layer_type=CubeRecommender,
                        fixed={"num_cards": num_cards},
                        seed_mod=17,
                        help="The model for the recommending changes to your cube",
                    ),
                }
            )
        if result["deck_builder_weight"] > 0:
            result.update(
                {
                    "deck_builder": hyper_config.get_sublayer(
                        "DeckBuilder",
                        sub_layer_type=DeckBuilder,
                        fixed={"num_cards": num_cards},
                        seed_mod=87,
                        help="The model for recommending how to build a deck out of a pool.",
                    ),
                }
            )
        if result["deck_adj_mtx_weight"] > 0:
            result.update(
                {
                    "deck_adj_mtx_reconstructor": hyper_config.get_sublayer(
                        "DeckAdjMtxReconstructor",
                        sub_layer_type=AdjMtxReconstructor,
                        fixed={"num_cards": num_cards},
                        seed_mod=23,
                        help="The model to reconstruct the deck adjacency matrix",
                    ),
                }
            )
        if result["cube_adj_mtx_weight"] > 0:
            result.update(
                {
                    "cube_adj_mtx_reconstructor": hyper_config.get_sublayer(
                        "CubeAdjMtxReconstructor",
                        sub_layer_type=AdjMtxReconstructor,
                        fixed={"num_cards": num_cards},
                        seed_mod=29,
                        help="The model to reconstruct the cube adjacency matrix",
                    ),
                }
            )
        return result

    def build(self, input_shapes):
        super(CombinedModel, self).build(input_shapes)
        self.embed_cards.build(input_shapes=(None, None))

    def call(self, inputs, training=False):
        if not isinstance(inputs[0], tuple):
            inputs = (tuple(inputs[:5]), (inputs[5],), tuple(inputs[6:8]), tuple(inputs[8:10]), tuple(inputs[10:12]))
        draftbot_loss = tf.constant(0, dtype=tf.float32)
        recommender_loss = tf.constant(0, dtype=tf.float32)
        deck_builder_loss = tf.constant(0, dtype=tf.float32)
        deck_adj_mtx_loss = tf.constant(0, dtype=tf.float32)
        cube_adj_mtx_loss = tf.constant(0, dtype=tf.float32)
        results = []
        with tf.experimental.async_scope():
            loss = 0
            if self.draftbots_weight > 0:
                if len(inputs[0]) > 6:
                    draftbot_loss = self.draftbots_weight * self.draftbots(
                        (*inputs[0][0:-2], self.embed_cards.embeddings, *inputs[0][-2:]), training=training
                    )
                else:
                    results.append(self.draftbots((*inputs[0], self.embed_cards.embeddings), training=training))
            if self.recommender_weight:
                if len(inputs[1]) > 1:
                    recommender_loss = self.recommender_weight * self.cube_recommender(
                        (*inputs[1], self.embed_cards.embeddings), training=training
                    )
                else:
                    results.append(self.cube_recommender((*inputs[1], self.embed_cards.embeddings), training=training))
            if self.deck_builder_weight:
                if len(inputs[2]) > 2:

                    deck_builder_loss = self.deck_builder_weight * self.deck_builder(
                        (*inputs[2][:2], self.embed_cards.embeddings, *inputs[2][2:]),
                        training=training,
                    )
                else:
                    results.append(
                        self.deck_builder(
                            (*inputs[2], self.embed_cards.embeddings),
                            training=training,
                        )
                    )
            if self.cube_adj_mtx_weight > 0:
                cube_adj_mtx_loss = self.cube_adj_mtx_weight * self.cube_adj_mtx_reconstructor(
                    (*inputs[3], self.embed_cards.embeddings), training=training
                )
            if self.deck_adj_mtx_weight > 0:
                deck_adj_mtx_loss = self.deck_adj_mtx_weight * self.deck_adj_mtx_reconstructor(
                    (*inputs[4], self.embed_cards.embeddings), training=training
                )
        if len(inputs[0]) > 6:
            loss = draftbot_loss + recommender_loss + deck_builder_loss + deck_adj_mtx_loss + cube_adj_mtx_loss
            self.add_loss(loss)
            tf.summary.scalar("loss", loss)
            return loss
        else:
            return results

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, 16], dtype=tf.int16),
            tf.TensorSpec(shape=[1, None], dtype=tf.int16),
            tf.TensorSpec(shape=[1, None, None], dtype=tf.int16),
            tf.TensorSpec(shape=[1, None, 4, 2], dtype=tf.int8),
            tf.TensorSpec(shape=[1, None, 4], dtype=tf.float32),
        ]
    )
    def call_draftbots(self, basics, pool, seen, seen_coords, seen_coord_weights):
        sublayer_scores, sublayer_weights = self.draftbots(
            (basics, pool, seen, seen_coords, seen_coord_weights, self.embed_cards.embeddings), training=False
        )
        return {
            "sublayer_scores": tf.identity(sublayer_scores, "sublayer_scores"),
            "sublayer_weights": tf.identity(sublayer_weights, "sublayer_weights"),
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, None], dtype=tf.int16)])
    def call_recommender(self, cube):
        decoded_noisy_cube, encoded_noisy_cube = self.cube_recommender(
            (cube, self.embed_cards.embeddings), training=False
        )
        return {
            "decoded_noisy_cube": tf.identity(decoded_noisy_cube, "decoded_noisy_cube"),
            "encoded_noisy_cube": tf.identity(encoded_noisy_cube, "encoded_noisy_cube"),
        }

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[1, None], dtype=tf.int16), tf.TensorSpec(shape=[1, None], dtype=tf.int16)]
    )
    def call_deck_builder(self, pool, instance_nums):
        card_scores = self.deck_builder((pool, instance_nums, self.embed_cards.embeddings), training=False)
        return {"card_scores": tf.identity(card_scores, "card_scores")}
