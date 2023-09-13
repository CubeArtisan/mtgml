import tensorflow as tf

from mtgml.constants import LARGE_INT
from mtgml.layers.bert import BERTDecoder, BERTEncoder
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.mlp import MLP
from mtgml.utils.masked import reduce_mean_masked


class DraftmancerDraftbots(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_cards = hyper_config.get_int(
            "num_cards", default=None, help="The number of cards. Should be provided in fixed."
        )
        card_embed_dims = hyper_config.get_int(
            "card_embed_dims",
            default=32,
            min=8,
            max=512,
            help="The number of dimensions to use for encoding card information.",
        )
        token_stream_dims = hyper_config.get_int(
            "token_stream_dims",
            default=128,
            min=8,
            max=512,
            help="The number of dimensions to use for the token stream.",
        )
        return {
            "card_embeddings": tf.keras.layers.Embedding(
                num_cards + 2, card_embed_dims, name="card_embeddings", mask_zero=True
            ),
            "token_stream_dims": token_stream_dims,
            "card_embed_dims": card_embed_dims,
            "use_pool_positions": hyper_config.get_bool(
                "use_pool_positions", default=True, help="Whether to use position embeddings for the pool."
            ),
            "encoder": hyper_config.get_sublayer(
                "Encoder",
                sub_layer_type=BERTEncoder,
                seed_mode=37,
                fixed={
                    "use_causal_mask": True,
                    "InitialDropout": {"blank_last_dim": False},
                    "use_position_embeds": True,
                    "num_positions": input_shapes[0][1] if input_shapes is not None else 1,
                },
                help="The encoding of the card embeddings in packs.",
            ),
            "decoder": hyper_config.get_sublayer(
                "Decoder",
                sub_layer_type=BERTDecoder,
                seed_mod=47,
                fixed={
                    "use_causal_mask": True,
                    "InitialDropout": {"blank_last_dim": False},
                    "use_position_embeds": True,
                    "num_positions": input_shapes[0][1] if input_shapes is not None else 1,
                },
                help="The decoding of the encoded packs and pool.",
            ),
            "scores_dense": hyper_config.get_sublayer(
                "FinalScores",
                sub_layer_type=MLP,
                seed_mod=59,
                fixed={"Final": {"dims": num_cards + 2}},
                help="The layer that outputs all the final scores.",
            ),
        }

    def call(self, inputs, training=False):
        # pack_ids is [batch_size, num_picks, cards_per_pack], pool_ids is [batch_size, num_picks]
        pack_ids, pool_ids = inputs
        pack_embeddings = self.card_embeddings(pack_ids, training=training)
        # This does weird things with masked out indices. Suggested implementation folows.
        # pack_embeddings = tf.reduce_mean(pack_embeddings, axis=2, name="mean_pack_embeddings")
        pack_embeddings = reduce_mean_masked(pack_embeddings, axis=2, mask=pack_ids > 0, name="mean_pack_embeddings")
        positions = tf.range(tf.shape(pack_embeddings)[1], dtype=tf.int32)
        transformer_encoder_output = self.encoder((pack_embeddings, positions), mask=pool_ids > 0, training=training)
        pool_embeddings = self.card_embeddings(pool_ids, training=training)
        transformer_decoder_output = self.decoder(
            (pool_embeddings, transformer_encoder_output, positions), mask=pool_ids > 1, training=training
        )
        scores = self.scores_dense(transformer_decoder_output, training=training)
        output_mask = ~(tf.reduce_sum(tf.one_hot(pack_ids, self.num_cards + 2), axis=-2) > 0) | tf.sequence_mask(
            2, self.num_cards + 2, dtype=tf.bool
        )
        softmax_output_masks = LARGE_INT * tf.cast(output_mask, self.compute_dtype)
        probs = tf.nn.softmax(scores - softmax_output_masks, axis=-1) * tf.cast(~output_mask, dtype=self.compute_dtype)
        return probs
