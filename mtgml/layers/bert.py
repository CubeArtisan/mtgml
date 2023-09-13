import numpy as np
import tensorflow as tf

from mtgml.constants import ACTIVATION_CHOICES
from mtgml.layers.configurable_layer import ConfigurableLayer
from mtgml.layers.extended_dropout import ExtendedDropout
from mtgml.layers.mlp import MLP
from mtgml.layers.wrapped import WDense, WMultiHeadAttention


class Transformer(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            "attention": hyper_config.get_sublayer(
                f"Attention",
                sub_layer_type=WMultiHeadAttention,
                seed_mod=39,
                fixed={
                    "use_causal_mask": hyper_config.get_bool(
                        "use_causal_mask",
                        default=False,
                        help="Ensure items only attend to items that came before them.",
                    ),
                },
                help=f"The initial attention layer",
            ),
            "final_mlp": hyper_config.get_sublayer(
                f"FinalMLP",
                sub_layer_type=MLP,
                seed_mod=47,
                fixed={"use_layer_norm": False, "use_batch_norm": False},
                help=f"The final transformation.",
            ),
            "layer_norm": tf.keras.layers.LayerNormalization(),
            "use_causal_mask": hyper_config.get_bool(
                "use_causal_mask", default=False, help="Ensure items only attend to items that came before them."
            ),
            "final_dropout": hyper_config.get_sublayer(
                "FinalDropout",
                sub_layer_type=ExtendedDropout,
                seed_mod=93,
                fixed={"noise_shape": None, "return_mask": False, "blank_last_dim": False},
                help="The dropout layer after applying the FinalMLP.",
            ),
            "supports_masking": True,
        }

    def call(self, inputs, training=False, mask=None):
        if len(inputs) == 1:
            tokens = inputs
            attended = self.attention(
                tokens, tokens, training=training, mask=mask, use_causal_mask=self.use_causal_mask
            )
        elif len(inputs) == 2:
            tokens, attention_mask = inputs
            tokens._keras_mask = None
            attended = self.attention(
                tokens,
                tokens,
                tokens,
                attention_mask=attention_mask,
                use_causal_mask=self.use_causal_mask,
                training=training,
                mask=mask,
            )
        elif len(inputs) == 3:
            query, tokens, attention_mask = inputs
            query._keras_mask = None
            tokens._keras_mask = None
            attended = self.attention(
                query,
                tokens,
                tokens,
                attention_mask=attention_mask,
                use_causal_mask=self.use_causal_mask,
                training=training,
                mask=mask,
            )
        else:
            raise ValueError("Invalid number of inputs.")
        transformed = self.final_mlp(attended, training=training, mask=mask)
        transformed = self.final_dropout(transformed, training=training, mask=mask)
        return self.layer_norm(transformed + tokens, training=training)


class BERTEncoder(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_layers = (
            hyper_config.get_int("num_hidden_layers", min=0, max=16, default=2, help="Number of transformer blocks.")
            + 1
        )
        use_causal_mask = hyper_config.get_bool(
            "use_causal_mask", default=False, help="Ensure items only attend to items that came before them."
        )
        original_token_dims = hyper_config.get_int(
            "token_stream_dims",
            default=128,
            min=8,
            max=1024,
            help="The size of the token embeddings passed between layers.",
        )
        use_bias = hyper_config.get_bool("use_bias", default=True, help="Use bias in the dense layers")
        num_heads = hyper_config.get_int(
            "num_heads", min=1, max=64, default=4, help="The number of separate heads of attention to use."
        )
        key_dims = hyper_config.get_int(
            "key_dims", min=1, max=64, default=32, help="Size of the attention head for query and key."
        )
        value_dims = hyper_config.get_int(
            "value_dims", min=1, max=64, default=32, help="Size of the attention head for value."
        )
        attention_output_dims = num_heads * value_dims
        attention_props = {
            "key_dims": key_dims,
            "num_heads": num_heads,
            "value_dims": value_dims,
            "dropout": hyper_config.get_float(
                "attention dropout_rate",
                min=0,
                max=1,
                default=0.25,
                help="The dropout rate for the attention layers of the transformer blocks.",
            ),
            "use_bias": use_bias,
            "output_dims": attention_output_dims,
        }
        num_hidden_dense = hyper_config.get_int(
            "num_hidden_dense", min=0, max=12, default=1, help="The number of hidden dense layers"
        )
        dense_dropout_rate = hyper_config.get_float(
            "dense dropout_rate",
            min=0,
            max=1,
            default=0.25,
            help="The dropout rate for the dense layers of the transformer blocks.",
        )

        activation = hyper_config.get_choice(
            "dense_activation",
            choices=ACTIVATION_CHOICES,
            default="selu",
            help="The activation function on the output of the layer.",
        )
        dense_props = (
            {
                "num_hidden": num_hidden_dense,
                "Final": {
                    "dims": original_token_dims,
                    "use_bias": use_bias,
                    "activation": activation,
                },
            }
            | {f"Dropout_{i}": {"rate": dense_dropout_rate} for i in range(num_hidden_dense)}
            | {
                f"Hidden_{i}": {
                    "dims": attention_output_dims * 2 ** (i + 1),
                    "use_bias": use_bias,
                    "activation": activation,
                }
                for i in range(num_hidden_dense)
            }
        )
        props = {
            "use_position_embeds": hyper_config.get_bool(
                "use_position_embeds",
                default=None,
                help="Whether to expect positions along with token indices that need to be embedded.",
            ),
            "initial_dropout": hyper_config.get_sublayer(
                "InitialDropout",
                sub_layer_type=ExtendedDropout,
                seed_mod=29,
                fixed={"return_mask": True, "noise_shape": None},
                help="The dropout to apply to the tokens before any other operations.",
            ),
            "seq_length": input_shapes[1] if input_shapes is not None else 1,
            # This should fix all the hyperparameters of the layer but is left for future proofing.
            "transform_initial_tokens": hyper_config.get_sublayer(
                "TransformInitialTokens",
                sub_layer_type=WDense,
                fixed={"dims": original_token_dims, "use_bias": False, "activation": "linear"},
                seed_mod=37,
                help="The layer to upscale or downscale the token embeddings.",
            ),
            "stream_dims": original_token_dims,
            "layers": tuple(
                hyper_config.get_sublayer(
                    f"Transformer_{i}",
                    sub_layer_type=Transformer,
                    seed_mod=23,
                    fixed={
                        "FinalMLP": dense_props,
                        "Attention": attention_props,
                        "FinalDropout": {"rate": dense_dropout_rate},
                        "use_causal_mask": use_causal_mask,
                        "dense_dropout": dense_dropout_rate,
                    },
                    help=f"The {i}th transformer layer.",
                )
                for i in range(num_layers)
            ),
            "supports_masking": True,
        }
        if props["use_position_embeds"]:
            props["num_positions"] = hyper_config.get_int(
                "num_positions", default=None, min=1, max=None, help="The number of position embeddings to create."
            )
        return props

    def build(self, input_shapes):
        super().build(input_shapes)
        if self.use_position_embeds:
            self.position_embeddings = self.add_weight(
                "copy_embeddings", shape=(self.num_positions + 1, self.stream_dims), trainable=True
            )

    def call(self, inputs, mask=None, training=False):
        if self.use_position_embeds:
            token_embeds = inputs[0]
        else:
            token_embeds = inputs
        if mask is None:
            mask = tf.ones(tf.shape(token_embeds)[:-1], dtype=tf.bool)
        token_embeds, dropout_mask = self.initial_dropout(token_embeds, training=training, mask=mask)
        if len(dropout_mask.shape) > len(mask.shape):
            mask = tf.math.reduce_any(dropout_mask, axis=-1)
        else:
            mask = dropout_mask
        token_embeds = self.transform_initial_tokens(token_embeds, training=training, mask=mask)
        if self.use_position_embeds:
            position_tokens = tf.gather(self.position_embeddings, inputs[1], name="position_tokens")
            token_embeds = token_embeds + position_tokens / tf.constant(
                np.sqrt(self.stream_dims), dtype=self.compute_dtype
            )
        embeddings = tf.expand_dims(tf.cast(mask, dtype=self.compute_dtype), -1) * token_embeds
        attention_mask = tf.logical_and(tf.expand_dims(mask, -1), tf.expand_dims(mask, -2), name="attention_mask")
        for layer in self.layers:
            embeddings = layer((embeddings, attention_mask), training=training, mask=mask)
        return embeddings


BERT = BERTEncoder


class BERTDecoder(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_layers = (
            hyper_config.get_int("num_hidden_layers", min=0, max=16, default=2, help="Number of transformer blocks.")
            + 1
        )
        use_causal_mask = hyper_config.get_bool(
            "use_causal_mask", default=False, help="Ensure items only attend to items that came before them."
        )
        original_token_dims = hyper_config.get_int(
            "token_stream_dims",
            default=128,
            min=8,
            max=1024,
            help="The size of the token embeddings passed between layers.",
        )
        use_bias = hyper_config.get_bool("use_bias", default=True, help="Use bias in the dense layers")
        num_heads = hyper_config.get_int(
            "num_heads", min=1, max=64, default=4, help="The number of separate heads of attention to use."
        )
        key_dims = hyper_config.get_int(
            "key_dims", min=1, max=64, default=32, help="Size of the attention head for query and key."
        )
        value_dims = hyper_config.get_int(
            "value_dims", min=1, max=64, default=32, help="Size of the attention head for value."
        )
        attention_output_dims = num_heads * value_dims
        attention_props = {
            "key_dims": key_dims,
            "num_heads": num_heads,
            "value_dims": value_dims,
            "dropout": hyper_config.get_float(
                "attention dropout_rate",
                min=0,
                max=1,
                default=0.25,
                help="The dropout rate for the attention layers of the transformer blocks.",
            ),
            "use_bias": use_bias,
            "output_dims": attention_output_dims,
        }
        num_hidden_dense = hyper_config.get_int(
            "num_hidden_dense", min=0, max=12, default=1, help="The number of hidden dense layers"
        )
        dense_dropout_rate = hyper_config.get_float(
            "dense dropout_rate",
            min=0,
            max=1,
            default=0.25,
            help="The dropout rate for the dense layers of the transformer blocks.",
        )

        activation = hyper_config.get_choice(
            "dense_activation",
            choices=ACTIVATION_CHOICES,
            default="selu",
            help="The activation function on the output of the layer.",
        )
        dense_props = (
            {
                "num_hidden": num_hidden_dense,
                "Final": {
                    "dims": original_token_dims,
                    "use_bias": use_bias,
                    "activation": activation,
                },
            }
            | {f"Dropout_{i}": {"rate": dense_dropout_rate} for i in range(num_hidden_dense)}
            | {
                f"Hidden_{i}": {
                    "dims": attention_output_dims * 2 ** (i + 1),
                    "use_bias": use_bias,
                    "activation": activation,
                }
                for i in range(num_hidden_dense)
            }
        )
        props = {
            "use_position_embeds": hyper_config.get_bool(
                "use_position_embeds",
                default=None,
                help="Whether to expect positions along with token indices that need to be embedded.",
            ),
            "initial_dropout": hyper_config.get_sublayer(
                "InitialDropout",
                sub_layer_type=ExtendedDropout,
                seed_mod=29,
                fixed={"return_mask": True, "noise_shape": None},
                help="The dropout to apply to the tokens before any other operations.",
            ),
            "seq_length": input_shapes[1] if input_shapes is not None else 1,
            # This should fix all the hyperparameters of the layer but is left for future proofing.
            "transform_initial_tokens": hyper_config.get_sublayer(
                "TransformInitialTokens",
                sub_layer_type=WDense,
                fixed={"dims": original_token_dims, "use_bias": False, "activation": "linear"},
                seed_mod=37,
                help="The layer to upscale or downscale the token embeddings.",
            ),
            "stream_dims": original_token_dims,
            "layers": tuple(
                (
                    hyper_config.get_sublayer(
                        f"SelfTransformer_{i}",
                        sub_layer_type=Transformer,
                        seed_mod=23,
                        fixed={
                            "FinalMLP": dense_props,
                            "Attention": attention_props,
                            "FinalDropout": {"rate": dense_dropout_rate},
                            "use_causal_mask": use_causal_mask,
                            "dense_dropout": dense_dropout_rate,
                        },
                        help=f"The {i}th transformer layer.",
                    ),
                    hyper_config.get_sublayer(
                        f"CrossTransformer_{i}",
                        sub_layer_type=Transformer,
                        seed_mod=23,
                        fixed={
                            "FinalMLP": dense_props,
                            "Attention": attention_props,
                            "FinalDropout": {"rate": dense_dropout_rate},
                            "use_causal_mask": use_causal_mask,
                            "dense_dropout": dense_dropout_rate,
                        },
                        help=f"The {i}th transformer layer.",
                    ),
                )
                for i in range(num_layers)
            ),
            "supports_masking": True,
        }
        if props["use_position_embeds"]:
            props["num_positions"] = hyper_config.get_int(
                "num_positions", default=None, min=1, max=None, help="The number of position embeddings to create."
            )
        return props

    def build(self, input_shapes):
        super().build(input_shapes)
        if self.use_position_embeds:
            self.position_embeddings = self.add_weight(
                "copy_embeddings", shape=(self.num_positions + 1, self.stream_dims), trainable=True
            )

    def call(self, inputs, mask=None, training=False):
        if self.use_position_embeds:
            token_embeds, encoder_embeds = inputs[:2]
        else:
            token_embeds, encoder_embeds = inputs
        if mask is None:
            mask = tf.ones(tf.shape(token_embeds)[:-1], dtype=tf.bool)
        token_embeds, dropout_mask = self.initial_dropout(token_embeds, training=training, mask=mask)
        if len(dropout_mask.shape) > len(mask.shape):
            mask = tf.math.reduce_any(dropout_mask, axis=-1)
        else:
            mask = dropout_mask
        token_embeds = self.transform_initial_tokens(token_embeds, training=training, mask=mask)
        if self.use_position_embeds:
            position_tokens = tf.gather(self.position_embeddings, inputs[1], name="position_tokens")
            token_embeds = token_embeds + position_tokens / tf.constant(
                np.sqrt(self.stream_dims), dtype=self.compute_dtype
            )
        embeddings = tf.expand_dims(tf.cast(mask, dtype=self.compute_dtype), -1) * token_embeds
        attention_mask = tf.logical_and(tf.expand_dims(mask, -1), tf.expand_dims(mask, -2), name="attention_mask")
        for self_layer, cross_layer in self.layers:
            embeddings = self_layer((embeddings, attention_mask), training=training, mask=mask)
            embeddings = cross_layer((encoder_embeds, embeddings, attention_mask), training=training, mask=mask)
        return embeddings
