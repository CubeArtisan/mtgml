import abc
from collections.abc import Sequence

import tensorflow as tf
from tensorboard.plugins.histogram.metadata import create_summary_metadata

from mtgml.config.hyper_config import HyperConfig
from mtgml.constants import should_log_histograms
from mtgml.tensorboard.timeseries import log_integer_histogram
from mtgml.utils.masked import (
    get_mask_for,
    histogram_masked,
    make_buckets,
    reduce_max_masked,
    reduce_min_masked,
    reduce_sum_masked,
)


class ConfigurableLayer(tf.keras.layers.Layer, metaclass=abc.ABCMeta):
    def __init__(self, hyper_config, **kwargs):
        super(ConfigurableLayer, self).__init__(**kwargs)
        self.hyper_config = hyper_config
        self.seed = self.hyper_config.seed
        self.built = False
        self.metrics_storage = {"counts": {}, "buckets": {}, "sum": {}, "value_count": {}}

    def get_config(self):
        config = super(ConfigurableLayer, self).get_config()
        config.update({"hyper_config": self.hyper_config.get_config()})
        return config

    @classmethod
    def from_config(cls, config):
        config["hyper_config"] = HyperConfig(layer_type=cls, data=config["hyper_config"])
        return cls(**config)

    def build(self, input_shapes):
        if self.built:
            return
        properties = self.get_properties(self.hyper_config, input_shapes=input_shapes)
        for key, prop in properties.items():
            setattr(self, key, prop)
        self.built = True

    def collapse_losses(
        self, losses_with_weights: Sequence[tuple[dict[str, tf.Tensor], tf.Tensor | None] | dict[str, tf.Tensor]]
    ) -> tuple[list[dict[str, tf.Tensor]], tf.Tensor]:
        prev_losses = {}
        prev_weights = None
        results = []
        for i, x in enumerate(losses_with_weights):
            if isinstance(x, (tuple, list)):
                losses, weights = x
            else:
                weights = None
                losses = x
            if prev_weights is None:
                new_losses = {
                    k: tf.reduce_mean(prev_loss, axis=-1, name=f"{k}_losses_{i}")
                    for k, prev_loss in prev_losses.items()
                }
            else:
                new_losses = {
                    k: tf.einsum("...a,...a->...", prev_loss, prev_weights, name=f"{k}_losses_{i}")
                    / tf.cast(tf.shape(prev_loss)[-1], dtype=prev_loss.dtype)
                    for k, prev_loss in prev_losses.items()
                }
            prev_losses = new_losses | losses
            results.append(prev_losses)
            prev_weights = weights
        loss = tf.add_n(
            [
                tf.math.multiply(self.lambdas[k], raw_loss, name=f"{k}_loss_weighted")
                for k, raw_loss in prev_losses.items()
            ],
            name="loss",
        )
        return results, loss

    def log_metrics(
        self,
        int_metrics: dict[str, tf.Tensor],
        float_metrics: dict[str, tf.Tensor],
        ranges: dict[str, tuple[int, int] | tuple[int, int, int]],
        saturate: frozenset[str] | set[str],
        mask: tf.Tensor | None = None,
    ):
        with tf.xla.experimental.jit_scope(False):
            all_metrics = float_metrics | int_metrics
            is_scalar = {name: value.shape == () or value.shape == (1,) for name, value in all_metrics.items()}
            all_metrics = {
                name: tf.reshape(value, (1,)) if is_scalar[name] else value for name, value in all_metrics.items()
            }
            replica_context = tf.distribute.get_replica_context()
            if mask is not None and replica_context is not None:
                mask = replica_context.all_gather(mask, axis=0)
            for name, value in all_metrics.items():
                if replica_context is not None:
                    value = replica_context.all_gather(value, axis=0)
                if not is_scalar[name] and should_log_histograms():
                    interval = (
                        ranges[name]
                        if name in ranges
                        else (
                            reduce_min_masked(value, mask=mask, axis=None, name=f"{name}_min"),
                            reduce_max_masked(value, mask=mask, axis=None, name=f"{name}_max"),
                        )
                    )
                    if name in int_metrics:
                        if len(interval) == 3:
                            start_index, max_index, bucket_size = interval
                        else:
                            start_index, max_index = interval
                            bucket_size = 1
                        real_max_index = max_index + bucket_size
                        repeats = 30 // ((real_max_index - start_index) // bucket_size)
                        edges = tf.range(
                            start_index, real_max_index + bucket_size, bucket_size, dtype=tf.int32, name=f"{name}_edges"
                        )
                        buckets = tf.stack([edges[:-1], edges[1:]], axis=0, name=f"{name}_buckets_initial")
                    else:
                        repeats = 1
                        buckets = make_buckets(
                            interval[0], end_range=interval[1], num_buckets=30, name=f"{name}_buckets_initial"
                        )
                    buckets = tf.cast(buckets, tf.float32, name="cast_buckets")
                if name not in self.metrics_storage["sum"]:
                    if not is_scalar[name] and should_log_histograms():
                        self.metrics_storage["counts"][name] = self.add_weight(
                            name=f"{name}_counts",
                            initializer=lambda *_1, **_2: tf.zeros_like(buckets[0]),
                            shape=(None,),
                            trainable=False,
                            collections=[],
                            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                        )
                        self.metrics_storage["buckets"][name] = self.add_weight(
                            name=f"{name}_buckets",
                            initializer=lambda *_1, **_2: buckets,
                            collections=[],
                            shape=(2, None),
                            trainable=False,
                            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                        )
                    self.metrics_storage["sum"][name] = self.add_weight(
                        name=f"{name}_summed",
                        initializer=lambda *_1, **_2: 0.0,
                        collections=[],
                        dtype=value.dtype,
                        trainable=False,
                        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                    )
                    self.metrics_storage["value_count"][name] = self.add_weight(
                        name=f"{name}_value_count",
                        initializer=lambda *_1, **_2: 0,
                        collections=[],
                        dtype=tf.int32,
                        trainable=False,
                        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                    )
                summed = self.metrics_storage["sum"][name] + reduce_sum_masked(
                    value, mask=None if is_scalar[name] else mask, name=f"{name}_summed", axis=None
                )
                value_count = self.metrics_storage["value_count"][name] + tf.cast(
                    tf.reduce_sum(
                        get_mask_for(value, None if is_scalar[name] else mask), name=f"{name}_value_count_sum"
                    ),
                    tf.int32,
                    name=f"{name}_value_count_update",
                )
                tf.summary.scalar(
                    f"{name}{'' if is_scalar[name] else '_mean'}",
                    tf.math.divide_no_nan(
                        summed,
                        tf.cast(value_count, summed.dtype, name=f"{name}_cast_value_count"),
                        name=f"{name}{'' if is_scalar[name] else '_mean'}",
                    ),
                )
                if isinstance(self.metrics_storage["sum"][name], tf.Variable):
                    self.metrics_storage["sum"][name].assign(
                        tf.cond(
                            tf.summary.should_record_summaries(),
                            lambda: tf.zeros_like(summed),
                            lambda: summed,
                            name=f"maybe_clear_{name}_sum",
                        ),
                        name=f"assign_clear_{name}_sum",
                    )
                    self.metrics_storage["value_count"][name].assign(
                        tf.cond(
                            tf.summary.should_record_summaries(),
                            lambda: tf.zeros_like(value_count),
                            lambda: value_count,
                            name=f"maybe_clear_{name}_value_count",
                        ),
                        name=f"assign_clear_{name}_value_count",
                    )
                if not is_scalar[name]:
                    if should_log_histograms():
                        cur_buckets = self.metrics_storage["buckets"][name]
                        counts = (
                            self.metrics_storage["counts"][name]
                            + histogram_masked(
                                value,
                                mask=mask,
                                buckets=cur_buckets,
                                saturate=name in saturate,
                                name=f"{name}_histogram",
                            )[:, 2]
                        )
                        with tf.summary.experimental.summary_scope(
                            f"{name}", "AggregatedHistogramOp", [counts, cur_buckets, repeats]
                        ) as (tag, _):

                            def calculate_tensor():
                                if name in int_metrics:
                                    step_size = (buckets[1, 0] - buckets[0, 0]) / tf.cast(repeats, dtype=tf.float32)
                                    labels = tf.range(
                                        buckets[0, 0],
                                        buckets[0, 0]
                                        + repeats
                                        * tf.cast(tf.shape(counts)[0], tf.float32, name="num_counts")
                                        * step_size,
                                        step_size,
                                        dtype=tf.float32,
                                        name=f"{name}_labels",
                                    )
                                    return tf.stack(
                                        [
                                            labels - step_size / 2,
                                            labels + step_size / 2,
                                            tf.repeat(
                                                counts / tf.cast(value_count, counts.dtype),
                                                axis=0,
                                                repeats=repeats,
                                                name=f"{name}_counts_repeated",
                                            ),
                                        ],
                                        axis=1,
                                        name=f"{name}_histogram_data",
                                    )
                                else:
                                    return tf.concat(
                                        [
                                            tf.transpose(cur_buckets),
                                            tf.expand_dims(counts / tf.cast(value_count, counts.dtype), 1),
                                        ],
                                        axis=1,
                                        name=f"{name}_histogram_data",
                                    )

                            tf.summary.write(
                                tag=tag,
                                tensor=calculate_tensor,
                                step=None,
                                metadata=create_summary_metadata(name, None),
                            )
                        if isinstance(self.metrics_storage["counts"][name], tf.Variable):
                            self.metrics_storage["counts"][name].assign(
                                tf.cond(
                                    tf.summary.should_record_summaries(),
                                    lambda: tf.zeros_like(buckets[0]),
                                    lambda: counts,
                                    name=f"maybe_clear_{name}_counts",
                                ),
                                name=f"assign_clear_{name}_counts",
                            )
                            self.metrics_storage["buckets"][name].assign(
                                tf.cond(
                                    tf.summary.should_record_summaries(),
                                    lambda: buckets,
                                    lambda: self.metrics_storage["buckets"][name].read_value(),
                                    name=f"maybe_clear_{name}_buckets",
                                ),
                                name=f"assign_clear_{name}_buckets",
                            )
