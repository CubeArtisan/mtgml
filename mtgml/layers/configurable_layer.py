import abc
from collections.abc import Sequence

import tensorflow as tf
from tensorboard.plugins.histogram.metadata import create_summary_metadata

from mtgml.config.hyper_config import HyperConfig
from mtgml.constants import should_log_histograms
from mtgml.metrics.accumulating_mean import AccumulatingMean
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
        all_metrics = float_metrics | int_metrics
        is_scalar = {name: value.shape == () or value.shape == (1,) for name, value in all_metrics.items()}
        all_metrics = {
            name: tf.reshape(value, (1,)) if is_scalar[name] else value for name, value in all_metrics.items()
        }
        for name, value in all_metrics.items():
            if not getattr(self, "_is_graph_network", False):
                with self._metrics_lock:
                    match = self._get_existing_metric(name)
                    if match:
                        metric_obj = match
                    else:
                        metric_obj = AccumulatingMean(
                            name,
                            is_scalar=is_scalar[name],
                            range=ranges.get(name, None),
                            saturate=name in saturate,
                            integer=name in int_metrics,
                        )
                        self._metrics.append(metric_obj)
                metric_obj.update_state(value, mask)
