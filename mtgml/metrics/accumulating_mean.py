import tensorflow as tf
from tensorboard.plugins.histogram.metadata import create_summary_metadata

from mtgml.constants import should_log_histograms
from mtgml.utils.masked import (
    get_mask_for,
    histogram_masked,
    make_buckets,
    reduce_max_masked,
    reduce_min_masked,
    reduce_sum_masked,
)


class AccumulatingMean(tf.keras.metrics.Metric):
    def __init__(
        self,
        name,
        is_scalar: bool = True,
        range: tuple[int, int] | tuple[int, int, int] | None = None,
        saturate: bool = False,
        integer: bool = False,
    ):
        super().__init__(name=name)
        self.is_scalar = is_scalar
        self.range = range
        self.sum = self.add_weight(
            "sum",
            shape=(),
            initializer="zeros",
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )
        self.count = self.add_weight(
            "count",
            shape=(),
            initializer="zeros",
            dtype=tf.int32,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )
        self.saturate = saturate
        if not is_scalar and should_log_histograms():
            self.bucket_counts = self.add_weight(
                name="bucket_counts",
                shape=(30,),
                initializer="zeros",
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            )
            self.buckets = self.add_weight(
                name="buckets",
                shape=(2, 30),
                initializer="zeros",
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            )

    def update_state(self, values, sample_weight=None):
        with tf.name_scope(self.name):
            if self.is_scalar:
                sample_weight = None
            if values.shape == ():
                values = tf.reshape(values, (1,))
            replica_context = tf.distribute.get_replica_context()
            if replica_context is not None:
                values = replica_context.all_gather(values, axis=0)
                if sample_weight is not None:
                    sample_weight = replica_context.all_gather(sample_weight, axis=0)
            summed = tf.math.add(
                self.sum,
                reduce_sum_masked(tf.cast(values, dtype=tf.float32), mask=sample_weight, name="summed", axis=None),
                name="new_sum",
            )
            count = tf.math.add(
                self.count,
                tf.cast(
                    tf.reduce_sum(get_mask_for(values, sample_weight), name="count_sum", axis=None),
                    dtype=tf.int32,
                    name="count_sum_update",
                ),
                name="new_count",
            )
            tf.summary.scalar(
                f"{self.name}{'' if self.is_scalar else '_mean'}",
                tf.math.divide_no_nan(
                    summed,
                    tf.cast(count, dtype=summed.dtype, name="cast_count_sum"),
                    name=f"{self.name}{'' if self.is_scalar else '_mean'}",
                ),
            )
            if isinstance(self.count, tf.Variable):
                self.count.assign(
                    tf.cond(
                        tf.summary.should_record_summaries(),
                        lambda: tf.zeros_like(count),
                        lambda: count,
                        name="maybe_clear_count",
                    ),
                    name="assign_clear_count",
                )
                self.sum.assign(
                    tf.cond(
                        tf.summary.should_record_summaries(),
                        lambda: tf.zeros_like(summed),
                        lambda: summed,
                        name="maybe_clear_sum",
                    ),
                    name="assign_clear_sum",
                )
            if not self.is_scalar and should_log_histograms():
                int_metric = values.dtype in (tf.int32, tf.int64)
                interval = (
                    self.range
                    if self.range is not None
                    else (
                        reduce_min_masked(values, mask=sample_weight, axis=None, name=f"min"),
                        reduce_max_masked(values, mask=sample_weight, axis=None, name=f"max"),
                    )
                )
                if len(interval) == 3:
                    start_index, max_index, bucket_size = interval
                    start_index = tf.convert_to_tensor(start_index, dtype=values.dtype, name="start_index")
                    max_index = tf.convert_to_tensor(max_index, dtype=values.dtype, name="max_index")
                    bucket_size = tf.convert_to_tensor(bucket_size, dtype=values.dtype, name="start_index")
                else:
                    start_index, max_index = interval
                    start_index = tf.convert_to_tensor(start_index, dtype=values.dtype, name="start_index")
                    max_index = tf.convert_to_tensor(max_index, dtype=values.dtype, name="max_index")
                    bucket_size = (
                        tf.constant(1, dtype=values.dtype)
                        if int_metric
                        else (max_index - start_index) / tf.constant(29, dtype=values.dtype)
                    )
                if int_metric:
                    real_max_index = max_index + bucket_size
                    repeats = tf.constant(30, dtype=values.dtype) // ((real_max_index - start_index) // bucket_size)
                    edges = tf.range(
                        start_index, real_max_index + bucket_size, bucket_size, dtype=tf.int32, name=f"edges"
                    )
                    buckets = tf.cast(
                        tf.stack([edges[:-1], edges[1:]], axis=0, name=f"new_buckets"),
                        dtype=self.buckets.dtype,
                        name=f"buckets_initial_cast",
                    )
                    step_size = (buckets[1, 0] - buckets[0, 0]) / tf.cast(repeats, dtype=tf.float32)
                    labels = tf.range(
                        buckets[0, 0],
                        buckets[0, 0]
                        + tf.cast(repeats, tf.float32)
                        * tf.cast(tf.shape(buckets)[1], tf.float32, name="num_counts")
                        * step_size,
                        step_size,
                        dtype=tf.float32,
                        name="labels",
                    )
                    buckets_for_hist = buckets
                    buckets = tf.stack(
                        [labels - step_size / 2, labels + step_size / 2],
                        axis=0,
                        name="stacked_labels",
                    )
                else:
                    repeats = 1
                    buckets = make_buckets(start_index, end_range=max_index, num_buckets=30, name="new_buckets")
                cur_buckets = tf.cond(
                    tf.reduce_max(self.buckets) - tf.reduce_min(self.buckets) > 0.0,
                    lambda: self.buckets,
                    lambda: buckets,
                    name="buckets_no_zero",
                )
                new_bucket_counts = tf.cast(
                    histogram_masked(
                        values,
                        mask=sample_weight,
                        buckets=cur_buckets if not int_metric else buckets_for_hist,
                        saturate=self.saturate,
                        name="histogram",
                    )[:, 2],
                    dtype=tf.float32,
                    name="histogram_cast",
                )
                if int_metric:
                    new_bucket_counts = tf.repeat(new_bucket_counts, repeats=repeats, name="repeated")
                bucket_counts = tf.math.add(
                    new_bucket_counts,
                    self.bucket_counts,
                    name="new_bucket_counts",
                )
                with tf.summary.experimental.summary_scope(
                    self.name, "AggregatedHistogramOp", [bucket_counts, cur_buckets, repeats]
                ) as (tag, _):

                    def calculate_tensor():
                        normalized = tf.math.divide_no_nan(
                            tf.cast(bucket_counts, dtype=tf.float32),
                            tf.cast(count, tf.float32),
                            name="normalized_bucket_counts",
                        )
                        return tf.concat(
                            [tf.transpose(cur_buckets), tf.expand_dims(normalized, 1)],
                            axis=1,
                            name="histogram_data",
                        )

                    tf.summary.write(
                        tag=tag,
                        tensor=calculate_tensor,
                        step=None,
                        metadata=create_summary_metadata(self.name, None),
                    )
                if isinstance(self.buckets, tf.Variable):
                    self.bucket_counts.assign(
                        tf.cond(
                            tf.summary.should_record_summaries(),
                            lambda: tf.zeros_like(bucket_counts),
                            lambda: bucket_counts,
                            name="maybe_clear_bucket_counts",
                        ),
                        name="assign_clear_bucket_counts",
                    )
                    self.buckets.assign(
                        tf.cond(
                            tf.summary.should_record_summaries(),
                            lambda: buckets,
                            lambda: cur_buckets,
                            name="maybe_clear_buckets",
                        ),
                        name="assign_clear_buckets",
                    )

    def result(self):
        return tf.math.divide_no_nan(tf.cast(self.sum, tf.float32), tf.cast(self.count, tf.float32))

    def reset_state(self):
        self.sum.assign(tf.zeros_like(self.sum))
        self.count.assign(tf.zeros_like(self.count))
        if not self.is_scalar and should_log_histograms():
            self.bucket_counts.assign(tf.zeros_like(self.bucket_counts))
