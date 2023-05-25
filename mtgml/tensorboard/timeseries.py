import logging

import tensorflow as tf
from tensorboard.plugins.histogram.metadata import create_summary_metadata


def log_timeseries(name, values_array, step=None, description=None, display_name=None, start_index=0):
    with tf.summary.experimental.summary_scope(name, "TimeSeriesSummaryOp", [values_array, step]) as (tag, _):

        def calculate_tensor():
            flat_values_array = tf.cast(tf.reshape(values_array, (-1,)), dtype=tf.float32)
            return tf.stack(
                [
                    tf.range(start_index, 30 + start_index, dtype=tf.float32) - 0.5,
                    tf.range(start_index, 30 + start_index, dtype=tf.float32) + 0.5,
                    tf.reduce_mean(tf.reshape(tf.repeat(flat_values_array, repeats=30), (30, -1)), axis=1),
                ],
                axis=1,
            )

        return tf.summary.write(
            tag=tag,
            tensor=calculate_tensor,
            step=step,
            metadata=create_summary_metadata(display_name, description),
        )


def log_integer_histogram(
    name: str,
    values_array: tf.Tensor,
    step: tf.Tensor | None = None,
    description: str | None = None,
    display_name: str | None = None,
    start_index: int = 0,
    max_index: int | None = None,
    bucket_size: int = 1,
    saturate: bool = False,
):
    with tf.summary.experimental.summary_scope(name, "IntegerHistogramOp", [values_array, step]) as (tag, _):

        def calculate_tensor():
            int_values_array = tf.cast(
                tf.reshape(
                    values_array,
                    (-1,),
                ),
                dtype=tf.int32,
            )
            real_max_index = (max_index + bucket_size) if max_index is not None else (30 * bucket_size + start_index)
            repeats = 30 // ((real_max_index - start_index) // bucket_size)
            assert repeats > 0, "You can't have more than 30 buckets."
            if saturate:
                match_ints = tf.range(start_index + bucket_size, real_max_index, bucket_size, dtype=tf.int32)
                left_match = (int_values_array < start_index + bucket_size)[None]
                right_match = (int_values_array >= real_max_index - bucket_size)[None]
                middle_match = (match_ints[:-1, None] <= int_values_array) & (int_values_array < match_ints[1:, None])
                matching = tf.concat([left_match, middle_match, right_match], axis=0, name="matching")
            else:
                match_ints = tf.range(start_index, real_max_index + bucket_size, bucket_size, dtype=tf.int32)
                matching = (match_ints[:-1, None] <= int_values_array) & (int_values_array < match_ints[1:, None])
            counts = tf.math.reduce_sum(tf.cast(matching, dtype=tf.float32), axis=1)
            if counts.shape[0] * repeats != 30:
                logging.warn(
                    f"Will have artifacts in histogram for {name} since it doesn't fit perfectly into 30 bins."
                )
            step_size = bucket_size / repeats
            labels = tf.range(start_index, start_index + 30 * step_size, step_size, dtype=tf.float32)
            return tf.stack(
                [labels - step_size / 2, labels + step_size / 2, tf.repeat(counts, repeats)],
                axis=1,
            )

        return tf.summary.write(
            tag=tag,
            tensor=calculate_tensor,
            step=step,
            metadata=create_summary_metadata(display_name, description),
        )
