import tensorflow as tf
from tensorboard.plugins.histogram.metadata import create_summary_metadata

def log_timeseries(name, values_array, step=None, description=None, display_name=None, start_index=0):
    with tf.summary.experimental.summary_scope(name, "TimeSeriesSummaryOp", [values_array, step]) as (tag, _):
        values_array = tf.cast(tf.reshape(values_array, (-1,)), dtype=tf.float32)
        num_values = values_array.shape[0]
        return tf.summary.write(
            tag=tag,
            tensor = tf.stack(
                [
                    tf.range(start_index, 30 + start_index, dtype=tf.float32) - 0.5,
                    tf.range(start_index, 30 + start_index, dtype=tf.float32) + 0.5,
                    tf.reduce_mean(tf.reshape(tf.repeat(values_array, repeats=30), (30, num_values)), axis=1),
                ],
                axis=1
            ),
            step=step,
            metadata=create_summary_metadata(display_name, description)
        )
