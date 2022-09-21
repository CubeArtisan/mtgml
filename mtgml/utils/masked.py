import tensorflow as tf

from mtgml.constants import EPSILON


def reduce_mean_masked(x, mask=None, axis=-1, keepdims=False, name=None, count=None, return_count=False):
    if name is None:
        name = "ReduceMeanMasked"
    if mask is None:
        mask = x._keras_mask
    with tf.name_scope(name) as scope:
        x = tf.convert_to_tensor(x, name='x')
        mask = tf.cast(mask, dtype=x.dtype, name='multiplicative_mask')
        sum = tf.reduce_sum(tf.math.multiply(x, mask, name='masked_x'), axis=axis, keepdims=keepdims, name='masked_sum')
        if count is None:
            count = tf.reduce_sum(mask, axis=axis, keepdims=keepdims, name='masked_count')
        else:
            tf.convert_to_tensor(count, name='masked_count')
        result = tf.math.divide_no_nan(sum, count, name=scope)
        if return_count:
            return result, count
        else:
            return result


def reduce_variance_masked(x, mask=None, axis=-1, keepdims=False, count=None, return_count=False, name=None) \
        -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
    if name is None:
        name = "ReduceVarianceMasked"
    if mask is None:
        mask = x._keras_mask
    with tf.name_scope(name) as scope:
        x = tf.convert_to_tensor(x, name='x')
        mask = tf.cast(mask, dtype=x.dtype, name='multiplicative_mask')
        mean, count = reduce_mean_masked(x, mask, axis=axis, keepdims=True, count=count, return_count=True,
                                         name='masked_mean')
        masked_squares = tf.math.multiply(tf.math.squared_difference(x, mean, name='squared_diff_from_mean'), mask,
                                          name='squared_diff_from_mean_masked')
        sum = tf.math.reduce_sum(masked_squares, axis=axis, keepdims=keepdims, name='masked_sum_squares')
        if not keepdims:
            count = tf.squeeze(count, axis=axis, name='squeezed_count')
        adjusted_count = tf.math.subtract(count, tf.constant(1, dtype=count.dtype), name='adjusted_count')
        result = tf.math.divide_no_nan(sum, adjusted_count, name=scope)
        if return_count:
            return result, count
        else:
            return result


def reduce_stddev_masked(x, mask=None, axis=-1, keepdims=False, count=None, return_count=False, name=None):
    if name is None:
        name = "ReduceStdDevMasked"
    with tf.name_scope(name) as scope:
        variance = reduce_variance_masked(x, mask, axis, keepdims, count, return_count, name='reduce_variance_masked')
        if isinstance(variance, tuple):
            variance, count = variance
        result = tf.math.sqrt(tf.math.add(variance, tf.constant(EPSILON, dtype=variance.dtype), name='add_epsilon'),
                              name=scope)
        if return_count:
            return result, count
        else:
            return result
