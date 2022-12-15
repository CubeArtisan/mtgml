from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast, overload
import tensorflow as tf

from mtgml.constants import EPSILON


@overload
def reduce_mean_masked(x, mask=None, axis: int | Sequence[int] = -1, keepdims=False, count: tf.Tensor | None = None,
                       return_count: Literal[False] = False, name: str | None = None) -> tf.Tensor:
    ...

@overload
def reduce_mean_masked(x, mask=None, axis: int | Sequence[int] = -1, keepdims=False, count: tf.Tensor | None = None,
                       return_count: Literal[True] = True, name: str | None = None) -> tuple[tf.Tensor, tf.Tensor]:
    ...


def reduce_mean_masked(x, mask=None, axis: int | Sequence[int] = -1, keepdims=False, count: tf.Tensor | None = None,
                           return_count: bool = False, name=None) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
    if name is None:
        name = "ReduceMeanMasked"
    if mask is None:
        mask = x._keras_mask
        if mask is None:
            mask = tf.ones_like(x)
    with tf.name_scope(name) as scope:
        x = tf.convert_to_tensor(x, name='x')
        mask = tf.cast(mask, dtype=x.dtype, name='multiplicative_mask')
        sum = tf.reduce_sum(tf.math.multiply(x, mask, name='masked_x'), axis=axis, keepdims=keepdims, name='masked_sum')
        if count is None:
            count = tf.reduce_sum(mask, axis=axis, keepdims=keepdims, name='masked_count')
        else:
            count = tf.convert_to_tensor(count, name='masked_count')
        if TYPE_CHECKING:
            assert count is not None
        result = tf.math.divide_no_nan(sum, count, name=scope)
        if return_count:
            if TYPE_CHECKING:
                result = cast(tf.Tensor, result)
            return result, count
        else:
            return result

@overload
def reduce_variance_masked(x, mask=None, axis: int | Sequence[int] = -1, keepdims=False, count: tf.Tensor | None = None,
                           return_count: Literal[False] = False, name: str | None = None) -> tf.Tensor:
    ...

@overload
def reduce_variance_masked(x, mask=None, axis: int | Sequence[int] = -1, keepdims=False, count: tf.Tensor | None = None,
                           return_count: Literal[True] = True, name: str | None = None) -> tuple[tf.Tensor, tf.Tensor]:
    ...


def reduce_variance_masked(x, mask=None, axis: int | Sequence[int] = -1, keepdims=False, count: tf.Tensor | None = None,
                           return_count: bool = False, name=None) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
    if name is None:
        name = "ReduceVarianceMasked"
    if mask is None:
        mask = x._keras_mask
        if mask is None:
            mask = tf.ones_like(x)
    with tf.name_scope(name) as scope:
        x = tf.convert_to_tensor(x, name='x')
        mask = tf.cast(mask, dtype=x.dtype, name='multiplicative_mask')
        mean, count = reduce_mean_masked(x, mask, axis=axis, keepdims=True, count=count, return_count=True,
                                         name='masked_mean')
        masked_squares = tf.math.multiply(tf.math.squared_difference(x, mean, name='squared_diff_from_mean'), mask,
                                          name='squared_diff_from_mean_masked')
        sum = tf.math.reduce_sum(masked_squares, axis=axis, keepdims=keepdims, name='masked_sum_squares')
        if TYPE_CHECKING:
            assert count is not None
        if not keepdims:
            count = tf.squeeze(count, axis)
        adjusted_count = tf.math.subtract(count, tf.constant(1, dtype=count.dtype), name='adjusted_count')
        result = tf.math.divide_no_nan(sum, adjusted_count, name=scope)
        if return_count:
            if TYPE_CHECKING:
                result = cast(tf.Tensor, result)
                count = cast(tf.Tensor, count)
            return result, count
        else:
            return result


@overload
def reduce_stddev_masked(x, mask=None, axis: int | Sequence[int] = -1, keepdims=False, count=None,
                           return_count: Literal[False] = False, name: str | None = None) -> tf.Tensor:
    ...

@overload
def reduce_stddev_masked(x, mask=None, axis: int | Sequence[int] = -1, keepdims=False, count=None,
                           return_count: Literal[True] = True, name: str | None = None) -> tuple[tf.Tensor, tf.Tensor]:
    ...


def reduce_stddev_masked(x, mask=None, axis: int | Sequence[int] = -1, keepdims=False, count: tf.Tensor | None = None,
                         return_count: bool = False, name: str | None = None) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
    if name is None:
        name = "ReduceStdDevMasked"
    with tf.name_scope(name) as scope:
        if return_count:
            variance, count = reduce_variance_masked(x, mask, axis, keepdims, count, True, name='reduce_variance_masked')
        else:
            variance = reduce_variance_masked(x, mask, axis, keepdims, count, False, name='reduce_variance_masked')
        result = tf.math.sqrt(tf.math.add(variance, tf.constant(EPSILON, dtype=variance.dtype), name='add_epsilon'),
                              name=scope)
        if return_count:
            if TYPE_CHECKING:
                result = cast(tf.Tensor, result)
                assert count is not None
            return result, count
        else:
            return result
