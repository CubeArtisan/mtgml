from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast, overload

import tensorflow as tf

from mtgml.constants import EPSILON, LARGE_INT


def get_mask_for(x, mask=None) -> tf.Tensor:
    if mask is None:
        mask = getattr(x, "_keras_mask", None)
        if mask is None:
            mask = tf.ones_like(x, name="default_mask")
    else:
        mask = tf.convert_to_tensor(mask, name="converted_mask")
    return tf.broadcast_to(
        tf.cast(mask, dtype=x.dtype, name="cast_mask"),
        tf.broadcast_dynamic_shape(tf.shape(mask), tf.shape(x)),
        name="mask",
    )


def reduce_sum_masked(
    x,
    mask=None,
    axis: int | Sequence[int] | None = -1,
    keepdims: bool = False,
    name: str | None = None,
) -> tf.Tensor:
    with tf.name_scope(name or "ReduceSumMasked") as scope:
        x = tf.convert_to_tensor(x, name="x")
        mask = get_mask_for(x, mask)
    return tf.math.reduce_sum(tf.math.multiply(x, mask, name="masked_x"), axis=axis, keepdims=keepdims, name=scope)


@overload
def reduce_mean_masked(
    x,
    mask=None,
    axis: int | Sequence[int] = -1,
    keepdims=False,
    count: tf.Tensor | None = None,
    return_count: Literal[False] = False,
    name: str | None = None,
) -> tf.Tensor:
    ...


@overload
def reduce_mean_masked(
    x,
    mask=None,
    axis: int | Sequence[int] = -1,
    keepdims=False,
    count: tf.Tensor | None = None,
    return_count: Literal[True] = True,
    name: str | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    ...


@tf.function
def reduce_mean_masked(
    x,
    mask=None,
    axis: int | Sequence[int] = -1,
    keepdims=False,
    count: tf.Tensor | None = None,
    return_count: bool = False,
    name=None,
) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
    with tf.name_scope(name or "ReduceMeanMasked") as scope:
        x = tf.convert_to_tensor(x, name="x")
        mask = get_mask_for(x, mask)
        sum = tf.reduce_sum(tf.math.multiply(x, mask, name="masked_x"), axis=axis, keepdims=keepdims, name="masked_sum")
        if count is None:
            count = tf.reduce_sum(mask, axis=axis, keepdims=keepdims, name="masked_count")
        else:
            count = tf.convert_to_tensor(count, name="masked_count")
        if TYPE_CHECKING:
            assert count is not None
        result = tf.math.divide_no_nan(sum, count, name=scope)
        if return_count:
            if TYPE_CHECKING:
                result = cast(tf.Tensor, result)
            return result, count
        else:
            return result


@tf.function
def reduce_max_masked(
    x,
    mask=None,
    axis: int | Sequence[int] = -1,
    keepdims: bool = False,
    name: str | None = None,
) -> tf.Tensor:
    with tf.name_scope(name or "ReduceMaxMasked") as scope:
        x = tf.convert_to_tensor(x, name="x")
        mask = get_mask_for(x, mask)
        additive_mask = (tf.constant(1, dtype=x.dtype) - mask) * tf.constant(LARGE_INT, dtype=x.dtype)
        return tf.math.multiply(
            tf.math.reduce_max(x - additive_mask, axis=axis, keepdims=keepdims, name="unmasked"),
            tf.cast(tf.math.reduce_any(tf.cast(mask, tf.bool), axis=axis, keepdims=keepdims), x.dtype),
            name=scope,
        )


@tf.function
def reduce_min_masked(
    x,
    mask=None,
    axis: int | Sequence[int] = -1,
    keepdims: bool = False,
    name: str | None = None,
) -> tf.Tensor:
    with tf.name_scope(name or "ReduceMinMasked") as scope:
        x = tf.convert_to_tensor(x, name="x")
        mask = get_mask_for(x, mask)
        additive_mask = (tf.constant(1, dtype=x.dtype) - mask) * tf.constant(LARGE_INT, dtype=x.dtype)
        return tf.math.multiply(
            tf.math.reduce_min(x + additive_mask, axis=axis, keepdims=keepdims, name="unmasked"),
            tf.cast(tf.math.reduce_any(tf.cast(mask, tf.bool), axis=axis, keepdims=keepdims), x.dtype),
            name=scope,
        )


@overload
def reduce_variance_masked(
    x,
    mask=None,
    axis: int | Sequence[int] = -1,
    keepdims=False,
    count: tf.Tensor | None = None,
    return_count: Literal[False] = False,
    name: str | None = None,
) -> tf.Tensor:
    ...


@overload
def reduce_variance_masked(
    x,
    mask=None,
    axis: int | Sequence[int] = -1,
    keepdims=False,
    count: tf.Tensor | None = None,
    return_count: Literal[True] = True,
    name: str | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    ...


@tf.function
def reduce_variance_masked(
    x,
    mask=None,
    axis: int | Sequence[int] = -1,
    keepdims=False,
    count: tf.Tensor | None = None,
    return_count: bool = False,
    name=None,
) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
    with tf.name_scope(name or "ReduceVarianceMasked") as scope:
        x = tf.convert_to_tensor(x, name="x")
        mask = get_mask_for(x, mask)
        mean, count = reduce_mean_masked(
            x, mask, axis=axis, keepdims=True, count=count, return_count=True, name="masked_mean"
        )
        masked_squares = tf.math.multiply(
            tf.math.squared_difference(x, mean, name="squared_diff_from_mean"),
            mask,
            name="squared_diff_from_mean_masked",
        )
        sum = tf.math.reduce_sum(masked_squares, axis=axis, keepdims=keepdims, name="masked_sum_squares")
        if TYPE_CHECKING:
            assert count is not None
        if not keepdims:
            count = tf.squeeze(count, axis)
        adjusted_count = tf.math.subtract(count, tf.constant(1, dtype=count.dtype), name="adjusted_count")
        result = tf.math.divide_no_nan(sum, adjusted_count, name=scope)
        if return_count:
            if TYPE_CHECKING:
                result = cast(tf.Tensor, result)
                count = cast(tf.Tensor, count)
            return result, count
        else:
            return result


@overload
def reduce_stddev_masked(
    x,
    mask=None,
    axis: int | Sequence[int] = -1,
    keepdims=False,
    count=None,
    return_count: Literal[False] = False,
    name: str | None = None,
) -> tf.Tensor:
    ...


@overload
def reduce_stddev_masked(
    x,
    mask=None,
    axis: int | Sequence[int] = -1,
    keepdims=False,
    count=None,
    return_count: Literal[True] = True,
    name: str | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    ...


def reduce_stddev_masked(
    x,
    mask=None,
    axis: int | Sequence[int] = -1,
    keepdims=False,
    count: tf.Tensor | None = None,
    return_count: bool = False,
    name: str | None = None,
) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
    if name is None:
        name = "ReduceStdDevMasked"
    with tf.name_scope(name) as scope:
        if return_count:
            variance, count = reduce_variance_masked(
                x, mask, axis, keepdims, count, True, name="reduce_variance_masked"
            )
        else:
            variance = reduce_variance_masked(x, mask, axis, keepdims, count, False, name="reduce_variance_masked")
        result = tf.math.sqrt(
            tf.math.add(variance, tf.constant(EPSILON, dtype=variance.dtype), name="add_epsilon"), name=scope
        )
        if return_count:
            if TYPE_CHECKING:
                result = cast(tf.Tensor, result)
                assert count is not None
            return result, count
        else:
            return result


def make_buckets(start_range, end_range=None, num_buckets=None, bucket_size=None, name: str | None = None):
    with tf.name_scope(name or "MakeBuckets") as scope:
        start_range = tf.convert_to_tensor(start_range, dtype=tf.float32)
        end_range = tf.convert_to_tensor(end_range, dtype=tf.float32)
        if num_buckets is None:
            if end_range is not None and bucket_size is not None:
                num_buckets = tf.math.ceil((end_range - start_range) / bucket_size, name="num_buckets")
            else:
                num_buckets = 30
        if end_range is None:
            if bucket_size is None:
                raise ValueError("One of end_range or bucket_size must be provided.")
            end_range = bucket_size * num_buckets + start_range
        edges = tf.concat(
            [tf.linspace(start_range, end_range, num_buckets + 1, name="linspace_edges")[:-1], [end_range]],
            axis=0,
            name="edges",
        )
        return tf.stack([edges[:-1], edges[1:]], axis=0, name=scope)


def histogram_masked(
    x,
    mask=None,
    start_range=None,
    end_range=None,
    num_buckets=None,
    bucket_size=None,
    buckets=None,
    saturate: bool = False,
    name: str | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    with tf.name_scope(name or "HistogramMasked") as scope:
        x = tf.convert_to_tensor(x, name="x")
        mask = get_mask_for(x, mask)
        if buckets is None:
            buckets = make_buckets(
                start_range, end_range=end_range, num_buckets=num_buckets, bucket_size=bucket_size, name="buckets"
            )
        buckets = tf.cast(buckets, dtype=x.dtype, name="cast_buckets")
        original_buckets = buckets
        for i in range(1, len(x.shape) + 3 - len(buckets.shape)):
            buckets = tf.expand_dims(buckets, -1, name=f"expand_bucket_{i}")
        expanded_x = tf.expand_dims(x, 0, name="expanded_x")
        expanded_mask = tf.expand_dims(mask, 0, name="expanded_mask")
        if not saturate:
            counts_initial = reduce_sum_masked(
                tf.cast((expanded_x >= buckets[0, :-1]) & (expanded_x < buckets[1, :-1]), dtype=x.dtype),
                mask=expanded_mask,
                axis=list(range(1, len(expanded_x.shape))),
                name="counts_initial",
            )
            counts_last = reduce_sum_masked(
                # Left and right edges of final interval
                tf.cast((x >= buckets[0, -1]) & (x <= buckets[1, -1]), dtype=x.dtype),
                mask=mask,
                axis=None,
                name="counts_last_interval",
            )
            counts = tf.concat([counts_initial, [counts_last]], axis=0, name="counts")
        else:
            counts_first = reduce_sum_masked(
                # Right edge (1) of first interval (0)
                tf.cast((x < buckets[1, 0]), dtype=x.dtype),
                mask=mask,
                axis=None,
                name="count_first_interval",
            )
            counts_last = reduce_sum_masked(
                # Left edge (0) of last interval (-1)
                tf.cast((x >= buckets[0, -1]), dtype=x.dtype),
                mask=mask,
                axis=None,
                name="count_last_interval",
            )
            counts_mid = reduce_sum_masked(
                tf.cast((expanded_x >= buckets[0, 1:-1]) & (expanded_x < buckets[1, 1:-1]), dtype=x.dtype),
                mask=expanded_mask,
                axis=list(range(1, len(expanded_x.shape))),
                name="count_middle_intervals",
            )
            counts = tf.concat([[counts_first], counts_mid, [counts_last]], axis=0, name="counts")
        return tf.transpose(
            tf.concat([original_buckets, tf.expand_dims(counts, 0, name="expanded_counts")], axis=0), name=scope
        )
