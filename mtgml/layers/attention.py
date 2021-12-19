import math

import tensorflow as tf


@tf.recompute_grad
def process_kv_chunk(query, key, value):
    with tf.name_scope('ProcessKVChunk'):
        # Paper divides query by sqrt of embed dims in key here.
        scores = tf.einsum('...qhd,...khd->...qhk', query, key, name='chunked_scores')
        max_score = tf.stop_gradient(tf.reduce_max(scores, axis=-1, keepdims=True, name='max_score'))
        exp_scores = tf.exp(tf.subtract(scores, max_score, name='shifted_scores'), name='exp_scores')
        # Value dim must equal key dim.
        chunk_value = tf.einsum('...vhf,...qhv->...qhf', value, exp_scores, name='chunk_value')
        sum_exp_score = tf.reduce_sum(exp_scores, axis=-1, name='sum_exp_score')
        max_score = tf.squeeze(max_score, -1)
        return (chunk_value, sum_exp_score, max_score)


def combine_kv_chunks(chunk_values, sum_exp_scores, max_scores):
    # combined values is (num_kv_chunks, query_chunk, num_heads, value_embed_dim)
    # max_score and sum_exp_score are (num_kv_chunks, query_chunk, num_heads)
    with tf.name_scope('CombineKVChunks') as scope:
        max_score = tf.reduce_max(max_scores, axis=0, keepdims=True, name='max_score')
        scaling_factors = tf.exp(max_scores - max_score, name='scaling_factors')
        chunk_values = tf.multiply(chunk_values, tf.expand_dims(scaling_factors, -1), name='scaled_chunk_values')
        sum_exp_scores = tf.multiply(sum_exp_scores, scaling_factors, name='scaled_sum_exp_scores')
        combined_values = tf.reduce_sum(chunk_values, axis=0, name='combined_values')
        combined_exp_scores = tf.expand_dims(tf.reduce_sum(sum_exp_scores, axis=0), -1, name='combined_exp_scores')
        result = tf.divide(combined_values, combined_exp_scores, name=scope)
        return result

def process_q_chunk(query, key_chunks, value_chunks):
    with tf.name_scope('ProccessQChunk'):
        chunk_results = tf.map_fn(lambda kv: process_kv_chunk(query, kv[0], kv[1]),
                                  (key_chunks, value_chunks), swap_memory=True, parallel_iterations=16,
                                  infer_shape=False, fn_output_signature=(tf.TensorSpec((*query.shape[:-1], value_chunks.shape[-1]), dtype=query.dtype),
                                                                          tf.TensorSpec(query.shape[:-1], dtype=query.dtype),
                                                                          tf.TensorSpec(query.shape[:-1], dtype=query.dtype)))

        return combine_kv_chunks(*chunk_results)


def calculate_partitions(size, chunk_size):
    partitions = [chunk_size for _ in range(size // chunk_size)]
    extra = size % chunk_size
    if extra > 0:
        partitions.append(extra)
    return partitions

@tf.function
def calculate_attention(query, key, value, query_chunk_size, kv_chunk_size, name=None):
    with tf.name_scope(name or 'MultiHeadAttention') as scope:
        q_partitions = calculate_partitions(query.shape[-3], query_chunk_size)
        kv_partitions = calculate_partitions(key.shape[-3], kv_chunk_size)
        query_chunks = tf.stack(tf.split(query, q_partitions, axis=-3, name='query_chunks'), axis=0)
        key_chunks = tf.stack(tf.split(key, kv_partitions, axis=-3, name='key_chunks'), axis=0)
        value_chunks = tf.stack(tf.split(value, kv_partitions, axis=-3, name='value_chunks'), axis=0)
        chunk_results = tf.map_fn(lambda q: process_q_chunk(q, key_chunks, value_chunks), query_chunks, swap_memory=True, parallel_iterations=16)
        chunk_results = tf.stack(chunk_results, axis=0, name='chunk_results_stacked')
        chunk_results = tf.reshape(chunk_results, (*query.shape[:-1], value.shape[-1]), name=scope)
        return chunk_results
