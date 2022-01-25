import math

import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer


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

def process_q_chunk(query, key_chunks, value_chunks, kv_partitions):
    with tf.name_scope('ProccessQChunk'):
        chunk_results = tf.map_fn(lambda kv: process_kv_chunk(query, kv[0], kv[1]),
                                  (key_chunks, value_chunks), swap_memory=True, parallel_iterations=kv_partitions,
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
        chunk_results = tf.map_fn(lambda q: process_q_chunk(q, key_chunks, value_chunks, len(kv_partitions)),
                                  query_chunks, swap_memory=True, parallel_iterations=len(q_partitions))
        chunk_results = tf.stack(chunk_results, axis=0, name='chunk_results_stacked')
        chunk_results = tf.reshape(chunk_results, (-1, *query.shape[1:-1], value.shape[-1]), name=scope)
        return chunk_results


def find_chunk_size(n):
    if n <= 32: return n
    p = 2 * int(math.sqrt(n))
    for i in range(p, 0, -1):
        if n % i == 0:
            if i < 64 and n // i > i: return n // i
            else: return i


class MultiHeadAttention(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        return {
            'num_heads': hyper_config.get_int('num_heads', min=1, max=64, default=8,
                                              help='The number of separate heads of attention to use.'),
            'key_dims': hyper_config.get_int('key_dims', min=1, max=64, default=8,
                                             help='Size of the attention head for query and key.'),
            'value_dims': hyper_config.get_int('value_dims', min=1, max=64, default=16,
                                               help='Size of the attention head for value.'),
            'output_dims': hyper_config.get_int('output_dims', min=8, max=512, default=64,
                                                help='The number of output dimensions from this layer.'),
            'query_sequence_length': input_shapes[0][-2] if input_shapes else None,
            'key_sequence_length': input_shapes[1][-2] if input_shapes else None,
        }

    def build(self, input_shapes):
        super(MultiHeadAttention, self).build(input_shapes)
        self.query_matrix = self.add_weight('query_matrix', shape=(input_shapes[0][-1], self.num_heads * self.key_dims),
                                            trainable=True)
        self.key_matrix = self.add_weight('key_matrix', shape=(input_shapes[1][-1], self.num_heads * self.key_dims),
                                          trainable=True)
        self.value_matrix = self.add_weight('value_matrix', shape=(input_shapes[2][-1], self.num_heads * self.value_dims),
                                            trainable=True)
        self.output_matrix = self.add_weight('output_matrix', shape=(self.num_heads * self.value_dims, self.output_dims), trainable=True)
        self.query_chunk_size = find_chunk_size(self.query_sequence_length)
        self.kv_chunk_size = self.key_sequence_length # find_chunk_size(self.key_sequence_length)

    def call(self, inputs, training=False):
        query, key, value = inputs
        query = tf.reshape(query @ self.query_matrix, (-1, self.query_sequence_length, self.num_heads, self.key_dims)) / tf.constant(self.key_dims, dtype=self.compute_dtype)
        key = tf.reshape(key @ self.key_matrix, (-1, self.key_sequence_length, self.num_heads, self.key_dims))
        value = tf.reshape(value @ self.value_matrix, (-1, self.key_sequence_length, self.num_heads, self.value_dims))
        attended = calculate_attention(query, key, value, self.query_chunk_size, self.kv_chunk_size, name='DistributedAttention')
        return tf.reshape(attended, (-1, self.key_sequence_length, self.value_dims * self.num_heads)) @ self.output_matrix
