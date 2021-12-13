import tensorflow as tf

from mtgml.layers.configurable_layer import ConfigurableLayer

class ItemRating(ConfigurableLayer):
    @classmethod
    def get_properties(cls, hyper_config, input_shapes=None):
        num_items = hyper_config.get_int('num_items', min=1, max=2**31 - 1, default=None,
                                         help='The number of items that should be given ratings')
        if not num_items and input_shapes:
            raise NotImplementedError('You must supply the number of items.')
        return {
            'num_items': num_items,
            'bounded': hyper_config.get_bool('bounded', default=False, help='Whether to bound the ratings to (0, 1).'),
        }

    def build(self, input_shape):
        super(ItemRating, self).build(input_shape)
        self.item_rating_logits = self.add_weight('item_rating_logits', shape=(self.num_items,),
                                                  initializer=tf.random_uniform_initializer(-0.05, 0.05,
                                                                                           seed=self.seed),
                                                  trainable=True)

    def call(self, inputs, training=False):
        zero_rating = tf.constant(0, shape=(1,), dtype=self.compute_dtype)
        if self.bounded:
            item_ratings = tf.concat([zero_rating, tf.nn.sigmoid(32 * self.item_rating_logits, name='item_ratings')],
                                     0, name='item_ratings')
        else:
            item_ratings = tf.concat([zero_rating, tf.nn.softplus(32 * self.item_rating_logits, name='item_ratings')],
                                     0, name='item_ratings')
        ratings = tf.gather(item_ratings, inputs, name='ratings')
        # Logging for Tensorboard
        tf.summary.histogram('weights/item_ratings', item_ratings)
        return ratings
