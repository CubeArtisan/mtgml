import tensorflow as tf


class RsqrtSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, multiplier=1.0, name='Rsqrt'):
        self.initial_lr = initial_lr
        self.delegate_initial = isinstance(initial_lr, tf.keras.optimizers.schedules.LearningRateSchedule)
        self.multiplier = multiplier
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name) as scope:
            desired = tf.math.rsqrt(tf.cast(step + 1, dtype=tf.float32)) * tf.constant(self.multiplier, dtype=tf.float32,
                                                                                       name='multiplier')
            if self.delegate_initial:
                initial = self.initial_lr(step)
            else:
                initial = tf.constant(self.initial_lr, dtype=tf.float32, name='initial_lr')
            return tf.minimum(desired, initial, name=scope)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "multiplier": self.multiplier,
            "name": self.name,
        }
