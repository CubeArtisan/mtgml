import tensorflow as tf


class LinearWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, warmed_up_lr, starting_lr = 0.0, name='LinearWarmup'):
        self.warmup_steps = warmup_steps
        self.delegate_once_warm = isinstance(warmed_up_lr, tf.keras.optimizers.schedules.LearningRateSchedule)
        self.warmed_up_lr = warmed_up_lr
        self.starting_lr = starting_lr
        self.name = name

    def __call__(self, step):

        def do_warmup():
            if self.delegate_once_warm:
                target_lr = self.warmed_up_lr(tf.constant(0, dtype=step.dtype))
            else:
                target_lr = tf.constant(self.warmed_up_lr, dtype=tf.float32, name='target_lr')
            warmup_steps = tf.constant(self.warmup_steps, dtype=tf.float32, name='warmup_steps')
            starting_lr = tf.constant(self.starting_lr, dtype=tf.float32, name='starting_lr')
            return starting_lr + (tf.cast(step, dtype=tf.float32, name='step') * (target_lr - starting_lr)
                                  / warmup_steps)

        def do_post_warmup():
            if self.delegate_once_warm:
                return self.warmed_up_lr(step - tf.constant(self.warmup_steps, dtype=step.dtype, name='warmup_steps'))
            else:
                return tf.constant(self.warmed_up_lr, dtype=tf.float32, name='warmed_up_lr')

        with tf.name_scope(self.name) as scope:
            return tf.cond(step < tf.constant(self.warmup_steps, dtype=step.dtype, name='warmup_steps'),
                           do_warmup,
                           do_post_warmup,
                           name=scope)

    def get_config(self):
        return {
            "warmup_steps": self.warmup_steps,
            "warmed_up_lr": self.warmed_up_lr,
            "starting_lr": self.starting_lr,
            "name": self.name,
        }
