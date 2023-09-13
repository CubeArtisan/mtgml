import time

import tensorflow as tf


class TimedStopping(tf.keras.callbacks.Callback):
    """Stop training when enough time has passed.
    # Arguments
        seconds: maximum time before stopping.
        verbose: verbosity mode.
    """

    def __init__(self, seconds=None, verbose=0):
        super(tf.keras.callbacks.Callback, self).__init__()

        self.start_time = 0
        self.seconds = seconds
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        if time.time() - self.start_time > self.seconds:
            self.model.stop_training = True
            if self.verbose:
                print("Stopping after %s seconds." % self.seconds)
