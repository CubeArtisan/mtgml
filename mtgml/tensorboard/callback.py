import tensorflow as tf


class TensorBoardFix(tf.keras.callbacks.TensorBoard):
    """
    This fixes incorrect step values when using the TensorBoard callback with custom summary ops
    """

    def on_train_begin(self, *args, **kwargs):
        super(TensorBoardFix, self).on_train_begin(*args, **kwargs)
        tf.summary.experimental.set_step(self._train_step)

    def on_test_begin(self, *args, **kwargs):
        super(TensorBoardFix, self).on_test_begin(*args, **kwargs)
        tf.summary.experimental.set_step(self._val_step)

    def _push_writer(self, writer, step):
        """Sets the default writer for custom batch-level summaries."""
        if self.update_freq == "epoch":
            return

        should_record = lambda: tf.equal(step % self.update_freq, self.update_freq - 1)
        # TODO(b/151339474): Fix deadlock when not using .value() here.
        summary_context = (
            writer.as_default(step.value()),
            tf.summary.record_if(should_record),
        )
        self._prev_summary_state.append(summary_context)
        summary_context[0].__enter__()
        summary_context[1].__enter__()


TensorBoardFix.__name__ = "TensorBoard"
