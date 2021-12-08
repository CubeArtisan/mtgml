import itertools

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

@tf.numpy_function
def plot_attention_scores(scores, multihead=False, name=''):
    if multihead:
        for i, subscores in enumerate(scores.split(scores, axis=-3)):
            plot_attention_scores(subscores, multihead=False, name=f'{name} {i}')
        return
    scores = scores.numpy()
    scores = scores.reshape((-1, *scores.shape[-2:]))
    scores = np.mean(scores, axis=0)
    figure = plt.figure(figsize=scores.shape)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(name)
    plt.colorbar()
    tick_marks = np.arrange(scores.shape[0])
    plt.xticks(tick_marks, [str(x) for x in tick_marks], rotation=45)
    plt.yticks(tick_marks, [str(x) for x in tick_marks])
    # Compute the labels from the normalized confusion matrix.
    labels = np.around(scores, decimals=2)
    # Use white text if squares are dark; otherwise black.
    threshold = scores.max() / 2.
    for i, j in itertools.product(range(scores.shape[0]), range(scores.shape[1])):
        color = "white" if scores[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    return figure
