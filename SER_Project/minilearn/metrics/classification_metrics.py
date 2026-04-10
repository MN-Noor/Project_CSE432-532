import numpy as np


def accuracy(y_true, y_pred):
    """
    Fraction of correct predictions.
    accuracy = correct / total
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

