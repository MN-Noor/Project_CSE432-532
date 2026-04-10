import numpy as np


def accuracy(y_true, y_pred):
    """
    Fraction of correct predictions.
    accuracy = correct / total
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)
def confusion_matrix(y_true, y_pred):
    """
    Square matrix of shape (n_classes, n_classes).
    Row = actual class, Column = predicted class.
    cm[i][j] = number of samples of class i predicted as class j.
    Diagonal = correct predictions.
    Off-diagonal = misclassifications.
    """
    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n       = len(classes)

    # map class labels to 0-indexed positions
    cls_map = {cls: i for i, cls in enumerate(classes)}

    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[cls_map[true], cls_map[pred]] += 1

    return cm, classes


def precision(y_true, y_pred, average='macro'):
    """
    Precision = TP / (TP + FP)
    How many of the predicted positives are actually positive.

    average:
        'macro'    — mean precision across classes (equal weight per class)
        'weighted' — mean weighted by class support (n samples per class)
        'per_class'— array of per-class precision values
    """
    cm, classes = confusion_matrix(y_true, y_pred)
    n_classes   = len(classes)

    per_class = np.zeros(n_classes)
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp          # predicted as i but not actually i
        per_class[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    if average == 'per_class':
        return per_class, classes
    elif average == 'macro':
        return np.mean(per_class)
    elif average == 'weighted':
        support = cm.sum(axis=1)           # actual count per class
        return np.average(per_class, weights=support)
    else:
        raise ValueError(f"average must be 'macro', 'weighted', or 'per_class'")


