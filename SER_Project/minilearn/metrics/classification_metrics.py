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


def recall(y_true, y_pred, average='macro'):
    """
    Recall = TP / (TP + FN)
    How many actual positives were correctly identified.

    average: same options as precision
    """
    cm, classes = confusion_matrix(y_true, y_pred)
    n_classes   = len(classes)

    per_class = np.zeros(n_classes)
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp          # actually i but predicted as something else
        per_class[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if average == 'per_class':
        return per_class, classes
    elif average == 'macro':
        return np.mean(per_class)
    elif average == 'weighted':
        support = cm.sum(axis=1)
        return np.average(per_class, weights=support)
    else:
        raise ValueError(f"average must be 'macro', 'weighted', or 'per_class'")


def f1_score(y_true, y_pred, average='macro'):
    """
    F1 = 2 * (precision * recall) / (precision + recall)
    Harmonic mean of precision and recall.
    Balances both — good metric when classes are imbalanced.

    average: same options as precision
    """
    cm, classes = confusion_matrix(y_true, y_pred)
    n_classes   = len(classes)

    per_class = np.zeros(n_classes)
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[i] = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    if average == 'per_class':
        return per_class, classes
    elif average == 'macro':
        return np.mean(per_class)
    elif average == 'weighted':
        support = cm.sum(axis=1)
        return np.average(per_class, weights=support)
    else:
        raise ValueError(f"average must be 'macro', 'weighted', or 'per_class'")


def classification_report(y_true, y_pred, labels=None):
    """
    Full report showing per-class and overall metrics.
    Mirrors sklearn's classification_report output.
    """
    cm, classes      = confusion_matrix(y_true, y_pred)
    prec, _          = precision(y_true, y_pred, average='per_class')
    rec,  _          = recall(y_true, y_pred,    average='per_class')
    f1,   _          = f1_score(y_true, y_pred,  average='per_class')
    support          = cm.sum(axis=1)

    # use provided labels or fall back to class values
    display_labels = labels if labels is not None else classes

    header = f"{'Class':>12}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}"
    print(header)
    print('-' * len(header))

    for i, cls in enumerate(classes):
        label = display_labels[i] if i < len(display_labels) else cls
        print(f"{str(label):>12}  {prec[i]:>10.4f}  "
              f"{rec[i]:>8.4f}  {f1[i]:>8.4f}  {support[i]:>8}")

    print('-' * len(header))
    print(f"{'macro avg':>12}  {np.mean(prec):>10.4f}  "
          f"{np.mean(rec):>8.4f}  {np.mean(f1):>8.4f}  "
          f"{support.sum():>8}")
    print(f"{'weighted avg':>12}  "
          f"{np.average(prec, weights=support):>10.4f}  "
          f"{np.average(rec,  weights=support):>8.4f}  "
          f"{np.average(f1,   weights=support):>8.4f}  "
          f"{support.sum():>8}")

    return {
        'precision_per_class' : prec,
        'recall_per_class'    : rec,
        'f1_per_class'        : f1,
        'support'             : support,
        'classes'             : classes,
        'macro_precision'     : np.mean(prec),
        'macro_recall'        : np.mean(rec),
        'macro_f1'            : np.mean(f1),
        'accuracy'            : accuracy(y_true, y_pred),
    }

def plot_confusion_matrix(y_true, y_pred, labels=None,
                           title='Confusion Matrix', ax=None):
    """
    Plot confusion matrix as a heatmap.
    Diagonal = correct, off-diagonal = misclassifications.
    """
    import matplotlib.pyplot as plt

    cm, classes    = confusion_matrix(y_true, y_pred)
    display_labels = labels if labels is not None else classes

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))

    # normalise so each row sums to 1 (shows proportions not raw counts)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(display_labels, rotation=45, ha='right')
    ax.set_yticklabels(display_labels)

    # annotate each cell with raw count and percentage
    thresh = cm_norm.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i,
                    f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                    ha='center', va='center', fontsize=8,
                    color='white' if cm_norm[i,j] > thresh else 'black')

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    return ax