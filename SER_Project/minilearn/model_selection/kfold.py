import numpy as np


class KFold:
    """
    K-Fold cross-validation.
    Splits data into k consecutive folds.
    Each fold is used once as validation, k-1 folds form the training set.

    Parameters
    ----------
    n_splits   : int  — number of folds (default 5)
    shuffle    : bool — shuffle before splitting (default True)
    random_state: int — seed for reproducibility
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2. Got {n_splits}")
        self.n_splits     = n_splits
        self.shuffle      = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        """
        (train_indices, val_indices) pairs for each fold.

        Usage:
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
        """
        X         = np.array(X)
        n_samples = len(X)
        indices   = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        # split indices into n_splits roughly equal chunks
        fold_sizes = np.full(self.n_splits,
                             n_samples // self.n_splits, dtype=int)
        # distribute remainder across first folds
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        folds   = []
        for size in fold_sizes:
            folds.append(indices[current: current + size])
            current += size

        for i in range(self.n_splits):
            val_idx   = folds[i]
            train_idx = np.concatenate(
                [folds[j] for j in range(self.n_splits) if j != i]
            )
            yield train_idx, val_idx


class StratifiedKFold:
    """
    Stratified K-Fold — preserves class proportions in every fold.
    Parameters
    ----------
    n_splits    : int  — number of folds (default 5)
    shuffle     : bool — shuffle within each class before splitting
    random_state: int  — seed
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2. Got {n_splits}")
        self.n_splits     = n_splits
        self.shuffle      = shuffle
        self.random_state = random_state

    def split(self, X, y):
        """
        (train_indices, val_indices) preserving class ratios.
        """
        X       = np.array(X)
        y       = np.array(y)
        classes = np.unique(y)
        rng     = np.random.default_rng(self.random_state)

        # collect per-class fold assignments
        class_folds = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            if self.shuffle:
                rng.shuffle(cls_idx)

            # split this class's indices into n_splits chunks
            cls_folds = np.array_split(cls_idx, self.n_splits)
            class_folds.append(cls_folds)

        # yield one fold at a time
        for fold_i in range(self.n_splits):
            val_idx = np.concatenate(
                [class_folds[c][fold_i] for c in range(len(classes))]
            )
            train_idx = np.concatenate([
                class_folds[c][j]
                for c in range(len(classes))
                for j in range(self.n_splits)
                if j != fold_i
            ])
            yield train_idx, val_idx


def cross_val_score(model, X, y, cv=5, metric='accuracy'):
    """
    Parameters
    ----------
    model  : any MiniLearn model with .fit() and .predict()
    X      : array-like, shape (n_samples, n_features)
    y      : array-like, shape (n_samples,)
    cv     : int or KFold/StratifiedKFold instance
    metric : 'accuracy' | 'f1' | 'precision' | 'recall'

    Returns
    -------
    np.ndarray of scores, one per fold
    """
    from minilearn.metrics import (
        accuracy, f1_score, precision, recall
    )

    X = np.array(X)
    y = np.array(y)

    # accept integer cv or a splitter object
    if isinstance(cv, int):
        splitter = StratifiedKFold(n_splits=cv, shuffle=True,
                                   random_state=42)
    else:
        splitter = cv

    metric_fn = {
        'accuracy' : accuracy,
        'f1'       : f1_score,
        'precision': precision,
        'recall'   : recall,
    }.get(metric)

    if metric_fn is None:
        raise ValueError(f"metric must be one of: accuracy, f1, precision, recall")

    scores = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score  = metric_fn(y_val, y_pred)
        scores.append(score)
        print(f"  Fold {fold+1}/{splitter.n_splits}  {metric}={score:.4f}")

    scores = np.array(scores)
    print(f"\n  Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")
    return scores