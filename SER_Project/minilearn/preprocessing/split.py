import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays into random train and test subsets.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    test_size : float, proportion of data for test set (default 0.2 = 20%)
    random_state : int or None, seed for reproducibility
    
    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = np.array(X)
    y = np.array(y)

    if len(X) != len(y):
        raise ValueError(f"X and y must have the same number of samples. "
                         f"Got X: {len(X)}, y: {len(y)}")

    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1. Got {test_size}")

    rng = np.random.default_rng(random_state)

    n_samples = len(X)
    n_test = int(np.ceil(n_samples * test_size))
    n_train = n_samples - n_test

    # Shuffle indices
    indices = rng.permutation(n_samples)

    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def stratified_train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Stratified split — preserves class proportions in both train and test.
    Essential for imbalanced datasets like RAVDESS (8 emotion classes).
    If neutral only has 96 samples and you do a random split,
    you might get only 5 neutral samples in test — too few to evaluate.
    Stratified guarantees ~20% of EACH class goes to test.

    Parameters
    ----------
    X            : array-like, shape (n_samples, n_features)
    y            : array-like, shape (n_samples,)
    test_size    : float, proportion for test set (default 0.2)
    random_state : int or None

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = np.array(X)
    y = np.array(y)

    if len(X) != len(y):
        raise ValueError(
            f"X and y must have same number of samples. "
            f"Got X: {len(X)}, y: {len(y)}"
        )

    if not (0.0 < test_size < 1.0):
        raise ValueError(
            f"test_size must be between 0 and 1. Got {test_size}"
        )

    rng = np.random.default_rng(random_state)

    train_indices = []
    test_indices  = []

    # process each class separately
    for cls in np.unique(y):
        # get all indices where this class appears
        cls_indices = np.where(y == cls)[0]

        # shuffle them
        rng.shuffle(cls_indices)

        # how many go to test for this class
        n_test = max(1, int(np.ceil(len(cls_indices) * test_size)))

        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])

    train_indices = np.array(train_indices)
    test_indices  = np.array(test_indices)

    # shuffle final indices so classes aren't in blocks
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]