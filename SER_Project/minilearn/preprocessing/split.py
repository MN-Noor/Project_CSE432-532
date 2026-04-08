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