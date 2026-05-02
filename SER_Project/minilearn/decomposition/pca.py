# minilearn/decomposition/pca.py
import numpy as np


class PCA:
    """
    Principal Component Analysis.

    Finds the directions of maximum variance in the data
    and projects onto a lower-dimensional subspace.

    Algorithm:
    1. Centre the data (subtract mean)
    2. Compute covariance matrix
    3. Eigendecomposition → eigenvectors = principal components
    4. Sort by eigenvalue (variance explained) descending
    5. Project data onto top n_components

    Parameters
    ----------
    n_components : int or None
        Number of components to keep.
        None = keep all.
    """

    def __init__(self, n_components=None):
        self.n_components       = n_components
        self.components_        = None   # shape (n_components, n_features)
        self.explained_variance_         = None
        self.explained_variance_ratio_   = None
        self.mean_              = None

    def fit(self, X):
        X    = np.array(X, dtype=float)
        n_samples, n_features = X.shape

        # centre the data
        self.mean_ = np.mean(X, axis=0)
        X_centred  = X - self.mean_

        # covariance matrix shape (n_features, n_features)
        cov = (X_centred.T @ X_centred) / (n_samples - 1)

        # eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # sort descending by eigenvalue
        order        = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]   # columns are eigenvectors

        # keep top n_components
        n = self.n_components or n_features
        self.components_      = eigenvectors[:, :n].T  # shape (n, n_features)
        self.explained_variance_       = eigenvalues[:n]
        self.explained_variance_ratio_ = (eigenvalues[:n]
                                           / eigenvalues.sum())
        return self

    def transform(self, X):
        """Project X onto principal components."""
        self._check_fitted()
        X_centred = np.array(X, dtype=float) - self.mean_
        return X_centred @ self.components_.T

    def fit_transform(self, X):
        """Fit and project in one step. Only use on training data."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed):
        """Project back to original feature space."""
        self._check_fitted()
        return X_transformed @ self.components_ + self.mean_

    def explained_variance_cumsum(self):
        """Cumulative explained variance — use to pick n_components."""
        return np.cumsum(self.explained_variance_ratio_)

    def _check_fitted(self):
        if self.components_ is None:
            raise RuntimeError("Call fit() before transform().")