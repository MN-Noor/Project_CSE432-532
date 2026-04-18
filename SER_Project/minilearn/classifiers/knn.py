# minilearn/classifiers/knn.py
import numpy as np
from collections import Counter


class KNearestNeighbors:
    """
    No training phase. Stores the training data, then at prediction
    time finds the K closest samples by distance and takes a majority vote.

    Parameters
    ----------
    k        : int    — number of neighbors to consider (default 5)
    metric   : str    — 'euclidean' | 'manhattan' | 'cosine' (default 'euclidean')
    weights  : str    — 'uniform' (equal vote) | 'distance' (closer = more weight)
    """

    def __init__(self, k=5, metric='euclidean', weights='uniform'):
        if k < 1:
            raise ValueError(f"k must be >= 1. Got {k}")
        if metric not in ('euclidean', 'manhattan', 'cosine'):
            raise ValueError(f"metric must be 'euclidean', 'manhattan', or 'cosine'. Got {metric}")
        if weights not in ('uniform', 'distance'):
            raise ValueError(f"weights must be 'uniform' or 'distance'. Got {weights}")

        self.k       = k
        self.metric  = metric
        self.weights = weights

        # set during fit()
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Store training data. KNN has no real training step — this just
        memorises X and y for use at prediction time.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        """
        self.X_train_ = np.array(X, dtype=float)
        self.y_train_ = np.array(y)
        self.classes_ = np.unique(self.y_train_)
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Predict class labels for each sample in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of predicted labels, shape (n_samples,)
        """
        self._check_fitted()
        X = np.array(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        """Predict label for a single sample x."""
        # step 1 — compute distance from x to every training sample
        distances = self._compute_distances(x)

        # step 2 — get indices of k smallest distances
        k_indices = np.argsort(distances)[:self.k]

        # step 3 — get labels of those k neighbors
        k_labels = self.y_train_[k_indices]

        # step 4 — vote
        if self.weights == 'uniform':
            return Counter(k_labels.tolist()).most_common(1)[0][0]
        else:
            return self._weighted_vote(k_labels, distances[k_indices])

    # ── predict_proba ─────────────────────────────────────────────────────────

    def predict_proba(self, X):
        """
        Class probability estimates based on neighbor vote fractions.
        Required for ROC/AUC computation in your notebook.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = np.array(X, dtype=float)
        proba = np.zeros((len(X), len(self.classes_)))

        for i, x in enumerate(X):
            distances  = self._compute_distances(x)
            k_indices  = np.argsort(distances)[:self.k]
            k_labels   = self.y_train_[k_indices]
            k_dists    = distances[k_indices]

            for j, cls in enumerate(self.classes_):
                if self.weights == 'uniform':
                    proba[i, j] = np.sum(k_labels == cls) / self.k
                else:
                    weights = self._get_weights(k_dists)
                    proba[i, j] = np.sum(weights[k_labels == cls])

        return proba

    # ── score ─────────────────────────────────────────────────────────────────

    def score(self, X, y):
        """Accuracy: fraction of correct predictions."""
        return np.mean(self.predict(X) == np.array(y))

    # ── distance functions ────────────────────────────────────────────────────

    def _compute_distances(self, x):
        """
        Compute distance from one sample x to ALL training samples at once.
        Returns array of shape (n_train_samples,).
        """
        if self.metric == 'euclidean':
            return self._euclidean(x)
        elif self.metric == 'manhattan':
            return self._manhattan(x)
        elif self.metric == 'cosine':
            return self._cosine(x)

    def _euclidean(self, x):
        """
        Euclidean distance: sqrt( sum( (x - X_train)^2 ) )

        Broadcasting trick: X_train_ has shape (n_train, n_features).
        Subtracting x (shape: n_features,) broadcasts across all rows.
        Result: one distance per training sample.
        """
        diff = self.X_train_ - x          # shape: (n_train, n_features)
        return np.sqrt(np.sum(diff ** 2, axis=1))

    def _manhattan(self, x):
        """
        Manhattan distance: sum( |x - X_train| )
        Same broadcasting as euclidean but sum of absolute values.
        """
        diff = self.X_train_ - x
        return np.sum(np.abs(diff), axis=1)

    def _cosine(self, x):
        """
        Cosine distance: 1 - cosine_similarity
        Measures angle between vectors, not magnitude.
        Useful when scale differences matter less than direction.

        Clipped to [0, 2] to handle floating point edge cases.
        """
        # dot product of x with every training row
        dot_products = self.X_train_ @ x                        # shape: (n_train,)

        # magnitudes (norms)
        norm_train = np.linalg.norm(self.X_train_, axis=1)      # shape: (n_train,)
        norm_x     = np.linalg.norm(x)                          # scalar

        # avoid division by zero for zero vectors
        denom = norm_train * norm_x
        denom = np.where(denom == 0, 1e-10, denom)

        similarity = dot_products / denom
        similarity = np.clip(similarity, -1.0, 1.0)             # numerical safety
        return 1.0 - similarity                                  # distance = 1 - similarity

    # ── weighted voting helpers ───────────────────────────────────────────────

    def _get_weights(self, distances):
        """
        Inverse distance weights: closer neighbors get more say.
        Exact ties get weight 1.0 to avoid division by zero.

        Returns normalised weights summing to 1.
        """
        # if any distance is exactly 0 (identical sample), give it weight 1
        # and all others weight 0
        if np.any(distances == 0):
            weights = (distances == 0).astype(float)
        else:
            weights = 1.0 / distances

        total = weights.sum()
        return weights / total if total > 0 else weights

    def _weighted_vote(self, k_labels, k_distances):
        """Return the class with the highest total weight among k neighbors."""
        weights = self._get_weights(k_distances)

        # sum weights per class
        class_weights = {}
        for label, weight in zip(k_labels, weights):
            class_weights[label] = class_weights.get(label, 0.0) + weight

        return max(class_weights, key=class_weights.get)

    # ── utility ───────────────────────────────────────────────────────────────

    def _check_fitted(self):
        """Raise a clear error if predict() is called before fit()."""
        if self.X_train_ is None:
            raise RuntimeError(
                "This KNearestNeighbors instance is not fitted yet. "
                "Call fit(X, y) before predict()."
            )

    def __repr__(self):
        return (f"KNearestNeighbors(k={self.k}, "
                f"metric='{self.metric}', weights='{self.weights}')")