# minilearn/clustering/kmeans.py
import numpy as np


class KMeans:
    """
    K-Means clustering 

    Algorithm:
    1. Initialise k centroids (k-means++ for better convergence)
    2. Assign each sample to nearest centroid
    3. Update centroids to mean of assigned samples
    4. Repeat 2-3 until convergence or max_iter reached

    Parameters
    ----------
    k            : int   — number of clusters 
    max_iter     : int   — maximum iterations 
    tol          : float — convergence threshold on centroid movement
    init         : str   — 'kmeans++' (smart init) or 'random'
    random_state : int   — seed
    """

    def __init__(self, k=8, max_iter=300, tol=1e-4,
                 init='kmeans++', random_state=None):
        self.k            = k
        self.max_iter     = max_iter
        self.tol          = tol
        self.init         = init
        self.random_state = random_state

        self.centroids_    = None   # shape (k, n_features)
        self.labels_       = None   # shape (n_samples,) — cluster assignment
        self.inertia_      = None   # sum of squared distances to centroids
        self.n_iter_       = 0      # iterations until convergence

    def fit(self, X):
        X   = np.array(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        # initialise centroids
        if self.init == 'kmeans++':
            self.centroids_ = self._init_kmeanspp(X, rng)
        else:
            idx             = rng.choice(len(X), size=self.k, replace=False)
            self.centroids_ = X[idx].copy()

        for i in range(self.max_iter):
            # assign step
            labels      = self._assign(X)

            # update step
            new_centroids = np.array([
                X[labels == j].mean(axis=0)
                if (labels == j).any()
                else self.centroids_[j]   # keep old centroid if cluster is empty
                for j in range(self.k)
            ])

            # check convergence — how much did centroids move
            shift = np.max(np.linalg.norm(new_centroids - self.centroids_,
                                           axis=1))
            self.centroids_ = new_centroids
            self.n_iter_    = i + 1

            if shift < self.tol:
                break

        self.labels_   = self._assign(X)
        self.inertia_  = self._compute_inertia(X)
        return self

    def predict(self, X):
        """Assign new samples to nearest centroid."""
        self._check_fitted()
        return self._assign(np.array(X, dtype=float))

    def fit_predict(self, X):
        """Fit and return cluster labels."""
        return self.fit(X).labels_

    # ── internal helpers ─────────────────────────────────

    def _assign(self, X):
        """
        Assign each sample to its nearest centroid.
        Uses vectorised squared distance computation.
        Returns array of cluster indices, shape (n_samples,)
        """
        # distances shape: (n_samples, k)
        distances = np.array([
            np.sum((X - centroid) ** 2, axis=1)
            for centroid in self.centroids_
        ]).T
        return np.argmin(distances, axis=1)

    def _compute_inertia(self, X):
        """Sum of squared distances from each point to its centroid."""
        total = 0.0
        for j in range(self.k):
            mask = self.labels_ == j
            if mask.any():
                diff   = X[mask] - self.centroids_[j]
                total += np.sum(diff ** 2)
        return total

    def _init_kmeanspp(self, X, rng):
        """
        K-Means++ initialisation — spreads initial centroids out.
        Each new centroid is chosen with probability proportional
        to its squared distance from the nearest existing centroid.
        This dramatically improves convergence vs random init.
        """
        n_samples  = len(X)
        centroids  = [X[rng.integers(0, n_samples)]]

        for _ in range(1, self.k):
            # distance from each point to nearest existing centroid
            dists = np.array([
                min(np.sum((x - c) ** 2) for c in centroids)
                for x in X
            ])
            # sample next centroid proportional to distance squared
            probs   = dists / dists.sum()
            cum_prob = np.cumsum(probs)
            r        = rng.random()
            idx      = np.searchsorted(cum_prob, r)
            centroids.append(X[idx])

        return np.array(centroids)

    def _check_fitted(self):
        if self.centroids_ is None:
            raise RuntimeError("Call fit() before predict().")