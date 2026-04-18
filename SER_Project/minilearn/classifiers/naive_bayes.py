# minilearn/classifiers/gaussian_naive_bayes.py
import numpy as np


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier 

    Assumes features are conditionally independent given the class,
    and that each feature follows a Gaussian (normal) distribution
    within each class.

    No gradient descent — training is just computing means, variances,
    and class priors from the data directly.

    Parameters
    ----------
    var_smoothing : float
        Adds a small value to all variances to prevent division by zero
        when a feature has zero variance in a class (e.g. constant feature).
        Default 1e-9 matches scikit-learn's default.
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

        # set during fit()
        self.classes_   = None   # unique class labels, shape (n_classes,)
        self.priors_    = None   # log P(class k), shape (n_classes,)
        self.means_     = None   # mean per class per feature, shape (n_classes, n_features)
        self.vars_      = None   # variance per class per feature, shape (n_classes, n_features)
        self.epsilon_   = None   # sklearn-style absolute variance smoothing value

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Compute class priors, feature means, and feature variances
        from training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.classes_         = np.unique(y)
        n_classes             = len(self.classes_)

        # allocate storage
        self.means_  = np.zeros((n_classes, n_features))
        self.vars_   = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()

        for i, cls in enumerate(self.classes_):
            # extract all training rows belonging to this class
            X_cls = X[y == cls]

            # mean of each feature for this class
            self.means_[i, :] = np.mean(X_cls, axis=0)

            # variance of each feature for this class
            # + epsilon_ prevents zero variance on constant features
            self.vars_[i, :]  = np.var(X_cls, axis=0) + self.epsilon_

            # log prior: log(count / total)
            # store as log to match the log-likelihood sum later
            self.priors_[i]   = np.log(len(X_cls) / n_samples)

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
        # compute log-posterior for every class for every sample
        log_posteriors = self._log_posteriors(X)     # shape (n_samples, n_classes)
        # pick the class with the highest log-posterior per sample
        best_indices   = np.argmax(log_posteriors, axis=1)
        return self.classes_[best_indices]

    def predict_proba(self, X):
        """
        Return class probability estimates.
        Converts log-posteriors to proper probabilities via softmax.
        Required for ROC/AUC in your notebook.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = np.array(X, dtype=float)
        log_posteriors = self._log_posteriors(X)
        # softmax over log-posteriors to get normalised probabilities
        return self._softmax(log_posteriors)

    # ── score ─────────────────────────────────────────────────────────────────

    def score(self, X, y):
        """Accuracy: fraction of correct predictions."""
        return np.mean(self.predict(X) == np.array(y))

    # ── core math ─────────────────────────────────────────────────────────────

    def _log_posteriors(self, X):
        """
        Compute log P(class k | x) for every sample and every class.

        By Bayes' theorem (ignoring the denominator which is class-independent):
            log P(class k | x) ∝ log P(class k) + sum_j log P(x_j | class k)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        n_samples  = X.shape[0]
        n_classes  = len(self.classes_)
        log_post   = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            # log likelihood of all features for class i
            # shape: (n_samples,)  — one total log-likelihood per sample
            log_likelihood = np.sum(
                self._log_gaussian(X, self.means_[i], self.vars_[i]),
                axis=1
            )
            # posterior = prior + likelihood  (both in log space → add)
            log_post[:, i] = self.priors_[i] + log_likelihood

        return log_post

    def _log_gaussian(self, X, mean, var):
        """
        Log of Gaussian probability density function.

        For a single feature value x with Gaussian(mean, var):
            log P(x) = -0.5 × log(2π × var) - (x - mean)² / (2 × var)

        Applied to the full matrix X (n_samples, n_features) at once,
        broadcasting mean and var (shape: n_features,) across all rows.

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
        Each element is log P(x_ij | class k) for feature j of sample i.
        """
        log_normaliser = -0.5 * np.log(2 * np.pi * var)
        exponent       = -0.5 * ((X - mean) ** 2) / var
        return log_normaliser + exponent

    @staticmethod
    def _softmax(Z):
        """
        Convert log-posteriors to normalised probabilities.
        Same numerical stability trick as logistic regression:
        subtract row max before exponentiating.
        """
        Z_stable = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z    = np.exp(Z_stable)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    # ── inspection helpers ────────────────────────────────────────────────────

    def most_discriminative_features(self, feature_names, top_n=10):
        """
        Return the features that differ most between classes.
        Useful for your notebook analysis section.

        Uses variance of class means — features where class means
        are spread far apart are most discriminative.

        Parameters
        ----------
        feature_names : list of str
        top_n         : int

        Returns
        -------
        list of (feature_name, score) tuples, sorted by score descending
        """
        self._check_fitted()
        # variance of class means per feature — high = means are spread out
        scores     = np.var(self.means_, axis=0)
        top_idx    = np.argsort(scores)[::-1][:top_n]
        return [(feature_names[i], scores[i]) for i in top_idx]

    # ── utility ───────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if self.classes_ is None:
            raise RuntimeError(
                "Model not fitted yet. Call fit(X, y) before predict()."
            )

    def __repr__(self):
        return f"GaussianNaiveBayes(var_smoothing={self.var_smoothing})"
