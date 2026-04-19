# minilearn/classifiers/svm.py
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Kernel functions
# ─────────────────────────────────────────────────────────────────────────────

def linear_kernel(X1, X2):
    """
    K(a, b) = a · b
    Just the dot product. No transformation.
    Fastest kernel — use when data is linearly separable in feature space.
    X1: (n, d), X2: (m, d) → (n, m)
    """
    return X1 @ X2.T


def rbf_kernel(X1, X2, gamma=0.1):
    """
    K(a, b) = exp(-gamma × ||a - b||²)
    Radial Basis Function — maps to infinite dimensions implicitly.
    gamma controls how far influence of a single point reaches:
      high gamma → tight, complex boundary (risk overfit)
      low gamma  → smooth, wide boundary (risk underfit)
    X1: (n, d), X2: (m, d) → (n, m)
    """
    # efficient squared distance: ||a-b||² = ||a||² + ||b||² - 2a·b
    sq_dist = (
        np.sum(X1 ** 2, axis=1, keepdims=True)   # shape (n, 1)
        + np.sum(X2 ** 2, axis=1)                 # shape (m,) broadcasts to (n, m)
        - 2 * (X1 @ X2.T)                         # shape (n, m)
    )
    return np.exp(-gamma * sq_dist)


def polynomial_kernel(X1, X2, degree=3, coef=1):
    """
    K(a, b) = (a · b + coef)^degree
    Captures feature interactions up to degree-th order.
    degree=2 captures pairwise interactions, degree=3 triple interactions.
    X1: (n, d), X2: (m, d) → (n, m)
    """
    return (X1 @ X2.T + coef) ** degree


# ─────────────────────────────────────────────────────────────────────────────
#  Binary SVM (one class vs another)
# ─────────────────────────────────────────────────────────────────────────────

class _BinarySVM:
    """
    Linear binary SVM trained with SGD on hinge loss + L2 regularisation.
    Used internally by SVM (multiclass) as one-vs-rest binary classifier.

    Parameters
    ----------
    lr         : float — learning rate
    reg_lambda : float — L2 regularisation (controls margin width)
    n_iter     : int   — gradient descent iterations
    """

    def __init__(self, lr=0.001, reg_lambda=0.01, n_iter=1000):
        self.lr         = lr
        self.reg_lambda = reg_lambda
        self.n_iter     = n_iter
        self.w_         = None
        self.b_         = 0.0

    def fit(self, X, y):
        """
        y must be binary: +1 or -1.
        Trains using full-batch gradient descent on hinge loss.
        """
        n_samples, n_features = X.shape
        self.w_ = np.zeros(n_features)
        self.b_ = 0.0

        for _ in range(self.n_iter):
            # raw decision scores for all samples
            scores = X @ self.w_ + self.b_           # shape (n_samples,)

            # hinge condition: y * score < 1 means violation
            margin = y * scores                       # shape (n_samples,)
            mask   = margin < 1                       # True where hinge fires

            # gradient of hinge loss w.r.t. w and b
            # regularisation gradient always applies
            dw = self.reg_lambda * self.w_
            db = 0.0

            # hinge gradient only applies to violating samples
            if mask.any():
                dw -= np.mean(y[mask, np.newaxis] * X[mask], axis=0)
                db -= np.mean(y[mask])

            self.w_ -= self.lr * dw
            self.b_ -= self.lr * db

        return self

    def decision_function(self, X):
        """Raw score: w·x + b. Sign = predicted class, magnitude = confidence."""
        return X @ self.w_ + self.b_

    def predict(self, X):
        """Returns +1 or -1 for each sample."""
        return np.sign(self.decision_function(X))


# ─────────────────────────────────────────────────────────────────────────────
#  Multiclass SVM — One-vs-Rest
# ─────────────────────────────────────────────────────────────────────────────

class SVM:
    """
    Multiclass SVM using One-vs-Rest (OvR) strategy.

    Trains one binary SVM per class: "is this emotion X or not?"
    Prediction = class whose binary SVM gives the highest confidence score.

    Supports linear kernel natively (w, b stored directly).
    For RBF/polynomial, uses kernel trick at prediction time.

    Parameters
    ----------
    kernel     : str   — 'linear' | 'rbf' | 'poly'
    lr         : float — learning rate for gradient descent
    reg_lambda : float — L2 regularisation strength (C = 1/reg_lambda)
    n_iter     : int   — gradient descent steps per binary SVM
    gamma      : float — RBF/poly kernel parameter
    degree     : int   — polynomial kernel degree
    """

    def __init__(self, kernel='linear', lr=0.001, reg_lambda=0.01,
                 n_iter=1000, gamma=0.1, degree=3):
        if kernel not in ('linear', 'rbf', 'poly'):
            raise ValueError(f"kernel must be 'linear', 'rbf', or 'poly'. Got {kernel}")

        self.kernel     = kernel
        self.lr         = lr
        self.reg_lambda = reg_lambda
        self.n_iter     = n_iter
        self.gamma      = gamma
        self.degree     = degree

        # set during fit()
        self.classes_   = None
        self.classifiers_= []    # one _BinarySVM per class
        self.X_train_   = None   # needed for kernel evaluation at predict time

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Train one binary SVM per class using One-vs-Rest.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,) — integer class labels
        """
        X = np.array(X, dtype=float)
        y = np.array(y)

        self.classes_    = np.unique(y)
        self.classifiers_= []
        self.X_train_    = X      # stored for kernel trick at predict time

        # apply kernel transformation to training data once
        X_kern = self._apply_kernel(X, X)   # shape (n_samples, n_samples)
                                             # or (n_samples, n_features) for linear

        for cls in self.classes_:
            # create binary labels: +1 for this class, -1 for all others
            y_binary = np.where(y == cls, 1, -1).astype(float)

            clf = _BinarySVM(
                lr=self.lr,
                reg_lambda=self.reg_lambda,
                n_iter=self.n_iter
            )
            clf.fit(X_kern, y_binary)
            self.classifiers_.append(clf)

        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Predict class labels.
        For each sample, compute confidence score from all 8 binary SVMs.
        Predict the class with the highest score.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of predicted class labels
        """
        self._check_fitted()
        scores = self.decision_function(X)           # shape (n_samples, n_classes)
        best   = np.argmax(scores, axis=1)           # index of highest score
        return self.classes_[best]

    def decision_function(self, X):
        """
        Raw confidence scores from each binary SVM for each sample.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        self._check_fitted()
        X      = np.array(X, dtype=float)
        X_kern = self._apply_kernel(X, self.X_train_)  # (n_test, n_train) or (n_test, n_features)

        # collect score from each binary classifier
        scores = np.column_stack([
            clf.decision_function(X_kern)
            for clf in self.classifiers_
        ])
        return scores

    def predict_proba(self, X):
        """
        Soft probability estimates via softmax over decision scores.
        Not true probabilities (Platt scaling would be needed for that)
        but good enough for ROC/AUC comparison.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        scores   = self.decision_function(X)
        # softmax for normalisation
        shifted  = scores - np.max(scores, axis=1, keepdims=True)
        exp_s    = np.exp(shifted)
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)

    # ── score ─────────────────────────────────────────────────────────────────

    def score(self, X, y):
        """Accuracy: fraction of correct predictions."""
        return np.mean(self.predict(X) == np.array(y))

    # ── kernel application ────────────────────────────────────────────────────

    def _apply_kernel(self, X1, X2):
        """
        Apply the chosen kernel to transform the input.

        Linear kernel: returns X1 directly (no transformation needed —
        the binary SVM's w already lives in feature space).

        RBF / poly: returns the kernel matrix K where K[i,j] = K(x_i, x_j).
        The binary SVM then operates in this implicit feature space.
        """
        if self.kernel == 'linear':
            return X1    # no transformation for linear
        elif self.kernel == 'rbf':
            return rbf_kernel(X1, X2, gamma=self.gamma)
        elif self.kernel == 'poly':
            return polynomial_kernel(X1, X2, degree=self.degree)

    # ── utility ───────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if self.classes_ is None:
            raise RuntimeError(
                "Model not fitted yet. Call fit(X, y) before predict()."
            )

    def __repr__(self):
        return (f"SVM(kernel='{self.kernel}', lr={self.lr}, "
                f"reg_lambda={self.reg_lambda}, n_iter={self.n_iter})")