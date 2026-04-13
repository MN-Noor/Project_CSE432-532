import numpy as np


class LogisticRegression:
    """
    Multiclass logistic regression trained with gradient descent.
    """

    def __init__(
        self,
        lr=0.1,
        n_iter=1000,
        reg_lambda=0.01,
        tol=1e-4,
        random_state=None,
        verbose=False,
    ):
        self.lr = lr
        self.n_iter = n_iter
        self.reg_lambda = reg_lambda
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        self.W_ = None
        self.b_ = None
        self.classes_ = None
        self.loss_curve_ = []

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        cls_map = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_idx = np.array([cls_map[label] for label in y])
        y_onehot = self._one_hot(y_idx, n_classes)

        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.normal(0, 0.01, size=(n_features, n_classes))
        self.b_ = np.zeros(n_classes)

        self.loss_curve_ = []
        prev_loss = np.inf

        for i in range(self.n_iter):
            logits = self._linear(X)
            probs = self._softmax(logits)

            loss = self._cross_entropy(y_onehot, probs) + self._l2_penalty()
            self.loss_curve_.append(loss)

            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Early stop at iteration {i}, loss={loss:.6f}")
                break
            prev_loss = loss

            dZ = (probs - y_onehot) / n_samples
            dW = X.T @ dZ + (self.reg_lambda / n_samples) * self.W_
            db = np.sum(dZ, axis=0)

            self.W_ -= self.lr * dW
            self.b_ -= self.lr * db

            if self.verbose and i % 100 == 0:
                print(f"iter {i:4d} loss={loss:.6f}")

        return self

    def predict(self, X):
        self._check_fitted()
        X = np.array(X, dtype=float)
        scores = self._softmax(self._linear(X))
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        self._check_fitted()
        X = np.array(X, dtype=float)
        return self._softmax(self._linear(X))

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))

    def _linear(self, X):
        return X @ self.W_ + self.b_

    def _softmax(self, Z):
        z_stable = Z - np.max(Z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _cross_entropy(self, y_onehot, probs):
        probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
        return -np.mean(np.sum(y_onehot * np.log(probs), axis=1))

    def _l2_penalty(self):
        return (self.reg_lambda / 2) * np.sum(self.W_ ** 2)

    @staticmethod
    def _one_hot(y_idx, n_classes):
        onehot = np.zeros((len(y_idx), n_classes))
        onehot[np.arange(len(y_idx)), y_idx] = 1.0
        return onehot

    def _check_fitted(self):
        if self.W_ is None:
            raise RuntimeError(
                "Model not fitted yet. Call fit(X, y) before predict()."
            )

    def __repr__(self):
        return (
            f"LogisticRegression(lr={self.lr}, n_iter={self.n_iter}, "
            f"reg_lambda={self.reg_lambda})"
        )
