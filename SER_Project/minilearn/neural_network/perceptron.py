
import numpy as np


class Perceptron:
    """
    Single-layer or one-hidden-layer ANN for multiclass classification.

    Architecture:
        hidden_size=0 : input → output (single-layer perceptron)
                        mathematically identical to logistic regression
        hidden_size>0 : input → hidden (ReLU) → output (softmax)
                        one hidden layer ANN

    Trained with gradient descent + cross-entropy loss + L2 regularisation.
    Backpropagation computes gradients for both layers when hidden_size > 0.

    Parameters
    ----------
    hidden_size  : int   — neurons in hidden layer. 0 = single-layer perceptron
    lr           : float — learning rate
    n_iter       : int   — training epochs
    reg_lambda   : float — L2 regularisation strength
    random_state : int   — seed for reproducibility
    verbose      : bool  — print loss every 100 iterations
    """

    def __init__(self, hidden_size=64, lr=0.01, n_iter=500,
                 reg_lambda=0.01, random_state=None, verbose=False):
        self.hidden_size  = hidden_size
        self.lr           = lr
        self.n_iter       = n_iter
        self.reg_lambda   = reg_lambda
        self.random_state = random_state
        self.verbose      = verbose

        # set during fit()
        self.W1_         = None   # input → hidden weights (hidden_size > 0 only)
        self.b1_         = None   # hidden bias
        self.W2_         = None   # hidden → output weights (or input → output)
        self.b2_         = None   # output bias
        self.classes_    = None
        self.loss_curve_ = []

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Train the network using gradient descent and backpropagation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,) — integer class labels
        """
        X   = np.array(X, dtype=float)
        y   = np.array(y)
        rng = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape
        self.classes_  = np.unique(y)
        n_classes      = len(self.classes_)

        # map original labels to 0-indexed integers
        cls_map = {c: i for i, c in enumerate(self.classes_)}
        y_idx   = np.array([cls_map[label] for label in y])
        Y_oh    = self._one_hot(y_idx, n_classes)   # shape (n_samples, n_classes)

        # initialise weights
        # small random values break symmetry — all-zero weights would learn nothing
        if self.hidden_size > 0:
            # two-layer network
            self.W1_ = rng.normal(0, 0.01, (n_features, self.hidden_size))
            self.b1_ = np.zeros(self.hidden_size)
            self.W2_ = rng.normal(0, 0.01, (self.hidden_size, n_classes))
            self.b2_ = np.zeros(n_classes)
        else:
            # single-layer perceptron — no hidden layer
            self.W2_ = rng.normal(0, 0.01, (n_features, n_classes))
            self.b2_ = np.zeros(n_classes)

        self.loss_curve_ = []

        for i in range(self.n_iter):

            # ── forward pass ──────────────────────────────────────────
            if self.hidden_size > 0:
                # hidden layer: linear → ReLU
                Z1 = X @ self.W1_ + self.b1_   # shape (n_samples, hidden_size)
                A1 = self._relu(Z1)             # shape (n_samples, hidden_size)
                # output layer: linear → softmax
                Z2 = A1 @ self.W2_ + self.b2_  # shape (n_samples, n_classes)
            else:
                # single layer: linear → softmax directly
                A1 = X
                Z2 = X @ self.W2_ + self.b2_   # shape (n_samples, n_classes)

            P = self._softmax(Z2)               # shape (n_samples, n_classes)

            # ── loss: cross-entropy + L2 regularisation ────────────────
            loss = self._cross_entropy(Y_oh, P)
            loss += (self.reg_lambda / 2) * np.sum(self.W2_ ** 2)
            if self.hidden_size > 0:
                loss += (self.reg_lambda / 2) * np.sum(self.W1_ ** 2)
            self.loss_curve_.append(loss)

            # ── backward pass (backpropagation) ────────────────────────
            # gradient of cross-entropy through softmax = P - Y_onehot
            # this is the elegant simplification of the full chain rule
            dZ2 = (P - Y_oh) / n_samples               # shape (n_samples, n_classes)
            dW2 = A1.T @ dZ2                            # shape (hidden, n_classes)
            dW2 += (self.reg_lambda / n_samples) * self.W2_
            db2 = np.sum(dZ2, axis=0)                   # shape (n_classes,)

            # update output layer weights
            self.W2_ -= self.lr * dW2
            self.b2_ -= self.lr * db2

            if self.hidden_size > 0:
                # backpropagate through hidden layer
                # chain rule: dLoss/dA1 = dZ2 @ W2.T
                # then through ReLU: dLoss/dZ1 = dLoss/dA1 * ReLU'(Z1)
                dA1 = dZ2 @ self.W2_.T                  # shape (n_samples, hidden)
                dZ1 = dA1 * self._relu_grad(Z1)         # shape (n_samples, hidden)
                dW1 = X.T @ dZ1                         # shape (n_features, hidden)
                dW1 += (self.reg_lambda / n_samples) * self.W1_
                db1 = np.sum(dZ1, axis=0)               # shape (hidden,)

                # update hidden layer weights
                self.W1_ -= self.lr * dW1
                self.b1_ -= self.lr * db1

            if self.verbose and i % 100 == 0:
                print(f'  iter {i:4d}  loss={loss:.6f}')

        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Predict class labels for each sample.

        Returns original class labels (e.g. 1-8 for RAVDESS).
        """
        proba = self.predict_proba(X)
        idx   = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        """
        Return class probability estimates via softmax.
        Required for ROC/AUC computation.

        Returns
        -------
        np.ndarray shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = np.array(X, dtype=float)

        if self.hidden_size > 0:
            A1 = self._relu(X @ self.W1_ + self.b1_)
            Z2 = A1 @ self.W2_ + self.b2_
        else:
            Z2 = X @ self.W2_ + self.b2_

        return self._softmax(Z2)

    def score(self, X, y):
        """Accuracy: fraction of correct predictions."""
        y_pred = self.predict(X)
        return np.mean(y_pred == np.array(y))

    # ── activation functions ──────────────────────────────────────────────────

    @staticmethod
    def _relu(Z):
        """
        ReLU activation: max(0, z)
        Used in hidden layer — introduces non-linearity.
        Gradient is simple: 1 if z > 0, else 0.
        """
        return np.maximum(0, Z)

    @staticmethod
    def _relu_grad(Z):
        """
        Gradient of ReLU — used in backpropagation.
        1 where Z > 0, 0 elsewhere.
        """
        return (Z > 0).astype(float)

    @staticmethod
    def _softmax(Z):
        """
        Softmax: converts raw scores to probabilities.
        Subtracts row max for numerical stability — prevents overflow.
        Each row sums to 1.0.
        """
        Z_stable = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z    = np.exp(Z_stable)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    @staticmethod
    def _cross_entropy(Y_onehot, P):
        """
        Cross-entropy loss: -mean( sum( Y * log(P) ) )
        Only the true class contributes — Y_onehot is 0 everywhere else.
        Clip P to prevent log(0) = -inf.
        """
        P_clipped = np.clip(P, 1e-12, 1 - 1e-12)
        return -np.mean(np.sum(Y_onehot * np.log(P_clipped), axis=1))

    @staticmethod
    def _one_hot(y_idx, n_classes):
        """
        Convert integer labels to one-hot matrix.
        y_idx=[0,2,1], n_classes=3 →
        [[1,0,0],
         [0,0,1],
         [0,1,0]]
        """
        Y = np.zeros((len(y_idx), n_classes))
        Y[np.arange(len(y_idx)), y_idx] = 1.0
        return Y

    # ── utility ───────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if self.W2_ is None:
            raise RuntimeError(
                "Model not fitted. Call fit(X, y) before predict()."
            )

    def __repr__(self):
        return (f"Perceptron(hidden_size={self.hidden_size}, "
                f"lr={self.lr}, n_iter={self.n_iter})")