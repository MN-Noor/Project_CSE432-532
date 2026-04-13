import numpy as np


class LinearRegression:
    """
    Ordinary Least Squares linear regression.
    """

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        X_design = self._add_intercept(X) if self.fit_intercept else X
        weights = np.linalg.pinv(X_design) @ y

        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.coef_ = weights

        return self

    def predict(self, X):
        self._check_fitted()
        X = np.array(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    @staticmethod
    def _add_intercept(X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def _check_fitted(self):
        if self.coef_ is None:
            raise RuntimeError(
                "Model not fitted yet. Call fit(X, y) before predict()."
            )

    def __repr__(self):
        return f"LinearRegression(fit_intercept={self.fit_intercept})"


class RidgeRegression:
    """
    Ridge Regression using the closed-form L2-regularized solution.
    """

    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        if self.fit_intercept:
            X = LinearRegression._add_intercept(X)

        n_features = X.shape[1]
        identity = np.eye(n_features)
        if self.fit_intercept:
            identity[0, 0] = 0.0

        weights = np.linalg.solve(X.T @ X + self.alpha * identity, X.T @ y)

        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.coef_ = weights

        return self

    def predict(self, X):
        self._check_fitted()
        return np.array(X, dtype=float) @ self.coef_ + self.intercept_

    def score(self, X, y):
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def _check_fitted(self):
        if self.coef_ is None:
            raise RuntimeError(
                "Model not fitted yet. Call fit(X, y) before predict()."
            )

    def __repr__(self):
        return (
            f"RidgeRegression(alpha={self.alpha}, "
            f"fit_intercept={self.fit_intercept})"
        )


class LassoRegression:
    """
    Lasso Regression trained with coordinate descent.
    """

    def __init__(self, alpha=0.01, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape

        if self.fit_intercept:
            self.intercept_ = np.mean(y)
            y = y - self.intercept_
            X_mean = np.mean(X, axis=0)
            X = X - X_mean
        else:
            X_mean = np.zeros(n_features)

        self.coef_ = np.zeros(n_features)
        col_norms_sq = np.sum(X ** 2, axis=0)

        for _ in range(self.max_iter):
            coef_old = self.coef_.copy()

            for j in range(n_features):
                if col_norms_sq[j] == 0:
                    continue

                residual = y - X @ self.coef_ + X[:, j] * self.coef_[j]
                rho_j = X[:, j] @ residual / col_norms_sq[j]
                threshold = self.alpha * n_samples / col_norms_sq[j]
                self.coef_[j] = self._soft_threshold(rho_j, threshold)

            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                break

        if self.fit_intercept:
            self.intercept_ -= X_mean @ self.coef_

        return self

    def predict(self, X):
        self._check_fitted()
        return np.array(X, dtype=float) @ self.coef_ + self.intercept_

    def score(self, X, y):
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    @staticmethod
    def _soft_threshold(value, threshold):
        if value > threshold:
            return value - threshold
        if value < -threshold:
            return value + threshold
        return 0.0

    def _check_fitted(self):
        if self.coef_ is None:
            raise RuntimeError(
                "Model not fitted yet. Call fit(X, y) before predict()."
            )

    def __repr__(self):
        return (
            f"LassoRegression(alpha={self.alpha}, max_iter={self.max_iter}, "
            f"fit_intercept={self.fit_intercept})"
        )
