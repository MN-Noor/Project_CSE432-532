# minilearn/preprocessing/scaler.py
import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """Compute mean and std from training data only."""
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0 # avoid division by zero for constant features
        return self

    def transform(self, X):
        """Apply z-score normalization using fitted stats."""
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        return (np.array(X) - self.mean_) / self.std_

    def fit_transform(self, X):
        """Convenience: fit on X then transform it (use on train set only)."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Reverse the scaling back to original space."""
        return np.array(X) * self.std_ + self.mean_