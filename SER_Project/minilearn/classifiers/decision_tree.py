# minilearn/classifiers/random_forest.py
import numpy as np
from collections import Counter


# ─────────────────────────────────────────────
#  single Decision Tree (CART)
# ─────────────────────────────────────────────

class _Node:
    """One node in a decision tree."""
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature   = feature    # which feature to split on
        self.threshold = threshold  # split value
        self.left      = left       # left child node
        self.right     = right      # right child node
        self.value     = value      # set only for leaf nodes (the class)


class _DecisionTree:
    """
    A single CART decision tree used internally by RandomForest.
    Supports max_depth and min_samples_split to control overfitting.
    """

    def __init__(self, max_depth=None, min_samples_split=2, random_state=None):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.random_state      = random_state
        self.root              = None
        self._rng              = np.random.default_rng(random_state)

   

    def fit(self, X, y):
        self.root = self._grow(X, y, depth=0)
        return self

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

   

    def _grow(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # stopping conditions → make a leaf
        if (self.max_depth is not None and depth >= self.max_depth) \
                or n_classes == 1 \
                or n_samples < self.min_samples_split:
            return _Node(value=self._majority(y))

        # pick a random subset of features (used by RandomForest)
        feat_indices = self._rng.choice(
            n_features,
            size=getattr(self, '_n_features_subset', n_features),
            replace=False
        )

        best_feat, best_thresh = self._best_split(X, y, feat_indices)

        if best_feat is None:           # no useful split found
            return _Node(value=self._majority(y))

        left_mask  = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left  = self._grow(X[left_mask],  y[left_mask],  depth + 1)
        right = self._grow(X[right_mask], y[right_mask], depth + 1)
        return _Node(feature=best_feat, threshold=best_thresh,
                     left=left, right=right)

    def _best_split(self, X, y, feat_indices):
        best_gain   = -1
        best_feat   = None
        best_thresh = None

        parent_gini = self._gini(y)

        for feat in feat_indices:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask  = X[:, feat] <= thresh
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                gain = self._information_gain(
                    y, y[left_mask], y[right_mask], parent_gini
                )

                if gain > best_gain:
                    best_gain   = gain
                    best_feat   = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    # ── metrics ─────────────────────────────────

    @staticmethod
    def _gini(y):
        """Gini impurity: 1 - sum(p_i^2). 0 = perfectly pure node."""
        n = len(y)
        if n == 0:
            return 0.0
        counts = np.bincount(y.astype(int))
        probs  = counts / n
        return 1.0 - np.sum(probs ** 2)

    def _information_gain(self, parent, left, right, parent_gini):
        """Weighted reduction in Gini after the split."""
        n  = len(parent)
        wl = len(left)  / n
        wr = len(right) / n
        return parent_gini - (wl * self._gini(left) + wr * self._gini(right))

    # ── prediction helpers ───────────────────────

    def _traverse(self, x, node):
        if node.value is not None:      # leaf
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    @staticmethod
    def _majority(y):
        return Counter(y.tolist()).most_common(1)[0][0]


# ─────────────────────────────────────────────
#  RandomForest
# ─────────────────────────────────────────────

class RandomForest:
    """
    Parameters
    ----------
    n_estimators      : int   — number of trees (default 100)
    max_depth         : int   — max depth per tree (None = grow fully)
    min_samples_split : int   — minimum samples to split a node (default 2)
    max_features      : str or int
                        'sqrt'  → sqrt(n_features)   [default, recommended]
                        'log2'  → log2(n_features)
                        int     → exact number of features
    random_state      : int or None — seed for reproducibility
    """

    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, max_features='sqrt',
                 random_state=None):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.random_state      = random_state

        self.trees_        = []          # fitted tree objects
        self.feature_importances_ = None # filled after fit()
        self._rng = np.random.default_rng(random_state)

    # ── fit ─────────────────────────────────────

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        # resolve max_features
        if self.max_features == 'sqrt':
            n_feat_subset = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            n_feat_subset = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            n_feat_subset = min(self.max_features, n_features)
        else:
            raise ValueError(f"max_features must be 'sqrt', 'log2', or int. "
                             f"Got {self.max_features}")

        self.trees_ = []
        importances = np.zeros(n_features)

        for i in range(self.n_estimators):
            # bootstrap sample — sample n_samples rows WITH replacement
            boot_idx = self._rng.integers(0, n_samples, size=n_samples)
            X_boot   = X[boot_idx]
            y_boot   = y[boot_idx]

            # build tree with a unique seed derived from parent seed
            tree = _DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self._rng.integers(0, 1_000_000)
            )
            tree._n_features_subset = n_feat_subset
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

            # accumulate feature importances from this tree
            importances += self._tree_feature_importances(tree.root, n_features)

        # normalise so they sum to 1
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances

        return self

    # ── predict ─────────────────────────────────

    def predict(self, X):
        """
        Majority vote across all trees.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of predicted class labels, shape (n_samples,)
        """
        X = np.array(X)
        # shape: (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.trees_])
        # majority vote per sample
        return np.array([
            Counter(all_preds[:, i].tolist()).most_common(1)[0][0]
            for i in range(X.shape[0])
        ])

    def predict_proba(self, X):
        X = np.array(X)
        classes     = np.unique(
            np.concatenate([tree.predict(X[:1]) for tree in self.trees_])
        )
        # collect all trees' predictions
        all_preds   = np.array([tree.predict(X) for tree in self.trees_])
        n_samples   = X.shape[0]
        n_classes   = len(self.classes_) if hasattr(self, 'classes_') \
                      else int(all_preds.max()) + 1

        proba = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            votes = all_preds[:, i]
            for cls in range(n_classes):
                proba[i, cls] = np.mean(votes == cls)
        return proba

    # ── score ────────────────────────────────────

    def score(self, X, y):
        """Accuracy: fraction of correct predictions."""
        return np.mean(self.predict(X) == np.array(y))

    # ── feature importance helper ────────────────

    def _tree_feature_importances(self, node, n_features):
        """
        Walk the tree recursively and accumulate Gini-based importance
        for each feature.  A feature used higher in the tree (larger
        subtrees) gets more credit.
        """
        importances = np.zeros(n_features)
        self._accumulate_importance(node, importances)
        return importances

    def _accumulate_importance(self, node, importances):
        if node is None or node.value is not None:
            return
        importances[node.feature] += 1   # simple count; proxy for importance
        self._accumulate_importance(node.left,  importances)
        self._accumulate_importance(node.right, importances)

    # ── oob score (out-of-bag) ───────────────────

    def oob_score(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples    = len(X)
        oob_votes    = [Counter() for _ in range(n_samples)]

        rng = np.random.default_rng(self.random_state)
        for tree in self.trees_:
            boot_idx = rng.integers(0, n_samples, size=n_samples)
            oob_idx  = np.setdiff1d(np.arange(n_samples), boot_idx)
            if len(oob_idx) == 0:
                continue
            preds = tree.predict(X[oob_idx])
            for idx, pred in zip(oob_idx, preds):
                oob_votes[idx][pred] += 1

        correct = 0
        total   = 0
        for i, votes in enumerate(oob_votes):
            if votes:
                pred = votes.most_common(1)[0][0]
                if pred == y[i]:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0