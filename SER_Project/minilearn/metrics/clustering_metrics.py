# minilearn/metrics/clustering_metrics.py
import numpy as np


def adjusted_rand_index(y_true, y_pred):
    """
    Adjusted Rand Index — measures agreement between true labels and clusters.
    Adjusted for chance: random clustering gives ARI=0.
    Range: [-1, 1], higher is better. 1.0 = perfect match.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n      = len(y_true)

    classes  = np.unique(y_true)
    clusters = np.unique(y_pred)

    # build contingency table
    cls_map  = {c: i for i, c in enumerate(classes)}
    clus_map = {c: i for i, c in enumerate(clusters)}
    contingency = np.zeros((len(classes), len(clusters)), dtype=int)

    for t, p in zip(y_true, y_pred):
        contingency[cls_map[t], clus_map[p]] += 1

    def comb2(n):
        return n * (n - 1) / 2

    sum_comb_c = np.sum([comb2(n_ij) for n_ij in contingency.flatten()])
    sum_comb_a = np.sum([comb2(a) for a in contingency.sum(axis=1)])
    sum_comb_b = np.sum([comb2(b) for b in contingency.sum(axis=0)])
    total_comb = comb2(n)

    expected = sum_comb_a * sum_comb_b / total_comb
    max_val  = (sum_comb_a + sum_comb_b) / 2

    if max_val - expected == 0:
        return 1.0
    return (sum_comb_c - expected) / (max_val - expected)


def normalized_mutual_info(y_true, y_pred):
    """
    Normalized Mutual Information — measures shared information
    between true labels and cluster assignments.
    Range: [0, 1]. 0=no shared info, 1=perfect match.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n      = len(y_true)

    classes  = np.unique(y_true)
    clusters = np.unique(y_pred)

    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        probs     = counts / len(labels)
        return -np.sum(probs * np.log(probs + 1e-10))

    h_true = entropy(y_true)
    h_pred = entropy(y_pred)

    # mutual information
    mi = 0.0
    for cls in classes:
        for clus in clusters:
            joint  = np.sum((y_true == cls) & (y_pred == clus)) / n
            p_cls  = np.sum(y_true == cls)  / n
            p_clus = np.sum(y_pred == clus) / n
            if joint > 0:
                mi += joint * np.log(joint / (p_cls * p_clus + 1e-10))

    denom = (h_true + h_pred) / 2
    return mi / denom if denom > 0 else 0.0