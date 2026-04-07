"""Evaluation metrics for drug-target affinity prediction.

All public functions accept array-like inputs and return scalar floats.
SciPy and scikit-learn are used when available; pure-NumPy fallbacks
are provided otherwise.

Functions
---------
concordance_index   Fraction of correctly ordered pairs (CI).
mse / rmse / mae     Standard error metrics.
pearson_correlation  Linear correlation coefficient.
spearman_correlation Rank correlation coefficient.
r_squared            Coefficient of determination (R²).
calculate_all_metrics  Convenience wrapper returning all of the above.
"""

import numpy as np
from rdkit.ML.Scoring.Scoring import CalcBEDROC

# Optional dependencies — pure-NumPy fallbacks are used when missing.
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import average_precision_score, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def concordance_index(y_true, y_pred, max_samples=5000):
    """Concordance Index (CI) via vectorised NumPy.

    Measures the fraction of concordant (correctly ordered) pairs.
    Perfect ranking = 1.0, random = 0.5.

    For datasets larger than *max_samples*, a random subset is used to
    keep the O(n²) calculation tractable.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    n = len(y_true)
    if n == 0:
        return 0.0
    
    # For large datasets, sample to avoid O(n²) memory/complexity
    if n > max_samples:
        indices = np.random.choice(n, max_samples, replace=False)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
        n = max_samples
    
    # Vectorized CI calculation using broadcasting
    # Create difference matrices
    true_diff = y_true[:, np.newaxis] - y_true[np.newaxis, :]  # n x n
    pred_diff = y_pred[:, np.newaxis] - y_pred[np.newaxis, :]  # n x n
    
    # Only consider upper triangle (pairs where i < j) and non-tied ground truth
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    non_tied = true_diff != 0
    valid_pairs = upper_mask & non_tied
    
    # Concordant: same ordering direction
    concordant = np.sum((true_diff > 0) & (pred_diff > 0) & valid_pairs) + \
                 np.sum((true_diff < 0) & (pred_diff < 0) & valid_pairs)
    
    total_pairs = np.sum(valid_pairs)
    
    if total_pairs == 0:
        return 0.5
    
    return concordant / total_pairs


def mse(y_true, y_pred):
    """Mean Squared Error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        float: MSE.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if HAS_SKLEARN:
        return mean_squared_error(y_true, y_pred)
    else:
        # Fallback implementation
        return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """Root Mean Squared Error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        float: RMSE.
    """
    return np.sqrt(mse(y_true, y_pred))


def pearson_correlation(y_true, y_pred):
    """Pearson linear correlation coefficient.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        float: Pearson *r* (0.0 if fewer than 2 samples).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) < 2:
        return 0.0
    
    if HAS_SCIPY:
        corr, _ = stats.pearsonr(y_true, y_pred)
        return corr
    else:
        # Fallback implementation using numpy
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        
        numerator = np.sum((y_true - mean_true) * (y_pred - mean_pred))
        denominator = np.sqrt(np.sum((y_true - mean_true) ** 2) * np.sum((y_pred - mean_pred) ** 2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


def r_squared(y_true, y_pred):
    """Coefficient of determination (R²).

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        float: R² (0.0 when total sum-of-squares is zero).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def mae(y_true, y_pred):
    """Mean Absolute Error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        float: MAE.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return np.mean(np.abs(y_true - y_pred))


def spearman_correlation(y_true, y_pred):
    """Spearman rank correlation coefficient.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        float: Spearman *ρ* (0.0 if fewer than 2 samples).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) < 2:
        return 0.0
    
    if HAS_SCIPY:
        corr, _ = stats.spearmanr(y_true, y_pred)
        return corr
    else:
        # Fallback: rank-based calculation
        def rankdata(arr):
            sorted_idx = np.argsort(arr)
            ranks = np.empty_like(sorted_idx, dtype=float)
            ranks[sorted_idx] = np.arange(1, len(arr) + 1)
            return ranks
        
        rank_true = rankdata(y_true)
        rank_pred = rankdata(y_pred)
        
        # Calculate Pearson correlation on ranks
        mean_true = np.mean(rank_true)
        mean_pred = np.mean(rank_pred)
        
        numerator = np.sum((rank_true - mean_true) * (rank_pred - mean_pred))
        denominator = np.sqrt(np.sum((rank_true - mean_true) ** 2) * np.sum((rank_pred - mean_pred) ** 2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


def calculate_all_metrics(y_true, y_pred):
    """Compute all metrics and return them as a dict.

    Keys: ``CI``, ``MSE``, ``RMSE``, ``MAE``, ``Pearson``, ``Spearman``, ``R2``.
    """
    return {
        'CI': concordance_index(y_true, y_pred),
        'MSE': mse(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'Pearson': pearson_correlation(y_true, y_pred),
        'Spearman': spearman_correlation(y_true, y_pred),
        'R2': r_squared(y_true, y_pred),
    }


def print_metrics(metrics_dict, prefix=""):
    """Pretty-print a metrics dict, optionally prefixed (e.g. 'Val')."""
    if prefix:
        print("\n{} Metrics:".format(prefix))
    else:
        print("\nMetrics:")
    
    print("-" * 40)
    for metric_name, value in metrics_dict.items():
        print("{}: {}".format(metric_name, value))
    print("-" * 40)


def _binary_arrays(labels, scores):
    labels = np.asarray(labels, dtype=int).flatten()
    scores = np.asarray(scores, dtype=float).flatten()
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    return labels, scores


def _top_n_from_fraction(total_count, top_fraction):
    if top_fraction <= 0:
        raise ValueError("top_fraction must be positive")
    if top_fraction < 1:
        return max(1, int(np.ceil(total_count * top_fraction)))
    return min(total_count, int(top_fraction))


def enrichment_factor(labels, scores, top_fraction):
    labels, scores = _binary_arrays(labels, scores)
    total_actives = int(labels.sum())
    if total_actives == 0 or labels.size == 0:
        return 0.0
    top_n = _top_n_from_fraction(labels.size, top_fraction)
    ranked = np.argsort(scores)[::-1][:top_n]
    observed = float(labels[ranked].sum())
    expected = float(total_actives) * float(top_n) / float(labels.size)
    if expected <= 0:
        return 0.0
    return observed / expected


def topk_recovery(labels, scores, top_fraction):
    labels, scores = _binary_arrays(labels, scores)
    total_actives = int(labels.sum())
    if total_actives == 0 or labels.size == 0:
        return 0.0
    top_n = _top_n_from_fraction(labels.size, top_fraction)
    ranked = np.argsort(scores)[::-1][:top_n]
    return float(labels[ranked].sum()) / float(total_actives)


def precision_recall_f1_at_fraction(labels, scores, top_fraction):
    labels, scores = _binary_arrays(labels, scores)
    top_n = _top_n_from_fraction(labels.size, top_fraction)
    ranked = np.argsort(scores)[::-1]
    predicted_positive = np.zeros_like(labels, dtype=bool)
    predicted_positive[ranked[:top_n]] = True

    tp = int(np.logical_and(predicted_positive, labels == 1).sum())
    fp = int(np.logical_and(predicted_positive, labels == 0).sum())
    fn = int(np.logical_and(~predicted_positive, labels == 1).sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "top_n": int(top_n),
    }


def bedroc(labels, scores, alpha=20.0):
    labels, scores = _binary_arrays(labels, scores)
    if labels.size == 0 or labels.sum() == 0:
        return 0.0
    ranked = np.argsort(scores)[::-1]
    score_rows = [[int(labels[idx])] for idx in ranked]
    return float(CalcBEDROC(score_rows, 0, float(alpha)))


def auroc(labels, scores):
    labels, scores = _binary_arrays(labels, scores)
    if labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        return 0.0
    if HAS_SKLEARN:
        return float(roc_auc_score(labels, scores))

    ranked = np.argsort(scores)
    ranks = np.empty_like(ranked, dtype=float)
    ranks[ranked] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = len(labels) - n_pos
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / max(1, n_pos * n_neg))


def auprc(labels, scores):
    labels, scores = _binary_arrays(labels, scores)
    if labels.size == 0 or labels.sum() == 0:
        return 0.0
    if HAS_SKLEARN:
        return float(average_precision_score(labels, scores))

    ranked = np.argsort(scores)[::-1]
    sorted_labels = labels[ranked]
    tp_cum = np.cumsum(sorted_labels)
    precision = tp_cum / np.arange(1, len(sorted_labels) + 1)
    recall = tp_cum / float(sorted_labels.sum())
    return float(np.sum(precision * np.diff(np.concatenate(([0.0], recall)))))


def reciprocal_rank(labels, scores):
    labels, scores = _binary_arrays(labels, scores)
    if labels.size == 0 or labels.sum() == 0:
        return 0.0
    ranked_labels = labels[np.argsort(scores)[::-1]]
    active_positions = np.flatnonzero(ranked_labels == 1)
    if active_positions.size == 0:
        return 0.0
    return float(1.0 / float(active_positions[0] + 1))


def mean_reciprocal_rank(rank_values):
    rank_values = np.asarray(rank_values, dtype=float).flatten()
    if rank_values.size == 0:
        return 0.0
    return float(rank_values.mean())


def paired_bootstrap_metric_delta(
    y_true,
    y_pred_a,
    y_pred_b,
    metric_fn,
    *,
    n_boot=1000,
    seed=1337,
):
    y_true = np.asarray(y_true).flatten()
    y_pred_a = np.asarray(y_pred_a).flatten()
    y_pred_b = np.asarray(y_pred_b).flatten()
    if not (y_true.shape == y_pred_a.shape == y_pred_b.shape):
        raise ValueError("bootstrap inputs must have matching shapes")
    if y_true.size == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = np.random.default_rng(seed)
    deltas = np.zeros(n_boot, dtype=float)
    for idx in range(n_boot):
        sample_idx = rng.integers(0, y_true.size, size=y_true.size)
        score_a = metric_fn(y_true[sample_idx], y_pred_a[sample_idx])
        score_b = metric_fn(y_true[sample_idx], y_pred_b[sample_idx])
        deltas[idx] = float(score_a) - float(score_b)

    observed = float(metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b))
    return {
        "delta": observed,
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
    }
