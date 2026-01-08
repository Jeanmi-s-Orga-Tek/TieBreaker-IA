import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, roc_auc_score

def evaluate_binary_prob_model(y_true, p_pred, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    p_pred = np.clip(p_pred, 1e-15, 1 - 1e-15)

    metrics = {
        "log_loss": log_loss(y_true, p_pred),
        "brier": brier_score_loss(y_true, p_pred),
        "accuracy": accuracy_score(y_true, (p_pred >= threshold).astype(int)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["auc"] = roc_auc_score(y_true, p_pred)
    else:
        metrics["auc"] = np.nan
    return metrics

def bootstrap_ci_diff(y_true, p_a, p_b, metric_fn, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)

    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(metric_fn(y_true[idx], p_a[idx]) - metric_fn(y_true[idx], p_b[idx]))
    diffs = np.asarray(diffs)
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)), float(diffs.mean())

y_test = np.array([1, 0, 1, 1, 0])
p_model1 = np.array([0.62, 0.40, 0.55, 0.80, 0.20])
p_model2 = np.array([0.58, 0.35, 0.60, 0.75, 0.30])

m1 = evaluate_binary_prob_model(y_test, p_model1)
m2 = evaluate_binary_prob_model(y_test, p_model2)

print("Model1:", m1)
print("Model2:", m2)

ci_lo, ci_hi, mean_diff = bootstrap_ci_diff(
    y_test, p_model1, p_model2,
    metric_fn=lambda y, p: log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))
)
print(f"log_loss diff (Model1-Model2): mean={mean_diff:.6f}, 95% CI=({ci_lo:.6f}, {ci_hi:.6f})")
