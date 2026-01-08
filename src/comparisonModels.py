import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, roc_auc_score


def evaluate_binary_prob_model(y_true, p_pred, threshold: float = 0.5) -> dict:
    """Compute common metrics for binary probabilistic predictions.

    Args:
        y_true: Iterable of 0/1 labels.
        p_pred: Iterable of predicted probabilities for class 1.
        threshold: Decision threshold used for accuracy.

    Returns:
        Dict with keys: log_loss, brier, accuracy, auc.
        AUC is NaN if only one class is present.
    """
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    p_pred = np.clip(p_pred, 1e-15, 1 - 1e-15)

    metrics = {
        "log_loss": float(log_loss(y_true, p_pred, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p_pred)),
        "accuracy": float(accuracy_score(y_true, (p_pred >= threshold).astype(int))),
    }

    if len(np.unique(y_true)) == 2:
        metrics["auc"] = float(roc_auc_score(y_true, p_pred))
    else:
        metrics["auc"] = float("nan")

    return metrics


def bootstrap_ci_diff(
    y_true,
    p_a,
    p_b,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap a confidence interval of a paired metric difference.

    Returns:
        (ci_low, ci_high, mean_diff) for (metric_a - metric_b).

    Notes:
        If a re-sample makes the metric undefined (e.g., only a single class for AUC/logloss etc.),
        that sample is skipped. If too many samples are skipped, CI might be wide.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)

    n = len(y_true)
    diffs: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        try:
            diffs.append(float(metric_fn(y_true[idx], p_a[idx]) - metric_fn(y_true[idx], p_b[idx])))
        except ValueError:
            # e.g. bootstrap resample with a single class for AUC/logloss etc.
            continue

    if not diffs:
        raise ValueError("Bootstrap produced no valid samples (metric undefined on all resamples).")

    diffs_arr = np.asarray(diffs, dtype=float)
    return (
        float(np.quantile(diffs_arr, 0.025)),
        float(np.quantile(diffs_arr, 0.975)),
        float(diffs_arr.mean()),
    )


def compare_models(
    model_a: dict,
    model_b: dict,
    labels: tuple[str, str] = ("model_a", "model_b"),
    report_out=None,
    y_true=None,
    p_a=None,
    p_b=None,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """High-level helper used by the CLI.

    Assumptions (lightweight):
    - If y_true/p_a/p_b are given, we compare on them directly.
    - Otherwise, this function expects the caller to supply already aligned arrays.
      (In this repo, wiring the real dataset is handled elsewhere.)
    """

    if y_true is None or p_a is None or p_b is None:
        raise ValueError(
            "compare_models requires y_true, p_a, p_b arrays. "
            "Provide them from your evaluation dataset before calling."
        )

    y_true = np.asarray(y_true).astype(int)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)

    m_a = evaluate_binary_prob_model(y_true, p_a)
    m_b = evaluate_binary_prob_model(y_true, p_b)

    # Bootstrap logloss diff (robust to one-class resamples via labels=[0,1]).
    ci_lo, ci_hi, mean_diff = bootstrap_ci_diff(
        y_true,
        p_a,
        p_b,
        metric_fn=lambda y, p: log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]),
        n_boot=n_boot,
        seed=seed,
    )

    report = {
        "model1": {"label": labels[0], "metrics": m_a},
        "model2": {"label": labels[1], "metrics": m_b},
        "log_loss_diff_mean": mean_diff,
        "log_loss_diff_ci95": [ci_lo, ci_hi],
        "n": int(len(y_true)),
    }

    if report_out is not None:
        from pathlib import Path
        import json

        out = Path(report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


if __name__ == "__main__":
    # This file is meant to be imported by the CLI.
    # Keep the module import side-effect free.
    pass
