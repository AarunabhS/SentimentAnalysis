"""Small, leakage-aware evaluation helpers used by this standalone project."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

SEED = 42


def binary_labels(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="raise")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError(f"{name} must contain explicit 0/1 labels; missing labels are not negatives.")
    if values.nunique() != 2 or values.value_counts().min() < 10:
        raise ValueError(f"{name} needs both classes with at least 10 examples each for evaluation.")
    return values.astype(int)


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def numeric_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    require_columns(frame, columns)
    result = frame[columns].apply(pd.to_numeric, errors="raise")
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError("Numeric features must not contain infinity.")
    if result.isna().all().any():
        raise ValueError("A required feature is entirely missing.")
    return result


def make_splits(y: pd.Series, groups: pd.Series | None = None) -> dict[str, np.ndarray]:
    """60/20/20 split; keep known people and repeated contexts together."""
    indices = np.arange(len(y))
    if groups is None:
        train, holdout = train_test_split(indices, test_size=0.4, stratify=y, random_state=SEED)
        validation, test = train_test_split(
            holdout, test_size=0.5, stratify=y.iloc[holdout], random_state=SEED
        )
    else:
        train, holdout = next(GroupShuffleSplit(
            n_splits=1, test_size=0.4, random_state=SEED
        ).split(indices, y, groups))
        a, b = next(GroupShuffleSplit(
            n_splits=1, test_size=0.5, random_state=SEED
        ).split(holdout, y.iloc[holdout], groups.iloc[holdout]))
        validation, test = holdout[a], holdout[b]
    result = {"train": train, "validation": validation, "test": test}
    for name, rows in result.items():
        if y.iloc[rows].nunique() != 2:
            raise ValueError(f"The {name} split does not contain both classes; use more data.")
    if groups is not None:
        group_sets = [set(groups.iloc[v]) for v in result.values()]
        assert not any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i))
    return result


def choose_threshold(y: pd.Series, scores: np.ndarray, beta: float = 1.0) -> float:
    """Maximize F-beta on validation only; ties prefer fewer alerts."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if len(thresholds) == 0:
        return 0.5
    b = beta ** 2
    objective = (1 + b) * precision[:-1] * recall[:-1] / np.maximum(
        b * precision[:-1] + recall[:-1], 1e-15
    )
    tied = np.flatnonzero(np.isclose(objective, objective.max(), rtol=0, atol=1e-12))
    return float(thresholds[tied[-1]])


def measure(y: pd.Series, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "log_loss": float(log_loss(y, np.clip(scores, 1e-12, 1 - 1e-12), labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, scores)),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "threshold": threshold, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "alerts": int(predictions.sum()), "positive_rate": float(y.mean()), "rows": len(y),
    }


def json_write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_experiment(X, y, models, output: Path, metadata: dict, *, splits=None, beta=1.0,
                   selection_metric="average_precision"):
    """Select using validation, then evaluate fixed candidates on untouched test data."""
    output.mkdir(parents=True, exist_ok=True)
    splits = make_splits(y) if splits is None else splits
    train, validation, test = (splits[k] for k in ["train", "validation", "test"])
    fitted, validation_scores, rows = {}, {}, []
    for name, model in models.items():
        model.fit(X.iloc[train], y.iloc[train])
        scores = model.predict_proba(X.iloc[validation])[:, 1]
        fitted[name], validation_scores[name] = model, scores
        threshold = 0.5 if name == "baseline" else choose_threshold(y.iloc[validation], scores, beta)
        rows.append({"model": name, "split": "validation", **measure(y.iloc[validation], scores, threshold)})
    candidates = [r for r in rows if r["model"] != "baseline"]
    if selection_metric == "log_loss":
        chosen = min(candidates, key=lambda row: row[selection_metric])
    else:
        chosen = max(candidates, key=lambda row: row[selection_metric])
    # This decision is frozen before any test labels are consulted.
    selected, threshold = chosen["model"], chosen["threshold"]
    predictions = pd.DataFrame({"row_id": test, "actual": y.iloc[test].to_numpy()})
    for name, model in fitted.items():
        scores = model.predict_proba(X.iloc[test])[:, 1]
        current_threshold = next(r["threshold"] for r in rows if r["model"] == name)
        rows.append({"model": name, "split": "test", **measure(y.iloc[test], scores, current_threshold)})
        predictions[name + "_score"] = scores
        if name == selected:
            predictions["predicted"] = (scores >= threshold).astype(int)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False, float_format="%.8f")
    predictions.to_csv(output / "predictions.csv", index=False, float_format="%.8f")
    assignments = np.empty(len(y), dtype=object)
    for name, indices in splits.items():
        assignments[indices] = name
    pd.DataFrame({"row_id": np.arange(len(y)), "split": assignments, "label": y}).to_csv(
        output / "splits.csv", index=False
    )
    summary = {
        **metadata, "seed": SEED, "selected_model": selected, "selected_threshold": threshold,
        "selection_metric": selection_metric, "threshold_objective": f"validation F{beta:g}",
        "splits": {k: {"rows": len(v), "positives": int(y.iloc[v].sum())} for k, v in splits.items()},
        "test": next(r for r in rows if r["model"] == selected and r["split"] == "test"),
        "baseline_test": next(r for r in rows if r["model"] == "baseline" and r["split"] == "test"),
        "python": platform.python_version(),
        "packages": {k: importlib.metadata.version(k) for k in ["numpy", "pandas", "scikit-learn", "matplotlib"]},
    }
    json_write(output / "metrics.json", summary)
    scores = predictions[selected + "_score"].to_numpy()
    p, r, _ = precision_recall_curve(y.iloc[test], scores)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    axes[0].step(r, p, where="post", color="#166c8b", lw=2, label=selected.replace("_", " "))
    axes[0].axhline(y.iloc[test].mean(), color="#a45a37", ls="--", label="no-skill prevalence")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Untouched test: precision vs. recall", xlim=(0, 1), ylim=(0, 1.03))
    axes[0].legend(fontsize=8)
    cm = confusion_matrix(y.iloc[test], predictions.predicted, labels=[0, 1])
    axes[1].imshow(cm, cmap="Blues")
    for (a, b), value in np.ndenumerate(cm):
        axes[1].text(b, a, str(value), ha="center", va="center", color="white" if value > cm.max()/2 else "#172635")
    class_names = metadata.get("class_names", ["0", "1"])
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=class_names, yticklabels=class_names,
                xlabel="Predicted", ylabel="Actual", title=f"Validation-selected threshold: {threshold:.3f}")
    fig.savefig(output / "evaluation.png", dpi=150)
    plt.close(fig)
    return summary, fitted[selected], predictions


def input_metadata(path: Path) -> dict:
    return {"input_file": path.name, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_report(output: Path, summary: dict, context: str, interpretation: str) -> None:
    test, baseline = summary["test"], summary["baseline_test"]
    report = f"""# {summary['title']} — reproducible results

{context}

## Evaluation design

Seed {summary['seed']}. Training, validation, and test row counts: {summary['splits']['train']['rows']:,},
{summary['splits']['validation']['rows']:,}, {summary['splits']['test']['rows']:,}.
Preprocessing is fitted inside the training pipeline. Model selection uses validation
{summary['selection_metric']}; the decision threshold maximizes {summary['threshold_objective']}.
The selected model is **{summary['selected_model']}**. No tuning uses the test labels.

## Untouched test results

| Measure | Selected model | Constant-prevalence baseline |
|---|---:|---:|
| Average precision | {test['average_precision']:.4f} | {baseline['average_precision']:.4f} |
| ROC AUC | {test['roc_auc']:.4f} | {baseline['roc_auc']:.4f} |
| Log loss (lower is better) | {test['log_loss']:.4f} | {baseline['log_loss']:.4f} |
| Brier score (lower is better) | {test['brier_score']:.4f} | {baseline['brier_score']:.4f} |
| Precision | {test['precision']:.4f} | {baseline['precision']:.4f} |
| Recall | {test['recall']:.4f} | {baseline['recall']:.4f} |
| F1 | {test['f1']:.4f} | {baseline['f1']:.4f} |
| Accuracy | {test['accuracy']:.4f} | {baseline['accuracy']:.4f} |

At threshold **{test['threshold']:.4f}**, the model produces **{test['alerts']:,}** positive flags:
**{test['tp']:,}** true positives and **{test['fp']:,}** false positives.
It misses **{test['fn']:,}** positives; **{test['tn']:,}** negatives are correctly rejected.
Test positive prevalence is **{test['positive_rate']:.2%}**, across {test['rows']:,} rows.

![Precision–recall curve and confusion matrix](evaluation.png)

## Interpretation and limits

{interpretation}

See [all candidate metrics](metrics.csv), [auditable test predictions](predictions.csv),
[split assignments](splits.csv), and [run metadata, input hash, and package versions](metrics.json).
Numbers here are generated by the script, not copied from the historical notebook.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
