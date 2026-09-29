"""Checks for split isolation, metric correctness, and validation-only decisions."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from workflow import binary_labels, choose_threshold, make_splits, measure, run_experiment


def test_unlabelled_rows_are_not_silently_negative():
    with pytest.raises(ValueError, match="explicit 0/1"):
        binary_labels(pd.Series([0]*10+[1]*10+[np.nan]), "target")


def test_repeated_people_never_cross_partitions():
    y = pd.Series([0, 1]*60)
    groups = pd.Series(np.repeat(np.arange(60), 2))
    splits = make_splits(y, groups)
    seen = [set(groups.iloc[v]) for v in splits.values()]
    assert not seen[0] & seen[1]
    assert not seen[0] & seen[2]
    assert not seen[1] & seen[2]
    assert sorted(np.concatenate(list(splits.values()))) == list(range(len(y)))


def test_all_negative_accuracy_does_not_hide_zero_recall():
    y = pd.Series([0]*99+[1])
    metric = measure(y, np.full(100, 0.01), 0.5)
    assert metric["accuracy"] == 0.99
    assert metric["recall"] == 0
    assert metric["fn"] == 1
    assert metric["average_precision"] == pytest.approx(0.01)


def test_threshold_is_selected_from_validation_scores():
    y = pd.Series([0, 0, 1, 1])
    assert choose_threshold(y, np.array([0.1, 0.2, 0.7, 0.8])) == pytest.approx(0.7)


def test_test_labels_do_not_change_model_selection(tmp_path):
    y = pd.Series([0, 1]*30)
    splits = {"train": np.arange(36), "validation": np.arange(36, 48), "test": np.arange(48, 60)}
    first = y.to_numpy()*0.8 + 0.1
    second = 1-first
    first[splits["test"]] = 1-first[splits["test"]]
    second[splits["test"]] = 1-second[splits["test"]]
    X = pd.DataFrame({"first": first, "second": second})
    class ColumnScore:
        def __init__(self, column):
            self.column = column
        def fit(self, X, y):
            return self
        def predict_proba(self, X):
            scores = X[self.column].to_numpy()
            return np.column_stack([1-scores, scores])
    def models():
        return {"baseline": DummyClassifier(strategy="prior"),
                "first": ColumnScore("first"), "second": ColumnScore("second")}
    a, _, _ = run_experiment(X, y, models(), tmp_path/"a", {"title":"fixture"}, splits=splits)
    changed = y.copy()
    changed.iloc[splits["test"]] = 1 - changed.iloc[splits["test"]]
    b, _, _ = run_experiment(X, changed, models(), tmp_path/"b", {"title":"fixture"}, splits=splits)
    assert a["selected_model"] == b["selected_model"]
    assert a["selected_model"] == "first"
    assert a["selected_threshold"] == b["selected_threshold"]
    assert a["test"]["average_precision"] < b["test"]["average_precision"]
