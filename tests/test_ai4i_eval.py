"""AI4I 評估協定的測試（先寫測試，再實作）。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import ai4i_eval as ev
import ai4i_features as f

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "evaluation_protocol.md"


# ---------- 測試資料 ----------

def _synthetic(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    y[:5] = 1
    y[5:10] = 0
    proba = np.where(y == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))
    return y, proba


# ---------- OOF 收集 ----------

def test_collect_oof_shapes_and_coverage():
    X, y = f.load_dataset()
    X, y = X.iloc[:3000], y.iloc[:3000]
    oof = ev.collect_oof(
        f.make_pipeline(LogisticRegression(max_iter=300, class_weight="balanced")),
        X, y, n_splits=3, random_state=42,
    )
    assert isinstance(oof, ev.OOFResult)
    assert oof.proba.shape == (len(y),)
    assert oof.y.shape == (len(y),)
    assert np.array_equal(oof.y, y.to_numpy())
    assert (oof.proba >= 0).all() and (oof.proba <= 1).all()
    # 每一筆都被某一個 fold 預測到，且剛好一次
    assert (oof.folds >= 0).all()
    counts = np.bincount(oof.folds)
    assert counts.sum() == len(y)
    assert counts.min() > 0.9 * len(y) / 3


def test_collect_oof_is_deterministic():
    X, y = f.load_dataset()
    X, y = X.iloc[:2000], y.iloc[:2000]
    pipe = lambda: f.make_pipeline(LogisticRegression(max_iter=300))
    a = ev.collect_oof(pipe(), X, y, n_splits=3, random_state=42)
    b = ev.collect_oof(pipe(), X, y, n_splits=3, random_state=42)
    np.testing.assert_array_equal(a.proba, b.proba)
    np.testing.assert_array_equal(a.folds, b.folds)


def test_collect_oof_rejects_bad_n_splits():
    X, y = f.load_dataset()
    X, y = X.iloc[:300], y.iloc[:300]
    with pytest.raises(ValueError):
        ev.collect_oof(f.make_pipeline(LogisticRegression()), X, y, n_splits=1)


# ---------- 閾值選擇（只准 OOF） ----------

def test_choose_threshold_rejects_raw_arrays():
    y, proba = _synthetic(200)
    with pytest.raises(TypeError, match="OOF"):
        ev.choose_threshold(proba, y)
    with pytest.raises(TypeError, match="OOF"):
        ev.choose_threshold(np.column_stack([1 - proba, proba]), y)


def test_choose_threshold_returns_metrics_and_cost():
    y, proba = _synthetic()
    oof = ev.OOFResult(proba=proba, y=y, folds=np.zeros(len(y), dtype=int))
    choice = ev.choose_threshold(oof, cost_fn=10.0, cost_fp=1.0)
    assert 0 <= choice.threshold <= 1
    assert choice.cost > 0
    assert choice.cost_fn == 10.0 and choice.cost_fp == 1.0
    assert choice.source == "oof"
    for key in ("precision", "recall", "f1", "pr_auc", "lift", "mcc"):
        assert key in choice.metrics


def test_choose_threshold_minimizes_weighted_cost():
    y, proba = _synthetic()
    oof = ev.OOFResult(proba=proba, y=y, folds=np.zeros(len(y), dtype=int))
    choice = ev.choose_threshold(oof, cost_fn=10.0, cost_fp=1.0)

    best = np.inf
    for t in np.unique(proba):
        pred = proba >= t
        fn = int(((y == 1) & ~pred).sum())
        fp = int(((y == 0) & pred).sum())
        best = min(best, 10.0 * fn + fp)
    assert abs(choice.cost - best) < 1e-6


def test_higher_false_negative_cost_increases_recall():
    y, proba = _synthetic()
    oof = ev.OOFResult(proba=proba, y=y, folds=np.zeros(len(y), dtype=int))
    cheap = ev.choose_threshold(oof, cost_fn=1.0, cost_fp=1.0)
    expensive = ev.choose_threshold(oof, cost_fn=50.0, cost_fp=1.0)
    assert expensive.metrics["recall"] >= cheap.metrics["recall"] - 1e-9


def test_choose_threshold_validates_costs():
    y, proba = _synthetic(100)
    oof = ev.OOFResult(proba=proba, y=y, folds=np.zeros(len(y), dtype=int))
    with pytest.raises(ValueError):
        ev.choose_threshold(oof, cost_fn=0)
    with pytest.raises(ValueError):
        ev.choose_threshold(oof, cost_fn=10, cost_fp=-1)


def test_oof_result_validates_lengths():
    with pytest.raises(ValueError):
        ev.OOFResult(proba=np.zeros(3), y=np.zeros(4), folds=np.zeros(3))


# ---------- 指標 ----------

def test_evaluate_known_confusion_values():
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.4, 0.6, 0.1])
    m = ev.evaluate(y, proba, threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["mcc"] == pytest.approx(0.0)
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, proba))
    assert m["primary_metric"] == "pr_auc"


def test_evaluate_lift_uses_prevalence():
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.4, 0.6, 0.1])
    m = ev.evaluate(y, proba, threshold=0.5)
    assert m["prevalence"] == pytest.approx(0.5)
    assert m["lift"] == pytest.approx(1.0)


def test_evaluate_handles_degenerate_predictions():
    y = np.array([0, 0, 1, 1])
    m = ev.evaluate(y, np.zeros(4), threshold=0.5)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert np.isfinite(m["mcc"])


def test_evaluate_validates_inputs():
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        ev.evaluate(y, np.array([0.1]), threshold=0.5)
    with pytest.raises(ValueError):
        ev.evaluate(y, np.array([0.1, 0.9]), threshold=1.5)
    with pytest.raises(ValueError):
        ev.evaluate(np.array([0, 2]), np.array([0.1, 0.9]), threshold=0.5)


# ---------- Bootstrap ----------

def test_bootstrap_ci_bounds_and_point():
    y, proba = _synthetic(1500)
    res = ev.bootstrap_ci(y, proba, metric="recall", threshold=0.5,
                          n_boot=300, random_state=42)
    assert res["metric"] == "recall"
    assert 0 <= res["lower"] <= res["upper"] <= 1
    assert res["lower"] <= res["point"] <= res["upper"]
    assert res["n_boot"] == 300


def test_bootstrap_ci_is_deterministic():
    y, proba = _synthetic(500)
    a = ev.bootstrap_ci(y, proba, metric="precision", n_boot=100, random_state=7)
    b = ev.bootstrap_ci(y, proba, metric="precision", n_boot=100, random_state=7)
    assert a == b


def test_bootstrap_ci_perfect_predictions():
    y = np.array([0] * 50 + [1] * 50)
    proba = y.astype(float)
    res = ev.bootstrap_ci(y, proba, metric="recall", threshold=0.5,
                          n_boot=200, random_state=0)
    assert res["point"] == 1.0
    assert res["lower"] == 1.0 and res["upper"] == 1.0


def test_bootstrap_ci_all_wrong_predictions():
    y = np.array([0] * 50 + [1] * 50)
    proba = 1.0 - y
    res = ev.bootstrap_ci(y, proba, metric="recall", threshold=0.5,
                          n_boot=200, random_state=0)
    assert res["point"] == 0.0
    assert res["lower"] == 0.0 and res["upper"] == 0.0


def test_bootstrap_ci_supports_pr_auc():
    y, proba = _synthetic(1200)
    res = ev.bootstrap_ci(y, proba, metric="pr_auc", n_boot=100, random_state=0)
    assert np.isfinite(res["point"])
    assert res["lower"] <= res["point"] <= res["upper"]


def test_bootstrap_ci_rejects_unknown_metric():
    y, proba = _synthetic(100)
    with pytest.raises(ValueError):
        ev.bootstrap_ci(y, proba, metric="accuracy_v2")


# ---------- 不平衡策略 ----------

def test_strategies_available_and_none():
    assert "none" in ev.STRATEGIES
    assert "class_weight" in ev.STRATEGIES
    pipe = ev.build_imbalance_pipeline("none", LogisticRegression(max_iter=200))
    assert pipe.named_steps["model"].get_params().get("class_weight") is None


def test_class_weight_strategy_sets_balanced():
    pipe = ev.build_imbalance_pipeline("class_weight", LogisticRegression(max_iter=200))
    assert pipe.named_steps["model"].class_weight == "balanced"


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        ev.build_imbalance_pipeline("upsample-everything", LogisticRegression())


def test_smote_optional_dependency_behavior():
    if ev.smote_available():
        pipe = ev.build_imbalance_pipeline("smote", LogisticRegression(max_iter=200))
        assert "smote" in pipe.named_steps
    else:
        with pytest.raises(ImportError, match="imbalanced-learn"):
            ev.build_imbalance_pipeline("smote", LogisticRegression(max_iter=200))
        assert "smote" not in ev.strategies_available()


# ---------- 基線 ----------

def test_dummy_baseline_metrics():
    y = np.array([1] * 30 + [0] * 970)
    base = ev.dummy_baseline(y)
    assert base["pr_auc"] == pytest.approx(0.03)
    assert base["precision"] == pytest.approx(0.03)
    assert base["recall"] == 1.0
    assert base["accuracy"] == pytest.approx(0.03)


# ---------- 文件 ----------

def test_evaluation_protocol_doc_covers_required_policy():
    assert DOCS.exists(), "缺少 docs/evaluation_protocol.md"
    text = DOCS.read_text(encoding="utf-8")
    for token in ("accuracy", "96.6", "OOF", "bootstrap", "10 : 1",
                  "SMOTE", "PR-AUC", "測試集只評估一次", "choose_threshold"):
        assert token in text, token
