"""AI4I 模型建立、比較與校準的測試（先寫測試，再實作）。"""

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import ai4i_features as f
import ai4i_train as tr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs" / "model_comparison.md"


def _small_data(n=800):
    X, y = f.load_dataset()
    return X.iloc[:n], y.iloc[:n]


# ---------- 模型註冊 ----------

def test_required_models_are_registered():
    names = tr.default_model_names()
    for name in ("dummy", "logistic", "random_forest", "hist_gradient_boosting"):
        assert name in names


def test_optional_models_follow_installation():
    import importlib.util

    names = tr.default_model_names()
    for name, module in (("xgboost", "xgboost"), ("lightgbm", "lightgbm")):
        installed = importlib.util.find_spec(module) is not None
        assert (name in names) == installed


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        tr.build_model_pipeline("magic-net", "none")


def test_dummy_cannot_use_class_weight():
    with pytest.raises(ValueError):
        tr.build_model_pipeline("dummy", "class_weight")


def test_pipeline_sets_class_weight():
    pipe = tr.build_model_pipeline("random_forest", "class_weight")
    assert pipe.named_steps["model"].class_weight == "balanced"


def test_pipeline_uses_tuned_params():
    pipe = tr.build_model_pipeline(
        "random_forest", "none", params={"n_estimators": 37, "max_depth": 3}
    )
    model = pipe.named_steps["model"]
    assert model.n_estimators == 37
    assert model.max_depth == 3


# ---------- 超參數搜尋 ----------

def test_search_spaces():
    assert tr.search_space("dummy") == {}
    for name in ("logistic", "random_forest", "hist_gradient_boosting"):
        space = tr.search_space(name)
        assert space, name
        assert all(k.startswith("model__") for k in space)


def test_tune_model_returns_best_params():
    X, y = _small_data(600)
    out = tr.tune_model("hist_gradient_boosting", X, y, n_iter=2, cv=2,
                        random_state=42)
    assert out["model"] == "hist_gradient_boosting"
    assert out["n_iter"] == 2
    assert isinstance(out["best_params"], dict)
    assert set(out["best_params"]) <= set(tr.search_space("hist_gradient_boosting"))
    assert out["best_score"] is not None
    # 供建模使用的參數不含 pipeline 前綴
    assert all(not k.startswith("model__") for k in out["estimator_params"])


def test_tune_model_without_search_space_is_noop():
    X, y = _small_data(400)
    out = tr.tune_model("dummy", X, y, n_iter=2, cv=2)
    assert out["best_params"] == {}
    assert out["n_iter"] == 0


# ---------- 比較 ----------

def test_compare_models_small_matrix():
    X, y = _small_data(800)
    result = tr.compare_models(
        X, y,
        models=["dummy", "logistic"],
        strategies=("none", "class_weight"),
        n_splits=3,
        n_boot=30,
        random_state=42,
    )
    entries = result["entries"]
    # dummy 只有 none；logistic 有兩種策略
    assert len(entries) == 3
    keys = {(e["model"], e["strategy"]) for e in entries}
    assert keys == {("dummy", "none"), ("logistic", "none"),
                    ("logistic", "class_weight")}

    for e in entries:
        for key in ("pr_auc_oof", "pr_auc_fold_mean", "pr_auc_fold_std",
                    "threshold", "precision", "recall", "f1", "lift",
                    "brier", "pr_auc_ci95"):
            assert key in e, key

    assert result["best"]["model"] == "logistic"
    assert result["config"]["n_splits"] == 3
    assert result["config"]["n_train"] == 800
    assert result["best"]["pr_auc_oof"] == max(e["pr_auc_oof"] for e in entries)
    json.dumps(result)  # 必須可直接序列化


def test_compare_models_does_not_accept_test_data():
    sig = inspect.signature(tr.compare_models)
    for bad in ("X_test", "y_test", "test_size"):
        assert bad not in sig.parameters


def test_compare_models_rejects_empty_models():
    X, y = _small_data(300)
    with pytest.raises(ValueError):
        tr.compare_models(X, y, models=[])


def test_compare_models_with_tuning_including_dummy():
    """回歸測試：dummy 沒有搜尋空間，best_score 為 None 時仍能完成比較。"""
    X, y = _small_data(600)
    result = tr.compare_models(
        X, y, models=["dummy"], strategies=("none",),
        n_splits=3, n_boot=20, tune=True,
    )
    assert len(result["entries"]) == 1
    assert result["best"]["model"] == "dummy"
    assert result["tuning"]["dummy"]["best_score"] is None


# ---------- 校準 ----------

def test_brier_score():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.6, 0.3])
    expected = np.mean((p - y) ** 2)
    assert tr.brier_score(y, p) == pytest.approx(expected)


def test_brier_score_validates_length():
    with pytest.raises(ValueError):
        tr.brier_score(np.array([0, 1]), np.array([0.5]))


# ---------- 測試集一次性守門 ----------

def test_test_evaluation_runs_only_once(tmp_path):
    X, y = _small_data(600)
    pipe = tr.build_model_pipeline("logistic", "none").fit(X, y)
    bundle = {"model": pipe, "threshold": 0.5, "feature_names": f.get_feature_names()}

    out_path = tmp_path / "test_metrics.json"
    pred_path = tmp_path / "test_predictions.csv"
    first = tr.run_test_evaluation(bundle, X, y, out_path=out_path, pred_path=pred_path)
    assert out_path.exists() and pred_path.exists()
    assert first["metrics"]["n"] == len(y)
    assert first["one_shot"] is True

    with pytest.raises(RuntimeError, match="只允許評估一次"):
        tr.run_test_evaluation(bundle, X, y, out_path=out_path, pred_path=pred_path)

    second = tr.run_test_evaluation(bundle, X, y, out_path=out_path,
                                    pred_path=pred_path, force=True)
    assert second["metrics"]["n"] == len(y)


# ---------- 存檔與載入 ----------

def test_save_and_load_model_roundtrip(tmp_path):
    X, y = _small_data(600)
    pipe = tr.build_model_pipeline("logistic", "none").fit(X, y)
    bundle = {
        "model": pipe,
        "threshold": 0.42,
        "feature_names": f.get_feature_names(),
        "meta": {"model": "logistic", "strategy": "none"},
    }
    path = tmp_path / "best_model.joblib"
    tr.save_model(bundle, path)
    loaded = tr.load_model(path)
    assert loaded["threshold"] == pytest.approx(0.42)
    np.testing.assert_allclose(
        pipe.predict_proba(X)[:, 1],
        loaded["model"].predict_proba(X)[:, 1],
    )


# ---------- Markdown ----------

def test_render_markdown_contains_numbers_and_best():
    comparison = {
        "config": {"n_splits": 5, "n_train": 8000, "n_pos": 271, "tune": True},
        "entries": [
            {"model": "logistic", "strategy": "none", "pr_auc_oof": 0.45123,
             "pr_auc_fold_mean": 0.44, "pr_auc_fold_std": 0.02,
             "pr_auc_ci95": [0.39, 0.51], "threshold": 0.5, "precision": 0.3,
             "recall": 0.6, "f1": 0.4, "lift": 8.8, "brier": 0.03},
            {"model": "hist_gradient_boosting", "strategy": "class_weight",
             "pr_auc_oof": 0.81234, "pr_auc_fold_mean": 0.80,
             "pr_auc_fold_std": 0.03, "pr_auc_ci95": [0.74, 0.88],
             "threshold": 0.6, "precision": 0.7, "recall": 0.81,
             "f1": 0.75, "lift": 20.5, "brier": 0.02},
        ],
        "best": {"model": "hist_gradient_boosting", "strategy": "class_weight",
                 "pr_auc_oof": 0.81234, "threshold": 0.6},
    }
    md = tr.render_markdown(comparison, calibration=None, test_metrics=None)
    assert "0.8123" in md
    assert "hist_gradient_boosting" in md
    assert "PR-AUC" in md
    assert "測試集" in md


# ---------- 產出檔案 ----------

def test_comparison_json_artifact_meets_acceptance():
    path = RESULTS / "model_comparison.json"
    assert path.exists(), "缺少 results/model_comparison.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data["entries"]
    assert len(entries) >= 6
    assert len({e["model"] for e in entries}) >= 3
    assert len({e["strategy"] for e in entries}) >= 2
    for e in entries:
        assert e["pr_auc_fold_std"] >= 0
        assert 0 <= e["pr_auc_oof"] <= 1


def test_best_model_artifact_loads():
    path = RESULTS / "best_model.joblib"
    assert path.exists(), "缺少 results/best_model.joblib"
    bundle = tr.load_model(path)
    for key in ("model", "threshold", "feature_names", "meta"):
        assert key in bundle
    assert 0 <= bundle["threshold"] <= 1


def test_model_comparison_doc_matches_json():
    assert DOCS.exists(), "缺少 docs/model_comparison.md"
    text = DOCS.read_text(encoding="utf-8")
    data = json.loads((RESULTS / "model_comparison.json").read_text(encoding="utf-8"))
    assert data["best"]["model"] in text
    assert f"{data['best']['pr_auc_oof']:.4f}" in text
    assert "accuracy" in text.lower()


def test_test_metrics_artifact_once_only():
    path = RESULTS / "test_metrics.json"
    assert path.exists(), "缺少 results/test_metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["one_shot"] is True
    assert 0 <= data["metrics"]["pr_auc"] <= 1
    assert data["metrics"]["n"] == 2000
