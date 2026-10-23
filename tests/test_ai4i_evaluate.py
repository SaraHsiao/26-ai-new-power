"""AI4I 評估視覺化、可解釋性與誤判分析的測試（先寫測試，再實作）。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

import ai4i_evaluate as ae
import ai4i_features as f
import ai4i_train as tr

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
FIGURES = [
    "ai4i_pr_curve.png",
    "ai4i_roc_curve.png",
    "ai4i_confusion.png",
    "ai4i_calibration.png",
    "ai4i_importance.png",
    "ai4i_physical.png",
    "ai4i_mode_recall.png",
]


# ---------- 指標與曲線 ----------

def test_confusion_counts_known_values():
    y = np.array([1, 1, 0, 0, 1, 0])
    pred = np.array([1, 0, 1, 0, 1, 0])
    c = ae.confusion_counts(y, pred)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 2)
    assert c["precision"] == pytest.approx(2 / 3)
    assert c["recall"] == pytest.approx(2 / 3)
    assert c["specificity"] == pytest.approx(2 / 3)
    assert c["accuracy"] == pytest.approx(4 / 6)


def test_confusion_counts_validates_length():
    with pytest.raises(ValueError):
        ae.confusion_counts(np.array([0, 1]), np.array([0]))


def test_precision_recall_summary():
    y = np.array([1, 1, 0, 0, 1, 0])
    proba = np.array([0.9, 0.4, 0.6, 0.1, 0.8, 0.2])
    out = ae.precision_recall_summary(y, proba)
    assert out["ap"] == pytest.approx(average_precision_score(y, proba))
    assert len(out["precision"]) == len(out["recall"])
    assert out["precision"].shape == out["recall"].shape
    assert 0 <= out["ap"] <= 1


def test_roc_summary():
    y = np.array([1, 1, 0, 0, 1, 0])
    proba = np.array([0.9, 0.4, 0.6, 0.1, 0.8, 0.2])
    out = ae.roc_summary(y, proba)
    assert out["auc"] == pytest.approx(roc_auc_score(y, proba))
    assert out["fpr"].shape == out["tpr"].shape


def test_calibration_summary():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(0.5 * y + rng.random(500) * 0.5, 0, 1)
    out = ae.calibration_summary(y, proba, n_bins=8)
    assert len(out["prob_pred"]) == len(out["prob_true"])
    assert out["counts"].sum() == len(y)
    assert out["brier"] == pytest.approx(np.mean((proba - y) ** 2))
    assert np.all((out["prob_true"] >= 0) & (out["prob_true"] <= 1))


# ---------- 可解釋性 ----------

def test_permutation_importance_summary():
    X, y = f.load_dataset()
    X, y = X.iloc[:400], y.iloc[:400]
    model = tr.build_model_pipeline("logistic", "none").fit(X, y)
    out = ae.permutation_importance_summary(model, X, y, n_repeats=2, random_state=42)
    assert len(out) == 6
    assert [d["feature"] for d in out] == sorted(
        [d["feature"] for d in out], key=lambda name: -next(
            d["mean"] for d in out if d["feature"] == name
        )
    )
    assert {d["feature"] for d in out} == set(f.BASE_FEATURES)
    assert all(d["std"] >= 0 for d in out)


# ---------- 失效模式 recall ----------

def test_per_mode_recall_synthetic():
    raw = pd.DataFrame({
        "UDI": [1, 2, 3, 4, 5, 6],
        "Type": ["L"] * 6,
        "Machine failure": [1, 1, 1, 0, 0, 1],
        "TWF": [1, 0, 0, 0, 0, 0],
        "HDF": [1, 1, 0, 0, 0, 0],
        "PWF": [0, 0, 1, 0, 0, 0],
        "OSF": [0, 0, 0, 0, 0, 0],
        "RNF": [0, 0, 0, 0, 0, 0],
    })
    pred = np.array([1, 0, 0, 0, 1, 0])
    out = ae.per_mode_recall(raw, pred)
    assert out["HDF"]["n"] == 2
    assert out["HDF"]["n_mf"] == 2
    assert out["HDF"]["predicted"] == 1
    assert out["HDF"]["recall"] == pytest.approx(0.5)
    assert out["HDF"]["recall_mf"] == pytest.approx(0.5)
    assert out["TWF"]["recall"] == pytest.approx(1.0)
    assert out["PWF"]["recall"] == pytest.approx(0.0)
    assert out["OSF"]["n"] == 0
    assert out["OSF"]["recall"] is None
    assert out["OSF"]["recall_mf"] is None
    assert out["RNF"]["n"] == 0


# ---------- 誤判分析 ----------

def test_build_error_cases_keeps_only_errors():
    raw = pd.DataFrame({
        "UDI": [1, 2, 3, 4],
        "Type": ["L", "M", "H", "L"],
        "Air temperature [K]": [298.0, 299.0, 300.0, 301.0],
        "Process temperature [K]": [308.0, 309.0, 310.0, 311.0],
        "Rotational speed [rpm]": [1500, 1400, 1300, 1200],
        "Torque [Nm]": [40.0, 45.0, 50.0, 55.0],
        "Tool wear [min]": [10, 20, 30, 40],
        "Machine failure": [1, 0, 1, 0],
        "TWF": [0, 0, 1, 0],
        "HDF": [0, 0, 0, 0],
        "PWF": [0, 0, 0, 0],
        "OSF": [0, 0, 0, 0],
        "RNF": [0, 0, 0, 0],
    })
    y = raw["Machine failure"].to_numpy()
    proba = np.array([0.2, 0.9, 0.7, 0.1])
    pred = np.array([0, 1, 1, 0])
    errors = ae.build_error_cases(raw, y, proba, pred, threshold=0.5)
    assert set(errors["error_type"]) == {"FN", "FP"}
    assert len(errors) == 2
    assert (errors["error_type"] == "FN").sum() == 1
    assert (errors["error_type"] == "FP").sum() == 1
    for col in ("UDI", "proba", "pred", "y_true", "Temp diff [K]",
                "Power [W]", "Torque x Tool wear"):
        assert col in errors.columns


def test_error_summary_counts_and_patterns():
    raw = pd.DataFrame({
        "UDI": [1, 2, 3, 4],
        "Type": ["L", "M", "H", "L"],
        "Air temperature [K]": [298.0, 299.0, 300.0, 301.0],
        "Process temperature [K]": [308.0, 309.0, 310.0, 311.0],
        "Rotational speed [rpm]": [1500, 1400, 1300, 1200],
        "Torque [Nm]": [40.0, 45.0, 50.0, 55.0],
        "Tool wear [min]": [10, 20, 30, 40],
        "Machine failure": [1, 0, 1, 0],
        "TWF": [0, 0, 1, 0],
        "HDF": [0, 0, 0, 0],
        "PWF": [0, 0, 0, 0],
        "OSF": [0, 0, 0, 0],
        "RNF": [0, 0, 0, 0],
    })
    y = raw["Machine failure"].to_numpy()
    proba = np.array([0.2, 0.9, 0.7, 0.1])
    pred = np.array([0, 1, 1, 0])
    errors = ae.build_error_cases(raw, y, proba, pred)
    summary = ae.error_summary(errors, raw, y, pred)
    assert summary["counts"] == {"FN": 1, "FP": 1}
    assert "feature_comparison" in summary
    assert "FN" in summary["modes"]
    assert summary["rates"]["fnr"] == pytest.approx(0.5)
    assert summary["rates"]["fpr"] == pytest.approx(0.5)


# ---------- 真實產出 ----------

def test_load_context_alignment():
    ctx = ae.load_context()
    assert len(ctx["y_test"]) == 2000
    assert int(ctx["y_test"].sum()) == 68
    saved = pd.read_csv(RESULTS / "test_predictions.csv", encoding="utf-8")
    np.testing.assert_allclose(ctx["proba"], saved["proba"].to_numpy(), rtol=1e-9)
    np.testing.assert_array_equal(ctx["pred"], saved["pred"].to_numpy())
    # raw_test 的順序必須與預測一致
    assert list(ctx["raw_test"]["UDI"]) == list(
        pd.read_csv(ROOT / "data" / "ai4i2020.csv", encoding="utf-8")
        .iloc[ctx["raw_test"].index]["UDI"]
    )


def test_metrics_artifact_consistency():
    path = RESULTS / "ai4i_metrics.json"
    assert path.exists(), "缺少 results/ai4i_metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    test_metrics = json.loads((RESULTS / "test_metrics.json").read_text(encoding="utf-8"))
    assert data["test"]["pr_auc"] == pytest.approx(test_metrics["metrics"]["pr_auc"])
    assert data["test"]["recall"] == pytest.approx(test_metrics["metrics"]["recall"])
    assert set(data["per_mode_recall"]) == {"TWF", "HDF", "PWF", "OSF", "RNF"}
    assert len(data["permutation_importance"]) == 6
    assert data["curves"]["roc_auc"] >= 0.5
    assert data["figures"]
    # RNF 旗標在測試集全部不是 Machine failure，不能拿來估計 recall 上限
    rnf = data["per_mode_recall"]["RNF"]
    assert rnf["n"] == 4
    assert rnf["n_mf"] == 0
    assert rnf["recall_mf"] is None
    assert data["unpredictable_failures"] == 2
    assert data["recall_ceiling"] == pytest.approx(1 - 2 / 68)


def test_report_discloses_twf_weakness_and_rnf_quirk():
    text = (DOCS / "evaluation_figures.md").read_text(encoding="utf-8")
    assert "TWF" in text
    assert "0.20" in text
    assert "RNF 旗標" in text
    assert "無模式" in text
    assert "已接近此上限" not in text
    assert "HDF 與 TWF 的 recall 最高" not in text


def test_error_cases_artifact():
    path = RESULTS / "error_cases.csv"
    assert path.exists(), "缺少 results/error_cases.csv"
    df = pd.read_csv(path, encoding="utf-8")
    assert set(df["error_type"]) <= {"FN", "FP"}
    assert len(df) > 0
    assert set(df["error_type"]) == {"FN", "FP"}
    for col in ("UDI", "proba", "pred", "y_true"):
        assert col in df.columns


def test_figures_exist_and_are_non_trivial():
    for name in FIGURES:
        path = FIG_DIR / name
        assert path.exists(), f"缺少圖表 {name}"
        assert path.stat().st_size > 10_000, f"{name} 檔案過小"


def test_report_mentions_figures_and_findings():
    path = DOCS / "evaluation_figures.md"
    assert path.exists(), "缺少 docs/evaluation_figures.md"
    text = path.read_text(encoding="utf-8")
    for token in FIGURES:
        assert token in text, token
    for token in ("PR", "ROC", "RNF", "permutation", "誤判", "recall"):
        assert token in text, token
