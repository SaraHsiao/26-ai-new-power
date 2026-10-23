"""AI4I 前處理與特徵工程的測試（先寫測試，再實作）。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

import ai4i_features as f

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "features.md"


# ---------- load_dataset ----------

def test_load_dataset_shape_and_columns():
    X, y = f.load_dataset()
    assert X.shape == (10000, 6)
    assert list(X.columns) == ["Type"] + f.SENSOR_COLUMNS
    assert len(y) == 10000
    assert int(y.sum()) == 339


def test_load_dataset_excludes_leakage_ids_and_target():
    X, _ = f.load_dataset()
    for col in f.FORBIDDEN_IN_X:
        assert col not in X.columns, f"{col} 不應出現在特徵中"
    f.assert_no_leakage(X.columns)  # 不應 raise


def test_load_dataset_returns_new_objects():
    X1, y1 = f.load_dataset()
    X2, y2 = f.load_dataset()
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


# ---------- 洩漏防護 ----------

@pytest.mark.parametrize("bad", ["TWF", "HDF", "PWF", "OSF", "RNF", "UDI",
                                 "Product ID", "Machine failure"])
def test_assert_no_leakage_rejects_forbidden_columns(bad):
    with pytest.raises(ValueError):
        f.assert_no_leakage(["Type", "Torque [Nm]", bad])


def test_forbidden_list_is_complete():
    assert set(f.LEAKAGE_COLUMNS) == {"TWF", "HDF", "PWF", "OSF", "RNF"}
    assert set(f.ID_COLUMNS) == {"UDI", "Product ID"}


# ---------- 特徵工程 ----------

def test_add_features_creates_expected_columns():
    X, _ = f.load_dataset()
    out = f.add_features(X)
    assert set(f.ENGINEERED_FEATURES) <= set(out.columns)
    assert out.shape[1] == X.shape[1] + len(f.ENGINEERED_FEATURES)


def test_add_features_formulas():
    X, _ = f.load_dataset()
    out = f.add_features(X)
    expected_power = out["Torque [Nm]"] * 2 * np.pi * out["Rotational speed [rpm]"] / 60
    expected_diff = out["Process temperature [K]"] - out["Air temperature [K]"]
    expected_strain = out["Torque [Nm]"] * out["Tool wear [min]"]
    assert np.allclose(out["Power [W]"], expected_power)
    assert np.allclose(out["Temp diff [K]"], expected_diff)
    assert np.allclose(out["Torque x Tool wear"], expected_strain)


def test_add_features_does_not_mutate_input():
    X, _ = f.load_dataset()
    before = X.copy()
    f.add_features(X)
    pd.testing.assert_frame_equal(X, before)


def test_add_features_rejects_leakage_columns():
    X, _ = f.load_dataset()
    bad = X.copy()
    bad["TWF"] = 0
    with pytest.raises(ValueError):
        f.add_features(bad)


# ---------- Preprocessor ----------

def test_preprocessor_shape_and_no_nan():
    X, _ = f.load_dataset()
    prep = f.build_preprocessor()
    Z = prep.fit_transform(X)
    assert Z.shape == (10000, 11)  # 5 感測 + 3 工程 + 3 one-hot
    assert not np.isnan(Z).any()


def test_preprocessor_scales_numeric_block_only():
    X, _ = f.load_dataset()
    Z = f.build_preprocessor(scale=True).fit_transform(X)
    numeric = Z[:, :8]
    assert np.allclose(numeric.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(numeric.std(axis=0), 1, atol=1e-6)

    onehot = Z[:, 8:]
    assert set(np.unique(onehot)) <= {0.0, 1.0}
    assert np.allclose(onehot.sum(axis=1), 1)


def test_preprocessor_scale_false_keeps_units():
    X, _ = f.load_dataset()
    Z = f.build_preprocessor(scale=False).fit_transform(X)
    air = Z[:, 0]
    assert 290 < air.mean() < 310  # 未標準化，仍是 K


def test_preprocessor_without_engineering():
    X, _ = f.load_dataset()
    Z = f.build_preprocessor(add_engineered=False).fit_transform(X)
    assert Z.shape == (10000, 8)  # 5 感測 + 3 one-hot


def test_preprocessor_drop_columns():
    X, _ = f.load_dataset()
    prep = f.build_preprocessor(drop_columns=["Power [W]"])
    Z = prep.fit_transform(X)
    assert Z.shape == (10000, 10)
    names = list(prep.get_feature_names_out())
    assert "Power [W]" not in names


def test_preprocessor_is_deterministic():
    X, _ = f.load_dataset()
    a = f.build_preprocessor().fit_transform(X)
    b = f.build_preprocessor().fit_transform(X)
    np.testing.assert_array_equal(a, b)


def test_feature_names():
    names = f.get_feature_names()
    assert len(names) == 11
    assert {"Power [W]", "Temp diff [K]", "Torque x Tool wear"} <= set(names)
    assert {"Type_L", "Type_M", "Type_H"} <= set(names)
    for col in f.SENSOR_COLUMNS:
        assert col in names


def test_unknown_type_does_not_crash():
    X, _ = f.load_dataset()
    prep = f.build_preprocessor().fit(X)
    odd = X.head(5).copy()
    odd.loc[odd.index[0], "Type"] = "XL"
    Z = prep.transform(odd)
    assert Z.shape == (5, 11)
    assert not np.isnan(Z).any()


# ---------- Pipeline 可用性 ----------

def test_make_pipeline_returns_pipeline_with_preprocessor():
    from sklearn.pipeline import Pipeline

    pipe = f.make_pipeline(LogisticRegression(max_iter=1000))
    assert isinstance(pipe, Pipeline)
    assert "prep" in pipe.named_steps
    assert "model" in pipe.named_steps


def test_pipeline_works_with_cross_val_score():
    X, y = f.load_dataset()
    pipe = f.make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision")
    assert scores.shape == (3,)
    assert np.all(scores >= 0) and np.all(scores <= 1)
    # 必須明顯優於隨機（故障率 0.0339）
    assert scores.mean() > 0.3


def test_preprocessor_fit_on_train_does_not_see_test():
    """標準化參數只由訓練資料決定。"""
    X, _ = f.load_dataset()
    numeric = f.SENSOR_COLUMNS + f.ENGINEERED_FEATURES
    prep = f.build_preprocessor().fit(X.iloc[:100])
    scaler = prep.named_steps["columns"].named_transformers_["num"].named_steps["scaler"]
    expected_train = f.add_features(X.iloc[:100])[numeric].mean().values
    expected_all = f.add_features(X)[numeric].mean().values
    assert np.allclose(scaler.mean_, expected_train)
    assert not np.allclose(scaler.mean_, expected_all)


# ---------- 文件 ----------

def test_features_doc_exists_and_covers_policy():
    assert DOCS.exists(), "缺少 docs/features.md"
    text = DOCS.read_text(encoding="utf-8")
    for token in ("Power [W]", "Temp diff [K]", "Torque x Tool wear",
                  "TWF", "洩漏", "one-hot", "cross_val_score"):
        assert token in text, token
