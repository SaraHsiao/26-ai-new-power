"""AI4I 2020 前處理與特徵工程。

對外 API：
    load_dataset()                           -> (X, y)
    add_features(X)                          -> DataFrame（加上 3 個物理特徵）
    build_preprocessor()                     -> Pipeline（特徵工程 + 編碼 + 標準化）
    make_pipeline(estimator)                 -> Pipeline（前處理 + 模型，可直接 cross_val_score）
    get_feature_names()                      -> list[str]
    assert_no_leakage(columns)               -> None（違規時 raise ValueError）

設計原則：**標籤洩漏必須由程式擋住**，不依賴人工檢查。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ai4i_data_prep import load_raw

TARGET = "Machine failure"

ID_COLUMNS = ["UDI", "Product ID"]
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FORBIDDEN_IN_X = ID_COLUMNS + LEAKAGE_COLUMNS + [TARGET]

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
ENGINEERED_FEATURES = ["Temp diff [K]", "Power [W]", "Torque x Tool wear"]
BASE_FEATURES = ["Type"] + SENSOR_COLUMNS
TYPE_CATEGORIES = ["H", "L", "M"]  # 固定順序，確保輸出欄名與結果可重現

TWO_PI = 2 * np.pi


# ---------------------------------------------------------------- 洩漏防護

def assert_no_leakage(columns: Iterable[str]) -> None:
    """特徵欄位中若出現識別碼、失效旗標或目標變數，立即 raise。"""
    bad = sorted(set(columns) & set(FORBIDDEN_IN_X))
    if bad:
        raise ValueError(
            f"偵測到不允許作為特徵的欄位：{bad}。"
            "UDI / Product ID 為識別碼；TWF/HDF/PWF/OSF/RNF 會洩漏標籤；"
            "Machine failure 為預測目標。"
        )


# ---------------------------------------------------------------- 資料載入

def load_dataset(path=None) -> tuple[pd.DataFrame, pd.Series]:
    """讀取原始資料，回傳 (X, y)；X 只含合法特徵欄位。"""
    df = load_raw(path)
    X = df[BASE_FEATURES].copy()
    y = df[TARGET].copy()
    assert_no_leakage(X.columns)
    return X, y


# ---------------------------------------------------------------- 特徵工程

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """加入物理特徵：溫差、功率、扭矩×磨耗。回傳新物件，不改動輸入。"""
    assert_no_leakage(df.columns)
    missing = [c for c in SENSOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少感測欄位：{missing}")

    out = df.copy()
    out["Temp diff [K]"] = out["Process temperature [K]"] - out["Air temperature [K]"]
    out["Power [W]"] = out["Torque [Nm]"] * TWO_PI * out["Rotational speed [rpm]"] / 60
    out["Torque x Tool wear"] = out["Torque [Nm]"] * out["Tool wear [min]"]
    return out


def _engineered_names(transformer, input_features) -> list[str]:
    return list(input_features) + ENGINEERED_FEATURES


# ---------------------------------------------------------------- Pipeline

def _numeric_columns(add_engineered: bool, drop_columns: Sequence[str]) -> list[str]:
    cols = list(SENSOR_COLUMNS) + (list(ENGINEERED_FEATURES) if add_engineered else [])
    unknown = [c for c in drop_columns if c not in cols]
    if unknown:
        raise ValueError(f"drop_columns 含有不存在的特徵：{unknown}")
    return [c for c in cols if c not in set(drop_columns)]


def build_preprocessor(
    scale: bool = True,
    add_engineered: bool = True,
    drop_columns: Sequence[str] = (),
) -> Pipeline:
    """建立前處理 Pipeline：特徵工程 → one-hot → （可選）標準化。"""
    numeric = _numeric_columns(add_engineered, drop_columns)

    feature_step = FunctionTransformer(
        add_features if add_engineered else None,
        feature_names_out=_engineered_names if add_engineered else "one-to-one",
        validate=False,
    )
    num_pipeline = Pipeline([("scaler", StandardScaler())]) if scale else "passthrough"
    columns_step = ColumnTransformer(
        [
            ("num", num_pipeline, numeric),
            ("cat", OneHotEncoder(
                categories=[TYPE_CATEGORIES],
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float64,
            ), ["Type"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("features", feature_step), ("columns", columns_step)])


def make_pipeline(estimator, **preprocessor_kwargs) -> Pipeline:
    """前處理 + 模型。可直接放進 cross_val_score，避免 CV 洩漏。"""
    return Pipeline([
        ("prep", build_preprocessor(**preprocessor_kwargs)),
        ("model", estimator),
    ])


def get_feature_names(
    add_engineered: bool = True,
    drop_columns: Sequence[str] = (),
) -> list[str]:
    """回傳轉換後的欄位名稱（順序與 Pipeline 輸出一致）。"""
    numeric = _numeric_columns(add_engineered, list(drop_columns))
    return numeric + [f"Type_{c}" for c in TYPE_CATEGORIES]
