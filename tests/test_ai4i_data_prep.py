"""AI4I 2020 資料取得與 EDA 的測試（先寫測試，再實作）。"""

import json
from pathlib import Path

import pandas as pd
import pytest

import ai4i_data_prep as prep

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ai4i2020.csv"
PROV = ROOT / "data" / "PROVENANCE.json"
REPORT = ROOT / "docs" / "eda_report_ai4i.md"
FIG_DIR = ROOT / "figures"

COLUMNS = [
    "UDI", "Product ID", "Type", "Air temperature [K]",
    "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
    "Tool wear [min]", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",
]
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FIGURE_FILES = [
    "ai4i_eda_target.png",
    "ai4i_eda_type.png",
    "ai4i_eda_modes.png",
    "ai4i_eda_correlation.png",
    "ai4i_eda_physical.png",
]


# ---------- 資料落地 ----------

def test_raw_csv_exists():
    assert DATA.exists(), "缺少 data/ai4i2020.csv，應先執行 src/ai4i_data_prep.py"


def test_shape_and_columns():
    df = prep.load_raw()
    assert df.shape == (10000, 14)
    assert list(df.columns) == COLUMNS


def test_no_missing_values():
    df = prep.load_raw()
    assert int(df.isna().sum().sum()) == 0


def test_no_duplicate_rows():
    df = prep.load_raw()
    assert int(df.duplicated().sum()) == 0


def test_udi_is_time_ordered():
    df = prep.load_raw()
    assert df["UDI"].is_monotonic_increasing
    assert df["UDI"].is_unique
    assert df["UDI"].min() == 1 and df["UDI"].max() == 10000


def test_product_id_prefix_matches_type():
    df = prep.load_raw()
    assert (df["Product ID"].str[0] == df["Type"]).all()


# ---------- 目標與失效模式（官方數值） ----------

def test_target_distribution():
    df = prep.load_raw()
    assert int(df["Machine failure"].sum()) == 339
    assert abs(df["Machine failure"].mean() - 0.0339) < 1e-4


def test_failure_mode_counts():
    df = prep.load_raw()
    counts = {m: int(df[m].sum()) for m in MODES}
    assert counts == {"TWF": 46, "HDF": 115, "PWF": 95, "OSF": 98, "RNF": 19}


def test_machine_failure_is_not_equivalent_to_mode_or():
    """記錄資料集的已知瑕疵，避免後續誤用五旗標推導標籤。"""
    df = prep.load_raw()
    checks = prep.validate_dataset(df)
    assert checks["mf_without_mode"] == 9
    assert checks["mode_without_mf"] == 18
    assert checks["or_mismatch_total"] == 27


def test_failure_mode_overlap():
    df = prep.load_raw()
    checks = prep.validate_dataset(df)
    assert checks["mode_overlap"] == {0: 9652, 1: 324, 2: 23, 3: 1}


def test_type_distribution_and_failure_rate():
    df = prep.load_raw()
    checks = prep.validate_dataset(df)
    assert checks["type_counts"] == {"L": 6000, "M": 2997, "H": 1003}
    rates = checks["failure_rate_by_type"]
    # 實測結果：L 最高、H 最低（非單調，與「等級越高越容易故障」的直覺相反）
    assert rates == {"H": 0.0209, "L": 0.0392, "M": 0.0277}
    assert max(rates, key=rates.get) == "L"


# ---------- 驗證函式本身的行為 ----------

def test_validate_dataset_returns_expected_keys():
    checks = prep.validate_dataset(prep.load_raw())
    for key in ("n_rows", "n_cols", "n_missing", "n_duplicates", "n_failure",
                "failure_rate", "udi_monotonic", "product_id_matches_type",
                "mf_without_mode", "mode_without_mf", "or_mismatch_total",
                "mode_overlap", "type_counts", "failure_rate_by_type"):
        assert key in checks, key


def test_validate_dataset_rejects_bad_frame():
    df = prep.load_raw().drop(columns=["Torque [Nm]"])
    with pytest.raises(ValueError):
        prep.validate_dataset(df)


def test_validate_dataset_rejects_duplicate_udi():
    df = prep.load_raw().copy()
    df.loc[0, "UDI"] = df.loc[1, "UDI"]
    with pytest.raises(ValueError):
        prep.validate_dataset(df)


# ---------- 來源紀錄 ----------

def test_provenance_matches_file():
    prov = json.loads(PROV.read_text(encoding="utf-8"))
    assert prov["license"] == "CC BY 4.0"
    assert "archive.ics.uci.edu" in prov["source_url"]
    assert prov["n_rows"] == 10000 and prov["n_cols"] == 14
    assert prov["sha256"] == prep.sha256_of(DATA)
    assert prov["download_date"] == prep.DOWNLOAD_DATE


def test_readme_documents_source_and_license():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "AI4I 2020" in readme
    assert "CC BY 4.0" in readme
    assert "archive.ics.uci.edu" in readme


# ---------- 報告與圖表 ----------

def test_report_contains_key_numbers():
    text = REPORT.read_text(encoding="utf-8")
    for token in ("10,000", "339", "3.39%", "TWF", "HDF", "PWF", "OSF", "RNF",
                  "CC BY 4.0", "UDI", "Product ID"):
        assert token in text, token
    # 必須揭露 OR 不等價的事實
    assert "9" in text and "18" in text


def test_report_is_deterministic():
    df = prep.load_raw()
    checks = prep.validate_dataset(df)
    first = prep.render_eda_report(df, checks)
    second = prep.render_eda_report(prep.load_raw(), prep.validate_dataset(prep.load_raw()))
    assert first == second
    assert first == REPORT.read_text(encoding="utf-8")


def test_figures_exist_and_are_non_trivial():
    for name in FIGURE_FILES:
        path = FIG_DIR / name
        assert path.exists(), f"缺少圖表 {name}"
        assert path.stat().st_size > 10_000, f"{name} 檔案過小，可能繪圖失敗"


def test_numeric_summary_matches_pandas():
    df = prep.load_raw()
    summary = prep.numeric_summary(df)
    assert set(summary) == {
        "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    }
    assert summary["Torque [Nm]"]["mean"] == pytest.approx(df["Torque [Nm]"].mean())


def test_engineering_features():
    df = prep.load_raw()
    feat = prep.add_engineering_features(df)
    assert "Temp diff [K]" in feat and "Power [W]" in feat
    expected_power = feat["Torque [Nm]"] * 2 * 3.141592653589793 * feat["Rotational speed [rpm]"] / 60
    assert (feat["Power [W]"] - expected_power).abs().max() < 1e-6
    assert (feat["Temp diff [K]"] - (feat["Process temperature [K]"] - feat["Air temperature [K]"])).abs().max() < 1e-9
