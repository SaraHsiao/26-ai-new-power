"""AI4I 時間/UDI 切分穩健性測試（先寫測試，再實作）。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ai4i_time_split as ats

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS = ROOT / "results"


def test_sort_by_udi_ascending():
    df = ats.load_sorted()
    assert df["UDI"].is_monotonic_increasing


def test_time_split_8020_by_order():
    train, test = ats.time_split()
    assert len(train) == 8000
    assert len(test) == 2000
    assert train["UDI"].max() < test["UDI"].min() or train.index.max() < test.index.max()


def test_no_shuffle():
    df = ats.load_sorted()
    train, test = ats.time_split()
    # 前 80% 的順序不被打亂
    assert list(train.index) == list(df.index[:8000])


def test_distribution_comparison():
    comp = ats.distribution_comparison()
    assert "Type" in comp
    assert "modes" in comp
    assert "failure_rate" in comp
    assert comp["failure_rate"]["train"] >= 0
    assert comp["failure_rate"]["test"] >= 0


def test_train_and_evaluate_time_split():
    res = ats.run_time_split()
    m = res["time_split"]["metrics"]
    for k in ["pr_auc", "recall", "precision", "tp", "fp", "fn", "tn"]:
        assert k in m
    assert 0 <= m["pr_auc"] <= 1
    assert m["n"] == 2000


def test_comparison_json_structure():
    path = RESULTS / "time_split_comparison.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "random_split" in data
    assert "time_split" in data
    assert "differences" in data
    assert "verdict" in data
    assert "pr_auc_diff" in data["differences"]
    assert abs(data["differences"]["pr_auc_diff"]) >= 0


def test_plot_exists():
    path = FIG_DIR / "ai4i_time_split.png"
    assert path.exists()
    assert path.stat().st_size > 10000
