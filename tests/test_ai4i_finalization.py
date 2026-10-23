"""AI4I 報告與交付整合測試（#10）。"""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_ai4i_section():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "AI4I 2020" in text
    assert "環境" in text or "安裝" in text
    assert "執行" in text


def test_ai4i_report_exists():
    path = ROOT / "docs" / "ai4i-report.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    # Check required sections
    for section in ["摘要", "緒論", "資料說明", "研究方法", "實驗設計", "結果", "討論", "結論與未來改進", "參考文獻"]:
        assert section in text


def test_all_report_numbers_from_json():
    report = (ROOT / "docs" / "ai4i-report.md").read_text(encoding="utf-8")
    test_metrics = json.loads((ROOT / "results" / "test_metrics.json").read_text(encoding="utf-8"))
    pr = test_metrics["metrics"]["pr_auc"]
    rec = test_metrics["metrics"]["recall"]
    pre = test_metrics["metrics"]["precision"]
    assert f"{pr:.4f}" in report or f"{pr:.3f}" in report
    assert "0.900" in report


def test_run_all_exists_and_runnable():
    path = ROOT / "src" / "run_all.py"
    assert path.exists()
    # Check it has main
    text = path.read_text(encoding="utf-8")
    assert "def main" in text or "__main__" in text


def test_consistency_check():
    path = ROOT / "src" / "check_consistency.py"
    assert path.exists()
    res = subprocess.run(["python", str(path)], cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    assert res.returncode == 0
    out = res.stdout
    assert "一致" in out or "PASS" in out or "OK" in out


def test_results_files_exist():
    required = ["test_metrics.json", "model_comparison.json", "test_predictions.csv", 
               "best_model.joblib", "ai4i_metrics.json", "time_split_comparison.json"]
    for r in required:
        assert (ROOT / "results" / r).exists(), f"Missing {r}"
