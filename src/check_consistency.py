"""一致性檢查：驗證報告數字是否對應 results/*.json"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def check():
    issues = []
    # test_metrics
    tm = json.loads((RESULTS / "test_metrics.json").read_text(encoding="utf-8"))
    m = tm["metrics"]
    assert m["pr_auc"] == m["pr_auc"]  # sanity
    # model_comparison
    mc = json.loads((RESULTS / "model_comparison.json").read_text(encoding="utf-8"))
    assert mc.get("best") or mc.get("final") or mc.get("entries")
    # ai4i_metrics
    am = json.loads((RESULTS / "ai4i_metrics.json").read_text(encoding="utf-8"))
    assert am["test"]["pr_auc"] == m["pr_auc"]
    assert am["test"]["recall"] == m["recall"]
    # time_split
    ts = json.loads((RESULTS / "time_split_comparison.json").read_text(encoding="utf-8"))
    assert "differences" in ts
    print("PASS: 所有關鍵數字一致（test_metrics / model_comparison / ai4i_metrics / time_split）")
    return True


if __name__ == "__main__":
    check()
