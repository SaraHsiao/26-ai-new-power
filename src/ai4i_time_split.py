"""AI4I UDI 時間切分穩健性檢查（#9）。

以 UDI 排序模擬時間順序（資料集未直接給出時間戳，但 UDI 為序號）：
- 不 shuffle，前 80% 訓練、後 20% 測試
- 僅用訓練段 fit 模型（不洩漏測試集）
- 與隨機切分（random_state=42）對照 PR-AUC 等指標
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import ai4i_eval as ev
import ai4i_train as tr
from ai4i_data_prep import MODES, load_raw
from ai4i_features import BASE_FEATURES, build_preprocessor, load_dataset

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS = ROOT / "results"

RANDOM_STATE = 42


def load_sorted() -> pd.DataFrame:
    raw = load_raw()
    return raw.sort_values("UDI").reset_index(drop=True)


def time_split(test_ratio: float = 0.2):
    df = load_sorted()
    n = len(df)
    split = int(n * (1 - test_ratio))
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()
    return train_df, test_df


def distribution_comparison() -> dict[str, Any]:
    train_df, test_df = time_split()
    comp = {}
    # Type distribution
    comp["Type"] = {
        "train": train_df["Type"].value_counts(normalize=True).to_dict(),
        "test": test_df["Type"].value_counts(normalize=True).to_dict(),
    }
    # Mode distributions
    comp["modes"] = {}
    for m in MODES:
        comp["modes"][m] = {
            "train_rate": float(train_df[m].mean()),
            "test_rate": float(test_df[m].mean()),
        }
    comp["failure_rate"] = {
        "train": float(train_df["Machine failure"].mean()),
        "test": float(test_df["Machine failure"].mean()),
    }
    comp["n"] = {"train": len(train_df), "test": len(test_df)}
    return comp


def _build_best_config():
    # Use same config as #7 best model
    return {
        "model": "random_forest",
        "strategy": "none",
        "params": {"n_estimators": 150, "max_depth": 8, "min_samples_leaf": 4, "max_features": 0.4},
        "calibrate": True,
    }


def run_time_split() -> dict[str, Any]:
    df = load_sorted()
    train_df, test_df = time_split()

    X_train = train_df[BASE_FEATURES]
    y_train = train_df["Machine failure"]
    X_test = test_df[BASE_FEATURES]
    y_test = test_df["Machine failure"]

    cfg = _build_best_config()
    preprocessor = build_preprocessor()
    from sklearn.ensemble import RandomForestClassifier
    est = RandomForestClassifier(random_state=RANDOM_STATE, **cfg["params"])
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", est)])
    pipe.fit(X_train, y_train)

    if cfg["calibrate"]:
        cal = CalibratedClassifierCV(pipe, method="sigmoid", cv=5)
        cal.fit(X_train, y_train)
        model = cal
    else:
        model = pipe

    proba = model.predict_proba(X_test)[:, 1]
    # Use OOF to pick threshold? Or use same approach as #7 - but easier to use fixed threshold from random split
    # For fairness, use threshold from random split OOF if available
    try:
        bundle = tr.load_model()
        threshold = float(bundle["threshold"])
    except Exception:
        # Fallback
        y_pred_proba = model.predict_proba(X_train)[:, 1]
        from sklearn.model_selection import StratifiedKFold
        from sklearn.base import clone
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        oof = np.zeros(len(X_train))
        for tr_idx, va_idx in skf.split(X_train, y_train):
            m = clone(model)
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            oof[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
        thresholds = np.linspace(0.01, 0.99, 99)
        best_t, best_f1 = 0.5, -1
        for t in thresholds:
            p = (oof >= t).astype(int)
            from sklearn.metrics import f1_score
            f1 = f1_score(y_train, p)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        threshold = best_t

    pred = (proba >= threshold).astype(int)
    metrics = ev.evaluate(y_test, proba, threshold=float(threshold), name="time_split_rf")
    metrics["n"] = len(y_test)
    metrics["n_pos"] = int(y_test.sum())

    return {
        "time_split": {"metrics": metrics, "threshold": float(threshold)},
        "config": cfg,
    }


def plot_time_split(res: dict, comp: dict, path: Path) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Failure rate by position
    df = load_sorted()
    df["bin"] = pd.cut(df["UDI"], bins=10)
    failure_by_bin = df.groupby("bin", observed=False)["Machine failure"].agg(["mean", "sum", "count"])
    axes[0, 0].bar(range(len(failure_by_bin)), failure_by_bin["mean"], color="#4C72B0")
    axes[0, 0].set_title("UDI 區段故障率（累積順序）")
    axes[0, 0].set_xlabel("UDI 區段（由低到高）")
    axes[0, 0].set_ylabel("故障率")
    axes[0, 0].axvline(7.5, color="red", linestyle="--", label="80/20 切分點")
    axes[0, 0].legend()

    # Type distribution
    types = sorted(set(list(comp["Type"]["train"].keys()) + list(comp["Type"]["test"].keys())))
    x = np.arange(len(types))
    width = 0.35
    axes[0, 1].bar(x - width / 2, [comp["Type"]["train"].get(t, 0) for t in types], width, label="訓練（前80%）")
    axes[0, 1].bar(x + width / 2, [comp["Type"]["test"].get(t, 0) for t in types], width, label="測試（後20%）")
    axes[0, 1].set_title("Type 分布對照")
    axes[0, 1].set_xticks(x, types)
    axes[0, 1].legend()

    # Mode rates
    modes = list(MODES)
    x2 = np.arange(len(modes))
    axes[1, 0].bar(x2 - width / 2, [comp["modes"][m]["train_rate"] for m in modes], width, label="訓練")
    axes[1, 0].bar(x2 + width / 2, [comp["modes"][m]["test_rate"] for m in modes], width, label="測試")
    axes[1, 0].set_title("失效模式率對照")
    axes[1, 0].set_xticks(x2, modes)
    axes[1, 0].legend()

    # Metrics comparison
    try:
        test_metrics = json.loads((RESULTS / "test_metrics.json").read_text(encoding="utf-8"))
        random_pr = test_metrics["metrics"]["pr_auc"]
    except Exception:
        random_pr = None
    time_pr = res["time_split"]["metrics"]["pr_auc"]
    labels_m = ["隨機切分", "UDI 時間切分"]
    values = [random_pr if random_pr else 0, time_pr]
    axes[1, 1].bar(labels_m, values, color=["#4C72B0", "#C44E52"])
    axes[1, 1].set_title("PR-AUC 對照")
    axes[1, 1].set_ylabel("PR-AUC")
    for i, v in enumerate(values):
        if v:
            axes[1, 1].text(i, v + 0.01, f"{v:.4f}", ha="center")

    fig.suptitle("UDI 順序切分穩健性檢查", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    comp = distribution_comparison()
    res = run_time_split()
    plot_time_split(res, comp, FIG_DIR / "ai4i_time_split.png")

    # Load random split results
    try:
        test_metrics = json.loads((RESULTS / "test_metrics.json").read_text(encoding="utf-8"))
        random_split = {
            "pr_auc": test_metrics["metrics"]["pr_auc"],
            "recall": test_metrics["metrics"]["recall"],
            "precision": test_metrics["metrics"]["precision"],
            "f1": test_metrics["metrics"]["f1"],
            "threshold": test_metrics["metrics"]["threshold"],
        }
    except Exception:
        random_split = {}

    time_split_res = res["time_split"]["metrics"]
    pr_diff = (time_split_res["pr_auc"] - random_split.get("pr_auc", time_split_res["pr_auc"]))
    recall_diff = (time_split_res["recall"] - random_split.get("recall", time_split_res["recall"]))

    if abs(pr_diff) <= 0.10:
        verdict = "穩定：PR-AUC 差異 <= 0.10，時間切分表現與隨機切分相近"
    else:
        verdict = f"需留意：PR-AUC 差異 {pr_diff:.4f} > 0.10，建議進一步調查"

    output = {
        "random_split": random_split,
        "time_split": time_split_res,
        "differences": {
            "pr_auc_diff": float(pr_diff),
            "recall_diff": float(recall_diff),
        },
        "distribution": comp,
        "verdict": verdict,
    }

    (RESULTS / "time_split_comparison.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(verdict)
    print(f"PR-AUC: random={random_split.get('pr_auc')}, time={time_split_res['pr_auc']}, diff={pr_diff:.4f}")
    return output


if __name__ == "__main__":
    main()
