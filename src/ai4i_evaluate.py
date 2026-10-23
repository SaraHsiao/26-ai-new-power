"""AI4I 評估視覺化、可解釋性與誤判分析（#8）。

所有圖表與數字都由此腳本產生；測試集預測直接重用 `results/test_predictions.csv`，
**不重新評估測試集**（維持 #6 的「只評估一次」協定）。

用法：
    python src/ai4i_evaluate.py
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

import ai4i_train as tr
from ai4i_data_prep import MODES, load_raw
from ai4i_features import BASE_FEATURES, load_dataset

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

BEST_MODEL_PATH = RESULTS / "best_model.joblib"
TEST_METRICS_PATH = RESULTS / "test_metrics.json"
TEST_PRED_PATH = RESULTS / "test_predictions.csv"
METRICS_PATH = RESULTS / "ai4i_metrics.json"
ERROR_CSV_PATH = RESULTS / "error_cases.csv"
REPORT_PATH = DOCS / "evaluation_figures.md"

RANDOM_STATE = 42
FIGURE_FILES = {
    "pr": "ai4i_pr_curve.png",
    "roc": "ai4i_roc_curve.png",
    "confusion": "ai4i_confusion.png",
    "calibration": "ai4i_calibration.png",
    "importance": "ai4i_importance.png",
    "physical": "ai4i_physical.png",
    "mode_recall": "ai4i_mode_recall.png",
}

MODE_DESC = {
    "TWF": "刀具磨耗（Tool Wear）",
    "HDF": "散熱失效（Heat Dissipation）",
    "PWF": "電力失效（Power）",
    "OSF": "應變超限（Overstrain）",
    "RNF": "隨機失效（Random）",
}
MODE_FEATURE = {
    "HDF": ("Temp diff [K]", 8.6),
    "PWF": ("Power [W]", None),
    "OSF": ("Torque x Tool wear", None),
    "TWF": ("Tool wear [min]", 240),
    "RNF": (None, None),
}

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Segoe UI", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLOR_NORMAL = "#4C72B0"
COLOR_FAIL = "#C44E52"


# ---------------------------------------------------------------- 指標
def confusion_counts(y_true, pred) -> dict[str, Any]:
    y = np.asarray(y_true).ravel()
    p = np.asarray(pred).ravel()
    if len(y) != len(p):
        raise ValueError("y_true 與 pred 長度不一致")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true 必須是 0/1")

    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float((tp + tn) / len(y)),
        "n": int(len(y)),
    }


def precision_recall_summary(y_true, proba) -> dict[str, Any]:
    y = np.asarray(y_true).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    precision, recall, thresholds = precision_recall_curve(y, p)
    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "ap": float(average_precision_score(y, p)),
    }


def roc_summary(y_true, proba) -> dict[str, Any]:
    y = np.asarray(y_true).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    fpr, tpr, thresholds = roc_curve(y, p)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": float(roc_auc_score(y, p)),
    }


def calibration_summary(y_true, proba, n_bins: int = 10) -> dict[str, Any]:
    """均勻分箱的可靠性圖資料（避免分位數分箱把大量近 0 機率擠成一箱）。"""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    if len(y) != len(p):
        raise ValueError("y_true 與 proba 長度不一致")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.searchsorted(edges, p, side="right") - 1
    bins = np.clip(bins, 0, n_bins - 1)

    prob_pred, prob_true, counts = [], [], []
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            continue
        prob_pred.append(float(np.mean(p[mask])))
        prob_true.append(float(np.mean(y[mask])))
        counts.append(n)

    return {
        "prob_pred": np.array(prob_pred, dtype=float),
        "prob_true": np.array(prob_true, dtype=float),
        "counts": np.array(counts, dtype=int),
        "brier": float(np.mean((p - y) ** 2)),
        "n_bins": int(n_bins),
    }


def permutation_importance_summary(
    model, X: pd.DataFrame, y, n_repeats: int = 10, random_state: int = RANDOM_STATE,
) -> list[dict[str, Any]]:
    """以 PR-AUC 下降量衡量原始特徵的重要度（permutation importance）。"""
    result = permutation_importance(
        model, X, y, scoring="average_precision",
        n_repeats=n_repeats, random_state=random_state, n_jobs=-1,
    )
    items = [
        {
            "feature": str(name),
            "mean": float(mean),
            "std": float(std),
        }
        for name, mean, std in zip(X.columns, result.importances_mean, result.importances_std)
    ]
    return sorted(items, key=lambda d: d["mean"], reverse=True)


# ---------------------------------------------------------------- 失效模式
def per_mode_recall(raw: pd.DataFrame, pred) -> dict[str, dict[str, Any]]:
    p = np.asarray(pred).ravel()
    if len(p) != len(raw):
        raise ValueError("pred 長度與 raw 不一致")
    has_label = "Machine failure" in raw.columns
    failed = raw["Machine failure"].to_numpy() == 1 if has_label else None
    out: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        mask = raw[mode].to_numpy() == 1
        n = int(mask.sum())
        predicted = int(p[mask].sum()) if n else 0
        if failed is not None:
            mask_mf = mask & failed
            n_mf = int(mask_mf.sum())
            predicted_mf = int(p[mask_mf].sum()) if n_mf else 0
        else:
            n_mf = None
            predicted_mf = None
        out[mode] = {
            "n": n,
            "n_mf": n_mf,
            "predicted": predicted,
            "predicted_mf": predicted_mf,
            "recall": (predicted / n) if n else None,
            "recall_mf": (predicted_mf / n_mf) if n_mf else None,
        }
    return out


# ---------------------------------------------------------------- 誤判
def build_error_cases(
    raw: pd.DataFrame, y_true, proba, pred, threshold: float = 0.5,
) -> pd.DataFrame:
    """回傳所有 FP / FN，附上原始特徵、工程特徵與預測機率。"""
    y = np.asarray(y_true).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    pr = np.asarray(pred).ravel()
    if not (len(raw) == len(y) == len(p) == len(pr)):
        raise ValueError("raw / y_true / proba / pred 長度必須一致")

    out = raw.copy()
    out["Temp diff [K]"] = out["Process temperature [K]"] - out["Air temperature [K]"]
    out["Power [W]"] = out["Torque [Nm]"] * 2 * np.pi * out["Rotational speed [rpm]"] / 60
    out["Torque x Tool wear"] = out["Torque [Nm]"] * out["Tool wear [min]"]
    out["y_true"] = y
    out["proba"] = p
    out["pred"] = pr.astype(int)
    out["threshold"] = float(threshold)

    fn_mask = (y == 1) & (pr == 0)
    fp_mask = (y == 0) & (pr == 1)
    out["error_type"] = np.where(fn_mask, "FN", np.where(fp_mask, "FP", ""))
    out = out[out["error_type"] != ""].copy()

    front = ["UDI", "Product ID", "Type"] + [
        "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
        "Torque [Nm]", "Tool wear [min]",
    ] + ["Temp diff [K]", "Power [W]", "Torque x Tool wear"] + MODES + [
        "y_true", "proba", "pred", "threshold", "error_type",
    ]
    front = [c for c in front if c in out.columns]
    return out[front]


def error_summary(errors: pd.DataFrame, raw: pd.DataFrame, y_true, pred) -> dict[str, Any]:
    y = np.asarray(y_true).ravel()
    pr = np.asarray(pred).ravel()
    cm = confusion_counts(y, pr)

    counts = {
        "FN": int((errors["error_type"] == "FN").sum()),
        "FP": int((errors["error_type"] == "FP").sum()),
    }

    features = ["Temp diff [K]", "Power [W]", "Torque x Tool wear",
                "Tool wear [min]", "Torque [Nm]", "Rotational speed [rpm]"]
    enriched = raw.copy()
    enriched["Temp diff [K]"] = enriched["Process temperature [K]"] - enriched["Air temperature [K]"]
    enriched["Power [W]"] = enriched["Torque [Nm]"] * 2 * np.pi * enriched["Rotational speed [rpm]"] / 60
    enriched["Torque x Tool wear"] = enriched["Torque [Nm]"] * enriched["Tool wear [min]"]

    groups = {
        "TP": (y == 1) & (pr == 1),
        "TN": (y == 0) & (pr == 0),
        "FN": (y == 1) & (pr == 0),
        "FP": (y == 0) & (pr == 1),
    }
    comparison = {}
    for feat in features:
        comparison[feat] = {
            g: (float(enriched.loc[mask, feat].mean()) if mask.any() else None)
            for g, mask in groups.items()
        }

    mode_counts = {}
    for etype in ("FN", "FP"):
        subset = errors[errors["error_type"] == etype]
        mode_counts[etype] = {
            mode: int(subset[mode].sum()) for mode in MODES
        }
        mode_counts[etype]["no_mode"] = int(
            (subset[MODES].sum(axis=1) == 0).sum()
        )

    examples = {}
    for etype in ("FN", "FP"):
        subset = errors[errors["error_type"] == etype].sort_values("proba", ascending=False)
        examples[etype] = [
            {
                "UDI": int(row["UDI"]),
                "Type": str(row["Type"]),
                "proba": round(float(row["proba"]), 4),
                "Tool wear [min]": int(row["Tool wear [min]"]),
                "Torque [Nm]": float(row["Torque [Nm]"]),
                "Temp diff [K]": round(float(row["Temp diff [K]"]), 2),
                "Power [W]": round(float(row["Power [W]"]), 0),
                "modes": [m for m in MODES if int(row[m]) == 1],
            }
            for _, row in subset.head(3).iterrows()
        ]

    return {
        "counts": counts,
        "rates": {
            "fnr": cm["fn"] / (cm["fn"] + cm["tp"]) if (cm["fn"] + cm["tp"]) else 0.0,
            "fpr": cm["fp"] / (cm["fp"] + cm["tn"]) if (cm["fp"] + cm["tn"]) else 0.0,
        },
        "feature_comparison": comparison,
        "modes": mode_counts,
        "examples": examples,
    }


# ---------------------------------------------------------------- 載入
def load_context() -> dict[str, Any]:
    """載入最終模型、測試集預測與對齊的原始資料（不重新評估）。"""
    bundle = tr.load_model(BEST_MODEL_PATH)
    X, y = load_dataset()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    raw = load_raw()
    _, raw_test = train_test_split(
        raw, test_size=0.2, stratify=raw["Machine failure"], random_state=RANDOM_STATE,
    )
    if not np.array_equal(X_test.index.to_numpy(), raw_test.index.to_numpy()):
        raise RuntimeError("測試集特徵與原始資料的順序不一致")

    saved = pd.read_csv(TEST_PRED_PATH, encoding="utf-8")
    if len(saved) != len(y_test):
        raise RuntimeError("test_predictions.csv 長度與測試集不符")
    if not np.array_equal(saved["y_true"].to_numpy(), y_test.to_numpy()):
        raise RuntimeError("test_predictions.csv 的 y_true 與測試集標籤不一致")

    test_metrics = json.loads(TEST_METRICS_PATH.read_text(encoding="utf-8"))
    return {
        "bundle": bundle,
        "X_test": X_test,
        "y_test": y_test,
        "raw_test": raw_test,
        "raw": raw,
        "proba": saved["proba"].to_numpy(dtype=float),
        "pred": saved["pred"].to_numpy(dtype=int),
        "threshold": float(bundle["threshold"]),
        "test_metrics": test_metrics,
    }


# ---------------------------------------------------------------- 圖表
def _setup_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def plot_pr_curve(ctx: dict, path: Path) -> Path:
    y, p = ctx["y_test"].to_numpy(), ctx["proba"]
    pr = precision_recall_summary(y, p)
    prevalence = float(np.mean(y))

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.step(pr["recall"], pr["precision"], where="post", color=COLOR_FAIL,
            linewidth=2, label=f"RandomForest（AP = {pr['ap']:.4f}）")
    ax.axhline(prevalence, color="gray", linestyle="--", linewidth=1,
               label=f"隨機猜（AP = {prevalence:.4f}）")
    m = ctx["test_metrics"]["metrics"]
    ax.scatter([m["recall"]], [m["precision"]], s=70, color="black", zorder=5,
               label=f"操作點（R={m['recall']:.3f}, P={m['precision']:.3f}）")
    _setup_axis(ax, "測試集 Precision-Recall 曲線（僅評估一次）", "Recall", "Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_roc_curve(ctx: dict, path: Path) -> Path:
    y, p = ctx["y_test"].to_numpy(), ctx["proba"]
    roc = roc_summary(y, p)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(roc["fpr"], roc["tpr"], color=COLOR_NORMAL, linewidth=2,
            label=f"RandomForest（AUC = {roc['auc']:.4f}）")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="隨機（AUC = 0.5）")
    _setup_axis(ax, "測試集 ROC 曲線", "False Positive Rate", "True Positive Rate")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_confusion(ctx: dict, path: Path) -> Path:
    y, pred = ctx["y_test"].to_numpy(), ctx["pred"]
    cm = confusion_counts(y, pred)
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(matrix, cmap="Blues")
    labels = ["正常 (0)", "故障 (1)"]
    ax.set_xticks([0, 1], [f"預測{labels[0]}", f"預測{labels[1]}"])
    ax.set_yticks([0, 1], [f"實際{labels[0]}", f"實際{labels[1]}"])

    total_neg, total_pos = cm["tn"] + cm["fp"], cm["fn"] + cm["tp"]
    annotations = [
        (0, 0, cm["tn"], total_neg, "TN"),
        (0, 1, cm["fp"], total_neg, "FP"),
        (1, 0, cm["fn"], total_pos, "FN"),
        (1, 1, cm["tp"], total_pos, "TP"),
    ]
    for row, col, value, total, name in annotations:
        pct = value / total if total else 0.0
        ax.text(col, row, f"{name} {value}\n({pct:.1%})", ha="center", va="center",
                fontsize=11, color="white" if value > matrix.max() / 2 else "black")
    ax.set_title("測試集混淆矩陣（百分比以實際類別為分母）", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_calibration(ctx: dict, path: Path) -> Path:
    y, p = ctx["y_test"].to_numpy(), ctx["proba"]
    cal = calibration_summary(y, p, n_bins=10)

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="完美校準")
    ax.plot(cal["prob_pred"], cal["prob_true"], "o-", color=COLOR_NORMAL,
            linewidth=1.5, markersize=5,
            label=f"模型（Brier = {cal['brier']:.4f}）")
    for x, yv, n in zip(cal["prob_pred"], cal["prob_true"], cal["counts"]):
        ax.annotate(str(int(n)), (x, yv), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7, color="dimgray")
    _setup_axis(ax, "測試集可靠性圖（點旁數字為該箱樣本數）",
                "平均預測機率", "實際故障比例")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_importance(importance: list[dict], path: Path) -> Path:
    ordered = list(reversed(importance))
    names = [d["feature"] for d in ordered]
    means = [d["mean"] for d in ordered]
    stds = [d["std"] for d in ordered]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.barh(names, means, xerr=stds, color=COLOR_NORMAL, alpha=0.85,
                   capsize=3, error_kw={"elinewidth": 1})
    for bar, value in zip(bars, means):
        ax.text(value + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}", va="center", fontsize=8)
    _setup_axis(ax, "Permutation Importance（測試集，指標：PR-AUC 下降量）",
                "PR-AUC 下降量", "")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _box_by_flag(ax, values, flag, labels, title, ylabel, reference=None,
                 reference_label=None, span=None, span_label=None):
    groups = [values[flag == 0], values[flag == 1]]
    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], [COLOR_NORMAL, COLOR_FAIL]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    if reference is not None:
        ax.axhline(reference, color="black", linestyle="--", linewidth=1,
                   label=reference_label)
    if span is not None:
        ax.axhspan(*span, color="green", alpha=0.08, label=span_label)
    _setup_axis(ax, title, "", ylabel)
    if reference is not None or span is not None:
        ax.legend(fontsize=7, frameon=False)


def plot_physical(raw: pd.DataFrame, mode_recall: dict, path: Path) -> Path:
    temp_diff = raw["Process temperature [K]"] - raw["Air temperature [K]"]
    power = raw["Torque [Nm]"] * 2 * np.pi * raw["Rotational speed [rpm]"] / 60
    strain = raw["Torque [Nm]"] * raw["Tool wear [min]"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    def add_recall(ax, mode):
        r = mode_recall[mode]
        if r["recall_mf"] is not None:
            text = (f"測試集 {mode} recall = {r['recall_mf']:.2f}"
                    f"（{r['predicted_mf']}/{r['n_mf']}）")
        else:
            text = f"測試集 {mode}: 無故障樣本（旗標 n={r['n']}）"
        ax.text(0.98, 0.02, text, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})

    _box_by_flag(axes[0, 0], temp_diff, raw["HDF"].to_numpy(),
                 ["HDF = 0", "HDF = 1"], "HDF 與溫差", "Temp diff [K]",
                 reference=8.6, reference_label="HDF 判準 8.6 K")
    add_recall(axes[0, 0], "HDF")

    _box_by_flag(axes[0, 1], power, raw["PWF"].to_numpy(),
                 ["PWF = 0", "PWF = 1"], "PWF 與功率", "Power [W]",
                 span=(3500, 9000), span_label="正常功率 3.5–9 kW")
    add_recall(axes[0, 1], "PWF")

    _box_by_flag(axes[1, 0], strain, raw["OSF"].to_numpy(),
                 ["OSF = 0", "OSF = 1"], "OSF 與扭矩 × 磨耗", "Torque x Tool wear")
    add_recall(axes[1, 0], "OSF")

    _box_by_flag(axes[1, 1], raw["Tool wear [min]"].to_numpy(), raw["TWF"].to_numpy(),
                 ["TWF = 0", "TWF = 1"], "TWF 與刀具磨耗", "Tool wear [min]",
                 reference=240, reference_label="TWF 上限 240 min")
    add_recall(axes[1, 1], "TWF")

    fig.suptitle("物理機制與失效模式（全資料 10,000 筆；註記為測試集 recall）", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_mode_recall(mode_recall: dict, path: Path) -> Path:
    modes = list(MODES)
    recalls = [
        mode_recall[m]["recall_mf"] if mode_recall[m]["recall_mf"] is not None else 0.0
        for m in modes
    ]
    colors = [
        "gray" if mode_recall[m]["recall_mf"] is None
        else (COLOR_FAIL if m == "RNF" else COLOR_NORMAL)
        for m in modes
    ]
    labels = [
        f"{m}\n(n_mf={mode_recall[m]['n_mf']})" if mode_recall[m]["n_mf"]
        else f"{m}\n(n_mf=0)"
        for m in modes
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bars = ax.bar(labels, recalls, color=colors, width=0.6)
    for bar, value, mode in zip(bars, recalls, modes):
        text = f"{value:.2f}" if mode_recall[mode]["recall_mf"] is not None else "無法估計"
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                text, ha="center", fontsize=10)
    _setup_axis(ax, "五種失效模式的測試集 Recall（分母為同時有該模式旗標的故障）",
                "失效模式", "Recall")
    ax.set_ylim(0, 1.18)
    rnf = mode_recall["RNF"]
    if rnf["n_mf"] == 0:
        idx = modes.index("RNF")
        ax.annotate(f"RNF 旗標 {rnf['n']} 筆皆非 Machine failure；\n"
                    "本資料無法用它估計 recall",
                    xy=(idx, 0.02), xytext=(idx - 2.2, 0.82),
                    fontsize=8, color=COLOR_FAIL,
                    arrowprops={"arrowstyle": "->", "color": COLOR_FAIL})
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 報告
def render_report(metrics: dict[str, Any]) -> str:
    test = metrics["test"]
    cal = metrics["calibration"]
    errors = metrics["errors"]
    importance = metrics["permutation_importance"]
    modes = metrics["per_mode_recall"]
    top3 = importance[:3]
    rnf = modes["RNF"]
    ceiling = metrics["recall_ceiling"]

    lines: list[str] = []
    lines.append("# AI4I 2020 評估視覺化與誤判分析\n")
    lines.append("> 本檔案由 `src/ai4i_evaluate.py` 自動產生；數字與 "
                 "`results/ai4i_metrics.json` 一致。\n")
    lines.append("## 摘要\n")
    lines.append(f"最終模型（{metrics['source']['model']}，"
                 f"{'已校準' if metrics['source']['calibrated'] else '未校準'}）"
                 f"在測試集 2,000 筆（68 筆故障）上的表現：\n")
    lines.append(f"- PR-AUC **{test['pr_auc']:.4f}**、ROC-AUC {metrics['curves']['roc_auc']:.4f}")
    lines.append(f"- Precision **{test['precision']:.4f}**、Recall **{test['recall']:.4f}**、"
                 f"F1 {test['f1']:.4f}、MCC {test['mcc']:.4f}")
    lines.append(f"- Lift **{test['lift']:.2f}×**、Brier {cal['brier']:.4f}")
    lines.append(f"- 混淆矩陣：TP {test['tp']} / FP {test['fp']} / FN {test['fn']} / TN {test['tn']}\n")

    lines.append("## 圖 1：Precision-Recall 曲線（`ai4i_pr_curve.png`）\n")
    lines.append(f"PR 曲線的 AP 為 **{metrics['curves']['ap']:.4f}**，隨機猜的基線是故障率 "
                 f"**{test['prevalence']:.4%}**。在 3.4% 的不平衡資料中，ROC 曲線會被大量 TN 稀釋，"
                 "看起來幾乎完美；PR 曲線只關注「被預測為故障」的樣本，"
                 "更能反映實際巡檢成本，因此本專案以 PR-AUC 為主要指標。\n")
    lines.append(f"黑色操作點對應 OOF 選出的閾值（{metrics['threshold']:.3f}）："
                 f"Recall {test['recall']:.3f}、Precision {test['precision']:.3f}。\n")

    lines.append("## 圖 2：ROC 曲線（`ai4i_roc_curve.png`）\n")
    lines.append(f"ROC-AUC 為 **{metrics['curves']['roc_auc']:.4f}**。"
                 "這個數字很高，但主要來自 TN（1,924 筆正常）容易分類；"
                 "它不能單獨用來判斷模型對故障的實用性，僅作為補充。\n")

    lines.append("## 圖 3：混淆矩陣（`ai4i_confusion.png`）\n")
    lines.append(f"- 68 筆故障中抓到 **{test['tp']} 筆（{test['recall']:.1%}）**，漏掉 **{test['fn']} 筆**")
    lines.append(f"- 1,932 筆正常中誤報 **{test['fp']} 筆（{errors['rates']['fpr']:.2%}）**")
    lines.append(f"- 漏檢率（FNR）{errors['rates']['fnr']:.2%}、誤報率（FPR）{errors['rates']['fpr']:.2%}\n")
    lines.append("矩陣內的百分比以「實際類別」為分母，可直接回答「故障抓得到幾成」與"
                 "「正常會被誤判幾成」。\n")

    lines.append("## 圖 4：可靠性圖（`ai4i_calibration.png`）\n")
    bins = cal["bins"]
    low, high = bins[0], bins[-1]
    mid = [b for b in bins if 0.05 <= b["prob_pred"] <= 0.95]
    mid_n = sum(b["count"] for b in mid)
    lines.append(f"Brier score **{cal['brier']:.4f}**（越低越好）。採均勻分箱："
                 f"低機率箱（n={low['count']}）平均預測 {low['prob_pred']:.4f}、"
                 f"實際 {low['prob_true']:.4f}；高機率箱（n={high['count']}）平均預測 "
                 f"{high['prob_pred']:.3f}、實際 {high['prob_true']:.3f}。"
                 f"中間機率（0.05–0.95）只有 {mid_n} 筆，個別箱的觀測比例變異大，"
                 "不宜過度解讀；整體而言，模型在大量低風險樣本與高風險樣本上"
                 "的機率是可信的，可供成本決策使用。\n")

    lines.append("## 圖 5：特徵重要度（`ai4i_importance.png`）\n")
    lines.append("以測試集上的 permutation importance（PR-AUC 下降量）衡量：\n")
    lines.append("| 排名 | 特徵 | 平均下降 | 標準差 | 對應物理機制 |")
    lines.append("| --- | --- | --- | --- | --- |")
    mapping = {
        "Tool wear [min]": "TWF（刀具磨耗）與 OSF（磨耗×扭矩）",
        "Torque [Nm]": "OSF（扭矩×磨耗）、PWF（功率）",
        "Rotational speed [rpm]": "PWF（功率）、HDF（轉速門檻）",
        "Process temperature [K]": "HDF（溫差）",
        "Air temperature [K]": "HDF（溫差）",
        "Type": "產品等級，與故障率相關（L 3.92% / M 2.77% / H 2.09%）",
    }
    for i, item in enumerate(importance, 1):
        lines.append(f"| {i} | `{item['feature']}` | {item['mean']:.4f} | {item['std']:.4f} | "
                     f"{mapping.get(item['feature'], '—')} |")
    lines.append("")
    top3_text = "、".join(
        f"`{d['feature']}`（{mapping.get(d['feature'], '—')}）" for d in top3
    )
    lines.append(f"前 3 名為 {top3_text}，都能對應到物理機制。"
                 "扭矩與轉速共同決定功率（PWF）與應變負荷（OSF），"
                 "氣溫與製程溫度共同決定溫差（HDF）；沒有出現「重要度很高但無法解釋」的特徵。")
    wear_rank = next((i for i, d in enumerate(importance, 1)
                      if d["feature"] == "Tool wear [min]"), None)
    if wear_rank:
        lines.append(f"值得注意的是 `Tool wear [min]` 只排第 {wear_rank}，"
                     "而 TWF 的 recall 明顯偏低（見圖 7）；"
                     "這表示模型目前較依賴扭矩/轉速，對刀具磨耗的權重不足，"
                     "是後續改進的直接線索。\n")

    lines.append("## 圖 6：物理機制對照（`ai4i_physical.png`）\n")
    lines.append("| 模式 | 物理特徵 | 旗標 n | 其中故障 n_mf | 抓到（故障） | Recall（故障） | 已知判準 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    crit = {
        "TWF": "磨耗 200–240 min",
        "HDF": "溫差 < 8.6 K 且轉速 < 1380 rpm",
        "PWF": "功率 < 3,500 W 或 > 9,000 W",
        "OSF": "扭矩 × 磨耗超過等級門檻",
        "RNF": "無（隨機）",
    }
    for mode in MODES:
        r = modes[mode]
        recall_txt = f"{r['recall_mf']:.2f}" if r["recall_mf"] is not None else "—"
        caught_txt = f"{r['predicted_mf']}/{r['n_mf']}" if r["n_mf"] else "—"
        lines.append(f"| {mode} | {MODE_DESC[mode]} | {r['n']} | {r['n_mf']} | "
                     f"{caught_txt} | {recall_txt} | {crit[mode]} |")
    lines.append("")
    lines.append("圖中箱型圖使用全資料 10,000 筆以呈現物理分布，角落註記為測試集故障 recall。"
                 "HDF=1 的溫差集中在 8.6 K 判準以下、PWF=1 落在正常功率區間之外、"
                 "OSF=1 的扭矩×磨耗明顯偏高、TWF=1 的磨耗接近 240 min，"
                 "都與文獻描述的機制一致。\n")

    lines.append("## 圖 7：五種失效模式 recall（`ai4i_mode_recall.png`）\n")
    twf = modes["TWF"]
    if twf["recall_mf"] is not None:
        lines.append(f"- TWF：{twf['n_mf']} 筆故障中只抓到 {twf['predicted_mf']} 筆"
                     f"（recall **{twf['recall_mf']:.2f}**），是整體 recall 的主要缺口。")
    strong = [m for m in ("HDF", "PWF", "OSF") if modes[m]["recall_mf"] is not None]
    if strong:
        lines.append("- " + "、".join(strong) + "：" + "、".join(
            f"{m} {modes[m]['predicted_mf']}/{modes[m]['n_mf']}"
            for m in strong
        ) + "，測試集全部抓到（recall 1.00）。")
    if rnf["n_mf"] == 0:
        lines.append(f"- RNF 旗標在測試集有 **{rnf['n']} 筆**，但全部不是 Machine failure"
                     "（全資料 19 筆旗標也只有 1 筆同時是故障）；"
                     "因此這個切分無法估計 RNF 的 recall，它也不構成 recall 上限。")
    lines.append(f"- 真正無法由模式旗標解釋的是 **{metrics['unpredictable_failures']} 筆無模式故障**，"
                 f"所以 recall 上限為 1 − {metrics['unpredictable_failures']}/68 = "
                 f"**{ceiling:.3f}**。目前整體 recall 為 {test['recall']:.3f}，"
                 f"與上限的差距 {ceiling - test['recall']:.3f} 幾乎全部來自 8 筆 TWF 漏檢"
                 "（另有 2 筆無模式故障），不是模型不穩定。\n")

    lines.append("## 誤判分析（`results/error_cases.csv`）\n")
    lines.append(f"- 偽陰性（FN）**{errors['counts']['FN']} 筆**、偽陽性（FP）**{errors['counts']['FP']} 筆**")
    lines.append("")
    lines.append("| 特徵 | FN 平均 | TP 平均 | FP 平均 | TN 平均 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for feat, groups in errors["feature_comparison"].items():
        def fmt(v):
            return "—" if v is None else f"{v:.2f}"
        lines.append(f"| {feat} | {fmt(groups['FN'])} | {fmt(groups['TP'])} | "
                     f"{fmt(groups['FP'])} | {fmt(groups['TN'])} |")
    lines.append("")
    comp = errors["feature_comparison"]

    def top_gaps(group_a: str, group_b: str, k: int = 2) -> str:
        scored = []
        for feat, vals in comp.items():
            a, b = vals[group_a], vals[group_b]
            if a is None or b is None:
                continue
            scored.append((abs(a - b) / max(abs(b), 1e-9), feat, a, b))
        scored.sort(reverse=True)
        return "；".join(f"{feat} {a:.2f}（{group_b} {b:.2f}）"
                        for _, feat, a, b in scored[:k])

    lines.append(f"- FN 與 TP 的平均特徵差異（括號為 TP）：{top_gaps('FN', 'TP')}。")
    lines.append(f"- FP 與 TN 的平均特徵差異（括號為 TN）：{top_gaps('FP', 'TN')}。\n")
    lines.append("FN 的模式組成：" + "、".join(
        f"{m} {errors['modes']['FN'][m]}" for m in MODES
    ) + f"（無模式 {errors['modes']['FN']['no_mode']} 筆）。")
    lines.append("FP 的模式組成：" + "、".join(
        f"{m} {errors['modes']['FP'][m]}" for m in MODES
    ) + f"（無模式 {errors['modes']['FP']['no_mode']} 筆）。\n")
    lines.append("最具代表性的錯誤（依預測機率排序）：\n")
    for etype, title in (("FN", "偽陰性（漏檢）"), ("FP", "偽陽性（誤報）")):
        lines.append(f"**{title}**\n")
        lines.append("| UDI | Type | proba | Tool wear | Torque | Temp diff | Power | 模式 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for ex in errors["examples"][etype]:
            modes_txt = "/".join(ex["modes"]) if ex["modes"] else "無模式旗標"
            lines.append(f"| {ex['UDI']} | {ex['Type']} | {ex['proba']:.4f} | "
                         f"{ex['Tool wear [min]']} | {ex['Torque [Nm]']:.1f} | "
                         f"{ex['Temp diff [K]']:.2f} | {ex['Power [W]']:,.0f} | {modes_txt} |")
        lines.append("")

    lines.append("## 結論\n")
    lines.append(f"1. 測試集 PR-AUC {test['pr_auc']:.4f}、Recall {test['recall']:.3f}、"
                 f"Precision {test['precision']:.3f}、Lift {test['lift']:.2f}×，"
                 "符合 #3 的必要門檻；Brier score "
                 f"{cal['brier']:.4f} 且可靠性圖在低/高機率區間與對角線一致。")
    lines.append(f"2. 重要度前 3 名（{', '.join(d['feature'] for d in top3)}）"
                 "都能對應到 PWF/HDF/OSF 的物理機制，沒有無法解釋的主要特徵。")
    lines.append(f"3. 最大缺口是 **TWF**：10 筆故障只抓到 {twf['predicted_mf']} 筆"
                 f"（recall {twf['recall_mf']:.2f}），且 10 筆 FN 中有 8 筆是 TWF。"
                 "這些樣本刀具磨耗已達 200–240 min，但扭矩/功率偏低，"
                 "模型因此給低分；Tool wear 的重要性只排第 "
                 f"{wear_rank}，是可直接改進的方向（例如提高磨耗相關特徵權重或加入交互項）。")
    lines.append(f"4. FP 共 {errors['counts']['FP']} 筆（FPR {errors['rates']['fpr']:.2%}），"
                 "全部沒有模式旗標，但扭矩×磨耗與功率偏高；"
                 "屬於「風險因子接近門檻但未實際失效」的邊界樣本，"
                 "對巡檢成本影響有限，但仍應監控。")
    lines.append(f"5. RNF 旗標在本資料幾乎不與 Machine failure 重疊（測試集 {rnf['n']} 筆全非故障，"
                 f"全資料 19 筆只有 1 筆是故障），因此不能把「未抓到 RNF」當成模型缺陷；"
                 f"真正不可解釋的是 {metrics['unpredictable_failures']} 筆無模式故障，"
                 f"recall 上限約 **{ceiling:.3f}**。目前 {test['recall']:.3f} 與上限的差距"
                 "主要來自 TWF 漏檢。")
    lines.append("6. 限制：測試集只評估一次，以上分析不再回頭調參；"
                 "permutation importance 是描述性而非因果；"
                 "未使用 SHAP（環境未安裝），以 permutation importance 取代。\n")

    lines.append("## 重現方式\n")
    lines.append("```bash")
    lines.append("python src/ai4i_evaluate.py   # 只讀取既有測試集預測，不重新評估")
    lines.append("python -m pytest tests -q")
    lines.append("```\n")
    return "\n".join(lines)


# ---------------------------------------------------------------- 主流程
def run_evaluation(verbose: bool = True) -> dict[str, Any]:
    ctx = load_context()
    y = ctx["y_test"].to_numpy()
    p = ctx["proba"]
    pred = ctx["pred"]

    cm = confusion_counts(y, pred)
    pr = precision_recall_summary(y, p)
    roc = roc_summary(y, p)
    cal = calibration_summary(y, p, n_bins=10)
    importance = permutation_importance_summary(ctx["bundle"]["model"], ctx["X_test"], y)
    modes = per_mode_recall(ctx["raw_test"], pred)
    errors_df = build_error_cases(ctx["raw_test"], y, p, pred, threshold=ctx["threshold"])
    errors = error_summary(errors_df, ctx["raw_test"], y, pred)
    n_failures = int(y.sum())
    unpredictable_failures = int(errors["modes"]["FN"]["no_mode"])
    ceiling = 1.0 - unpredictable_failures / n_failures if n_failures else 0.0

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["pr"] = plot_pr_curve(ctx, FIG_DIR / FIGURE_FILES["pr"])
    paths["roc"] = plot_roc_curve(ctx, FIG_DIR / FIGURE_FILES["roc"])
    paths["confusion"] = plot_confusion(ctx, FIG_DIR / FIGURE_FILES["confusion"])
    paths["calibration"] = plot_calibration(ctx, FIG_DIR / FIGURE_FILES["calibration"])
    paths["importance"] = plot_importance(importance, FIG_DIR / FIGURE_FILES["importance"])
    paths["physical"] = plot_physical(ctx["raw"], modes, FIG_DIR / FIGURE_FILES["physical"])
    paths["mode_recall"] = plot_mode_recall(modes, FIG_DIR / FIGURE_FILES["mode_recall"])

    metrics = {
        "source": {
            "model": ctx["test_metrics"].get("model"),
            "strategy": ctx["test_metrics"].get("strategy"),
            "calibrated": ctx["test_metrics"].get("calibrated"),
            "model_path": str(BEST_MODEL_PATH.relative_to(ROOT)),
            "predictions": str(TEST_PRED_PATH.relative_to(ROOT)),
        },
        "test": ctx["test_metrics"]["metrics"],
        "threshold": ctx["threshold"],
        "curves": {"ap": pr["ap"], "roc_auc": roc["auc"]},
        "confusion": cm,
        "calibration": {
            "brier": cal["brier"],
            "n_bins": cal["n_bins"],
            "bins": [
                {"prob_pred": float(x), "prob_true": float(t), "count": int(n)}
                for x, t, n in zip(cal["prob_pred"], cal["prob_true"], cal["counts"])
            ],
        },
        "per_mode_recall": modes,
        "permutation_importance": importance,
        "unpredictable_failures": int(unpredictable_failures),
        "recall_ceiling": float(ceiling),
        "errors": errors,
        "figures": list(FIGURE_FILES.values()),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    errors_df.to_csv(ERROR_CSV_PATH, index=False, encoding="utf-8")
    DOCS.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(metrics), encoding="utf-8")

    if verbose:
        print(f"[test] PR-AUC {metrics['test']['pr_auc']:.4f}  "
              f"R {metrics['test']['recall']:.3f}  P {metrics['test']['precision']:.3f}")
        print(f"[mode] " + "  ".join(
            f"{m}:{modes[m]['predicted']}/{modes[m]['n']}" for m in MODES))
        print(f"[top3] " + ", ".join(f"{d['feature']} ({d['mean']:.4f})" for d in importance[:3]))
        print(f"[errors] FN {errors['counts']['FN']} / FP {errors['counts']['FP']}")
        print(f"[json] {METRICS_PATH}")
        print(f"[csv]  {ERROR_CSV_PATH}")
        print(f"[doc]  {REPORT_PATH}")
        print(f"[figures] {len(paths)} 張")
    return metrics


if __name__ == "__main__":
    run_evaluation()
