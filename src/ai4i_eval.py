"""AI4I 評估協定：OOF 收集、閾值選擇、指標與 bootstrap 信賴區間。

核心規則（對應 #6 驗收標準）：

1. `choose_threshold()` **只接受** `collect_oof()` 產生的 `OOFResult`；
   傳入原始陣列（例如測試集預測）會直接 raise `TypeError`。
2. 測試集只評估一次；本模組不提供「拿測試集調閾值」的 API。
3. 主要指標是 PR-AUC；accuracy 僅作參考，因為全猜正常就有 96.61%。

用法：
    oof = collect_oof(make_pipeline(model), X_train, y_train)
    choice = choose_threshold(oof, cost_fn=10.0, cost_fp=1.0)
    metrics = evaluate(y_test, test_proba, threshold=choice.threshold)
    ci = bootstrap_ci(y_test, test_proba, metric="pr_auc", n_boot=1000)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from ai4i_features import build_preprocessor, make_pipeline

PRIMARY_METRIC = "pr_auc"
ALLOWED_METRICS = ("pr_auc", "precision", "recall", "f1", "lift", "accuracy", "mcc")
STRATEGIES = ("none", "class_weight", "smote")

# 預設成本假設：漏檢（FN）成本是誤報（FP）的 10 倍
DEFAULT_COST_FN = 10.0
DEFAULT_COST_FP = 1.0


# ---------------------------------------------------------------- OOF
@dataclass(frozen=True)
class OOFResult:
    """Out-of-fold 預測結果。只能由 collect_oof() 產生並用於閾值選擇。"""

    proba: np.ndarray
    y: np.ndarray
    folds: np.ndarray

    def __post_init__(self) -> None:
        proba = np.asarray(self.proba, dtype=float).ravel()
        y = np.asarray(self.y).ravel()
        folds = np.asarray(self.folds).ravel()
        if not (len(proba) == len(y) == len(folds)):
            raise ValueError("proba、y、folds 長度必須一致")
        if len(y) == 0:
            raise ValueError("OOFResult 不可為空")
        object.__setattr__(self, "proba", proba)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "folds", folds)

    def __len__(self) -> int:
        return len(self.y)


def collect_oof(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> OOFResult:
    """以 Stratified K-fold 收集 out-of-fold 預測機率。

    每個 fold 的模型只在訓練資料上 fit；回傳的 `proba` 中每一筆
    都來自「沒看過該筆資料」的模型。
    """
    if n_splits < 2:
        raise ValueError("n_splits 必須 >= 2")
    y_arr = np.asarray(y)
    proba = np.full(len(y_arr), np.nan, dtype=float)
    folds = np.full(len(y_arr), -1, dtype=int)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for k, (train_idx, valid_idx) in enumerate(cv.split(X, y_arr)):
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y_arr[train_idx])
        proba[valid_idx] = fold_model.predict_proba(X.iloc[valid_idx])[:, 1]
        folds[valid_idx] = k

    if np.isnan(proba).any() or (folds < 0).any():
        raise RuntimeError("有樣本未被 OOF 預測涵蓋")
    return OOFResult(proba=proba, y=y_arr, folds=folds)


# ---------------------------------------------------------------- 指標
def evaluate(
    y_true,
    proba,
    threshold: float = 0.5,
    name: str = "model",
) -> dict[str, Any]:
    """以固定閾值計算指標；accuracy 僅作參考，主要指標為 `pr_auc`。"""
    y = np.asarray(y_true).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    if len(y) != len(p):
        raise ValueError("y_true 與 proba 長度不一致")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true 必須是 0/1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold 必須介於 0 與 1")

    pred = p >= threshold
    tp = int(np.sum((y == 1) & pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    tn = int(np.sum((y == 0) & ~pred))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y)
    prevalence = float(np.mean(y))
    lift = precision / prevalence if prevalence > 0 else 0.0

    n_pos = int(np.sum(y == 1))
    if 0 < n_pos < len(y):
        pr_auc = float(average_precision_score(y, p))
    else:
        pr_auc = 0.0

    if len(np.unique(pred)) == 1:
        mcc = 0.0
    else:
        mcc = float(matthews_corrcoef(y, pred.astype(int)))

    return {
        "name": name,
        "threshold": float(threshold),
        "n": int(len(y)),
        "n_pos": n_pos,
        "prevalence": prevalence,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "mcc": mcc,
        "pr_auc": pr_auc,
        "lift": float(lift),
        "primary_metric": PRIMARY_METRIC,
    }


# ---------------------------------------------------------------- 閾值選擇
@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    cost: float
    cost_per_sample: float
    cost_fn: float
    cost_fp: float
    metrics: dict[str, Any] = field(default_factory=dict)
    source: str = "oof"


def choose_threshold(
    oof,
    y=None,
    *,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> ThresholdChoice:
    """用 OOF 預測挑閾值，最小化 `cost_fn × FN + cost_fp × FP`。

    僅接受 `OOFResult`。若傳入原始陣列（例如測試集預測），立即 raise，
    以防止「用測試集調閾值」。
    """
    if y is not None or not isinstance(oof, OOFResult):
        raise TypeError(
            "choose_threshold() 只接受 OOFResult（由 collect_oof() 產生），"
            "不可傳入原始預測陣列或測試集預測。"
        )
    if cost_fn <= 0 or cost_fp <= 0:
        raise ValueError("cost_fn 與 cost_fp 必須為正數")

    y_arr = oof.y
    p = oof.proba
    n_pos = int(y_arr.sum())
    n_neg = len(y_arr) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("OOF 標籤必須同時包含 0 與 1")

    # 依機率由大到小排序，累積 TP / FP；候選閾值為每個不同的機率值
    order = np.argsort(-p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y_arr[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)

    last_of_group = np.r_[
        np.flatnonzero(p_sorted[1:] != p_sorted[:-1]),
        len(p_sorted) - 1,
    ]
    tp_at = tp_cum[last_of_group]
    fp_at = fp_cum[last_of_group]
    costs = cost_fn * (n_pos - tp_at) + cost_fp * fp_at

    best = int(np.argmin(costs))
    threshold = float(p_sorted[last_of_group[best]])
    cost = float(costs[best])

    return ThresholdChoice(
        threshold=threshold,
        cost=cost,
        cost_per_sample=cost / len(y_arr),
        cost_fn=float(cost_fn),
        cost_fp=float(cost_fp),
        metrics=evaluate(y_arr, p, threshold=threshold, name="oof"),
    )


# ---------------------------------------------------------------- Bootstrap
def _metric_value(metric: str, y: np.ndarray, proba: np.ndarray, threshold: float) -> float:
    if metric == "pr_auc":
        if y.sum() == 0 or y.sum() == len(y):
            return float("nan")
        return float(average_precision_score(y, proba))
    return float(evaluate(y, proba, threshold=threshold)[metric])


def bootstrap_ci(
    y_true,
    proba,
    metric: str = "recall",
    threshold: float = 0.5,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, Any]:
    """以 bootstrap 重抽樣計算指標的 (1-alpha) 信賴區間。"""
    if metric not in ALLOWED_METRICS:
        raise ValueError(f"不支援的指標 {metric}；可用：{ALLOWED_METRICS}")
    if n_boot < 1:
        raise ValueError("n_boot 必須 >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha 必須介於 0 與 1")

    y = np.asarray(y_true).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    point = _metric_value(metric, y, p, threshold)

    rng = np.random.default_rng(random_state)
    n = len(y)
    values = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        values[b] = _metric_value(metric, y[idx], p[idx], threshold)

    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        raise ValueError("所有 bootstrap 樣本都無法計算該指標")

    lower, upper = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "metric": metric,
        "point": float(point),
        "lower": float(lower),
        "upper": float(upper),
        "n": int(n),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------- 不平衡策略
def smote_available() -> bool:
    try:
        import imblearn  # noqa: F401
    except ImportError:
        return False
    return True


def strategies_available() -> list[str]:
    return [s for s in STRATEGIES if s != "smote" or smote_available()]


def build_imbalance_pipeline(
    strategy: str,
    estimator,
    **preprocessor_kwargs,
):
    """依策略建立 Pipeline：none / class_weight / smote（選配）。

    SMOTE 需安裝 imbalanced-learn；未安裝時 raise ImportError，
    並在訊息中說明安裝方式。
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"未知策略 {strategy}；可用：{STRATEGIES}")

    model = clone(estimator)

    if strategy == "class_weight":
        if "class_weight" not in model.get_params():
            raise ValueError(f"{type(model).__name__} 不支援 class_weight")
        model.set_params(class_weight="balanced")

    if strategy == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
        except ImportError as exc:
            raise ImportError(
                "SMOTE 需要 imbalanced-learn，請執行 `pip install imbalanced-learn`。"
            ) from exc
        return ImbPipeline([
            ("prep", build_preprocessor(**preprocessor_kwargs)),
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ])

    return make_pipeline(model, **preprocessor_kwargs)


# ---------------------------------------------------------------- 基線
def dummy_baseline(y_true) -> dict[str, Any]:
    """全猜「故障」的基線：PR-AUC 等於故障率，Recall = 1。"""
    y = np.asarray(y_true).ravel()
    prevalence = float(np.mean(y))
    return {
        "name": "dummy_all_positive",
        "n": int(len(y)),
        "prevalence": prevalence,
        "pr_auc": prevalence,
        "precision": prevalence,
        "recall": 1.0,
        "f1": 2 * prevalence / (1 + prevalence) if prevalence > 0 else 0.0,
        "accuracy": prevalence,
        "lift": 1.0,
        "primary_metric": PRIMARY_METRIC,
    }
