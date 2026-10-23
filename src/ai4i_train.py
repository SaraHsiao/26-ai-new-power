"""AI4I 模型建立、比較與校準。

流程（嚴格遵守 #6 協定）：

1. 只在訓練集上做 RandomizedSearchCV（輕量）與 5-fold OOF 比較。
2. 用 OOF 選閾值（`choose_threshold`，成本 10:1）。
3. 以 OOF Brier score 決定是否採用機率校準。
4. 模型定案後，測試集**只評估一次**（`run_test_evaluation` 有檔案守門）。

用法：
    python src/ai4i_train.py                 # 完整訓練（含超參數搜尋）
    python src/ai4i_train.py --no-tune       # 跳過搜尋，較快
    python src/ai4i_train.py --force-test    # 明知風險才重跑測試集
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

import ai4i_eval as ev
from ai4i_features import get_feature_names, load_dataset

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
COMPARISON_PATH = RESULTS / "model_comparison.json"
BEST_MODEL_PATH = RESULTS / "best_model.joblib"
TEST_METRICS_PATH = RESULTS / "test_metrics.json"
TEST_PRED_PATH = RESULTS / "test_predictions.csv"
MODEL_DOC_PATH = DOCS / "model_comparison.md"

RANDOM_STATE = 42
COST_FN = 10.0
COST_FP = 1.0

REQUIRED_MODELS = ("dummy", "logistic", "random_forest", "hist_gradient_boosting")
OPTIONAL_MODELS = ("xgboost", "lightgbm")


# ---------------------------------------------------------------- 模型註冊
def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def default_model_names() -> list[str]:
    names = list(REQUIRED_MODELS)
    for name in OPTIONAL_MODELS:
        if _installed(name):
            names.append(name)
    return names


def _make_estimator(name: str, random_state: int = RANDOM_STATE):
    if name == "dummy":
        return DummyClassifier(strategy="prior", random_state=random_state)
    if name == "logistic":
        return LogisticRegression(max_iter=2000, random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, n_jobs=-1,
            random_state=random_state,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, random_state=random_state,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            n_jobs=-1, random_state=random_state,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
            random_state=random_state, verbose=-1,
        )
    raise ValueError(f"未知模型 {name}；可用：{default_model_names()}")


def build_model_pipeline(
    name: str,
    strategy: str = "none",
    random_state: int = RANDOM_STATE,
    params: dict | None = None,
    **preprocessor_kwargs,
):
    """建立「前處理 + 不平衡策略 + 模型」的完整 Pipeline。"""
    if name == "dummy" and strategy != "none":
        raise ValueError("DummyClassifier 不支援 class_weight / SMOTE，僅能用 strategy='none'")

    estimator = _make_estimator(name, random_state=random_state)
    if params:
        estimator.set_params(**params)
    return ev.build_imbalance_pipeline(strategy, estimator, **preprocessor_kwargs)


def search_space(name: str) -> dict[str, list]:
    """輕量超參數搜尋空間（key 已含 Pipeline 前綴）。"""
    if name == "dummy":
        return {}
    if name == "logistic":
        return {"model__C": [0.01, 0.1, 1.0, 10.0]}
    if name == "random_forest":
        return {
            "model__n_estimators": [150, 250, 400],
            "model__max_depth": [None, 8, 14, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", 0.4],
        }
    if name == "hist_gradient_boosting":
        return {
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [20, 40, 80],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        }
    if name in ("xgboost", "lightgbm"):
        return {
            "model__n_estimators": [200, 300, 500],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [3, 4, 6],
        }
    raise ValueError(f"未知模型 {name}")


def _n_combinations(space: dict[str, list]) -> int:
    n = 1
    for values in space.values():
        n *= len(values)
    return n


def tune_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 8,
    cv: int = 3,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """在訓練資料上做輕量 RandomizedSearchCV；不使用測試集。"""
    space = search_space(name)
    if not space:
        return {
            "model": name, "best_params": {}, "estimator_params": {},
            "best_score": None, "search_space": {}, "n_iter": 0, "cv": cv,
        }

    pipe = build_model_pipeline(name, "none", random_state=random_state)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=space,
        n_iter=min(n_iter, _n_combinations(space)),
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
        refit=False,
    )
    search.fit(X, y)

    best_params = dict(search.best_params_)
    estimator_params = {
        key[len("model__"):]: value for key, value in best_params.items()
    }
    return {
        "model": name,
        "best_params": best_params,
        "estimator_params": estimator_params,
        "best_score": float(np.max(search.cv_results_["mean_test_score"])),
        "search_space": space,
        "n_iter": int(search.n_iter),
        "cv": cv,
    }


# ---------------------------------------------------------------- 指標輔助
def brier_score(y_true, proba) -> float:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    if len(y) != len(p):
        raise ValueError("y_true 與 proba 長度不一致")
    return float(np.mean((p - y) ** 2))


def _fold_pr_auc(oof, n_splits: int) -> list[float]:
    scores = []
    for k in range(n_splits):
        mask = oof.folds == k
        yk = oof.y[mask]
        if 0 < yk.sum() < len(yk):
            scores.append(float(average_precision_score(yk, oof.proba[mask])))
    return scores


# ---------------------------------------------------------------- 比較
def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: list[str] | None = None,
    strategies: tuple[str, ...] = ("none", "class_weight"),
    n_splits: int = 5,
    n_boot: int = 1000,
    tune: bool = False,
    n_iter: int = 8,
    tune_cv: int = 3,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> dict[str, Any]:
    """模型 × 不平衡策略的 OOF 比較；測試集不參與。"""
    names = list(models) if models is not None else default_model_names()
    if not names:
        raise ValueError("至少需要一個模型")

    tuned: dict[str, dict] = {}
    tuning: dict[str, dict] = {}
    for name in names:
        if tune:
            info = tune_model(name, X, y, n_iter=n_iter, cv=tune_cv,
                              random_state=random_state)
            tuned[name] = info["estimator_params"]
            tuning[name] = {
                "best_params": info["best_params"],
                "best_score": info["best_score"],
                "n_iter": info["n_iter"],
                "search_space": info["search_space"],
            }
            if verbose:
                score = info["best_score"]
                suffix = f" (AP {score:.4f})" if score is not None else ""
                print(f"[tune] {name}: {info['best_params']}{suffix}")
        else:
            tuned[name] = {}

    entries: list[dict] = []
    for name in names:
        for strategy in strategies:
            if name == "dummy" and strategy != "none":
                continue
            pipe = build_model_pipeline(name, strategy, random_state=random_state,
                                        params=tuned[name])
            oof = ev.collect_oof(pipe, X, y, n_splits=n_splits,
                                 random_state=random_state)
            choice = ev.choose_threshold(oof, cost_fn=COST_FN, cost_fp=COST_FP)
            metrics = choice.metrics
            pr_auc = float(average_precision_score(oof.y, oof.proba))
            ci = ev.bootstrap_ci(oof.y, oof.proba, metric="pr_auc",
                                 n_boot=n_boot, random_state=random_state)
            folds = _fold_pr_auc(oof, n_splits)

            entry = {
                "model": name,
                "strategy": strategy,
                "n_splits": n_splits,
                "pr_auc_oof": pr_auc,
                "pr_auc_fold_mean": float(np.mean(folds)) if folds else 0.0,
                "pr_auc_fold_std": float(np.std(folds, ddof=1)) if len(folds) > 1 else 0.0,
                "pr_auc_ci95": [float(ci["lower"]), float(ci["upper"])],
                "threshold": float(choice.threshold),
                "cost_per_sample": float(choice.cost_per_sample),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mcc": metrics["mcc"],
                "lift": metrics["lift"],
                "accuracy": metrics["accuracy"],
                "brier": brier_score(oof.y, oof.proba),
                "params": tuned[name],
            }
            entries.append(entry)
            if verbose:
                print(f"[oof] {name:24s} {strategy:12s} "
                      f"PR-AUC {pr_auc:.4f} "
                      f"(fold {entry['pr_auc_fold_mean']:.4f}±{entry['pr_auc_fold_std']:.4f}) "
                      f"R {entry['recall']:.3f} P {entry['precision']:.3f}")

    best = max(entries, key=lambda e: (e["pr_auc_oof"], e["recall"]))
    return {
        "entries": entries,
        "best": best,
        "tuning": tuning,
        "config": {
            "models": names,
            "strategies": list(strategies),
            "n_splits": n_splits,
            "n_train": int(len(y)),
            "n_pos": int(np.sum(y)),
            "tune": bool(tune),
            "n_iter": int(n_iter),
            "random_state": random_state,
            "n_boot": int(n_boot),
            "cost_fn": COST_FN,
            "cost_fp": COST_FP,
        },
    }


# ---------------------------------------------------------------- 校準
def calibrate_best(
    comparison: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    method: str = "sigmoid",
    cv: int = 3,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """比較最佳模型校準前後的 Brier score 與 PR-AUC（皆用 OOF）。"""
    best = comparison["best"]
    params = best.get("params", {})
    base = build_model_pipeline(best["model"], best["strategy"],
                                random_state=random_state, params=params)

    oof_before = ev.collect_oof(base, X, y, n_splits=n_splits,
                                random_state=random_state)
    calibrated = CalibratedClassifierCV(
        build_model_pipeline(best["model"], best["strategy"],
                             random_state=random_state, params=params),
        method=method,
        cv=cv,
    )
    oof_after = ev.collect_oof(calibrated, X, y, n_splits=n_splits,
                               random_state=random_state)

    choice_before = ev.choose_threshold(oof_before, cost_fn=COST_FN, cost_fp=COST_FP)
    choice_after = ev.choose_threshold(oof_after, cost_fn=COST_FN, cost_fp=COST_FP)

    brier_before = brier_score(oof_before.y, oof_before.proba)
    brier_after = brier_score(oof_after.y, oof_after.proba)
    use_calibrated = brier_after < brier_before

    return {
        "model": best["model"],
        "strategy": best["strategy"],
        "method": method,
        "cv": cv,
        "brier_before": brier_before,
        "brier_after": brier_after,
        "delta_brier": brier_after - brier_before,
        "pr_auc_before": float(average_precision_score(oof_before.y, oof_before.proba)),
        "pr_auc_after": float(average_precision_score(oof_after.y, oof_after.proba)),
        "threshold_before": float(choice_before.threshold),
        "threshold_after": float(choice_after.threshold),
        "recall_before": choice_before.metrics["recall"],
        "recall_after": choice_after.metrics["recall"],
        "precision_before": choice_before.metrics["precision"],
        "precision_after": choice_after.metrics["precision"],
        "use_calibrated": bool(use_calibrated),
        "chosen_threshold": float(choice_after.threshold if use_calibrated else choice_before.threshold),
        "chosen_oof": oof_after if use_calibrated else oof_before,
    }


# ---------------------------------------------------------------- 測試集（一次）
def run_test_evaluation(
    bundle: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_path: Path = TEST_METRICS_PATH,
    pred_path: Path = TEST_PRED_PATH,
    force: bool = False,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """在測試集上評估**一次**；預設會拒絕重複評估。"""
    out_path = Path(out_path)
    pred_path = Path(pred_path)
    if out_path.exists() and not force:
        raise RuntimeError(
            f"測試集只允許評估一次：{out_path} 已存在。"
            "若確定要重跑，請明確設定 force=True（並在報告中說明原因）。"
        )

    proba = bundle["model"].predict_proba(X_test)[:, 1]
    threshold = float(bundle["threshold"])
    metrics = ev.evaluate(y_test, proba, threshold=threshold, name="test")
    ci_auc = ev.bootstrap_ci(y_test, proba, metric="pr_auc",
                             n_boot=n_boot, random_state=RANDOM_STATE)
    ci_recall = ev.bootstrap_ci(y_test, proba, metric="recall", threshold=threshold,
                                n_boot=n_boot, random_state=RANDOM_STATE)

    meta = bundle.get("meta", {})
    payload = {
        "metrics": metrics,
        "pr_auc_ci95": [float(ci_auc["lower"]), float(ci_auc["upper"])],
        "recall_ci95": [float(ci_recall["lower"]), float(ci_recall["upper"])],
        "threshold": threshold,
        "model": meta.get("model"),
        "strategy": meta.get("strategy"),
        "calibrated": bool(meta.get("calibrated", False)),
        "one_shot": True,
        "protocol": "測試集僅在模型與閾值定案後評估一次（#6 協定）",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    pd.DataFrame({
        "y_true": np.asarray(y_test).ravel(),
        "proba": proba,
        "pred": (proba >= threshold).astype(int),
    }).to_csv(pred_path, index=False, encoding="utf-8")
    return payload


# ---------------------------------------------------------------- 存檔
def save_model(bundle: dict[str, Any], path: Path = BEST_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model(path: Path = BEST_MODEL_PATH) -> dict[str, Any]:
    return joblib.load(Path(path))


# ---------------------------------------------------------------- Markdown
def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def render_markdown(
    comparison: dict[str, Any],
    calibration: dict[str, Any] | None = None,
    test_metrics: dict[str, Any] | None = None,
) -> str:
    cfg = comparison["config"]
    entries = sorted(comparison["entries"], key=lambda e: e["pr_auc_oof"],
                     reverse=True)
    best = dict(comparison["best"])
    for entry in entries:
        if (entry["model"], entry["strategy"]) == (best["model"], best["strategy"]):
            for key, value in entry.items():
                best.setdefault(key, value)
            break

    lines: list[str] = []
    lines.append("# AI4I 2020 模型比較與校準\n")
    lines.append("> 本檔案由 `src/ai4i_train.py` 自動產生；數字與 "
                 "`results/model_comparison.json` 一致。\n")

    lines.append("## 1. 實驗設定\n")
    lines.append(f"- 訓練資料：**{cfg['n_train']:,}** 筆，其中故障 **{cfg['n_pos']}** 筆")
    lines.append(f"- 交叉驗證：Stratified **{cfg['n_splits']}-fold**，`random_state={cfg.get('random_state', RANDOM_STATE)}`")
    lines.append(f"- 不平衡策略：{', '.join(f'`{s}`' for s in cfg.get('strategies', ['none', 'class_weight']))}")
    lines.append(f"- 超參數搜尋：{'RandomizedSearchCV（n_iter=' + str(cfg.get('n_iter', 8)) + '）' if cfg.get('tune') else '未執行'}")
    lines.append(f"- 閾值規則：OOF 上最小化 `{cfg.get('cost_fn', COST_FN):.0f}×FN + {cfg.get('cost_fp', COST_FP):.0f}×FP`")
    lines.append("- 測試集：**僅在模型定案後評估一次**，未參與任何選擇\n")
    lines.append("> accuracy 僅供參考：全部猜「正常」即可得到 96.61%，因此不作為主要指標。\n")

    lines.append("## 2. 模型比較（依 OOF PR-AUC 排序）\n")
    header = ("模型", "策略", "PR-AUC (OOF)", "fold mean ± std", "95% CI",
              "閾值", "Precision", "Recall", "F1", "Lift", "Brier")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for e in entries:
        star = " ⭐" if (e["model"], e["strategy"]) == (best["model"], best["strategy"]) else ""
        lines.append("| " + " | ".join([
            e["model"] + star,
            e["strategy"],
            _fmt(e["pr_auc_oof"]),
            f"{_fmt(e['pr_auc_fold_mean'])} ± {_fmt(e['pr_auc_fold_std'])}",
            f"{_fmt(e['pr_auc_ci95'][0])} – {_fmt(e['pr_auc_ci95'][1])}",
            _fmt(e["threshold"], 3),
            _fmt(e["precision"], 3),
            _fmt(e["recall"], 3),
            _fmt(e["f1"], 3),
            _fmt(e["lift"], 2),
            _fmt(e["brier"], 4),
        ]) + " |")
    lines.append("")

    lines.append("## 3. 最佳模型\n")
    lines.append(f"- 模型：**{best['model']}** ＋ 策略 `{best['strategy']}`")
    lines.append(f"- OOF PR-AUC：**{_fmt(best['pr_auc_oof'])}**"
                 f"（95% CI {_fmt(best['pr_auc_ci95'][0])} – {_fmt(best['pr_auc_ci95'][1])}）")
    lines.append(f"- 操作點：閾值 **{_fmt(best['threshold'], 3)}**、"
                 f"Precision **{_fmt(best['precision'], 3)}**、Recall **{_fmt(best['recall'], 3)}**、"
                 f"Lift **{_fmt(best['lift'], 2)}×**")
    if best.get("params"):
        lines.append(f"- 超參數：`{json.dumps(best['params'], ensure_ascii=False)}`")
    lines.append("")

    if calibration:
        lines.append("## 4. 機率校準（OOF）\n")
        lines.append(f"- 方法：`{calibration['method']}`（cv={calibration['cv']}）")
        lines.append("")
        lines.append("| 指標 | 校準前 | 校準後 | 差異 |")
        lines.append("| --- | --- | --- | --- |")
        lines.append(f"| Brier score（越低越好） | {_fmt(calibration['brier_before'])} | "
                     f"{_fmt(calibration['brier_after'])} | {_fmt(calibration['delta_brier'])} |")
        lines.append(f"| PR-AUC | {_fmt(calibration['pr_auc_before'])} | "
                     f"{_fmt(calibration['pr_auc_after'])} | "
                     f"{_fmt(calibration['pr_auc_after'] - calibration['pr_auc_before'])} |")
        lines.append(f"| Recall @ 選定閾值 | {_fmt(calibration['recall_before'], 3)} | "
                     f"{_fmt(calibration['recall_after'], 3)} | — |")
        lines.append(f"| Precision @ 選定閾值 | {_fmt(calibration['precision_before'], 3)} | "
                     f"{_fmt(calibration['precision_after'], 3)} | — |")
        lines.append("")
        decision = "採用校準後模型" if calibration["use_calibrated"] else "保留未校準模型"
        lines.append(f"決策：Brier score {'下降' if calibration['use_calibrated'] else '未下降'}，"
                     f"因此**{decision}**（決策僅根據訓練集 OOF，未看測試集）。\n")

    lines.append("## 5. 測試集評估（僅一次）\n")
    if test_metrics:
        m = test_metrics["metrics"]
        lines.append(f"- 測試筆數：**{m['n']:,}**（故障 {m['n_pos']} 筆）")
        lines.append(f"- 閾值：**{_fmt(test_metrics['threshold'], 3)}**"
                     "（由 OOF 決定，未用測試集調整）")
        lines.append("")
        lines.append("| 指標 | 數值 |")
        lines.append("| --- | --- |")
        lines.append(f"| PR-AUC | **{_fmt(m['pr_auc'])}**（95% CI "
                     f"{_fmt(test_metrics['pr_auc_ci95'][0])} – {_fmt(test_metrics['pr_auc_ci95'][1])}） |")
        lines.append(f"| Precision | {_fmt(m['precision'])} |")
        lines.append(f"| Recall | {_fmt(m['recall'])}（95% CI "
                     f"{_fmt(test_metrics['recall_ci95'][0])} – {_fmt(test_metrics['recall_ci95'][1])}） |")
        lines.append(f"| F1 | {_fmt(m['f1'])} |")
        lines.append(f"| MCC | {_fmt(m['mcc'])} |")
        lines.append(f"| Lift | {_fmt(m['lift'], 2)}× |")
        lines.append(f"| accuracy（僅參考） | {_fmt(m['accuracy'])} |")
        lines.append(f"| 混淆矩陣 | TP {m['tp']} / FP {m['fp']} / FN {m['fn']} / TN {m['tn']} |")
        lines.append("")
    else:
        lines.append("尚未評估（依協定於模型定案後執行一次）。\n")

    lines.append("## 6. 重現方式\n")
    lines.append("```bash")
    lines.append("python src/ai4i_train.py        # 重新訓練、比較、校準（測試集若已評估會拒絕重跑）")
    lines.append("python -m pytest tests -q       # 全部測試")
    lines.append("```\n")
    lines.append("載入模型：\n")
    lines.append("```python")
    lines.append("from ai4i_train import load_model")
    lines.append("bundle = load_model('results/best_model.joblib')")
    lines.append("proba = bundle['model'].predict_proba(X_new)[:, 1]")
    lines.append("pred = (proba >= bundle['threshold']).astype(int)")
    lines.append("```\n")

    lines.append("## 7. 限制\n")
    lines.append("- RNF（19 筆隨機失效）無法由感測特徵預測，recall 存在上限（約 0.94）。")
    lines.append("- 測試集僅 68 筆故障，指標的信賴區間偏寬；已用 bootstrap 呈現不確定性。")
    lines.append("- 隨機切分假設資料可交換；時間漂移的影響見 #9 的時間切分對照。")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- 主流程
def run_training(
    tune: bool = True,
    n_iter: int = 8,
    n_boot: int = 1000,
    force_test: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    if verbose:
        print(f"[data] train {len(y_train):,}（故障 {int(y_train.sum())}）／"
              f"test {len(y_test):,}（故障 {int(y_test.sum())}）")

    comparison = compare_models(
        X_train, y_train, tune=tune, n_iter=n_iter, n_boot=n_boot,
        random_state=RANDOM_STATE, verbose=verbose,
    )

    calibration = calibrate_best(
        comparison, X_train, y_train, n_splits=5,
        method="sigmoid", cv=3, random_state=RANDOM_STATE,
    )
    chosen_oof = calibration.pop("chosen_oof")
    if verbose:
        print(f"[calibration] Brier {calibration['brier_before']:.4f} -> "
              f"{calibration['brier_after']:.4f}"
              f"（{'採用' if calibration['use_calibrated'] else '不採用'}）")

    best = comparison["best"]
    final_base = build_model_pipeline(
        best["model"], best["strategy"], random_state=RANDOM_STATE,
        params=best.get("params", {}),
    )
    if calibration["use_calibrated"]:
        final_model = CalibratedClassifierCV(
            final_base, method=calibration["method"], cv=calibration["cv"],
        )
    else:
        final_model = final_base
    final_model.fit(X_train, y_train)

    bundle = {
        "model": final_model,
        "threshold": calibration["chosen_threshold"],
        "feature_names": get_feature_names(),
        "meta": {
            "model": best["model"],
            "strategy": best["strategy"],
            "params": best.get("params", {}),
            "calibrated": calibration["use_calibrated"],
            "calibration_method": calibration["method"] if calibration["use_calibrated"] else None,
            "threshold_source": "OOF (cost 10:1)",
            "random_state": RANDOM_STATE,
        },
    }
    save_model(bundle)
    if verbose:
        print(f"[model] 已存檔 {BEST_MODEL_PATH}")

    if TEST_METRICS_PATH.exists() and not force_test:
        test_metrics = json.loads(TEST_METRICS_PATH.read_text(encoding="utf-8"))
        if verbose:
            print("[test] 已存在評估結果，維持「只評估一次」不重跑")
    else:
        test_metrics = run_test_evaluation(
            bundle, X_test, y_test, force=force_test, n_boot=n_boot,
        )
        if verbose:
            print(f"[test] PR-AUC {test_metrics['metrics']['pr_auc']:.4f} "
                  f"R {test_metrics['metrics']['recall']:.3f} "
                  f"P {test_metrics['metrics']['precision']:.3f}")

    payload = dict(comparison)
    payload["calibration"] = calibration
    payload["final"] = {
        "model": best["model"],
        "strategy": best["strategy"],
        "calibrated": calibration["use_calibrated"],
        "threshold": bundle["threshold"],
        "model_path": str(BEST_MODEL_PATH.relative_to(ROOT)),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    DOCS.mkdir(parents=True, exist_ok=True)
    MODEL_DOC_PATH.write_text(
        render_markdown(payload, calibration=calibration, test_metrics=test_metrics),
        encoding="utf-8",
    )
    if verbose:
        print(f"[json] {COMPARISON_PATH}")
        print(f"[doc]  {MODEL_DOC_PATH}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="AI4I 模型訓練與比較")
    parser.add_argument("--no-tune", action="store_true", help="跳過超參數搜尋")
    parser.add_argument("--n-iter", type=int, default=8, help="RandomizedSearchCV 次數")
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap 次數")
    parser.add_argument("--force-test", action="store_true",
                        help="已評估過測試集仍強制重跑（不建議）")
    args = parser.parse_args()
    run_training(
        tune=not args.no_tune,
        n_iter=args.n_iter,
        n_boot=args.n_boot,
        force_test=args.force_test,
    )


if __name__ == "__main__":
    main()
