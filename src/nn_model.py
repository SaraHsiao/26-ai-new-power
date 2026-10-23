"""步驟 3：類神經網路（MLP）分類模型訓練與評估。"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from utils import (CLASS_NAMES, FEATURES, FIG_DIR, RANDOM_STATE, RESULT_DIR,
                   TARGET, ensure_dirs, load_iris_df)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Segoe UI", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CONFIG = {
    "hidden_layer_sizes": (32, 16),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "learning_rate_init": 1e-3,
    "batch_size": 16,
    "max_iter": 600,
    "early_stopping": True,
    "n_iter_no_change": 30,
    "validation_fraction": 0.15,
    "random_state": RANDOM_STATE,
}


def main() -> None:
    ensure_dirs()
    df = load_iris_df()
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    print(f"[split] train={len(X_train)}  test={len(X_test)}")

    # 訓練集 5 折交叉驗證，確認模型穩定性
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(MLPClassifier(**CONFIG), X_train_s, y_train, cv=cv)
    print(f"[cv] 5-fold accuracy = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model = MLPClassifier(**CONFIG)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                   digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)
    print(f"[nn] test accuracy = {acc:.4f}  (epochs run = {len(model.loss_curve_)})")

    # --- 訓練曲線 ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(range(1, len(model.loss_curve_) + 1), model.loss_curve_, color="#C44E52")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training loss")
    axes[0].set_title("訓練損失曲線")
    axes[0].grid(alpha=0.3)

    if model.validation_scores_:
        axes[1].plot(range(1, len(model.validation_scores_) + 1),
                     model.validation_scores_, color="#4C72B0")
    axes[1].axhline(acc, color="#55A868", linestyle="--", label=f"test accuracy = {acc:.3f}")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("驗證集正確率與測試集表現")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nn_training_curves.png", dpi=170)
    plt.close(fig)

    # --- Confusion matrix ---
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(f"測試集混淆矩陣（accuracy = {acc:.3f}）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nn_confusion_matrix.png", dpi=170)
    plt.close(fig)

    # --- 各類別 F1 長條圖 ---
    fig, ax = plt.subplots(figsize=(6.4, 4))
    f1 = [report[c]["f1-score"] for c in CLASS_NAMES]
    bars = ax.bar(CLASS_NAMES, f1, color=["#4C72B0", "#DD8452", "#55A868"], alpha=0.85)
    for bar, v in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1-score")
    ax.set_title("各類別 F1-score（測試集）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nn_f1_scores.png", dpi=170)
    plt.close(fig)

    result = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CONFIG.items()},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "epochs_run": int(len(model.loss_curve_)),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "cv_scores": [round(float(s), 4) for s in cv_scores],
        "test_accuracy": round(float(acc), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "misclassified": [
            {"index": int(i), "true": str(t), "pred": str(p)}
            for i, (t, p) in enumerate(zip(y_test, y_pred)) if t != p
        ],
    }
    (RESULT_DIR / "nn_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: result[k] for k in
                      ("test_accuracy", "cv_accuracy_mean", "cv_accuracy_std", "epochs_run")},
                     ensure_ascii=False, indent=2))
    print(f"[nn] 結果已輸出到 {RESULT_DIR} 與 {FIG_DIR}")


if __name__ == "__main__":
    main()
