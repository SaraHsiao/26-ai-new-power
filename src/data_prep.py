"""步驟 1：取得 Iris Dataset 並保存為 CSV。

資料來源：R. A. Fisher (1936) 的 Iris 資料集，此處透過
scikit-learn 內建的 `load_iris` 取得（與 UCI Iris 相同內容）。
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

from utils import (CLASS_NAMES, CSV_PATH, FEATURES, FIG_DIR, RANDOM_STATE,
                   TARGET, ensure_dirs, md_table)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Segoe UI", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def export_iris_csv() -> pd.DataFrame:
    ensure_dirs()
    iris = load_iris(as_frame=True)
    df = iris.data.copy()
    df.columns = FEATURES
    df[TARGET] = [CLASS_NAMES[i] for i in iris.target]
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    return df


def explore(df: pd.DataFrame) -> str:
    """基本探索與檢查，回傳 Markdown 摘要。"""
    lines = []
    lines.append("# Iris 資料集探索報告\n")
    lines.append("> 本檔案由 `src/data_prep.py` 自動產生。\n")

    lines.append("## 1. 資料形狀與型別\n")
    lines.append(f"- 資料筆數：**{df.shape[0]}**")
    lines.append(f"- 欄位數：**{df.shape[1]}**（4 個特徵 + 1 個類別標籤）")
    lines.append(f"- 特徵欄位：`{'`、`'.join(FEATURES)}`")
    lines.append(f"- 類別欄位：`{TARGET}`\n")
    lines.append("| 欄位 | 型別 | 非空值數 |")
    lines.append("| --- | --- | --- |")
    for col in df.columns:
        lines.append(f"| {col} | {df[col].dtype} | {df[col].notna().sum()} |")
    lines.append("")

    missing = int(df.isna().sum().sum())
    dup = int(df.duplicated().sum())
    lines.append("## 2. 缺失值與重複值\n")
    lines.append(f"- 缺失值總數：**{missing}**")
    lines.append(f"- 完全重複的資料列：**{dup}**")
    if dup:
        lines.append("\n> 重複列對應的是不同觀測個體剛好量到相同數值，因此保留不刪除。")
    lines.append("")

    lines.append("## 3. 敘述統計\n")
    desc = df[FEATURES].describe().T
    desc["range"] = desc["max"] - desc["min"]
    desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "range"]]
    desc.index.name = "特徵"
    lines.append(md_table(desc.round(3), index=True))
    lines.append("")

    lines.append("## 4. 類別分布\n")
    counts = df[TARGET].value_counts().reindex(CLASS_NAMES)
    lines.append("| 類別 | 筆數 | 比例 |")
    lines.append("| --- | --- | --- |")
    for name, cnt in counts.items():
        lines.append(f"| {name} | {cnt} | {cnt / len(df):.1%} |")
    lines.append("\n三類別各 50 筆，屬**完全平衡**資料集。\n")

    lines.append("## 5. 離群值檢查（IQR 準則）\n")
    lines.append("| 特徵 | Q1 | Q3 | IQR | 下界 | 上界 | 離群值數 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for col in FEATURES:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        lines.append(
            f"| {col} | {q1:.2f} | {q3:.2f} | {iqr:.2f} | {low:.2f} | {high:.2f} | {n_out} |"
        )
    lines.append("")

    lines.append("## 6. 各特徵與類別的關係（分組平均）\n")
    grouped = df.groupby(TARGET)[FEATURES].mean().reindex(CLASS_NAMES)
    lines.append(md_table(grouped.round(3), index=True))
    lines.append("")

    lines.append("## 7. 相關係數矩陣（Pearson）\n")
    corr = df[FEATURES].corr().round(3)
    corr.index.name = "特徵"
    lines.append(md_table(corr, index=True))
    lines.append("")

    return "\n".join(lines)


def plot_eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, col in zip(axes.ravel(), FEATURES):
        data = [df.loc[df[TARGET] == c, col].values for c in CLASS_NAMES]
        bp = ax.boxplot(data, tick_labels=CLASS_NAMES, patch_artist=True, widths=0.55)
        for patch, color in zip(bp["boxes"], ["#4C72B0", "#DD8452", "#55A868"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(col.replace("_", " ").title())
        ax.set_ylabel("cm")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Iris 各特徵於三個類別的箱型圖", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_boxplots.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, col in zip(axes.ravel(), FEATURES):
        for name, color in zip(CLASS_NAMES, ["#4C72B0", "#DD8452", "#55A868"]):
            ax.hist(df.loc[df[TARGET] == name, col], bins=12, alpha=0.6,
                    label=name, color=color, edgecolor="white")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("cm")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Iris 各特徵的分布（依類別）", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_distributions.png", dpi=160)
    plt.close(fig)


def main() -> None:
    df = export_iris_csv()
    print(f"[data] 已輸出 {CSV_PATH} ({len(df)} rows)")
    print(f"[data] 類別分布: {df[TARGET].value_counts().to_dict()}")

    report = explore(df)
    out = FIG_DIR.parent / "docs" / "eda_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"[eda] 探索報告已寫入 {out}")

    plot_eda(df)
    print(f"[eda] 圖表已輸出到 {FIG_DIR}")


if __name__ == "__main__":
    main()
