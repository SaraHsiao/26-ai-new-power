"""步驟 2：PCA 降維與 3D 特徵分布圖。"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import (CLASS_NAMES, FEATURES, FIG_DIR, RESULT_DIR, TARGET,
                   ensure_dirs, load_iris_df)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Segoe UI", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = ["#4C72B0", "#DD8452", "#55A868"]
MARKERS = ["o", "s", "^"]


def main() -> None:
    ensure_dirs()
    df = load_iris_df()
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    ratios = pca.explained_variance_ratio_
    print("[pca] 解釋變異量比例:", np.round(ratios, 4), "累積:", round(ratios.sum(), 4))

    # --- 3D 分布圖 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for name, color, marker in zip(CLASS_NAMES, COLORS, MARKERS):
        mask = y == name
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                   s=48, alpha=0.85, label=name, color=color,
                   marker=marker, edgecolor="white", linewidth=0.6)
    ax.set_xlabel(f"PC1 ({ratios[0]:.1%})", labelpad=8)
    ax.set_ylabel(f"PC2 ({ratios[1]:.1%})", labelpad=8)
    ax.set_zlabel(f"PC3 ({ratios[2]:.1%})", labelpad=8)
    ax.set_title("Iris 資料集 PCA 3D 特徵分布（前三個主成分）", fontsize=14, pad=18)
    ax.view_init(elev=22, azim=45)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_3d.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- 解釋變異量 ---
    fig, ax = plt.subplots(figsize=(7, 4.2))
    comps = np.arange(1, 5)
    full = PCA(n_components=4, random_state=42).fit(X_scaled)
    full_ratios = full.explained_variance_ratio_
    bars = ax.bar(comps, full_ratios * 100, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
                  alpha=0.85)
    for bar, r in zip(bars, full_ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{r:.1%}", ha="center", fontsize=9)
    ax.plot(comps, np.cumsum(full_ratios) * 100, "o--", color="#333333",
            label="累積解釋變異量")
    ax.set_xticks(comps, [f"PC{i}" for i in comps])
    ax.set_ylabel("解釋變異量 (%)")
    ax.set_ylim(0, 110)
    ax.set_title("各主成分解釋變異量")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_variance.png", dpi=170)
    plt.close(fig)

    result = {
        "explained_variance_ratio": [round(float(r), 4) for r in full_ratios],
        "cumulative_ratio": [round(float(r), 4) for r in np.cumsum(full_ratios)],
        "components_3d": [[round(float(v), 4) for v in row] for row in pca.components_],
    }
    (RESULT_DIR / "pca_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[pca] 圖表與結果已輸出（{FIG_DIR}）")


if __name__ == "__main__":
    main()
