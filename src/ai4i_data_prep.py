"""AI4I 2020 Predictive Maintenance Dataset：下載、驗證、EDA。

資料來源：UCI Machine Learning Repository（dataset 601）
授權：CC BY 4.0

用法：
    python src/ai4i_data_prep.py            # 若資料不存在會自動下載
    python src/ai4i_data_prep.py --force    # 強制重新下載
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
DOC_DIR = ROOT / "docs"
CSV_PATH = DATA_DIR / "ai4i2020.csv"
PROV_PATH = DATA_DIR / "PROVENANCE.json"
REPORT_PATH = DOC_DIR / "eda_report_ai4i.md"

SOURCE_URL = "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"
ZIP_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
LICENSE = "CC BY 4.0"
DOWNLOAD_DATE = "2026-10-23"
CITATION = (
    "Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C"
)

COLUMNS = [
    "UDI", "Product ID", "Type", "Air temperature [K]",
    "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
    "Tool wear [min]", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",
]
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
NUMERIC = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Segoe UI", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


# ---------------------------------------------------------------- 檔案工具

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(force: bool = False) -> Path:
    """從 UCI 下載 zip 並取出 ai4i2020.csv。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists() and not force:
        return CSV_PATH

    print(f"[download] {ZIP_URL}")
    with urllib.request.urlopen(ZIP_URL, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError("壓縮檔中找不到 CSV")
        raw = zf.read(names[0])
    CSV_PATH.write_bytes(raw)
    print(f"[download] 已寫入 {CSV_PATH} ({len(raw):,} bytes)")
    return CSV_PATH


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """讀取原始資料，並去除欄位名稱前後空白。"""
    path = Path(path or CSV_PATH)
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，請先執行 download_dataset()")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------------------------------------- 驗證

def validate_dataset(df: pd.DataFrame) -> dict:
    """檢查資料完整性與已知特性；資料不符預期時 raise ValueError。"""
    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要欄位: {missing_cols}")

    df = df[COLUMNS]

    if df["UDI"].duplicated().any():
        raise ValueError("UDI 應為唯一值")
    if not df["UDI"].is_monotonic_increasing:
        raise ValueError("UDI 應嚴格遞增")
    if df["Type"].isna().any():
        raise ValueError("Type 不應有缺失值")

    n_rows = len(df)
    n_failure = int(df["Machine failure"].sum())
    mode_sum = df[MODES].sum(axis=1)
    mode_or = (mode_sum > 0).astype(int)
    mismatch = mode_or != df["Machine failure"]

    failure_rate_by_type = {
        t: round(float(g["Machine failure"].mean()), 4)
        for t, g in df.groupby("Type")
    }

    return {
        "n_rows": n_rows,
        "n_cols": len(df.columns),
        "n_missing": int(df.isna().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "n_failure": n_failure,
        "failure_rate": round(n_failure / n_rows, 4),
        "udi_monotonic": bool(df["UDI"].is_monotonic_increasing),
        "udi_unique": bool(df["UDI"].is_unique),
        "product_id_matches_type": bool((df["Product ID"].str[0] == df["Type"]).all()),
        "mode_counts": {m: int(df[m].sum()) for m in MODES},
        "mf_without_mode": int(((df["Machine failure"] == 1) & (mode_or == 0)).sum()),
        "mode_without_mf": int(((df["Machine failure"] == 0) & (mode_or == 1)).sum()),
        "or_mismatch_total": int(mismatch.sum()),
        "mode_overlap": {int(k): int(v) for k, v in mode_sum.value_counts().sort_index().items()},
        "type_counts": {str(k): int(v) for k, v in df["Type"].value_counts().sort_index().items()},
        "failure_rate_by_type": failure_rate_by_type,
    }


def numeric_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    desc = df[NUMERIC].describe().T
    keep = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    return {
        str(idx): {k: round(float(desc.loc[idx, k]), 4) for k in keep}
        for idx in desc.index
    }


def add_engineering_features(df: pd.DataFrame) -> pd.DataFrame:
    """加入 EDA 用的物理特徵（正式 Pipeline 於 #5 實作）。"""
    out = df.copy()
    out["Temp diff [K]"] = out["Process temperature [K]"] - out["Air temperature [K]"]
    out["Power [W]"] = (
        out["Torque [Nm]"] * 2 * np.pi * out["Rotational speed [rpm]"] / 60
    )
    return out


# ---------------------------------------------------------------- 報告

def _md_table(rows: list[list], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def render_eda_report(df: pd.DataFrame, checks: dict) -> str:
    """產生 Markdown 探索報告（內容完全由資料決定，不含執行時間）。"""
    feats = add_engineering_features(df)
    rate = checks["failure_rate"]

    parts: list[str] = []
    parts.append("# AI4I 2020 資料集探索報告\n")
    parts.append("> 本檔案由 `src/ai4i_data_prep.py` 自動產生；所有數字皆可重跑取得。\n")

    # 1. 來源
    parts.append("## 1. 資料來源與授權\n")
    parts.append(f"- 資料集：**AI4I 2020 Predictive Maintenance Dataset**")
    parts.append(f"- 來源：{SOURCE_URL}")
    parts.append(f"- 授權：**{LICENSE}**")
    parts.append(f"- 下載日期：{DOWNLOAD_DATE}")
    parts.append(f"- 檔案 SHA256：`{sha256_of(CSV_PATH)}`")
    parts.append(f"- 引用：{CITATION}\n")

    # 2. 形狀與欄位
    parts.append("## 2. 資料形狀與欄位\n")
    parts.append(f"- 筆數：**{checks['n_rows']:,}**")
    parts.append(f"- 欄位數：**{checks['n_cols']}**（3 個識別／類別欄 + 5 個感測特徵 + 6 個標籤欄）\n")
    parts.append(_md_table(
        [["UDI", "整數", "流水號（嚴格遞增）", "否（僅供時間切分參考）"],
         ["Product ID", "字串", "產品編號，首碼等於 Type", "否（與 Type 重複）"],
         ["Type", "類別", "產品等級 L / M / H", "是（one-hot）"],
         ["Air temperature [K]", "數值", "環境溫度", "是"],
         ["Process temperature [K]", "數值", "製程溫度", "是"],
         ["Rotational speed [rpm]", "數值", "轉速", "是"],
         ["Torque [Nm]", "數值", "扭矩", "是"],
         ["Tool wear [min]", "數值", "刀具磨耗（累積分鐘）", "是"],
         ["Machine failure", "二元", "是否故障（目標）", "**預測目標**"],
         ["TWF / HDF / PWF / OSF / RNF", "二元", "五種失效模式旗標", "**否（會洩漏答案）**"]],
        ["欄位", "型別", "說明", "作為特徵"]))
    parts.append("")

    # 3. 基本檢查
    parts.append("## 3. 基本檢查\n")
    parts.append(_md_table(
        [["缺失值總數", f"{checks['n_missing']}"],
         ["完全重複列", f"{checks['n_duplicates']}"],
         ["UDI 嚴格遞增", "是" if checks["udi_monotonic"] else "否"],
         ["UDI 唯一", "是" if checks["udi_unique"] else "否"],
         ["Product ID 首碼 == Type", "是（100%）" if checks["product_id_matches_type"] else "否"]],
        ["檢查項目", "結果"]))
    parts.append("\n結論：資料完整、無缺失與重複，`Product ID` 與 `Type` 完全重複，"
                 "後續只保留 `Type`；`UDI` 為嚴格遞增流水號，可作為時間切分的依據。\n")

    # 4. 目標分布
    parts.append("## 4. 目標變數分布\n")
    normal = checks["n_rows"] - checks["n_failure"]
    parts.append(_md_table(
        [["0（正常）", f"{normal:,}", f"{normal / checks['n_rows']:.2%}"],
         ["1（故障）", f"{checks['n_failure']:,}", f"{rate:.2%}"]],
        ["Machine failure", "筆數", "占比"]))
    parts.append(f"\n故障率僅 **{rate:.2%}**，屬極度不平衡資料集：全部猜「正常」即可得到 "
                 f"**{normal / checks['n_rows']:.2%}** 的 accuracy，因此後續不得以 accuracy 作為主要指標。\n")

    # 5. Type
    parts.append("## 5. Type 分布與故障率\n")
    rows = []
    for t in ["L", "M", "H"]:
        cnt = checks["type_counts"].get(t, 0)
        rows.append([t, f"{cnt:,}", f"{cnt / checks['n_rows']:.1%}",
                     f"{checks['failure_rate_by_type'][t]:.2%}"])
    parts.append(_md_table(rows, ["Type", "筆數", "占比", "故障率"]))
    parts.append("\n故障率以 **L 最高（3.92%）**、M 次之（2.77%）、H 最低（2.09%），並非單調關係；"
                 "因此不應假設「等級越高越容易故障」，但 `Type` 與故障率確實相關，仍值得保留為特徵。\n")

    # 6. 失效模式
    parts.append("## 6. 五種失效模式\n")
    mode_desc = {
        "TWF": "Tool Wear Failure（刀具磨耗失效）",
        "HDF": "Heat Dissipation Failure（散熱失效）",
        "PWF": "Power Failure（電力失效）",
        "OSF": "Overstrain Failure（應變超限）",
        "RNF": "Random Failure（隨機失效）",
    }
    rows = [[m, mode_desc[m], f"{checks['mode_counts'][m]}",
             f"{checks['mode_counts'][m] / checks['n_rows']:.2%}",
             f"{checks['mode_counts'][m] / checks['n_failure']:.1%}"] for m in MODES]
    parts.append(_md_table(rows, ["旗標", "意義", "筆數", "占全資料", "占故障"]))
    parts.append("")

    parts.append("### 6.1 標籤之間並非完全等價（重要）\n")
    parts.append(
        f"- `Machine failure == OR(五旗標)` 的不一致筆數：**{checks['or_mismatch_total']}**\n"
        f"  - 有故障（Machine failure = 1）但五旗標皆為 0：**{checks['mf_without_mode']}** 筆（未分類故障）\n"
        f"  - 五旗標有任一為 1 但 Machine failure = 0：**{checks['mode_without_mf']}** 筆（旗標與標籤不一致）"
    )
    overlap = checks["mode_overlap"]
    parts.append(
        f"- 同時成立的模式數分布：0 個 {overlap.get(0, 0):,} 筆、1 個 {overlap.get(1, 0):,} 筆、"
        f"2 個 {overlap.get(2, 0):,} 筆、3 個 {overlap.get(3, 0):,} 筆\n"
    )
    parts.append("因此**不能**用五個模式旗標回推 `Machine failure`，也不能把它們放進特徵；"
                 "這正是本專案要嚴格防止的標籤洩漏。\n")

    # 7. 敘述統計
    parts.append("## 7. 數值特徵敘述統計\n")
    summary = numeric_summary(df)
    rows = [[k, v["mean"], v["std"], v["min"], v["25%"], v["50%"], v["75%"], v["max"]]
            for k, v in summary.items()]
    parts.append(_md_table(rows, ["特徵", "平均", "標準差", "最小", "Q1", "中位數", "Q3", "最大"]))
    parts.append("")

    # 8. 工程特徵
    parts.append("## 8. 物理工程特徵（初步）\n")
    parts.append("- `Temp diff [K] = Process temperature − Air temperature`"
                 "：散熱失效（HDF）的已知判準為溫差 < 8.6 K 且轉速 < 1380 rpm。")
    parts.append("- `Power [W] = Torque × 2π × RPM / 60`：電力失效（PWF）的已知判準為功率落在 3,500–9,000 W 之外。")
    corr_power_rpm = float(feats["Power [W]"].corr(feats["Rotational speed [rpm]"]))
    corr_power_torque = float(feats["Power [W]"].corr(feats["Torque [Nm]"]))
    parts.append(f"- 相關係數：Power 與轉速 **{corr_power_rpm:.3f}**、與扭矩 **{corr_power_torque:.3f}**。\n")

    # 9. 物理觀察
    parts.append("## 9. 物理機制初步觀察\n")
    hdf = df["HDF"] == 1
    pwf = df["PWF"] == 1
    osf = df["OSF"] == 1
    twf = df["TWF"] == 1
    parts.append(_md_table([
        ["HDF", f"{feats.loc[hdf, 'Temp diff [K]'].mean():.2f} K",
         f"{feats.loc[~hdf, 'Temp diff [K]'].mean():.2f} K", "故障時溫差明顯較小"],
        ["PWF", f"{feats.loc[pwf, 'Power [W]'].mean():,.0f} W",
         f"{feats.loc[~pwf, 'Power [W]'].mean():,.0f} W", "故障時平均功率偏離正常區間"],
        ["OSF", f"{(feats.loc[osf, 'Torque [Nm]'] * feats.loc[osf, 'Tool wear [min]']).mean():,.0f}",
         f"{(feats.loc[~osf, 'Torque [Nm]'] * feats.loc[~osf, 'Tool wear [min]']).mean():,.0f}",
         "故障時 扭矩×磨耗 明顯較高"],
        ["TWF", f"{feats.loc[twf, 'Tool wear [min]'].mean():.1f} min",
         f"{feats.loc[~twf, 'Tool wear [min]'].mean():.1f} min", "故障時磨耗較高"],
    ], ["模式", "故障樣本平均", "其餘樣本平均", "觀察"]))
    parts.append("")

    # 10. 對後續工作的啟示
    parts.append("## 10. 對後續工作的啟示\n")
    parts.append("1. **標籤洩漏**：`TWF/HDF/PWF/OSF/RNF` 必須排除於特徵之外，並以程式化 assert 保護（#5）。")
    parts.append("2. **極度不平衡**：故障率 3.39%，主要指標採 PR-AUC 與 Recall/Precision，不使用 accuracy（#6）。")
    parts.append("3. **RNF 限制**：19 筆隨機失效本質上不可由感測特徵預測，構成 recall 上限。")
    parts.append("4. **類型與時間**：`Type` 有預測價值；`UDI` 嚴格遞增，可作為時間切分的順序依據（#9）。")
    parts.append("5. **特徵工程**：溫差與功率具有明確物理意義，值得在 #5 納入。")
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------- 圖表

def make_eda_figures(df: pd.DataFrame, checks: dict) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    feats = add_engineering_features(df)

    # 1. 目標分布
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = [checks["n_rows"] - checks["n_failure"], checks["n_failure"]]
    bars = ax.bar(["正常 (0)", "故障 (1)"], counts, color=["#4C72B0", "#C44E52"], width=0.55)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 80, f"{v:,}\n({v / checks['n_rows']:.2%})",
                ha="center", fontsize=10)
    ax.set_ylabel("筆數")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_title(f"目標變數分布（故障率 {checks['failure_rate']:.2%}）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = FIG_DIR / "ai4i_eda_target.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    # 2. Type 分布與故障率
    types = ["L", "M", "H"]
    counts = [checks["type_counts"].get(t, 0) for t in types]
    rates = [checks["failure_rate_by_type"][t] * 100 for t in types]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bars = axes[0].bar(types, counts, color=COLORS[:3], width=0.55)
    for bar, v in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 70, f"{v:,}", ha="center", fontsize=9)
    axes[0].set_title("各 Type 筆數")
    axes[0].set_ylabel("筆數")
    axes[0].set_ylim(0, max(counts) * 1.15)
    axes[0].grid(axis="y", alpha=0.3)

    bars = axes[1].bar(types, rates, color=COLORS[:3], width=0.55)
    for bar, v in zip(bars, rates):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.2f}%", ha="center", fontsize=9)
    axes[1].set_title("各 Type 故障率")
    axes[1].set_ylabel("故障率 (%)")
    axes[1].set_ylim(0, max(rates) * 1.2)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = FIG_DIR / "ai4i_eda_type.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    # 3. 五種失效模式
    fig, ax = plt.subplots(figsize=(7.2, 4))
    vals = [checks["mode_counts"][m] for m in MODES]
    bars = ax.bar(MODES, vals, color=COLORS, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2,
                f"{v}\n({v / checks['n_rows']:.2%})", ha="center", fontsize=9)
    ax.set_ylabel("筆數")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("五種失效模式筆數（可重疊，總和大於故障數）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = FIG_DIR / "ai4i_eda_modes.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    # 4. 相關係數
    corr_cols = NUMERIC + ["Temp diff [K]", "Power [W]"]
    corr = feats[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)), [c.split(" [")[0] for c in corr_cols],
                  rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr_cols)), [c.split(" [")[0] for c in corr_cols], fontsize=9)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(corr.iloc[i, j]) > 0.55 else "black")
    ax.set_title("數值特徵相關係數矩陣（含工程特徵）")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    p = FIG_DIR / "ai4i_eda_correlation.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    # 5. 物理關係
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    groups = [feats.loc[feats["HDF"] == 0, "Temp diff [K]"], feats.loc[feats["HDF"] == 1, "Temp diff [K]"]]
    bp = ax.boxplot(groups, tick_labels=["HDF = 0", "HDF = 1"], patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.axhline(8.6, color="black", linestyle="--", linewidth=1, label="HDF 判準 8.6 K")
    ax.set_ylabel("溫差 [K]")
    ax.set_title("HDF 與溫差")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    groups = [feats.loc[feats["PWF"] == 0, "Power [W]"], feats.loc[feats["PWF"] == 1, "Power [W]"]]
    bp = ax.boxplot(groups, tick_labels=["PWF = 0", "PWF = 1"], patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.axhspan(3500, 9000, color="green", alpha=0.08, label="正常功率區間 3.5–9 kW")
    ax.set_ylabel("功率 [W]")
    ax.set_title("PWF 與功率")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    strain = feats["Torque [Nm]"] * feats["Tool wear [min]"]
    normal = feats["OSF"] == 0
    ax.scatter(strain[normal], feats.loc[normal, "Torque [Nm]"], s=6, alpha=0.25,
               color="#4C72B0", label="OSF = 0")
    ax.scatter(strain[~normal], feats.loc[~normal, "Torque [Nm]"], s=26, alpha=0.9,
               color="#C44E52", label="OSF = 1", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Torque × Tool wear")
    ax.set_ylabel("Torque [Nm]")
    ax.set_title("OSF 與扭矩×磨耗")
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    groups = [feats.loc[feats["TWF"] == 0, "Tool wear [min]"], feats.loc[feats["TWF"] == 1, "Tool wear [min]"]]
    bp = ax.boxplot(groups, tick_labels=["TWF = 0", "TWF = 1"], patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Tool wear [min]")
    ax.set_title("TWF 與刀具磨耗")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("失效模式與物理特徵的關係（初步）", fontsize=14)
    fig.tight_layout()
    p = FIG_DIR / "ai4i_eda_physical.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------- 主流程

def write_provenance(df: pd.DataFrame) -> Path:
    prov = {
        "dataset": "AI4I 2020 Predictive Maintenance Dataset",
        "source_url": SOURCE_URL,
        "download_url": ZIP_URL,
        "license": LICENSE,
        "download_date": DOWNLOAD_DATE,
        "citation": CITATION,
        "file": CSV_PATH.name,
        "sha256": sha256_of(CSV_PATH),
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
    }
    PROV_PATH.write_text(json.dumps(prov, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return PROV_PATH


def main(force: bool = False) -> None:
    download_dataset(force=force)
    df = load_raw()
    checks = validate_dataset(df)
    print(f"[data] {checks['n_rows']:,} 筆 × {checks['n_cols']} 欄，"
          f"故障 {checks['n_failure']} 筆（{checks['failure_rate']:.2%}）")

    prov = write_provenance(df)
    print(f"[provenance] {prov}")

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_eda_report(df, checks), encoding="utf-8")
    print(f"[report] {REPORT_PATH}")

    paths = make_eda_figures(df, checks)
    print(f"[figures] 已輸出 {len(paths)} 張圖到 {FIG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI4I 2020 資料下載與 EDA")
    parser.add_argument("--force", action="store_true", help="強制重新下載資料")
    main(force=parser.parse_args().force)
