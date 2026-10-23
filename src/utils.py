"""共用工具：路徑設定與資料載入。"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"
DOC_DIR = ROOT / "docs"

CSV_PATH = DATA_DIR / "iris.csv"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]
RANDOM_STATE = 42


def ensure_dirs() -> None:
    for d in (DATA_DIR, FIG_DIR, RESULT_DIR, DOC_DIR):
        d.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, index: bool = False, floatfmt: str = "{:.3f}") -> str:
    """把 DataFrame 轉成 Markdown 表格（避免依賴 tabulate）。"""
    cols = [str(c) for c in df.columns]
    header = ([str(df.index.name or "index")] if index else []) + cols
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for idx, row in df.iterrows():
        cells = ([str(idx)] if index else [])
        for value in row:
            if isinstance(value, float):
                cells.append(floatfmt.format(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_iris_df() -> pd.DataFrame:
    """從專案內的 CSV 讀取資料；若不存在則先由 scikit-learn 匯出。"""
    if not CSV_PATH.exists():
        from data_prep import export_iris_csv  # noqa: PLC0415

        export_iris_csv()
    return pd.read_csv(CSV_PATH)
