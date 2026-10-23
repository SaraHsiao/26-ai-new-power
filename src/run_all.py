"""一鍵重現：依序執行資料前處理、PCA 視覺化與模型訓練。"""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

for script in ("data_prep", "pca_3d", "nn_model"):
    print(f"\n=== {script} ===")
    runpy.run_module(script, run_name="__main__")
