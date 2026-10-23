"""一鍵執行全流程：資料準備 -> 特徵 -> 評估協定驗證 -> 訓練比較 -> 視覺化與誤判 -> 時間切分穩健性"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(cmd):
    print(f"\n{'='*60}")
    print(f"執行: {' '.join(cmd)}")
    print('='*60)
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        print(f"警告: {' '.join(cmd)} 回傳碼 {res.returncode}")
    return res.returncode


def main():
    steps = [
        [sys.executable, "src/ai4i_data_prep.py"],
        [sys.executable, "src/ai4i_train.py"],
        [sys.executable, "src/ai4i_evaluate.py"],
        [sys.executable, "src/ai4i_time_split.py"],
        [sys.executable, "src/check_consistency.py"],
        [sys.executable, "-m", "pytest", "tests", "-q"],
    ]
    for cmd in steps:
        run(cmd)
    print("\n全流程完成")


if __name__ == "__main__":
    main()
