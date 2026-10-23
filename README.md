# Iris Dataset 分析與類神經網路分類

以 Iris 資料集完成「資料取得 → 探索 → PCA 3D 視覺化 → 類神經網路分類 → 小論文」的完整流程。

> 對應 Issue：[#2 [ML] Iris Dataset 分析與類神經網路分類](https://github.com/SaraHsiao/26-ai-new-power/issues/2)

## 專案結構

```
.
├─ data/iris.csv              # 原始資料（150 筆，含特徵與 species）
├─ src/
│  ├─ utils.py                # 共用路徑與 Markdown 表格工具
│  ├─ data_prep.py            # 匯出 CSV、EDA、箱型圖與分布圖
│  ├─ pca_3d.py               # PCA 降維與 3D 特徵分布圖
│  ├─ nn_model.py             # MLP 訓練、交叉驗證、評估與視覺化
│  └─ run_all.py              # 一鍵重現全部結果
├─ figures/                   # 輸出圖表（PCA 3D、訓練曲線、混淆矩陣…）
├─ results/                   # 評估結果 JSON（pca_result.json、nn_metrics.json）
├─ docs/
│  ├─ eda_report.md           # 自動產生的資料探索報告
│  └─ iris-nn-paper.md        # 小論文
└─ README.md
```

## 環境需求

- Python 3.12（3.10 以上應可運作）

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

主要套件：`numpy`、`pandas`、`scikit-learn`、`matplotlib`。

## 執行方式

```bash
python src/run_all.py
```

或分步執行：

```bash
python src/data_prep.py   # 產生 data/iris.csv、docs/eda_report.md、EDA 圖表
python src/pca_3d.py      # 產生 figures/pca_3d.png、figures/pca_variance.png
python src/nn_model.py    # 訓練 MLP，產生訓練曲線、混淆矩陣、results/nn_metrics.json
```

所有隨機性均固定 `random_state = 42`，重複執行可得到相同結果。

## 資料說明

| 欄位 | 說明 |
| --- | --- |
| `sepal_length` | 花萼長度（cm） |
| `sepal_width` | 花萼寬度（cm） |
| `petal_length` | 花瓣長度（cm） |
| `petal_width` | 花瓣寬度（cm） |
| `species` | 物種：`setosa`、`versicolor`、`virginica` |

三類各 50 筆，共 150 筆；無缺失值。

## 結果摘要

| 項目 | 結果 |
| --- | --- |
| PCA 前三分量解釋變異量 | 72.96% / 22.85% / 3.67%（累積 99.48%） |
| 訓練集 5-fold CV accuracy | 95.83% ± 4.56% |
| 測試集 accuracy | **96.67%**（29/30） |
| Macro F1 | 0.967 |
| 誤判 | 1 筆 versicolor → virginica |

詳細分析與圖表討論請見 [`docs/iris-nn-paper.md`](docs/iris-nn-paper.md)。

## 圖表

| 檔案 | 內容 |
| --- | --- |
| `figures/pca_3d.png` | PCA 三維特徵分布圖 |
| `figures/pca_variance.png` | 各主成分解釋變異量 |
| `figures/nn_training_curves.png` | 訓練損失與驗證正確率曲線 |
| `figures/nn_confusion_matrix.png` | 測試集混淆矩陣 |
| `figures/nn_f1_scores.png` | 各類別 F1-score |
| `figures/eda_boxplots.png` | 各特徵於三類別的箱型圖 |
| `figures/eda_distributions.png` | 各特徵分布直方圖 |

## 資料來源

- Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179–188.
- UCI Machine Learning Repository — Iris Data Set。
- 本專案透過 `sklearn.datasets.load_iris` 取得資料後匯出為 `data/iris.csv`。
