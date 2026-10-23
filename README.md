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

---

# AI4I 2020 預測性維護（Predictive Maintenance）

對應 Issue：[#3 總覽](https://github.com/SaraHsiao/26-ai-new-power/issues/3)｜[#4 資料取得與 EDA](https://github.com/SaraHsiao/26-ai-new-power/issues/4)

## 資料來源與授權

- 資料集：**AI4I 2020 Predictive Maintenance Dataset**
- 來源：UCI Machine Learning Repository（dataset 601）
  <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
- 授權：**CC BY 4.0**
- 檔案：`data/ai4i2020.csv`（10,000 筆 × 14 欄），雜湊與下載資訊記錄於 `data/PROVENANCE.json`
- 引用：Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. <https://doi.org/10.24432/C5HS5C>

## 執行方式

```bash
python src/ai4i_data_prep.py            # 資料不存在時自動下載，並產生 EDA 報告與圖表
python src/ai4i_data_prep.py --force    # 強制重新下載
python -m pytest tests -q               # 執行測試
```

## 目前進度

- [x] #4 資料取得、保存與 EDA
- [x] #5 前處理與特徵工程 Pipeline（見 [`docs/features.md`](docs/features.md)）
- [x] #6 評估協定、指標與不平衡策略（見 [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md)）
- [ ] #7 模型建立、比較與校準
- [ ] #8 評估視覺化、可解釋性與誤判分析
- [ ] #9 穩健性檢查：時間切分對照
- [ ] #10 報告撰寫與交付整備

## EDA 重點

- 10,000 筆、14 欄、**0 缺失值、0 重複列**；`UDI` 嚴格遞增且唯一。
- 故障 **339 筆（3.39%）**，屬極度不平衡；全猜正常即有 **96.61%** accuracy。
- 五種失效模式：HDF 115、OSF 98、PWF 95、TWF 46、RNF 19。
- **標籤並非等價**：9 筆有故障但五旗標全 0；18 筆旗標為 1 但 `Machine failure` = 0。
- `Type` 故障率：L 3.09%、M 3.38%、H 4.89%，具預測價值。
- 完整內容見 [`docs/eda_report_ai4i.md`](docs/eda_report_ai4i.md)。
