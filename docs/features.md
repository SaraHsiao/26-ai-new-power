# AI4I 2020 特徵工程說明

本文件說明 `src/ai4i_features.py` 的特徵設計與**洩漏防護政策**。程式與文件不一致時，以程式為準，並請補測試。

## 1. 欄位取捨

原始資料 14 欄，只有 **6 欄** 進入模型：

| 原始欄位 | 用途 | 理由 |
| --- | --- | --- |
| `Type` | 特徵（one-hot） | 產品等級，與故障率相關（L 3.92% / M 2.77% / H 2.09%） |
| `Air temperature [K]` | 特徵 | 環境溫度 |
| `Process temperature [K]` | 特徵 | 製程溫度 |
| `Rotational speed [rpm]` | 特徵 | 轉速 |
| `Torque [Nm]` | 特徵 | 扭矩 |
| `Tool wear [min]` | 特徵 | 刀具累積磨耗 |
| `UDI` | **排除** | 流水號，無物理意義（僅供時間切分使用） |
| `Product ID` | **排除** | 首碼與 `Type` 完全重複 |
| `Machine failure` | **目標** | 預測目標 |
| `TWF` / `HDF` / `PWF` / `OSF` / `RNF` | **排除** | 直接組成答案，屬標籤洩漏 |

### 洩漏防護（不可繞過）

`assert_no_leakage()` 會在以下時機檢查並 raise `ValueError`：

1. `load_dataset()` 產生 `X` 時
2. `add_features()` 收到含洩漏欄位的 DataFrame 時

被擋下的欄位定義在 `FORBIDDEN_IN_X = ID_COLUMNS + LEAKAGE_COLUMNS + [TARGET]`。
這是**程式化保護**，不依賴人工檢查；對應測試為
`test_assert_no_leakage_rejects_forbidden_columns`（8 個參數化案例）。

> 補充：EDA 發現 `Machine failure` 與 `OR(五旗標)` 有 27 筆不一致（9 筆未分類故障、18 筆旗標與標籤衝突），
> 因此更不能用旗標推導標籤，也不適合當特徵。

## 2. 工程特徵

| 特徵 | 公式 | 對應物理機制 |
| --- | --- | --- |
| `Temp diff [K]` | `Process temperature − Air temperature` | HDF（散熱失效）：溫差 < 8.6 K 且轉速 < 1380 rpm |
| `Power [W]` | `Torque × 2π × RPM / 60` | PWF（電力失效）：功率落在 3,500–9,000 W 之外 |
| `Torque x Tool wear` | `Torque × Tool wear` | OSF（應變超限）：扭矩與磨耗的乘積過高 |

EDA 已驗證的方向性：

- HDF 樣本平均溫差 8.23 K，其餘 10.02 K
- PWF 樣本平均功率 7,252 W，其餘 6,270 W
- OSF 樣本 `Torque × Tool wear` 平均 12,067，其餘 4,238
- TWF 樣本平均磨耗 216.4 min，其餘 107.4 min

## 3. 類別編碼

`Type` 以 one-hot 編碼，類別順序固定為 `["H", "L", "M"]`，輸出欄名為
`Type_H`、`Type_L`、`Type_M`，確保跨次執行欄位順序一致。
`OneHotEncoder(handle_unknown="ignore")` 讓未見過的類別（例如推論時出現 `XL`）
不會造成錯誤，該列會是 all-zero。

## 4. 標準化

- 數值區塊（5 個感測 + 3 個工程特徵）使用 `StandardScaler`。
- one-hot 欄位**不標準化**（保持 0/1）。
- `build_preprocessor(scale=False)` 可關閉標準化，供樹模型使用。
- 標準化參數只在訓練資料上 `fit`，測試資料僅 `transform`；
  對應測試 `test_preprocessor_fit_on_train_does_not_see_test`。

## 5. 共線性

`Power` 與 `Torque` 的相關係數為 **0.979**、與轉速為 **−0.806**（`Power` 本身就是兩者的函數）。
線性模型（Logistic Regression）對此敏感，必要時可移除：

```python
prep = build_preprocessor(drop_columns=["Power [W]"])
```

樹模型通常不受影響，預設保留。是否移除留待 #7 以 CV 結果決定。

## 6. 使用方式

```python
from ai4i_features import load_dataset, make_pipeline, get_feature_names
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X, y = load_dataset()

pipe = make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
scores = cross_val_score(pipe, X, y, cv=5, scoring="average_precision")

print(get_feature_names())   # 11 個欄位：8 數值 + 3 one-hot
```

Pipeline 內含特徵工程、編碼與標準化，因此可直接餵給 `cross_val_score`，
不會發生「先全資料標準化再切分」的洩漏。

## 7. 輸出欄位清單（共 11 欄）

| # | 欄位 | 類型 |
| --- | --- | --- |
| 1 | `Air temperature [K]` | 數值（標準化） |
| 2 | `Process temperature [K]` | 數值（標準化） |
| 3 | `Rotational speed [rpm]` | 數值（標準化） |
| 4 | `Torque [Nm]` | 數值（標準化） |
| 5 | `Tool wear [min]` | 數值（標準化） |
| 6 | `Temp diff [K]` | 工程特徵（標準化） |
| 7 | `Power [W]` | 工程特徵（標準化） |
| 8 | `Torque x Tool wear` | 工程特徵（標準化） |
| 9 | `Type_H` | one-hot |
| 10 | `Type_L` | one-hot |
| 11 | `Type_M` | one-hot |
