# AI4I 2020 評估協定

> 本文件定義「怎麼評分、怎麼挑閾值、怎麼比較」。程式實作於 `src/ai4i_eval.py`，
> 規則若有爭議，以程式與測試為準。

## 1. 為什麼不用 accuracy

資料集故障率 **3.39%**（339 / 10,000）。一個永遠預測「正常」的模型：

- accuracy = **96.61%**
- 但 recall = 0，完全抓不到任何故障

因此 accuracy 在本專案**不作為主要指標**，只保留為輔助欄位，用來對照「看似很高但沒有用」的模型。
主要指標是 **PR-AUC（Average Precision）**，並輔以 Precision / Recall / Lift。

## 2. 指標定義

| 指標 | 定義 | 用途 |
| --- | --- | --- |
| **PR-AUC** | `average_precision_score(y, proba)` | **主要指標**；對不平衡資料比 ROC-AUC 敏感 |
| Precision | TP / (TP + FP) | 誤報有多少 |
| Recall | TP / (TP + FN) | 漏檢有多少 |
| F1 | 2PR / (P + R) | 單一平衡分數 |
| MCC | Matthews correlation coefficient | 考慮 TN 的整體相關性 |
| Lift | Precision / prevalence | 相對「隨機猜」提升幾倍 |
| accuracy | (TP + TN) / N | **僅參考**，不可作為驗收依據 |

`evaluate()` 回傳的字典包含上表所有指標、混淆矩陣（tp/fp/fn/tn）與
`primary_metric = "pr_auc"`。

### Dummy 基線

全猜「故障」的基線：PR-AUC = 故障率（0.0339）、Recall = 1、Lift = 1。
任何模型的 PR-AUC 若沒有明顯高於 0.0339，就等於沒有學到東西。

## 3. 切分與 OOF 協定

- **隨機切分**：80% 訓練 / 20% 測試，`stratify=y`、`random_state=42`。
  測試集約有 **68 筆故障**（339 × 20%）。
- **交叉驗證**：訓練集內用 Stratified 5-fold（`random_state=42`）。
- **OOF 預測**：`collect_oof()` 讓每一筆訓練資料都被「沒看過它」的模型預測一次，
  用來選閾值與比較模型；`OOFResult` 內含每筆資料所屬的 fold。
- **時間切分**（#9）：以 `UDI` 排序後切分，作為穩健性對照，不取代隨機切分。

## 4. 閾值選擇規則（強制）

`choose_threshold()` 只接受 `collect_oof()` 產生的 `OOFResult`。
若傳入原始陣列（包含測試集預測），會直接 raise `TypeError`：

```python
choice = choose_threshold(oof, cost_fn=10.0, cost_fp=1.0)
```

- 在 OOF 預測上，對每個候選閾值計算混淆矩陣。
- 最小化加權成本：`cost = cost_fn × FN + cost_fp × FP`。
- 預設成本假設為 **漏檢 : 誤報 = 10 : 1**（`cost_fn=10.0`、`cost_fp=1.0`），
  可依場域調整；提高 `cost_fn` 會把操作點推向高 recall。
- 回傳 `ThresholdChoice`，內含閾值、成本、每筆平均成本與該點的完整指標。

> 不允許用測試集選閾值、挑模型或調整任何超參數。

## 5. 測試集使用規範

- 測試集**只評估一次**：模型與閾值都定案後才載入測試集。
- 測試集不得用於特徵選擇、模型選擇、閾值調整或早期停止。
- 報告中的數字必須能對應 `results/*.json`，不可手動抄寫。
- 若需要來回實驗，一律在訓練集的 CV / OOF 上進行。

## 6. Bootstrap 信賴區間

`bootstrap_ci()` 以 1,000 次重抽樣（可調整）估計指標的 95% 百分位區間：

```python
ci = bootstrap_ci(y_test, test_proba, metric="pr_auc", n_boot=1000, random_state=42)
# -> {"point": ..., "lower": ..., "upper": ..., ...}
```

- 支援 `pr_auc`、`precision`、`recall`、`f1`、`lift`、`accuracy`、`mcc`。
- 重抽樣若只剩單一類別，`pr_auc` 無定義，該次會跳過（不計入百分位）。
- 邊界情況（全對、全錯）已有單元測試：全對的 recall CI 為 `[1, 1]`，
  全錯為 `[0, 0]`。
- 隨機種子固定，結果可重現。

## 7. 不平衡策略

| 策略 | 實作 | 說明 |
| --- | --- | --- |
| `none` | 原始模型 | 對照組 |
| `class_weight` | `class_weight="balanced"` | 依類別頻率加權，不需額外套件 |
| `SMOTE` | `imblearn`（選配） | 未安裝時 `build_imbalance_pipeline("smote", ...)` 會 raise `ImportError` 並提示安裝 |

SMOTE 只會在**訓練 fold 內**執行（`imblearn.pipeline.Pipeline`），
不會把合成樣本混進驗證或測試資料。

## 8. 成本假設表

| 情況 | 代價 | 預設權重 |
| --- | --- | --- |
| 漏檢（FN，實際故障但未預警） | 停機、維修、安全風險 | 10 |
| 誤報（FP，無故障卻預警） | 多一次檢查 | 1 |

成本可調整，例如 `cost_fn=50` 代表漏檢代價是誤報的 50 倍。
`ThresholdChoice.metrics` 會直接給出該操作點的 Precision / Recall / F1，
方便與現場人員討論「這個閾值要抓多少、誤報多少」。

## 9. 驗收門檻（對應 #3）

| 層級 | 條件 |
| --- | --- |
| 必要 | PR-AUC ≥ 0.80，且 bootstrap 95% 下界 ≥ 0.70 |
| 必要 | Lift ≥ 10×（相對故障率 3.39%） |
| 必要 | Recall ≥ 0.80 時 Precision ≥ 0.60 |
| 加分 | 時間切分（#9）下仍維持合理水準 |
| 誠實揭露 | RNF（19 筆隨機失效）不可預測，recall 上限約 0.94；必須在報告中說明 |

## 10. 使用範例

```python
from ai4i_features import load_dataset, make_pipeline
from ai4i_eval import (collect_oof, choose_threshold, evaluate,
                       bootstrap_ci, build_imbalance_pipeline)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)

# 1) 只用訓練集 + OOF 選閾值
pipe = build_imbalance_pipeline("class_weight", LogisticRegression(max_iter=1000))
oof = collect_oof(pipe, X_train, y_train, n_splits=5, random_state=42)
choice = choose_threshold(oof, cost_fn=10.0, cost_fp=1.0)

# 2) 測試集只評估一次
pipe.fit(X_train, y_train)
test_proba = pipe.predict_proba(X_test)[:, 1]
metrics = evaluate(y_test, test_proba, threshold=choice.threshold)

# 3) 不確定性
ci = bootstrap_ci(y_test, test_proba, metric="recall",
                  threshold=choice.threshold, n_boot=1000)
```
