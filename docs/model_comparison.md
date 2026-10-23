# AI4I 2020 模型比較與校準

> 本檔案由 `src/ai4i_train.py` 自動產生；數字與 `results/model_comparison.json` 一致。

## 1. 實驗設定

- 訓練資料：**8,000** 筆，其中故障 **271** 筆
- 交叉驗證：Stratified **5-fold**，`random_state=42`
- 不平衡策略：`none`, `class_weight`
- 超參數搜尋：RandomizedSearchCV（n_iter=8）
- 閾值規則：OOF 上最小化 `10×FN + 1×FP`
- 測試集：**僅在模型定案後評估一次**，未參與任何選擇

> accuracy 僅供參考：全部猜「正常」即可得到 96.61%，因此不作為主要指標。

## 2. 模型比較（依 OOF PR-AUC 排序）

| 模型 | 策略 | PR-AUC (OOF) | fold mean ± std | 95% CI | 閾值 | Precision | Recall | F1 | Lift | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest ⭐ | none | 0.9016 | 0.9026 ± 0.0261 | 0.8715 – 0.9310 | 0.151 | 0.772 | 0.860 | 0.813 | 22.78 | 0.0073 |
| hist_gradient_boosting | class_weight | 0.8911 | 0.8919 ± 0.0192 | 0.8589 – 0.9218 | 0.589 | 0.922 | 0.830 | 0.874 | 27.22 | 0.0084 |
| random_forest | class_weight | 0.8845 | 0.8862 ± 0.0264 | 0.8527 – 0.9134 | 0.584 | 0.632 | 0.863 | 0.730 | 18.67 | 0.0210 |
| hist_gradient_boosting | none | 0.8826 | 0.8831 ± 0.0233 | 0.8489 – 0.9139 | 0.165 | 0.877 | 0.841 | 0.859 | 25.89 | 0.0078 |
| logistic | none | 0.4958 | 0.5005 ± 0.0351 | 0.4287 – 0.5560 | 0.115 | 0.324 | 0.642 | 0.431 | 9.57 | 0.0229 |
| logistic | class_weight | 0.4552 | 0.4586 ± 0.0466 | 0.3955 – 0.5181 | 0.786 | 0.320 | 0.635 | 0.426 | 9.46 | 0.1094 |
| dummy | none | 0.0338 | 0.0339 ± 0.0003 | 0.0294 – 0.0378 | 0.034 | 0.034 | 0.797 | 0.065 | 1.00 | 0.0327 |

## 3. 最佳模型

- 模型：**random_forest** ＋ 策略 `none`
- OOF PR-AUC：**0.9016**（95% CI 0.8715 – 0.9310）
- 操作點：閾值 **0.151**、Precision **0.772**、Recall **0.860**、Lift **22.78×**
- 超參數：`{"n_estimators": 150, "min_samples_leaf": 4, "max_features": 0.4, "max_depth": 8}`

## 4. 機率校準（OOF）

- 方法：`sigmoid`（cv=3）

| 指標 | 校準前 | 校準後 | 差異 |
| --- | --- | --- | --- |
| Brier score（越低越好） | 0.0073 | 0.0071 | -0.0002 |
| PR-AUC | 0.9016 | 0.8974 | -0.0042 |
| Recall @ 選定閾值 | 0.860 | 0.852 | — |
| Precision @ 選定閾值 | 0.772 | 0.834 | — |

決策：Brier score 下降，因此**採用校準後模型**（決策僅根據訓練集 OOF，未看測試集）。

## 5. 測試集評估（僅一次）

- 測試筆數：**2,000**（故障 68 筆）
- 閾值：**0.085**（由 OOF 決定，未用測試集調整）

| 指標 | 數值 |
| --- | --- |
| PR-AUC | **0.9008**（95% CI 0.8323 – 0.9552） |
| Precision | 0.8788 |
| Recall | 0.8529（95% CI 0.7571 – 0.9286） |
| F1 | 0.8657 |
| MCC | 0.8611 |
| Lift | 25.85× |
| accuracy（僅參考） | 0.9910 |
| 混淆矩陣 | TP 58 / FP 8 / FN 10 / TN 1924 |

## 6. 重現方式

```bash
python src/ai4i_train.py        # 重新訓練、比較、校準（測試集若已評估會拒絕重跑）
python -m pytest tests -q       # 全部測試
```

載入模型：

```python
from ai4i_train import load_model
bundle = load_model('results/best_model.joblib')
proba = bundle['model'].predict_proba(X_new)[:, 1]
pred = (proba >= bundle['threshold']).astype(int)
```

## 7. 限制

- RNF（19 筆隨機失效）無法由感測特徵預測，recall 存在上限（約 0.94）。
- 測試集僅 68 筆故障，指標的信賴區間偏寬；已用 bootstrap 呈現不確定性。
- 隨機切分假設資料可交換；時間漂移的影響見 #9 的時間切分對照。
