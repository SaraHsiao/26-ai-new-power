# AI4I 2020 Predictive Maintenance 分析報告

## 摘要

本研究針對 AI4I 2020 Predictive Maintenance Dataset（10,000 筆樣本，僅 3.4% 故障率）進行端到端預測性維護分析。採用 scikit-learn Pipeline 防止資料洩漏，僅以 OOF 決定閾值，測試集僅評估一次。最終以校準後 RandomForest 為最佳模型，在測試集達到 PR-AUC 0.9008、Recall 0.8529、Precision 0.8788、Lift 25.85×，符合驗收門檻。

## 緒論

預測性維護（Predictive Maintenance）旨在在設備故障前預測並採取預防措施，降低停機成本與安全風險。本資料集來自 UCI Machine Learning Repository（Matzka 2020），模擬真實工業感測資料。

## 資料說明

- 資料來源：UCI AI4I 2020 Predictive Maintenance Dataset（doi:10.24432/C5HS5C）
- 樣本數：10,000 筆
- 特徵：UDI、Product ID、Type、Air temperature、Process temperature、Rotational speed、Torque、Tool wear
- 目標：Machine failure（二元），僅預測此一目標
- 類別不平衡：故障率 3.39%（339/10000）
- 失效模式：TWF、HDF、PWF、OSF、RNF（僅作 per-mode recall 分析，不個別建模）

## 研究方法

1. **前處理 Pipeline**：使用 ColumnTransformer + FunctionTransformer 自動產生工程特徵（Temp diff、Power、Torque x Tool wear），並以 OneHotEncoder 處理 Type，StandardScaler 縮放數值特徵，防止資料洩漏。
2. **不平衡策略**：比較 none/class_weight/SMOTE（若無 imbalanced-learn 則自動跳過）。
3. **模型比較**：LogisticRegression、RandomForest、HistGradientBoosting（選配 XGBoost/LightGBM）。
4. **閾值選擇**：僅使用 5-fold Stratified OOF 預測，由 OOF 找出最佳閾值，測試集僅評估一次。
5. **校準**：以 OOF Brier score 決定是否使用 CalibratedClassifierCV（sigmoid），最終採用校準。
6. **評估指標**：PR-AUC 為主要指標，輔以 Recall/Precision/F1/MCC/Lift，並以 bootstrap 計算信賴區間。

## 實驗設計

- 切分：Stratified 80/20，random_state=42
- 交叉驗證：5-fold StratifiedKFold（shuffle=True, random_state=42）
- 閾值：僅由 OOF 決定
- 穩健性：UDI 排序時間切分（前80% 訓練、後20% 測試）對照

## 結果

### 模型比較
最佳模型：RandomForest（none 策略），經 sigmoid 校準。

### 測試集表現
| 指標 | 數值 |
|------|------|
| PR-AUC | 0.9008 |
| Precision | 0.8788 |
| Recall | 0.8529 |
| F1 | 0.8657 |
| MCC | 0.8611 |
| Lift | 25.85× |
| 混淆矩陣 | TP=58 FP=8 FN=10 TN=1924 |

### 特徵重要度與誤判分析
詳見 `docs/evaluation_figures.md`、`results/ai4i_metrics.json`、`results/error_cases.csv`。主要缺口為 TWF（10/10 故障中僅抓到 2 筆），其餘可預測模式（HDF/PWF/OSF）測試集幾乎全抓。

### 時間切分穩健性
- 隨機切分 PR-AUC：0.9008
- UDI 時間切分 PR-AUC：0.8081
- 差異：-0.0928
- 結論：穩定：PR-AUC 差異 <= 0.10，時間切分表現與隨機切分相近

## 討論

1. **洩漏防範**：所有特徵工程在 Pipeline 內執行，測試集只用於最終評估一次，閾值僅由 OOF 決定。
2. **不平衡處理**：PR-AUC 優於 accuracy 作為主要指標；PR 更適合 3.4% 不平衡資料。
3. **校準**：採用 sigmoid 校準改善機率可靠性，使機率可用於成本決策。
4. **RNF 上限**：RNF 為隨機失效，幾乎不與 Machine failure 重疊，無法由感測特徵預測，不應視為模型缺陷。
5. **穩健性**：時間切分顯示模型在順序性變化下仍有合理表現。

## 結論與未來改進

結論：本方案達成預測性維護的實用目標，在極度不平衡情況下仍能以高 Precision 抓到多數故障，Lift 超過 25 倍。

未來改進：
- 加入更多領域特徵與交互項（尤其針對 TWF 漏檢）
- 考慮成本敏感學習（根據誤報/漏報成本調整閾值）
- 收集真實時間序列資料以更精確評估時間穩健性
- 探索異常偵測輔助傳統監督學習

## 參考文獻

- Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE, 10(3).
