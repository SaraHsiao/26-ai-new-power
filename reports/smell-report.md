# 靜態檢查報告（Code Smell Report）

> 產生時間：2026-10-23 ｜ 執行範圍：`docs/` 與 `scripts/` ｜ 指令：`npm run check`（ESLint + jscpd + knip，另外單獨跑一份放寬格式的 jscpd 附錄）
>
> 網頁版：<a href="../docs/reports/smell-report.html">docs/reports/smell-report.html（靜態頁「分析結果」分頁）</a> ｜ 追蹤事件單：**#11（單一完整 issue**；原 #12–#15 已整併進 #11**）**

**本報告只陳述檢查結果，未修改任何既有程式。** 檢查範圍內的 JS 檔共 3 個：

| 檔案 | 行數 | 說明 |
|---|---:|---|
| `docs/app.js` | 173 | 手繪風前端：讀取模型 JSON、前向推論、推論表單事件 |
| `docs/site.js` | 2 | `tools/build_site.py` 產生的報告清單／圖表清單常數 |
| `scripts/check_lint.mjs` | 182 | Lint 工具彙總腳本（本次檢查工具本身，非既有產品程式） |

---

## 1. 各項警告總數（依規則分組）

ESLint（`docs/`＋`scripts/`）總計 **5 個 error、0 個 warning**：

| 規則 | 門檻 | docs/ | scripts/ | 總數 |
|---|---|---:|---:|---:|
| `complexity` | ≤10 | 0 | 1 | **1** |
| `max-depth` | ≤3 | 0 | 1 | **1** |
| `max-lines-per-function` | ≤50 | 0 | 0 | **0** |
| `sonarjs/cognitive-complexity` | ≤15 | 0 | 0 | **0** |
| `sonarjs/no-nested-template-literals` | — | 0 | 2 | **2** |
| `sonarjs/no-os-command-from-path` | — | 0 | 1 | **1** |
| （sonarjs recommended 其餘規則） | — | 0 | 0 | **0** |
| **合計** | | **0** | **5** | **5** |

明細（全部落在 `scripts/check_lint.mjs` 這支檢查工具腳本；實際前端程式 `docs/app.js`、`docs/site.js` 全數通過，`docs/` 目前是 0 警告）：

| 位置 | 規則 | 訊息 |
|---|---|---|
| `scripts/check_lint.mjs:151` | `complexity` | Arrow function has a complexity of 20. Maximum allowed is 10. |
| `scripts/check_lint.mjs:71` | `max-depth` | Blocks are nested too deeply (4). Maximum allowed is 3. |
| `scripts/check_lint.mjs:174` | `sonarjs/no-nested-template-literals` | Refactor this code to not use nested template literals. |
| `scripts/check_lint.mjs:189` | `sonarjs/no-nested-template-literals` | Refactor this code to not use nested template literals. |
| `scripts/check_lint.mjs:133` | `sonarjs/no-os-command-from-path` | Make sure the "PATH" variable only contains fixed, unwriteable directories. |

追蹤事件單：**#11（單一完整 issue）**——原 #12（complexity）、#13（max-depth）、#14（巢狀 template literals）、#15（os-command-from-path）已整併進 #11。

> 註：ESLint 依 eslint.config.mjs 分範圍取全域 —— `docs/**` 用 `globals.browser`、`scripts/**`／`tests/**` 用 `globals.node`，`no-undef` 已確認沒有誤報（`fetch`、`document` 等瀏覽器全域通過）。

## 2. 重複率與重複片段位置

**JS 範圍（docs/*.js、scripts/*.mjs）：重複率 0%**（jscpd 來源 2 檔、0 個重複片段、重複 0 行，`minTokens=30`）。

附錄 —— 把 `docs/` 內生成的 HTML 報告、`results.json`、CSS、Python 一併送檢（較嚴格檢視）：

| 指標 | 數值 |
|---|---|
| 來源檔數 | 29 |
| 重複片段（clones） | 19 |
| 重複行數 | 169 |
| **重複率** | **4.37%** |

重複片段位置（皆為 `minTokens≥30`、行數≥5 的片段）：

| 片段 A | 片段 B | 行數／token |
|---|---|---|
| reports/ai4i-report.html:1–6 | reports/model_comparison.html:1–6 | 6 / 54 |
| reports/ai4i-report.html:6–15 | reports/model_comparison.html:6–15 | 10 / 132 |
| reports/features.html:1–6 | reports/model_comparison.html:1–6 | 6 / 54 |
| reports/features.html:6–15 | reports/model_comparison.html:6–15 | 10 / 132 |
| reports/evaluation_protocol.html:1–6 | reports/model_comparison.html:1–6 | 6 / 54 |
| reports/evaluation_protocol.html:6–15 | reports/model_comparison.html:6–15 | 10 / 132 |
| reports/evaluation_figures.html:1–6 | reports/model_comparison.html:1–6 | 6 / 54 |
| reports/evaluation_figures.html:6–15 | reports/model_comparison.html:6–15 | 10 / 132 |
| reports/eda_report_ai4i.html:1–6 | reports/model_comparison.html:1–6 | 6 / 54 |
| reports/eda_report_ai4i.html:6–15 | reports/model_comparison.html:6–15 | 10 / 132 |
| reports/eda_report.html:1–6 | reports/model_comparison.html:1–6 | 6 / 50 |
| reports/eda_report.html:6–15 | reports/eda_report_ai4i.html:6–15 | 10 / 129 |
| reports/iris-nn-paper.html:1–6 | reports/model_comparison.html:1–6 | 6 / 50 |
| reports/iris-nn-paper.html:6–15 | reports/model_comparison.html:6–15 | 10 / 128 |
| docs/results.json:314–341 | docs/results.json:205–232 | 28 / 192 |
| docs/results.json:520–539 | docs/results.json:478–497 | 20 / 146 |
| docs/inference.html:85–90 | docs/inference.html:45–50 | 6 / 47 |
| docs/index.html:1–6 | reports/model_comparison.html:1–6 | 6 / 52 |
| docs/index.html:15–30 | docs/inference.html:100–115 | 16 / 263 |

**解讀**：HTML 重複主要來自 `tools/build_site.py` 生成的報告頁共用頁首／頁尾樣板（9 份報告 × 2 段），屬生成產物、非手寫程式；`results.json` 內部幾段結構相似的 metrics 區塊；`inference.html` 自身 2 個表單卡片結構相同（6 行 CSS）；`index.html:15–30` 與 `inference.html:100–115` 共用頁首 layout。生成內容的重複可以接受（改動應修 `build_site.py` 樣板或 `style.css`，不是改 HTML）。

## 3. 未使用的檔案與函式

**未使用檔案（knip）：0 個**。交叉驗證 `<script src>` 引用與 npm scripts：

| 檔案 | 被誰載入 |
|---|---|
| `docs/site.js` | `docs/index.html`（內嵌 `<script src="site.js">`） |
| `docs/app.js` | `docs/inference.html`（`<script src="./app.js">`） |
| `scripts/check_lint.mjs` | `package.json` 的 `"check"` / `"lint"` scripts 與 GitHub issue #11 之後的 CI 流程 |

**未使用函式：0 個**。範圍內共 10 個具名函式，全部有實際呼叫點：

| 函式 | 定義位置 | 使用位置 |
|---|---|---|
| `loadModels` | app.js:8 | app.js:136（DOMContentLoaded） |
| `softmax` | app.js:23 | app.js:66（forwardIris） |
| `forwardIris` | app.js:33 | app.js:150（iris 表單 submit） |
| `treePredict` | app.js:69 | app.js:119（predictAI4I） |
| `predictAI4I` | app.js:88 | app.js:166（ai4i 表單 submit） |
| `showResult` | app.js:131 | app.js:142、160、165、171 |
| `run`／`nl`／`devName` | check_lint.mjs | check_lint.mjs 內部互呼 |
| `card`／`fmt` | index.html（內嵌 script） | index.html 內嵌 main() |

> knip 只檢「未使用匯出／檔案」；函式層級以上表手動掃描補齊（含 `index.html` 內嵌 script 的 `card`、`fmt`、`main()`）。

## 4. 「前端推論」相關程式規模

推論程式集中在 **`docs/app.js`**（推論表單頁 `docs/inference.html` 的小寫互動邏輯）：

| 面向 | 行數 | 函式數 |
|---|---:|---:|
| **推論核心**（模型前向計算） | **104** | **4** |
| 推論整體（含模型載入、表單事件與結果顯示） | 173 | 7 |

推論核心 4 個函式：

| 函式 | 位置 | 行數 | 內容 |
|---|---|---:|---|
| `softmax` | app.js:23–31 | 9 行 | 多類別 softmax 轉機率 |
| `forwardIris` | app.js:33–67 | 35 行 | Iris MLP 手工前向（標準化＋逐層 ReLU） |
| `treePredict` | app.js:69–86 | 18 行 | 單棵樹（平坦陣列 5 值節點）推論 |
| `predictAI4I` | app.js:88–129 | 42 行 | 8 欄特徵工程＋ 5 折森林平均＋ sigmoid 校準 |

輔助層：`loadModels`（模型 JSON 載入，14 行）、`showResult`（結果渲染，3 行）、`DOMContentLoaded` listener（表單事件與兩張推論表的 submit 處理，39 行）——合計 56 行、3 個函式單元。`docs/site.js` 不含推論邏輯（2 行生成常數、0 函式）。

## 小結

- 既有前端程式（`docs/app.js`、`docs/site.js`）在四項複雜度/長度門檻與 sonarjs 規則下**全部乾淨**（0 警告）；目前唯一的 5 個 ESLint error 都在檢查工具腳本 `scripts/check_lint.mjs` 自身（#12–#15 已開單）。
- JS 重複率 0%；若把生成產物也算，4.37% 的重複全部可歸因於 `build_site.py` 樣板與共通頁首（建議有需求時在樣板層調整）。
- 無死碼：3 個 JS 檔與所有函式均有實際使用點。
- 追蹤事件單：#11（單一完整 issue；原 #12–#15 已整併進 #11）。

## 修復紀錄（2026-10-23，Lab4 五項依序修復）

> 修正範圍僅 `scripts/check_lint.mjs`（檢查工具腳本自身；`docs/` 前端程式與所有測試檔未動）。每項修復後先跑完整測試（pytest 131 ＋ node:test 4），全綠才進行下一項；修復不使用 eslint-disable、不改任何測試斷言。「警告數」＝ ESLint error 總數（修復前基準 5）。

### 修復 1／5：`complexity`（原行號 151；Arrow function complexity 20 > 10）

- **修了什麼**：knip `problems` 的 flatMap arrow 內含 9 段「清列表→push」迴圈，複雜度 20。改成宣告式對照表 `ISSUE_KIND_FIELDS`（8 個陣列欄位 → 種類字串 → 目標值抽法）＋ 新函式 `issueProblemItems()`；`enumMembers` 是物件鍵而非陣列，保留原樣走迴圈。呼叫端改為 `problems: issues.flatMap(issueProblemItems)`。行為逐一比對不變：同 9 種 kind 字串與 target 抽法（`devName`、`b.name`、`e.symbol`…）。
- **警告數變化**：5 → **4**（complexity 歸零）
- **測試數**：pytest **131 passed** ＋ node:test **4 passed**；`npm run check` 其餘指標不變（jscpd 0%、knip 未使用檔案 0／未使用匯出 0，knip `problems` 仍為 0）。

### 修復 2／5：`max-depth`（原行號 71；try → if → for → for 嵌套 4 層 > 3）

- **修了什麼**：ESLint 段落為 `try { if (existsSync) { for f { for m { … } } } }`，最深 4 層。把「解析報表 → 依規則計數 + findings 彙總」抽成頂層函式 `eslintSummary(results)`（含既有 `nl()`、`ROOT` 去前綴邏輯，行為不變）；呼叫端只剩 `if (existsSync) { summary.eslint = eslintSummary(...) }`。`.reports/summary.json` 的 `byRule / findings / errorCount` 結構逐一比對不變。
- **警告數變化**：4 → **3**
- **測試數**：pytest **131 passed** ＋ node:test **4 passed**；jscpd 0%、knip 死碼 0 不變。

### 修復 3／5：`sonarjs/no-nested-template-literals`（原行號 174、189，共 2 處）

- **修了什麼**：同 issue 的兩處「template literal 內再插一層 template literal」——knip 異常分支的 `${cond ? \`；輸出：…\` : ""}` 與彙總段的 `${others.map(→\`${k}=${v}\`)…}`。改法都是**先求值到區域變數再插值**：`knipDetail`、`otherText`。輸出字串內容逐字不變。
- **警告數變化**：3 → **1**（no-nested-template-literals 歸零）
- **測試數**：pytest **131 passed** ＋ node:test **4 passed**；jscpd 0%、knip 死碼 0 不變。

### 修復 4／5：`sonarjs/no-os-command-from-path`（原行號 133；用 PATH 上的 `node` 啟動子行程）

- **修了什麼**：knip 段落直接 `spawnSync("node", …)`（被規則標記），另兩處相同的 smell 一併修正：eslint、jscpd 都是「`"node"` 字串餵給 `run()` → `spawnSync(command, …)`」，子行程全走 PATH 查找。三處改為 `process.execPath`（目前 node 執行檔的絕對路徑，不經 PATH 解析）；eslint / jscpd binary 本已用 `path.join(ROOT, "node_modules", …)` 給定，knip 同。行為等價（同一個 node binary），工具仍正常產出三份報表。
- **警告數變化**：1 → **0**
- **測試數**：pytest **131 passed** ＋ node:test **4 passed**；`npm run check` 三段檢查在 `process.execPath` 下全部照常運作（eslint/jscpd/knip 報表都重新產生）。

### 修復後總結（5 項全數完成）

| 項次 | 規則（原行號 #12–#15 對應） | 改法摘要 | 警告數 |
|---|---|---|---|
| 1 | `complexity`（151） | 9 段 push 迴圈（8 個陣列欄位進對照表、enumMembers 字典鍵留迴圈）→ 對照表 `ISSUE_KIND_FIELDS` ＋ `issueProblemItems()` | 5→4 |
| 2 | `max-depth`（71） | 彙總迴圈抽頂層函式 `eslintSummary()`，try/if 下不再疊 4 層 | 4→3 |
| 3 | `no-nested-template-literals`（174、189） | 內層模板先求值入 `knipDetail`／`otherText` | 3→1 |
| 4 | `no-os-command-from-path`（133） | 三處子行程 `"node"` → `process.execPath` | 1→0 |

> 註：smell-report 分組表把 5 個 error 對應到 4 個 issue（#14 nested-template 一張單含 2 處），故「五項」以 5 個 error 計，實修為 4 個 issue、4 次提交式修復循環（每次循環含完整測試）。

- **最終 ESLint：errors=0、warnings=0**（含門檻類 complexity / max-depth / max-lines-per-function / cognitive-complexity 全為 0）；jscpd 0%；knip 未使用檔案 0／匯出 0。
- 全程未使用 eslint-disable、未修改任何測試（斷言原封不動），每項修復後完整測試全綠才繼續；`docs/` 前端程式零修改。
- 本表記錄的「原行號」為修復前基準；第 1、2 項的中途行號位移屬正常（抽函式使檔案增長）。
- HTML 版 `docs/reports/smell-report.html` 為 build_site 產物，本次未重建；修復紀錄以本檔為準。
