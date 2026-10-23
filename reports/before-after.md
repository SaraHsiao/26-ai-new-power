# 修復前後對照報告（Before / After）

> 產生時間：2026-10-23 ｜ 對應事件：Lab4 挑出的 5 項 ESLint error 依序修復（追蹤於 #11，原 #12–#15）
> 修復對象：`scripts/check_lint.mjs`（檢查工具腳本自身）；`docs/` 前端程式與所有測試檔零修改。
> 過程紀錄（每項：修了什麼／警告數變化／測試數）見 `reports/smell-report.md` 之「修復紀錄」章節。

---

## 1. 指標修正前後對照

| 指標 | 修復前 | 修復後 | 變化 |
|---|---|---|---|
| **ESLint error 總數** | **5** | **0** | −5 |
| — `complexity`（門檻 ≤10） | 1（單一函式 complexity **20**） | 0 | −1 |
| — `max-depth`（門檻 ≤3） | 1（最深嵌套 **4** 層） | 0 | −1 |
| — `sonarjs/no-nested-template-literals` | 2 | 0 | −2 |
| — `sonarjs/no-os-command-from-path` | 1（`spawnSync("node")` 走 PATH） | 0 | −1 |
| — `max-lines-per-function`（≤50） | 0 | 0 | — |
| — `sonarjs/cognitive-complexity`（≤15） | 0 | 0 | — |
| ESLint warning 總數 | 0 | 0 | — |
| ESLint 檢查檔案數 | 6 | 6 | — |
| ESLint findings（summary.json） | 5 筆 | 0 筆 | −5 |
| **jscpd JS 重複率**（來源 4 檔，minTokens 30） | **0%**（0 個重複片段） | **0%**（0 個重複片段） | — |
| 附錄：放寬口徑 jscpd（含生成產物，報告 §2） | 4.37%（19 片段） | 未重跑（本次僅動 `scripts/check_lint.mjs`，不影響生成產物） | — |
| **knip 未使用檔案 / 未使用匯出** | 0 / 0 | 0 / 0 | — |
| knip `problems`（summary.json，行為不變的直接證據） | 0 | 0 | — |
| **測試數** | pytest **131** ＋ node:test **4** | pytest **131** ＋ node:test **4** | — |
| 測試執行結果 | 全綠 | 全綠（**5 次執行 × 135/135**：4 個修復循環＋最終驗證，每次都全綠才進下一項） | — |
| `scripts/check_lint.mjs` 行數 | 195 | 219 | +24 |

> 附註：修復六步的 ESLint error 計數軌跡：**5 → 4 → 3 → 1 → 0**（第 3 項一次清掉同規則的 2 個 error），每步測試均全綠。

---

## 2. 五項 diff 摘要

> 口徑：unified diff（僅計修改行）。五項合計 **+59 行 / −35 行**，與檔案 195 → 219 行（淨 +24）對賬吻合。行號為修復前基準。

| # | 規則／原位置 | 檔案 | 增／刪行數 | 重點內容 |
|---|---|---|---|---|
| 1 | `complexity:151`（arrow complexity 20） | `scripts/check_lint.mjs` | **+24 / −14** | functions 區新增 `ISSUE_KIND_FIELDS` 對照表（10 行）＋ `issueProblemItems()`（10 行）＝ +22；knip `problems` 由 14 行（problems 開頭＋`f`／`items` 宣告＋**9 段**清列表/push 迴圈＋`return`＋收尾）改為 1 行單行 `flatMap(issueProblemItems)` ＝ +1 / −14 |
| 2 | `max-depth:71`（try→if→for→for 共 4 層） | `scripts/check_lint.mjs` | **+28 / −16** | `summary` 後新增 `eslintSummary(results)` 函式（註解 2 行＋函式本體 25 行＝ +27）；ESLint 段落內 16 行（`byRule`／`findings` 宣告、雙層迴圈、`rel` 拼路徑、summary 組裝）抽走，改 1 行 `summary.eslint = eslintSummary(results)` ＝ +1 / −16 |
| 3 | `no-nested-template-literals:174、189`（2 個 error，同 issue 一併修） | `scripts/check_lint.mjs` | **+4 / −2** | 兩處各把內層模板先求值進變數再插值：knip 1 行 → `knipDetail` 變數＋console.log 2 行；彙總段 1 行 → `otherText` 變數＋console.log 2 行。輸出字串逐字不變 |
| 4 | `no-os-command-from-path:133` | `scripts/check_lint.mjs` | **+3 / −3** | 三處 1 行替換（1:1）：knip `spawnSync("node",…)` → `spawnSync(process.execPath,…)`；eslint、jscpd 兩處 `run(…, "node", …)` → `run(…, process.execPath, …)`（同一 smell 的同構呼叫點） |
| 合計 | | | **+59 / −35** | |

（註：smell-report 分組表中 5 個 error 對應 4 個 issue——#14 nested-template 一張單含 2 處；上表以「5 項」＝5 個 error 揭列，實際執行為 4 個修復循環。）

---

## 3. 哪一項最難修？—— 第 1 項 `complexity`

**難度排序：項 1（complexity）＞ 項 4 ＞ 項 2 ＞ 項 3。** 項 1 最難，理由有四：

1. **它是唯一「改變結構形狀」的修復，其他項只是搬家或替換**。項 2 是把程式碼原封搬到新函式（位置變更）、項 3 是先行求值（等價轉換）、項 4 是一個 token 的替換（`"node"` → `process.execPath`，但跨三處）；只有項 1 要把 **9 段異質迴圈**收斂成一張資料表＋一個函式，是真正的設計判斷。
2. **每段迴圈有三個獨立變化軸，不能收錯**：列表欄位（`dependencies`…`duplicates`）、種類字串（`unlisted-dependency`…`duplicate-export`）、目標值抽法（`devName`／`b.name`／`e.symbol`／identity 共 4 種）——9 個 kind × 對應抽法必須逐一對齊，收錯一列 knip summary 就會出錯，而且**沒有測試會抓到**。
3. **做對了之後要另行驗證，因為它落在測試護網之外**：`npm test`／pytest 只涵蓋 `docs/` 與 `src/`，`scripts/check_lint.mjs` 沒有單元測試。本次以「修復前後各跑一次 `npm run check`，比對 `.reports/summary.json` 的欄位逐一相同（`problems`、`unusedDevDependencies`、`unresolvedPlugins`…，兩次皆為空輸出）」作為行為不變的證據——這是本次修復中唯一需要自建驗證方式的項目。
4. **語意陷阱最多**：`enumMembers` 是「物件鍵」而非陣列，無法直接進對照表（獨立留迴圈）；`devDependencies` 在檔案裡出現兩次、兩處語意不同（`problems` 內的 `unused-devDependency` vs 獨立的 `unusedDevDependencies` 摘要欄），容易誤併成同一張表；identity 抽法要用解構預設參數放在表內才乾淨；且新表＋函式必須定義在 `const summary` 之前（否則立即執行的初始化直接 ReferenceError——檢查工具自己壞掉，後續所有驗證就都失去依據）。

項 4 的難度在第二位：替換本身只是一行，但要辨認「同構 smell 在三處」（knip 直接 spawn、eslint/jscpd 經 `run()` 間接 spawn），只修被標記的那一行可以過 lint，卻會留下不一致；且要選對正面解法（`process.execPath` 給絕對路徑，而不是用 `resolve` 把相對名稱轉絕對）。

---

## 4. 驗證方式備忘

- **每次修復後的完整測試**：`python -X utf8 -m pytest tests/ -q`（131 passed）＋ `npm test`（node:test 4 passed）→ 全綠才繼續下一項。
- **工具自身行為不變**：每項修復後重跑 `npm run check`，eslint / jscpd / knip 三段照常產生報表，`summary.json` 欄位逐一相同。
- **關鍵 diff 無行為差異的檢查點**：項 1 後 `knip.problems`／`knip.unusedDevDependencies` 結構與值不變（0＝0）；項 2 後 `summary.eslint.byRule`／`findings` 欄位相同且立即被新基準（3 筆 findings）正確填入——即修復後第一次執行就把當時剩餘的 3 個 error 正確收錄，證明抽函式後邏輯等價；項 3 後 console 輸出字串逐字相同；項 4 後三個工具照常運行（同一 node binary）。
- **紅燒條款**：全程未使用 `eslint-disable`，未修改任何測試或斷言，零紅燈發生。
