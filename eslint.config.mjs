// ESLint flat config（CLI 用，無需編輯器整合）
//
// 檔案範圍分兩組，各自宣告全域，避免 no-undef 誤報：
//   [1] Node 端：scripts/、tests/（JS 檔）與 Lint 設定 -> node 全域
//   [2] 瀏覽器端：docs/ 底下前端 JS（GitHub Pages，<script> 載入）-> browser 全域
//
// 門檻（需求）：complexity 10、max-depth 3、max-lines-per-function 50、cognitive-complexity 15
import globals from "globals";
import sonarjs from "eslint-plugin-sonarjs";

const SONARJS_RECOMMENDED = { ...sonarjs.configs.recommended };

const THRESHOLD_RULES = {
  complexity: ["error", 10],
  "max-depth": ["error", 3],
  "max-lines-per-function": [
    "error",
    { max: 50, skipBlankLines: true, skipComments: true, IIFEs: false },
  ],
  "sonarjs/cognitive-complexity": ["error", 15],
};

export default [
  // [1] Node 端：scripts/、tests/（僅 JS 檔，Python 不適用）、根目錄腳本
  {
    files: ["scripts/**/*.{js,cjs}", "tests/**/*.{js,cjs}", "*.js"],
    ...SONARJS_RECOMMENDED,
    languageOptions: {
      ecmaVersion: 2024,
      sourceType: "commonjs",
      globals: { ...globals.node },
    },
  },
  {
    files: ["scripts/**/*.{js,cjs}", "tests/**/*.{js,cjs}", "*.js"],
    rules: THRESHOLD_RULES,
  },
  // Node 端 ESM（.mjs）
  {
    files: ["scripts/**/*.mjs", "tests/**/*.mjs", "*.mjs"],
    ...SONARJS_RECOMMENDED,
    languageOptions: {
      ecmaVersion: 2024,
      sourceType: "module",
      globals: { ...globals.node },
    },
  },
  {
    files: ["scripts/**/*.mjs", "tests/**/*.mjs", "*.mjs"],
    rules: THRESHOLD_RULES,
  },

  // [2] 瀏覽器端：docs/ 底下前端 JS
  {
    files: ["docs/**/*.{js,mjs}"],
    ...SONARJS_RECOMMENDED,
    languageOptions: {
      ecmaVersion: 2024,
      sourceType: "script",
      globals: { ...globals.browser },
    },
  },
  {
    files: ["docs/**/*.{js,mjs}"],
    rules: THRESHOLD_RULES,
  },

];
