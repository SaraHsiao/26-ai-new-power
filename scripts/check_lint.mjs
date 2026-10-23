#!/usr/bin/env node
// 一次跑完所有靜態檢查：ESLint、jscpd（重複率）、knip（死碼）。
//
// 設計重點：只產生報表與彙總數字（輸出到 .reports/），
// 不因為檢查結果讓整體流程失敗——先「跑得動、跑得出數字」，
// 之後決定門檻是否要改成會擋 CI。
//
// 用法：npm run check
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, rmSync, writeFileSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const REPORTS = path.join(ROOT, ".reports");

if (!existsSync(path.join(ROOT, "node_modules"))) {
  console.error("[check] 找不到 node_modules，請先在專案根目錄執行：npm install");
  process.exit(1);
}
rmSync(REPORTS, { recursive: true, force: true });
mkdirSync(REPORTS, { recursive: true });

function run(name, command, args) {
  const res = spawnSync(command, args, {
    cwd: ROOT,
    encoding: "utf8",
    maxBuffer: 32 * 1024 * 1024,
  });
  if (res.error) {
    console.error(`[check] ${name} 執行失敗：${res.error.message}`);
    return null;
  }
  const noise = `${res.stderr || ""}`.trim();
  if (noise && name === "eslint") console.log(`[check] ${name} stderr: ${noise.split(/\r?\n/)[0]}`);
  return res.status;
}

function nl(msg) {
  return String(msg).replace(/\r?\n/g, " ");
}

// knip 的 devDependencies 名稱可能為字串或 { name } 物件
function devName(dep) {
  return typeof dep === "string" ? dep : dep && dep.name;
}

const summary = {
  generatedAt: new Date().toISOString(),
  thresholds: { complexity: 10, "max-depth": 3, "max-lines-per-function": 50, "sonarjs/cognitive-complexity": 15 },
  eslint: null,
  jscpd: null,
  knip: null,
};

// ---------- 1) ESLint ----------
try {
  run("eslint", "node", [
    path.join(ROOT, "node_modules", "eslint", "bin", "eslint.js"),
    ".",
    "--format", "json",
    "--output-file", path.join(".reports", "eslint.json"),
  ]);
  const file = path.join(REPORTS, "eslint.json");
  if (existsSync(file)) {
    const results = JSON.parse(readFileSync(file, "utf8"));
    const byRule = {};
    const findings = [];
    for (const f of results) {
      const rel = f.filePath.replace(ROOT + path.sep, "").replace(/\\/g, "/");
      for (const m of f.messages) {
        byRule[m.ruleId || "internal"] = (byRule[m.ruleId || "internal"] || 0) + 1;
        findings.push({ file: rel, line: m.line, column: m.column, ruleId: m.ruleId, severity: m.severity, message: nl(m.message) });
      }
    }
    summary.eslint = {
      checkedFiles: results.length,
      errorCount: results.reduce((a, f) => a + f.errorCount, 0),
      warningCount: results.reduce((a, f) => a + f.warningCount, 0),
      byRule,
      findings,
    };
    console.log(
      `[eslint] 檢查 ${results.length} 個檔案；errors=${summary.eslint.errorCount}, warnings=${summary.eslint.warningCount}`
    );
  } else {
    console.error("[eslint] 沒有產生報表檔。");
  }
} catch (e) {
  console.error(`[eslint] 解析失敗：${e.message}`);
}

// ---------- 2) jscpd（重複率）----------
try {
  run("jscpd", "node", [
    path.join(ROOT, "node_modules", "jscpd", "bin", "jscpd"),
    "docs", "scripts", "tests",
    "--reporters", "json",
    "--output", path.join(".reports", "jscpd"),
  ]);
  const file = path.join(REPORTS, "jscpd", "jscpd-report.json");
  if (existsSync(file)) {
    const r = JSON.parse(readFileSync(file, "utf8"));
    const total = (r.statistics && r.statistics.total) || {};
    summary.jscpd = {
      sources: total.sources,
      duplicatedLines: total.duplicatedLines,
      percent: total.percentage,
      clones: total.clones,
    };
    if (Array.isArray(r.duplicates)) {
      summary.jscpd.items = r.duplicates.map((d) => ({
        firstFile: (d.firstFile?.name || "").replace(/\\/g, "/"),
        firstLine: d.firstFile?.start,
        secondFile: (d.secondFile?.name || "").replace(/\\/g, "/"),
        secondLine: d.secondFile?.start,
        tokens: d.tokens,
        lines: d.lines,
      }));
    }
    console.log(
      `[jscpd] JS 來源檔=${summary.jscpd.sources}；重複率=${summary.jscpd.percent}%（重複區塊 ${summary.jscpd.clones}，重複行數 ${summary.jscpd.duplicatedLines}）`
    );
  } else {
    console.error("[jscpd] 沒有產生報表檔。");
  }
} catch (e) {
  console.error(`[jscpd] 解析失敗：${e.message}`);
}

// ---------- 3) knip（死碼）----------
try {
  const res = spawnSync("node", [path.join(ROOT, "node_modules", "knip", "bin", "knip.js"), "--reporter", "json"], {
    cwd: ROOT, encoding: "utf8", maxBuffer: 32 * 1024 * 1024,
  });
  const file = path.join(REPORTS, "knip-report.json");
  let r = null;
  try {
    r = JSON.parse((res.stdout || "").slice((res.stdout || "").indexOf("{")));
  } catch {
    /* stdout 不是 JSON 時退回 exit code */
  }
  if (r) writeFileSync(file, JSON.stringify(r, null, 2));
  let knip = null;
  if (r) {
    knip = {
      unusedFiles: (r.files || []).map((f) => (typeof f === "string" ? f : f.name || JSON.stringify(f))),
      unusedExports: (r.issues?.exports ? Object.entries(r.issues.exports).flatMap(([f, v]) => (v.exports || []).map((e) => `${f}:${e.symbol}`)) : []),
      unresolvedPlugins: (r.unresolvedPlugins || []).map((p) => (typeof p === "string" ? p : JSON.stringify(p))),
      // knip 的 issues JSON：每個檔案一筆（併入 entry.file），含 dependencies / devDependencies / exports 等
      problems: (r.issues || []).flatMap((issue) => {
        const f = issue.file || "package.json";
        const items = [];
        for (const dep of issue.dependencies || []) items.push({ file: f, kind: "unlisted-dependency", target: devName(dep) });
        for (const dep of issue.devDependencies || []) items.push({ file: f, kind: "unused-devDependency", target: devName(dep) });
        for (const dep of issue.optionalPeerDependencies || []) items.push({ file: f, kind: "optional-peer-dependency", target: devName(dep) });
        for (const b of issue.binaries || []) items.push({ file: f, kind: "unlisted-binary", target: b.name });
        for (const u of issue.unresolved || []) items.push({ file: f, kind: "unresolved-import", target: u });
        for (const e of issue.exports || []) items.push({ file: f, kind: "unused-export", target: e.symbol });
        for (const t of issue.types || []) items.push({ file: f, kind: "unused-export-type", target: t.symbol });
        for (const name of Object.keys(issue.enumMembers || {})) items.push({ file: f, kind: "unused-enumMember", target: name });
        for (const d of issue.duplicates || []) items.push({ file: f, kind: "duplicate-export", target: d.name });
        return items;
      }),
      unusedDevDependencies: (r.issues || []).flatMap((issue) => issue.devDependencies || []).map((d) => devName(d)),
    };
  } else {
    knip = { exitStatus: res.status, stderr: (res.stderr || "").split(/\r?\n/)[0] || "" };
  }
  summary.knip = knip;
  if (knip && Array.isArray(knip.unusedFiles)) {
    console.log(`[knip] 未使用檔案=${knip.unusedFiles.length}；未使用匯出=${knip.unusedExports.length}` + (knip.unresolvedPlugins.length ? `；未解析插件=${knip.unresolvedPlugins.length}` : ""));
  } else {
    console.log(`[knip] 結束，exit code=${res.status}${knip && knip.stderr ? `；輸出：${knip.stderr}` : ""}`);
  }
} catch (e) {
  console.error(`[knip] 解析失敗：${e.message}`);
}

// ---------- 彙總輸出 ----------
writeFileSync(path.join(REPORTS, "summary.json"), JSON.stringify(summary, null, 2));
console.log("\n================ 檢查彙總 ================");
if (summary.eslint) {
  console.log("ESLint 門檻類違規：");
  for (const k of ["complexity", "max-depth", "max-lines-per-function", "sonarjs/cognitive-complexity"]) {
    console.log(`  ${k}: ${summary.eslint.byRule[k] || 0}`);
  }
  const others = Object.entries(summary.eslint.byRule).filter(([k]) => !["complexity", "max-depth", "max-lines-per-function", "sonarjs/cognitive-complexity"].includes(k));
  if (others.length) console.log(`  其他規則：${others.map(([k, v]) => `${k}=${v}`).join(", ")}`);
}
if (summary.jscpd) console.log(`jscpd 重複率：${summary.jscpd.percent}%（重複區塊 ${summary.jscpd.clones ?? summary.jscpd.duplicates}）`);
if (summary.knip && summary.knip.unusedFiles) {
  console.log(`knip 死碼：未使用檔案 ${summary.knip.unusedFiles.length} 個、未使用匯出 ${summary.knip.unusedExports.length} 個`);
}
console.log(`詳細報表：.reports/eslint.json、.reports/jscpd/jscpd-report.json、.reports/knip-report.json、.reports/summary.json`);
