"""把既有產出搬成靜態網站（純複製與轉檔，不訓練、不重算、不編數字）。

做了什麼：
1. figures/*.png            -> docs/figures/          （逐位元複製，不重畫）
2. results/*.json           -> docs/results.json      （原封不動巢狀收錄，並紀錄來源）
3. docs/*.md                -> docs/reports/*.html    （純 Python Markdown 轉 HTML）
4. 首頁 docs/index.html     （純 HTML + 原生 JS fetch results.json，無 framework、無 build）

樣式主題：溫馨手繪風（暖色紙張背景、歪歪的卡片邊框、膠帶貼紙、手寫字體）。

用法：python tools/build_site.py
"""

from __future__ import annotations

import html
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
SITE_FIG = DOCS / "figures"
SITE_REPORTS = DOCS / "reports"

# 首頁會用到的既有 JSON；key -> results/ 底下的檔名
RESULT_FILES = {
    "nn_metrics": "nn_metrics.json",
    "pca_result": "pca_result.json",
    "model_comparison": "model_comparison.json",
    "test_metrics": "test_metrics.json",
    "ai4i_metrics": "ai4i_metrics.json",
    "time_split": "time_split_comparison.json",
}

MD_REPORTS = [
    "ai4i-report.md",
    "iris-nn-paper.md",
    "eda_report.md",
    "eda_report_ai4i.md",
    "features.md",
    "evaluation_protocol.md",
    "model_comparison.md",
    "evaluation_figures.md",
]

# 圖表的友善名稱（抓不到就退回原檔名，不另外發明數字）
FIGURE_LABELS = {
    "ai4i_pr_curve.png": "AI4I PR 曲線",
    "ai4i_roc_curve.png": "AI4I ROC 曲線",
    "ai4i_confusion.png": "AI4I 混淆矩陣",
    "ai4i_calibration.png": "AI4I 可靠性圖",
    "ai4i_importance.png": "AI4I 特徵重要度",
    "ai4i_physical.png": "AI4I 物理對照",
    "ai4i_mode_recall.png": "AI4I 失效模式 recall",
    "ai4i_time_split.png": "AI4I 時間切分對照",
    "ai4i_eda_correlation.png": "AI4I EDA 相關性",
    "ai4i_eda_modes.png": "AI4I EDA 失效模式",
    "ai4i_eda_physical.png": "AI4I EDA 物理量",
    "ai4i_eda_target.png": "AI4I EDA 目標",
    "ai4i_eda_type.png": "AI4I EDA Type",
    "eda_boxplots.png": "Iris EDA 箱型圖",
    "eda_distributions.png": "Iris EDA 分布",
    "nn_training_curves.png": "Iris NN 訓練曲線",
    "nn_confusion_matrix.png": "Iris NN 混淆矩陣",
    "nn_f1_scores.png": "Iris NN F1",
    "pca_3d.png": "Iris PCA 3D",
    "pca_variance.png": "Iris PCA 變異量",
}

NA = "—"


# ---------------------------------------------------------------- 1. 圖複製
def copy_figures() -> list[str]:
    SITE_FIG.mkdir(parents=True, exist_ok=True)
    copied = []
    for png in sorted(FIGURES.glob("*.png")):
        dest = SITE_FIG / png.name
        shutil.copyfile(png, dest)
        copied.append(png.name)
    if not copied:
        raise RuntimeError(f"{FIGURES} 找不到任何 .png")
    print(f"[figures] 複製 {len(copied)} 張 -> {SITE_FIG}")
    return copied


# ---------------------------------------------------------------- 2. results.json
def build_results_json() -> dict:
    payload: dict = {
        "_sources": {},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for key, filename in RESULT_FILES.items():
        path = RESULTS / filename
        if not path.exists():
            raise RuntimeError(f"缺少既有產出 {path}（腳本不自行補數字，請先執行原流程）")
        data = json.loads(path.read_text(encoding="utf-8"))
        payload[key] = data
        payload["_sources"][key] = str(path.relative_to(ROOT)).replace("\\", "/")
    out = DOCS / "results.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[results] 收錄 {len(RESULT_FILES)} 個指標檔 -> {out}")
    return payload


# ---------------------------------------------------------------- 3. Markdown -> HTML
def md_inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", r'<img src="\2" alt="\1">', text)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def md_to_html(md_text: str) -> str:
    lines = md_text.splitlines()
    out: list[str] = []
    i, n = 0, len(lines)
    in_code = False
    code_buf: list[str] = []
    list_tag: str | None = None  # 目前開著的 <ul>/<ol>

    def close_list():
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    while i < n:
        line = lines[i]

        if line.strip().startswith("```"):
            if in_code:
                out.append("<pre><code>" + html.escape("\n".join(code_buf)) + "</code></pre>")
                code_buf, in_code = [], False
            else:
                close_list()
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        stripped = line.strip()
        if not stripped:
            close_list()
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            close_list()
            level = len(m.group(1))
            out.append(f"<h{level}>{md_inline(m.group(2))}</h{level}>")
            i += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|") and i + 1 < n \
                and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            close_list()
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            out.append("<table>")
            out.append("<thead><tr>" + "".join(f"<th>{md_inline(c)}</th>" for c in rows[0]) + "</tr></thead>")
            out.append("<tbody>")
            for row in rows[2:]:
                out.append("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in row) + "</tr>")
            out.append("</tbody></table>")
            continue

        m = re.match(r"^[-*]\s+(.*)$", stripped)
        if m:
            if list_tag != "ul":
                close_list()
                out.append("<ul>")
                list_tag = "ul"
            out.append(f"<li>{md_inline(m.group(1))}</li>")
            i += 1
            continue

        m = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if m:
            if list_tag != "ol":
                close_list()
                out.append("<ol>")
                list_tag = "ol"
            out.append(f"<li>{md_inline(m.group(1))}</li>")
            i += 1
            continue

        if stripped.startswith(">"):
            close_list()
            quote = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("<blockquote><p>" + md_inline(" ".join(quote)) + "</p></blockquote>")
            continue

        if stripped.startswith("---") or stripped.startswith("***"):
            close_list()
            out.append("<hr>")
            i += 1
            continue

        close_list()
        para = [stripped]
        i += 1
        while i < n and lines[i].strip() and not re.match(r"^(#{1,6}\s|\||[-*]\s|\d+[.)]\s|>|```)", lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{md_inline(' '.join(para))}</p>")

    close_list()
    if in_code:  # 防呆：未關閉的 code fence
        out.append("<pre><code>" + html.escape("\n".join(code_buf)) + "</code></pre>")
    return "\n".join(out)


HTML_SHELL = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Patrick+Hand&family=LXGW+WenKai+TC&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../style.css">
</head>
<body>
<main class="report">
<p class="crumb"><a href="../index.html">← 回首頁</a></p>
{body}
</main>
</body>
</html>
"""


def convert_reports() -> list[dict]:
    SITE_REPORTS.mkdir(parents=True, exist_ok=True)
    made = []
    for md_name in MD_REPORTS:
        src = DOCS / md_name
        if not src.exists():
            raise RuntimeError(f"缺少既有報告 {src}")
        md_text = src.read_text(encoding="utf-8")
        body = md_to_html(md_text)
        title = re.search(r"^#\s+(.+)$", md_text, re.M)
        title = title.group(1).strip() if title else md_name
        dest = SITE_REPORTS / (src.stem + ".html")
        dest.write_text(
            HTML_SHELL.format(title=html.escape(title), body=body), encoding="utf-8"
        )
        made.append({"md": md_name, "html": f"reports/{dest.name}", "title": title})
    print(f"[reports] 轉出 {len(made)} 份 HTML -> {SITE_REPORTS}")
    return made


# ---------------------------------------------------------------- 4. 首頁
def build_index(figures: list[str], reports: list[dict]) -> None:
    index_path = DOCS / "index.html"
    index_path.write_text(INDEX_HTML, encoding="utf-8")

    titles = {r["html"]: r["title"] for r in reports}
    figs = [{"file": f, "label": FIGURE_LABELS.get(f, f)} for f in figures]
    site_js = DOCS / "site.js"
    site_js.write_text(
        "const REPORT_TITLES = " + json.dumps(titles, ensure_ascii=False) + ";\n"
        "const FIGURES = " + json.dumps(figs, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    print(f"[site] 產生 {index_path} 與 {site_js}（首頁僅 render，資料全部來自 results.json）")


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Iris × AI4I 2020 — 手工小站</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Patrick+Hand&family=LXGW+WenKai+TC&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css">
</head>
<body>
<div class="layout">
<header>
<svg class="doodles" viewBox="0 0 190 44" width="190" height="44" aria-hidden="true">
  <g fill="none" stroke="#e8b04b" stroke-width="2" stroke-linecap="round">
    <circle cx="22" cy="22" r="7"/>
    <path d="M22 9v-6M22 35v6M9 22H3M35 22h6M13 13l-4-4M31 13l4-4M13 31l-4 4M31 31l4 4"/>
  </g>
  <g fill="none" stroke="#8aa9c4" stroke-width="2" stroke-linecap="round">
    <circle cx="92" cy="22" r="9"/>
    <path d="M92 9v-6M92 35v6M79 22h-6M105 22h6M82 12l-4-4M102 12l4-4M82 32l-4 4M102 32l4 4"/>
    <circle cx="92" cy="22" r="3"/>
  </g>
  <g fill="none" stroke="#7fae8a" stroke-width="2" stroke-linecap="round">
    <path d="M150 36c0-12 8-22 20-25-2 13-11 23-20 25z"/>
    <path d="M150 36c-11-1-15-9-16-16 9 1 14 7 16 16z"/>
  </g>
</svg>
<h1>🌷 Iris × AI4I 2020<br>手工小站</h1>
<p class="muted">純 HTML ＋ 原生 JavaScript 讀取 results.json 顯示；不訓練、不重算，
所有數字皆來自 results/ 下的既有輸出。✋ 全部由 tools/build_site.py 逐檔搬運。
<span id="generatedAt"></span></p>
<nav>
<a href="inference.html">✏️ 互動推論表單</a>
<a href="reports/ai4i-report.html">📖 期末報告</a>
<a href="reports/model_comparison.html">🧮 模型比較</a>
<a href="reports/evaluation_figures.html">🖼 圖表說明</a>
</nav>
</header>

<section id="content" aria-live="polite"><p class="muted">載入 results.json …</p></section>

<section>
<h2>📚 報告（由 docs/*.md 轉成）</h2>
<ul id="reportLinks" class="links"></ul>
</section>

<section>
<h2>🖼 圖表（自 figures/ 原檔複製）</h2>
<div id="figGrid" class="fig-grid"></div>
</section>
</div>
<script src="site.js"></script>
<script>
const fmt = (v, d = 4) => (v == null ? "—" : (typeof v === "number" ? Number(v).toFixed(d) : v));

function card(title, rows) {
  return `<div class="kpi"><h3>${title}</h3>` + rows.map(
    ([k, v]) => `<div class="row"><span>${k}</span><span class="dots"></span><strong>${v}</strong></div>`
  ).join("") + "</div>";
}

async function main() {
  const res = await fetch("results.json");
  const data = await res.json();
  if (data.generated_at) document.getElementById("generatedAt").textContent =
    `產生時間：${fmt(data.generated_at)}`;

  const nn = data.nn_metrics || {};
  const pca = data.pca_result || {};
  const cmp = data.model_comparison || {};
  const best = cmp.best || {};
  const tm = data.test_metrics || {};
  const test = tm.metrics || {};
  const ev = data.ai4i_metrics || {};
  const ts = data.time_split || {};
  const src = data._sources || {};

  const irisHtml = [
    card("🌸 Iris 模型", [
      ["測試 accuracy", fmt(nn.test_accuracy)],
      ["CV accuracy", `${fmt(nn.cv_accuracy_mean)} ± ${fmt(nn.cv_accuracy_std)}`],
      ["epochs", nn.epochs_run ?? "—"],
    ]),
    card("🌿 Iris PCA", [
      ["第 1 主成分", fmt((pca.explained_variance_ratio || [])[0])],
      ["第 3 主成分累積", fmt((pca.cumulative_ratio || [])[2])],
    ]),
  ];

  const ai4iHtml = [
    card("🔧 AI4I 模型", [
      ["最佳模型", `${best.model ?? "—"}（${best.strategy ?? "—"}）`],
      ["OOF PR-AUC", fmt(best.pr_auc_oof)],
      ["閾值（OOF）", fmt(best.threshold)],
    ]),
    card("🧪 AI4I 測試集", [
      ["PR-AUC", fmt(test.pr_auc)],
      ["Precision", fmt(test.precision)],
      ["Recall", fmt(test.recall)],
      ["閾值", fmt(test.threshold)],
      ["ROC-AUC", fmt(ev.curves && ev.curves.roc_auc)],
      ["Brier", fmt(ev.calibration && ev.calibration.brier)],
    ]),
    card("⏳ 時間切分對照", [
      ["time-split PR-AUC", fmt(ts.time_split && ts.time_split.pr_auc)],
      ["PR-AUC 差異", fmt(ts.differences && ts.differences.pr_auc_diff)],
      ["結論", ts.verdict ?? "—"],
    ]),
  ];

  document.getElementById("content").innerHTML = `
    <section><h2>🌸結果摘要</h2>
      <div class="grid">${irisHtml.join("")}</div>
    </section>
    <section><h2>🔧 AI4I 2020（隨機森林 ＋ sigmoid 校準）</h2>
      <div class="grid">${ai4iHtml.join("")}</div>
    </section>
    <section><h2>🔗 數字的來源</h2>
      <ul class="links">${Object.entries(src || {}).map(
        ([k, v]) => `<li><code>${k}</code> ← <code>${v}</code></li>`).join("")}</ul>
    </section>`;

  document.getElementById("reportLinks").innerHTML =
    (typeof REPORT_TITLES !== "undefined" ? Object.entries(REPORT_TITLES) : [])
      .map(([file, title]) => `<li><a href="${file}">📄 ${title}</a></li>`).join("");

  document.getElementById("figGrid").innerHTML =
    (typeof FIGURES !== "undefined" ? FIGURES : [])
      .map(f => `<figure><img loading="lazy" src="figures/${f.file}" alt="${f.file}"><figcaption>${f.label}<br><span class="fname">${f.file}</span></figcaption></figure>`)
      .join("");
}

main().catch(err => {
  document.getElementById("content").innerHTML =
    `<p class="err">載入 results.json 失敗：${err}</p>`;
});
</script>
</body>
</html>
"""


STYLE_CSS = r"""
/* ============================================================
   溫馨手繪風 · warm hand-drawn theme
   暖色紙張 ─ 歪歪的邊框 ─ 膠帶貼紙 ─ 手寫字體
   ============================================================ */
:root{
  --paper:#fbf3e4;
  --card:#fffdf7;
  --ink:#4a3b2a;
  --ink-soft:#8a755c;
  --line:#a3805a;
  --line-soft:#d8c4a6;
  --red:#e07a5f;
  --yellow:#f2c14e;
  --green:#81b29a;
  --blue:#6d9dc5;
  --tape:#f9dcc4;
  --wobbly: 18px 225px 255px 25px / 255px 25px 225px 18px;
  --wobbly2: 255px 25px 225px 15px / 25px 225px 255px 25px;
}

body{
  font-family:"Patrick Hand","LXGW WenKai TC","Segoe Print","Comic Sans MS","Microsoft JhengHei",sans-serif;
  color:var(--ink);
  background-color:var(--paper);
  background-image:radial-gradient(#e6d6bb 1.1px, transparent 1.1px);
  background-size:26px 26px;
  margin:0;
  line-height:1.6;
}

a{ color:var(--red); text-decoration:underline wavy rgba(224,122,95,.55); text-underline-offset:3px; }
a:hover{ text-decoration:underline wavy var(--line); filter:brightness(.95); }

h1, h2, h3{ font-weight:600; line-height:1.3; }
h1{ font-size:1.7rem; margin:6px 0 4px; }
h2{ font-size:1.3rem; margin:28px 0 14px; }
h3{ font-size:1.05rem; margin:0 0 8px; }

h1::after, h2::after{
  content:"";
  display:block;
  height:7px;
  margin-top:6px;
  background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='72' height='7' viewBox='0 0 72 7'%3E%3Cpath d='M2 4 Q 10 1.5 18 4 T 34 4 T 50 4 T 66 4 T 82 4' fill='none' stroke='%23e07a5f' stroke-width='2.2' stroke-linecap='round'/%3E%3C/svg%3E") repeat-x;
  background-size:72px 7px;
}
.muted{ color:var(--ink-soft); font-size:.95rem; }

/* ---------- 首頁 ---------- */
.layout{ max-width:960px; margin:0 auto; padding:26px 18px 72px; }
header{ text-align:center; padding:8px 0 4px; }
.doodles{ display:block; margin:0 auto 4px; }
nav{ display:flex; gap:10px; justify-content:center; flex-wrap:wrap; margin:14px 0 4px; }
nav a{
  font-size:1rem;
  color:var(--ink);
  background:var(--card);
  border:2px solid var(--line);
  border-radius:var(--wobbly2);
  padding:6px 16px;
  text-decoration:none;
  box-shadow:2px 3px 0 rgba(163,128,90,.22);
  transition:transform .12s ease, box-shadow .12s ease;
}
nav a:hover{ transform:translateY(-2px) rotate(-1.2deg); text-decoration:none; }

.grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:18px; }

.kpi{
  position:relative;
  background:var(--card);
  border:2px solid var(--line);
  border-radius:var(--wobbly);
  padding:16px 18px 12px;
  box-shadow:3px 4px 0 rgba(163,128,90,.18);
}
.kpi:nth-child(odd){ transform:rotate(-.7deg); }
.kpi:nth-child(even){ transform:rotate(.7deg); }

.kpi::before, .fig-grid figure::before{
  content:"";
  position:absolute;
  top:-10px; left:50%;
  transform:translateX(-50%) rotate(-3deg);
  width:66px; height:20px;
  background:var(--tape);
  opacity:.9;
  border-left:1.5px dashed rgba(255,255,255,.75);
  border-right:1.5px dashed rgba(255,255,255,.75);
  border-radius:2px;
  box-shadow:0 1px 2px rgba(0,0,0,.06);
}
.kpi:nth-child(4n+2)::before{ background:#cfe3d2; }
.kpi:nth-child(4n+3)::before{ background:#fde2b8; }
.kpi:nth-child(4n+4)::before{ background:#d6e3f5; }

.kpi h3{ border-bottom:1.5px dashed var(--line-soft); padding-bottom:6px; }
.row{ display:flex; align-items:baseline; gap:8px; padding:3.5px 0; font-size:.98rem; }
.row span{ color:var(--ink-soft); flex:0 0 auto; }
.row .dots{ flex:1 1 auto; border-bottom:2px dotted var(--line-soft); transform:translateY(-3px); }
.row strong{ color:var(--ink); font-variant-numeric:tabular-nums; flex:0 0 auto; }

.links{ margin:6px 0 0; padding-left:0; list-style:none; }
.links li{ padding:3px 0; }
.links li::before{ content:"✏️ "; }
.links code{
  background:transparent;
  padding:0 2px;
  font-family:inherit;
  font-size:.95em;
  color:#7b6a54;
}
.links li:nth-child(even){ transform:rotate(.35deg); }
.links li:nth-child(odd){ transform:rotate(-.3deg); }

.err{ color:var(--red); font-size:1.05rem; }

/* ---------- 圖表貼紙 ---------- */
.fig-grid{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:22px; }
.fig-grid figure{
  position:relative;
  margin:10px 0 0;
  background:var(--card);
  border:2px solid var(--line);
  border-radius:var(--wobbly2);
  padding:12px 12px 10px;
  box-shadow:3px 4px 0 rgba(163,128,90,.18);
}
.fig-grid figure:nth-child(odd){ transform:rotate(-.8deg); }
.fig-grid figure:nth-child(even){ transform:rotate(.8deg); }
.fig-grid img{ width:100%; height:auto; border-radius:10px; }
.fig-grid figcaption{
  border-top:1.5px dashed var(--line-soft);
  margin-top:10px; padding-top:6px;
  font-size:.92rem; text-align:center;
}
.fig-grid .fname{ display:block; font-size:.8rem; color:var(--ink-soft); }

/* ---------- 報告頁 ---------- */
main.report{ max-width:840px; margin:0 auto; padding:26px 18px 80px; }
.crumb{ font-size:1rem; margin:0 0 20px; }

main.report img{
  background:#fff;
  border:2px solid var(--line-soft);
  border-radius:14px;
  padding:6px;
  box-shadow:2px 3px 0 rgba(163,128,90,.15);
  max-width:100%;
  height:auto;
}
main.report h1{ font-size:1.55rem; }

table{
  border-collapse:separate;
  border-spacing:0;
  width:100%;
  margin:12px 0;
  background:var(--card);
  border:2px dashed var(--line-soft);
  border-radius:14px;
  overflow:hidden;
}
th, td{
  padding:7px 12px;
  border-bottom:1.5px dashed #ecdfca;
  font-size:.96rem;
  text-align:left;
  font-variant-numeric:tabular-nums;
}
tr:last-child td{ border-bottom:none; }
th{ background:#f6ead2; font-weight:600; }

@media (max-width:720px){
  .report table, table{ display:block; overflow-x:auto; }
}

code{
  background:#f3e8d2;
  padding:1px 7px;
  border-radius:8px;
  font-size:.9em;
}
pre code{
  display:block;
  padding:12px 14px;
  overflow-x:auto;
  background:#f3e8d2;
  border:1.5px dashed var(--line-soft);
  border-radius:12px;
}
blockquote{
  position:relative;
  border-left:6px solid var(--yellow);
  background:#fdf6e0;
  border-radius:10px;
  margin:12px 0;
  padding:10px 16px;
  color:#5c4c39;
}
hr{ border:none; border-top:2px dashed var(--line-soft); margin:22px 0; }

ul, ol{ padding-left:1.5em; }
li{ margin:3px 0; }
"""


def main() -> None:
    figures = copy_figures()
    build_results_json()
    reports = convert_reports()
    build_index(figures, reports)
    (DOCS / "style.css").write_text(STYLE_CSS, encoding="utf-8")
    print("[site] 完成：figures 複製、results.json、reports/*.html、index.html、style.css")


if __name__ == "__main__":
    main()
