"""把既有產出搬成靜態網站（純複製與轉檔，不訓練、不重算、不編數字）。

做了什麼：
1. figures/*.png            -> docs/figures/          （逐位元複製，不重畫）
2. results/*.json           -> docs/results.json      （原封不動巢狀收錄，並紀錄來源）
3. docs/*.md                -> docs/reports/*.html    （純 Python Markdown 轉 HTML）
4. 首頁 docs/index.html     （純 HTML + 原生 JS fetch results.json，無 framework、無 build）

用法：python tools/build_site.py
"""

from __future__ import annotations

import html
import json
import re
import shutil
import sys
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
    text = re.sub(r"`", r"`", text)
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
    site_js = DOCS / "site.js"
    site_js.write_text(
        "const REPORT_TITLES = " + json.dumps(titles, ensure_ascii=False) + ";\n"
        "const FIGURES = " + json.dumps(figures, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    print(f"[site] 產生 {index_path} 與 {site_js}（首頁僅 render，資料全部來自 results.json）")


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Iris × AI4I 2020 - 靜態結果網站</title>
<link rel="stylesheet" href="style.css">
<style>
.layout{max-width:960px;margin:0 auto;padding:0 16px 64px;}
</style>
</head>
<body>
<div class="layout">
<header>
<h1>Iris × AI4I 2020 — 既有結果靜態展示</h1>
<p class="muted">純 HTML + 原生 JavaScript 讀取 results.json 顯示；不訓練、不重算，所有數字皆來自 results/ 下的既有輸出。
<span id="generatedAt"></span></p>
<nav>
<a href="inference.html">互動推論表單</a> ·
<a href="reports/iris-nn-paper.html">小論文</a> ·
<a href="reports/model_comparison.html">模型比較</a> ·
<a href="reports/evaluation_figures.html">評估圖表說明</a>
</nav>
</header>

<section id="content" aria-live="polite"><p class="muted">載入 results.json …</p></section>

<section>
<h2>報告（由 docs/*.md 轉成）</h2>
<ul id="reportLinks" class="links"></ul>
</section>

<section>
<h2>圖表（自 figures/ 原檔複製）</h2>
<div id="figGrid" class="fig-grid"></div>
</section>
</div>
<script src="site.js"></script>
<script>
const fmt = (v, d = 4) => (v == null ? "—" : (typeof v === "number" ? Number(v).toFixed(d) : v));

function card(title, rows) {
  return `<div class="kpi"><h3>${title}</h3>` + rows.map(
    ([k, v]) => `<div class="row"><span>${k}</span><strong>${v}</strong></div>`
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
    card("Iris NN", [
      ["測試 accuracy", fmt(nn.test_accuracy)],
      ["CV accuracy", `${fmt(nn.cv_accuracy_mean)} ± ${fmt(nn.cv_accuracy_std)}`],
      ["epochs", nn.epochs_run ?? "—"],
    ]),
    card("Iris PCA", [
      ["第 1 主成分", fmt((pca.explained_variance_ratio || [])[0])],
      ["第 3 主成分累積", fmt((pca.cumulative_ratio || [])[2])],
    ]),
  ];

  const ai4iHtml = [
    card("AI4I 模型", [
      ["最佳模型", `${best.model ?? "—"}（${best.strategy ?? "—"}）`],
      ["OOF PR-AUC", fmt(best.pr_auc_oof)],
      ["閾值（OOF）", fmt(best.threshold)],
    ]),
    card("AI4I 測試集", [
      ["PR-AUC", fmt(test.pr_auc)],
      ["Precision", fmt(test.precision)],
      ["Recall", fmt(test.recall)],
      ["閾值", fmt(test.threshold)],
      ["ROC-AUC", fmt(ev.curves && ev.curves.roc_auc)],
      ["Brier", fmt(ev.calibration && ev.calibration.brier)],
    ]),
    card("時間切分對照", [
      ["time-split PR-AUC", fmt(ts.time_split && ts.time_split.pr_auc)],
      ["PR-AUC 差異", fmt(ts.differences && ts.differences.pr_auc_diff)],
      ["結論", ts.verdict ?? "—"],
    ]),
  ];

  document.getElementById("content").innerHTML = `
    <section><h2>結果摘要</h2>
      <div class="grid">${irisHtml.join("")}</div>
    </section>
    <section><h2>AI4I 2020（隨機森林 + sigmoid 校準）</h2>
      <div class="grid">${ai4iHtml.join("")}</div>
    </section>
    <section><h2>數字來源</h2>
      <ul>${Object.entries(src || {}).map(
        ([k, v]) => `<li><code>${k}</code> ← <code>${v}</code></li>`).join("")}</ul>
    </section>`;

  document.getElementById("reportLinks").innerHTML =
    (typeof REPORT_TITLES !== "undefined" ? Object.entries(REPORT_TITLES) : [])
      .map(([file, title]) => `<li><a href="${file}">${title}</a></li>`).join("");

  document.getElementById("figGrid").innerHTML = (typeof FIGURES !== "undefined" ? FIGURES : [])
    .map(name => `<figure><img loading="lazy" src="figures/${name}" alt="${name}"><figcaption>${name}</figcaption></figure>`)
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


STYLE_CSS = r""":root { color-scheme: light; }
body { font-family: system-ui, "Microsoft JhengHei", sans-serif; margin: 0;
       background: #f7f8fa; color: #22303f; }
a { color: #1a6fd4; }
.muted { color: #68788c; }
header { padding: 24px 0 12px; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 24px 0 12px; }
section { margin-bottom: 28px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
.kpi { background: #fff; border: 1px solid #e2e7ee; border-radius: 10px; padding: 14px 16px; }
.kpi h3 { margin: 0 0 8px; font-size: 0.95rem; }
.row { display: flex; justify-content: space-between; gap: 12px; padding: 3px 0; font-size: 0.92rem; }
.row strong { font-variant-numeric: tabular-nums; }
.links li { margin: 3px 0; }
.fig-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }
.fig-grid figure { margin: 0; background: #fff; border: 1px solid #e2e7ee; border-radius: 10px; padding: 10px; }
.fig-grid img { width: 100%; height: auto; border-radius: 6px; }
.fig-grid figcaption { font-size: 0.8rem; color: #68788c; margin-top: 6px; word-break: break-all; }
nav { margin: 10px 0 4px; font-size: 0.95rem; }
.err { color: #b3261e; }

/* 報告頁 */
main.report { max-width: 860px; margin: 0 auto; padding: 24px 16px 64px; }
main.report h1 { font-size: 1.45rem; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; background: #fff; }
th, td { border: 1px solid #dde3ea; padding: 6px 10px; font-size: 0.9rem; text-align: left; }
th { background: #eef3f9; }
code { background: #eef1f5; padding: 1px 5px; border-radius: 4px; font-size: 0.88em; }
pre code { display: block; padding: 10px; overflow-x: auto; }
blockquote { border-left: 4px solid #c9d6e6; margin: 10px 0; padding: 4px 14px; color: #47586d; background: #fff; }
img { max-width: 100%; }
.crumb { margin-bottom: 18px; }
"""


def main() -> None:
    figures = copy_figures()
    build_results_json()
    reports = convert_reports()
    build_index(figures, reports)
    (DOCS / "style.css").write_text(STYLE_CSS, encoding="utf-8")
    print("[site] 完成：docs/figure 複製、results.json、reports/*.html、index.html、style.css")


if __name__ == "__main__":
    main()
