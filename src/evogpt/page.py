"""Static HTML leaderboard for GitHub Pages. Standard library only."""

from __future__ import annotations

from datetime import datetime
from html import escape
import math
from typing import Any

# Fixed categorical order, light and dark steps (validated adjacent-pair CVD safe).
SERIES = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"),
          ("#eda100", "#c98500"), ("#e87ba4", "#d55181"), ("#008300", "#008300"),
          ("#6250d6", "#9085e9"), ("#e34948", "#e66767")]
TICKS = [0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10, 30, 100, 300, 1000]


def _x(value: float) -> str:
    return f"{value:.2f}×"


def _tick(value: float) -> str:
    return f"{value:g}×"


def _when(stamp: str) -> str:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).strftime("%a %d %b %Y, %H:%M UTC")


def _flops(value: int) -> str:
    exponent = round(math.log10(value))
    if 10 ** exponent == value:
        return "10" + "".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[int(digit)] for digit in str(exponent))
    return f"{value:.2e}"


def _short(entry_id: str) -> tuple[str, str]:
    recipe, revision = entry_id.split("@")
    return recipe, revision[:7]


def _entry_cell(entry_id: str, color: int | None) -> str:
    recipe, revision = _short(entry_id)
    swatch = f'<span class="sw" style="--c:var(--s{color})"></span>' if color is not None else '<span class="sw none"></span>'
    return (f'{swatch}<span class="name">{escape(recipe)}</span>'
            f'<code title="{escape(entry_id)}">{escape(revision)}</code>')


def _chart_width(board: dict[str, Any]) -> int:
    return 148 + 150 * max(len(board["active_scale_ids"]), 2)


def _chart(board: dict[str, Any], colors: dict[str, int]) -> str:
    # Only open scales: later ones have no scores yet.
    scale_ids = board["active_scale_ids"]
    points: dict[str, list[tuple[int, float, dict[str, Any]]]] = {entry: [] for entry in colors}
    for index, scale_id in enumerate(scale_ids):
        for row in board["scales"][scale_id]["rows"]:
            if row["entry_id"] in points and row["score"] is not None:
                points[row["entry_id"]].append((index, row["score"], row))
    values = [score for series in points.values() for _, score, _ in series] + [1.0]
    low, high = math.log(min(values)), math.log(max(values))
    pad = max((high - low) * 0.12, math.log(1.25))
    low, high = low - pad, high + pad
    left, right, top, bottom = 52, 96, 16, 44
    width, height = _chart_width(board), 320
    plot_w, plot_h = width - left - right, height - top - bottom

    def px(index: int) -> float:
        return left + plot_w * (index + 0.5) / len(scale_ids)

    def py(score: float) -> float:
        return top + plot_h * (high - math.log(score)) / (high - low)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="chart-title">',
             '<title id="chart-title">Compute multiplier at each scale, log scale; 1× is the baseline</title>']
    for tick in TICKS:
        if low <= math.log(tick) <= high:
            y = py(tick)
            cls = "base" if tick == 1 else "grid"
            parts.append(f'<line class="{cls}" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
            parts.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{_tick(tick)}</text>')
    for index, scale_id in enumerate(scale_ids):
        scale = board["scales"][scale_id]
        parts.append(f'<text class="tick" x="{px(index):.1f}" y="{height - 24}" text-anchor="middle">{escape(scale_id)}</text>')
        parts.append(f'<text class="tick" x="{px(index):.1f}" y="{height - 8}" text-anchor="middle">{_flops(scale["max_flops"])} FLOPs</text>')
    parts.append(f'<text class="tick" x="{width - right + 8}" y="{py(1.0) + 4:.1f}">baseline</text>')
    direct = len(points) <= 4
    for entry, series in points.items():
        if not series:
            continue
        color = f"var(--s{colors[entry]})"
        path = " ".join(f"{'M' if i == 0 else 'L'}{px(index):.1f},{py(score):.1f}" for i, (index, score, _) in enumerate(series))
        parts.append(f'<path class="line" d="{path}" style="stroke:{color}"/>')
        recipe, revision = _short(entry)
        for index, score, row in series:
            loss = "—" if row["median_loss"] is None else f"{row['median_loss']:.4f}"
            tip = f"{recipe} @{revision} · {scale_ids[index]}: {_x(score)} · median loss {loss}"
            parts.append(f'<g class="pt"><circle class="hit" cx="{px(index):.1f}" cy="{py(score):.1f}" r="12"/>'
                         f'<circle class="dot" cx="{px(index):.1f}" cy="{py(score):.1f}" r="5" style="fill:{color}"/>'
                         f'<title>{escape(tip)}</title></g>')
        if direct:
            index, score, _ = series[-1]
            parts.append(f'<text class="label" x="{px(index) + 10:.1f}" y="{py(score) + 4:.1f}">{escape(recipe)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(board: dict[str, Any]) -> str:
    scale_ids = list(board["scales"])
    active = board["active_scale_ids"]
    scored = [row for row in board["overall"] if row["score"] is not None]
    colors = {row["entry_id"]: index + 1 for index, row in enumerate(scored[:len(SERIES)])}
    by_scale = {scale_id: {row["entry_id"]: row for row in board["scales"][scale_id]["rows"]} for scale_id in scale_ids}
    top_n = board.get("top_n")
    demo = any(scale["baseline_is_fixture"] for scale in board["scales"].values())

    head = [f'<th class="num">{escape(scale_id)}<span>{_flops(board["scales"][scale_id]["max_flops"])}</span></th>' for scale_id in active]
    rows, rank = [], 0
    for row in board["overall"]:
        has_score = row["score"] is not None
        rank += int(has_score)
        cells = []
        for scale_id in active:
            scale_row = by_scale[scale_id].get(row["entry_id"])
            score = None if scale_row is None else scale_row["score"]
            worst = has_score and score == row["score"]
            cells.append(f'<td class="num{" worst" if worst else ""}">{_x(score) if score is not None else "—"}</td>')
        promoted = has_score and top_n is not None and rank <= top_n and board["status"] == "active"
        cls = ' class="promote"' if promoted else ""
        overall = f"<strong>{_x(row['score'])}</strong>" if has_score else f'<span class="status">{escape(row["status"])}</span>'
        rows.append(f'<tr{cls}><td class="num">{rank if has_score else "—"}</td>'
                    f'<td>{_entry_cell(row["entry_id"], colors.get(row["entry_id"]))}</td>'
                    f'<td class="num">{overall}</td>{"".join(cells)}'
                    f'<td class="num">{_x(row["mean_score"]) if has_score else "—"}</td>'
                    f'<td class="num">{row["completed_scales"]}/{row["required_scales"]}</td></tr>')
    if not rows:
        rows.append(f'<tr><td colspan="{len(active) + 5}" class="empty">No entries yet.</td></tr>')

    details = []
    for scale_id in active:
        scale = board["scales"][scale_id]
        lines = []
        for r in scale["rows"]:
            loss = "—" if r["median_loss"] is None else f"{r['median_loss']:.6f}"
            lines.append(f'<tr><td>{_entry_cell(r["entry_id"], colors.get(r["entry_id"]))}</td>'
                         f'<td class="num">{loss}</td>'
                         f'<td class="num">{_x(r["score"]) if r["score"] is not None else "—"}</td>'
                         f'<td class="num">{len(r["seeds"])}</td></tr>')
        baseline = "uncalibrated" if scale["baseline_loss"] is None else f"{scale['baseline_loss']:.6f}"
        details.append(
            f'<details><summary><strong>{escape(scale_id)}</strong> · {_flops(scale["max_flops"])} FLOPs · '
            f'baseline loss {baseline} · {len(scale["rows"])} entries</summary>'
            '<div class="scroll"><table><thead><tr><th>Entry</th><th class="num">Median val loss</th>'
            '<th class="num">Multiplier</th><th class="num">Seeds</th></tr></thead>'
            f'<tbody>{"".join(lines) or "<tr><td colspan=4 class=empty>No runs yet.</td></tr>"}</tbody></table></div></details>')

    legend = "".join(f'<li>{_entry_cell(entry, color)}</li>' for entry, color in colors.items())
    light = "".join(f"--s{i + 1}:{pair[0]};" for i, pair in enumerate(SERIES))
    dark = "".join(f"--s{i + 1}:{pair[1]};" for i, pair in enumerate(SERIES))
    status = {"active": "Active", "complete": "Complete", "uncalibrated": "Uncalibrated"}.get(board["status"], board["status"])
    banner = ('<p class="banner">Synthetic demo. These scores are fixtures, not measured training runs. '
              'The real season appears here once its baseline is calibrated.</p>') if demo else ""
    promote_note = (f"Shaded rows are the current top {top_n}; they advance to the next scale at the deadline."
                    if top_n and board["status"] == "active" else "")
    chart = (f'<ul class="legend">{legend}</ul><div class="chart">{_chart(board, colors)}</div>'
             if colors else '<p class="empty">The chart appears once an entry is scored at every active scale.</p>')
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EvoGPT Leaderboard</title>
<style>
:root {{ color-scheme: light; --bg:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --ink3:#8a8984;
  --rule:#e6e5e0; --grid:#eeede8; --hi:#f3f1ea; --warn:#fff4d6; --warn-ink:#6b4e00; {light} }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --bg:#141413; --panel:#1a1a19;
  --ink:#f0efec; --ink2:#c3c2b7; --ink3:#8d8c84; --rule:#2e2e2b; --grid:#262624; --hi:#232320; --warn:#3a2f10; --warn-ink:#f2d27a; {dark} }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --bg:#141413; --panel:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --ink3:#8d8c84;
  --rule:#2e2e2b; --grid:#262624; --hi:#232320; --warn:#3a2f10; --warn-ink:#f2d27a; {dark} }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:960px; margin:0 auto; padding:32px 16px 64px; }}
h1 {{ font-size:28px; margin:0; letter-spacing:-0.01em; }}
h2 {{ font-size:17px; margin:36px 0 10px; }}
.lede {{ color:var(--ink2); margin:4px 0 16px; font-size:17px; }}
.meta {{ color:var(--ink2); display:flex; flex-wrap:wrap; gap:6px 18px; margin:0 0 8px; padding:0; list-style:none; }}
.meta b {{ color:var(--ink); font-weight:600; }}
.banner {{ background:var(--warn); color:var(--warn-ink); padding:10px 14px; border-radius:6px; }}
.note {{ color:var(--ink2); font-size:14px; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
th, td {{ padding:8px 10px; border-bottom:1px solid var(--rule); text-align:left; white-space:nowrap; }}
th {{ font-size:13px; color:var(--ink2); font-weight:600; vertical-align:bottom; }}
th span {{ display:block; font-weight:400; color:var(--ink3); }}
.num {{ text-align:right; }}
tr.promote td {{ background:var(--hi); }}
td.worst {{ text-decoration:underline; text-decoration-color:var(--ink3); text-underline-offset:3px; }}
.status, .empty {{ color:var(--ink3); }}
code {{ font:12px ui-monospace,Menlo,monospace; color:var(--ink3); margin-left:6px; }}
.sw {{ display:inline-block; width:10px; height:10px; border-radius:50%; background:var(--c); margin-right:8px; vertical-align:baseline; }}
.sw.none {{ background:none; box-shadow:inset 0 0 0 1.5px var(--ink3); }}
.legend {{ list-style:none; padding:0; margin:0 0 6px; display:flex; flex-wrap:wrap; gap:4px 18px; font-size:14px; }}
.chart svg {{ width:100%; max-width:{_chart_width(board)}px; height:auto; display:block; }}
.chart .grid {{ stroke:var(--grid); stroke-width:1; }}
.chart .base {{ stroke:var(--ink3); stroke-width:1.5; stroke-dasharray:4 4; }}
.chart .tick {{ fill:var(--ink2); font-size:12px; }}
.chart .label {{ fill:var(--ink); font-size:12px; }}
.chart .line {{ fill:none; stroke-width:2; stroke-linejoin:round; }}
.chart .dot {{ stroke:var(--bg); stroke-width:2; }}
.chart .hit {{ fill:transparent; }}
.chart .pt:hover .dot {{ r:7; }}
details {{ border-bottom:1px solid var(--rule); padding:10px 0; }}
summary {{ cursor:pointer; color:var(--ink2); }}
summary strong {{ color:var(--ink); }}
details table {{ margin-top:8px; }}
footer {{ margin-top:40px; color:var(--ink3); font-size:13px; }}
footer code {{ margin:0; word-break:break-all; white-space:normal; }}
</style>
</head>
<body>
<main>
<h1>EvoGPT</h1>
<p class="lede">Which training algorithm wins with scaling?</p>
<ul class="meta">
<li>Season <b>{escape(board["contest_id"])}</b></li>
<li>Status <b>{escape(status)}</b></li>
<li>Round <b>{board["round"]}</b></li>
<li>Next promotion <b>{escape(_when(board["deadline"]))}</b></li>
<li>Open scales <b>{escape(", ".join(active))}</b></li>
</ul>
{banner}
<h2>Compute multiplier by scale</h2>
<p class="note">How many times a scale's FLOP budget the baseline needs to reach the same median validation loss. Above 1× beats the baseline.</p>
{chart}
<h2>Overall</h2>
<p class="note">Ranked by the weakest open scale (underlined), then by the geometric mean. {escape(promote_note)}</p>
<div class="scroll"><table>
<thead><tr><th class="num">#</th><th>Entry</th><th class="num">Score</th>{"".join(head)}<th class="num">Geo mean</th><th class="num">Scales</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table></div>
<h2>Per scale</h2>
{"".join(details)}
<footer>Loss is cross-entropy per byte on the season's held-out validation split, median over seeds. FLOPs are counted from executed operations.
Configuration digest <code>{escape(board["config_digest"])}</code></footer>
</main>
</body>
</html>
"""
