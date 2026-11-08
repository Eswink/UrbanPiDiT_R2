"""Render Stage C2's result page FROM readings.json, never from typed numbers.

Every number in the page is read out of the readings file; nothing is transcribed. The page is
written only when readings.json exists, so a placeholder can never be committed and mistaken
for a measurement.
"""
import json
import sys
from pathlib import Path


def _mark(value) -> str:
    """Render a boolean as the page's bold Chinese true/false."""
    return "**真**" if value else "**假**"


def _risk_tables(r: dict, ex: dict, seeds: list[str]) -> list[str]:
    """The §2 tables: per-seed paired 72 h skill, then all six arms' five leads."""
    rows = ["## §2 逐 seed 的 72 h t2m skill（同 seed 配对）", "",
            "| seed | off | on | Δ(on−off) |", "| --- | --- | --- | --- |"]
    for seed in seeds:
        off = r["seeds"][seed]["off"]["skill"]["72"]
        on = r["seeds"][seed]["on"]["skill"]["72"]
        rows.append(f"| {seed} | {off:+.6f} | {on:+.6f} | {on - off:+.6f} |")
    rows += ["", "六个臂的五个 lead 全部 skill（t2m，分母为注册 D2 值）：", "",
             "| seed | 臂 | 6 h | 12 h | 24 h | 48 h | 72 h |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for seed in seeds:
        for arm in ("off", "on"):
            skill = r["seeds"][seed][arm]["skill"]
            cells = " | ".join(f"{skill[str(lead)]:+.6f}" for lead in (6, 12, 24, 48, 72))
            rows.append(f"| {seed} | {arm} | {cells} |")
    rows += ["", "---", ""]
    return rows


def _verdict_lines(ex: dict) -> list[str]:
    """§3: the verdict text is READ from exit_conditions, never decided here."""
    if ex["all"]:
        return ["四项预声明出口全真 ⇒ 按预声明 **进入独立复核**（`advance-to-independent-review`）。",
                "",
                "**本页不自宣科学通过**：这只是一轮工程出口，72 h skill 的配对均值不退化**不等于**"
                "该目标项有科学价值；独立复核与用户裁定仍在后。"]
    return ["至少一项预声明出口为假 ⇒ 按预声明**永久关闭「导数域目标匹配」这一支**"
            "（`permanently-close-derivative-matching-branch`）：不调权重、不换项、不加 seed、不换名重开。",
            "",
            "**如实记录**：本页不解释机制，也不主张该方向在别的形式下同样无效——预声明关闭的是**这一支**。"]


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 3:
        raise SystemExit("usage: render_c2_result.py <readings.json> <protocol.json> <out.md>")
    readings_path, protocol_path, out_path = (Path(value) for value in arguments)
    r = json.loads(readings_path.read_text(encoding="utf-8"))
    body = json.loads(protocol_path.read_text(encoding="utf-8"))
    ex = r["exit_conditions"]
    seeds = sorted(r["seeds"], key=int)

    lines = [
        "# S3 Stage C2（导数域目标匹配 on/off）读数 + 预声明出口 (a)–(d)",
        "",
        "- **日期**：2026-11-08",
        f"- **状态**：预声明出口 (a)–(d) 逐条求值；`all = {ex['all']}`、`verdict = {ex['verdict']}`",
        "- **`scientific_claim: false`**；`test_read: false`；`r = 0`",
        f"- **协议**：`{r['protocol_sha256']}`",
        f"- **代码**：`{body['code_commit']}`；`model_code_sha256 {body['model_code_sha256'][:8]}…`、"
        f"`training_code_sha256 {body['training_code_sha256'][:8]}…`",
        "",
        "---",
        "",
        "## §1 预声明工程出口 (a)–(d)",
        "",
        "| 判据 | 要求 | 实测 | 结果 |",
        "| --- | --- | --- | --- |",
        f"| (a) | 72 h t2m skill 配对均值相对同代码 off 臂**不退化** | 逐 seed Δ："
        + "、".join(f"{s}={ex['delta_72h'][s]:+.6f}" for s in seeds)
        + f"；均值 `{sum(ex['delta_72h'].values())/len(ex['delta_72h']):+.6f}` | {_mark(ex['a_72h_skill_mean_delta'])} |",
        f"| (b) | on 臂逐步记录两项匹配且**严格为正** | preflight 四行 total 全正："
        f"{all(row['total'] > 0.0 for row in r['preflight']['rows'])}；三 seed 的 on 臂 "
        f"`constraint_terms_recorded`："
        + "、".join(f"{s}={r['seeds'][s]['on'].get('constraint_terms_recorded')}" for s in seeds)
        + f" | {_mark(ex['b_matching_bit'])} |",
        f"| (c) | off 路径与注册算术**逐位等价** | 由 dispatcher 的 `None` 早退与逐位比对支撑 | {_mark(ex['c_off_path_unchanged'])} |",
        f"| (d) | 无判据被改动 | `scientific_claim={r['scientific_claim']}`、"
        f"`test_read={r['test_read']}` | {_mark(ex['d_no_criterion_changed'])} |",
        "",
        f"`exit_conditions`（驱动自算、未手改）：`{json.dumps(ex, ensure_ascii=False, sort_keys=True)}`",
        "",
        "---",
        "",
        "## §2 逐 seed 的 72 h t2m skill（同 seed 配对）",
        "",
        "| seed | off | on | Δ(on−off) |",
        "| --- | --- | --- | --- |",
    ]
    for s in seeds:
        off = r["seeds"][s]["off"]["skill"]["72"]
        on = r["seeds"][s]["on"]["skill"]["72"]
        lines.append(f"| {s} | {off:+.6f} | {on:+.6f} | {on - off:+.6f} |")

    lines += _risk_tables(r, ex, seeds)
    lines += ["## §3 判决与后续", ""]
    lines += _verdict_lines(ex)
    lines += [
        "",
        "---",
        "",
        "## §4 限制（不得省略）",
        "",
        "- `scientific_claim: false`、`test_read: false`、`r = 0`；未读 2023 test。",
        "- 本页所有数字**逐项取自 `readings.json`**，未手抄；`readings.json` 由驱动的 `readings()` 计算。",
        "- 单 seed 单次运行，GPU 训练不保证逐位可复现；checkpoint/源码/数据身份是严格校验的。",
        "- 72 h skill 的分母是 train-only 的 48 桶 D2 reference；只有 t2m 有注册分母。",
        "- 预声明的 (a) 是**保号判据**（均值 Δ ≥ 0），不是效应量判据；一个接近零的正值不代表有意义的改进。",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
