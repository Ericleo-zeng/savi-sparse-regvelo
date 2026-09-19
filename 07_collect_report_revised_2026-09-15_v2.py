#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""07_collect_report.py — 汇总全部验证结果为一份报告（2026-09-15 v2）。

修订要点（对应 09-15 V5 验证的教训）：
    1. 敏感性对比由 glob 全收改为【显式登记制】：每条注明性质
       （判决依据 / 参考 / 旁证），含混淆的对比（如 3a/T0.5）只能以
       旁证身份出现，不得与判决依据平权；
    2. 主表新增 V5 / V8 独立行，V5 的 FAIL 与 caveat 一眼可见；
    3. 新增"已知问题与 caveats"节（T0.4 / T0.5 / V5 caveat / 显存标度律 /
       硬件差异），内容同步进 SUMMARY.json；
    4. 利用 06 v2 写入的 generated_at / ref / tests 溯源字段展示对比来源；
       无溯源字段的 legacy 文件标注"(legacy, 无溯源)"。
    5. glob 到但未登记的 COMPARE 文件进"未登记"警示区，提示人工归类，
       防止垃圾判决（如崩溃半成品的对比结果）静默进入报告。
       EXCLUDED_COMPARES 中的条目不再重复出现在警示区（2026-09-15 补丁）。

用法：
    python 07_collect_report.py

输出：
    results/REPORT.md      — 人类可读汇总（含每项判决与关键数字）
    results/SUMMARY.json   — 机器可读汇总
"""
from __future__ import annotations

import glob
import json
import os

import config as C

ITEMS = [
    ("V1", "ODE 传播残差", "V1_ode_residual/V1_result.json",
     "median < 2e-3 且 P99 < 1e-2"),
    ("V2", "batch-size 不变性", "V2_batch_invariance/V2_result.json",
     "median < 1e-5，P99 < 1e-4，route 分布不随 bs 变化"),
    ("V3", "CPU/GPU 一致性", "V3_cpu_gpu/V3_result.json",
     "FP32 下 max rel < 1e-4 为 PASS，>= 1e-2 为 FAIL"),
    ("V1b", "ODE 残差·生产 n=5000（稳定性）", "V1_ode_residual_prod_n5000/V1_result.json",
     "同基线判据；v_p99=1.17e-2，FAIL 非抽样噪声"),
    ("V1c", "ODE 残差·子集 faithful（对照）", "V1_subset_faithful/V1_result.json",
     "同基线判据；v_p99=9.36e-3 擦线 PASS——v 尾部预算贴验收线（见 caveats）"),
]

# ---------------- 敏感性对比显式登记（性质字段控制报告中的地位） ----------------
COMPARE_ITEMS = [
    ("V5_rate_max", "l3_adaptive__vs__l3_r20_l3_r100/COMPARE_result.json",
     "判决依据", "3b：同数据重拟合，唯一变量 rate_max（有效对比）"),
    ("V8_K_sensitivity", "V8_K_sensitivity/COMPARE_result.json",
     "判决依据", "K=10/20 vs K=40；基于 T0.5 拼接 checkpoint 的 L3 数值自洽性"),
    ("V5_K20_probe", "l3_adaptive_srcprobe__vs__l3_adaptive_K20probe/COMPARE_result.json",
     "参考", "K=20 vs K=10 @ faithful adaptive checkpoint；一次性验证产物"),
]

# 明确不进报告的对比（旁证/含混淆），报告须交代去向
EXCLUDED_COMPARES = {
    "l3_K10": ("3a 对比（ref=T0.5 拼接 checkpoint），含拟合数据量混淆，"
               "仅旁证，不作判决依据——见 known_issues T0.5"),
}

KNOWN_ISSUES = [
    ("T0.4（已修）", "sparse_regvelo_engine.py：--rate-max 20 因 20.0 兼任 "
     "argparse 默认值与合法请求值被静默忽略（走 adaptive）；187/224/309/312/572 "
     "五处已修，真实 rerun 验证 requested=actual=20/mode=manual"),
    ("T0.5（已记录）", "子集参考 checkpoint = 生产 per-gene 参数 + 子集 per-cell "
     "量的拼接物（config.ensure_subset_checkpoint 设计行为）；仅适用 L3 数值验证，"
     "禁止作 L2 参数级对比参考；metadata 已补 params_source 等字段"),
    ("V5 caveat", "rate_max 高度敏感：rm20 v cosine 中位 0.386 / rm100 0.699（3b 干净"
     "对比）；100 vs 107.31 亦 FAIL——adaptive 系数(p99.5×1.2)是决定性超参，"
     "生产结论依赖 adaptive 的选择，报告讨论节须保留此 caveat"),
    ("显存标度律", "L3 峰值显存 ∝ K×batch_size：K=10×bs=1024 → 5.82 GB；"
     "K=20×bs=1024 在 11.5 GB 卡 OOM，bs=512 通过。V8（K≤40, bs=2048）当年在"
     "更大显存机器上运行，跨机对比须注明硬件"),
    ("溯源链（已加固）", "2026-09-15 起：run_l3_only 将 source_checkpoint 块"
     "（path+hash+rate_max_*）写入 l3_metadata.json；06 v2 对比前强制完整性检查"
     "并提示 ref/test 配置差异；05 脚本 checkpoint 守卫校验 metadata 内容"),
]


def _cmp_row(tag: str, j: dict, role: str, note: str, rel_path: str) -> list[str]:
    verdict = j.get("verdict", "?")
    legacy = "generated_at" not in j
    src = ""
    if not legacy:
        tests = "+".join(j.get("tests", {}).keys())
        src = f"{os.path.basename(j.get('ref',''))} vs {tests} @ {j.get('generated_at','')}"
    suffix = " (legacy, 无溯源)" if legacy else ""
    return [f"| {tag}{suffix} | {verdict} | {role} | {note} | {src} |"]


def main() -> int:
    lines = ["# Sparse RegVelo 验证报告", "",
             f"- 验证目录：`{C.VALIDATION_DIR}`",
             f"- production 规模：{C.EXPECTED_N} cells × {C.EXPECTED_G} genes",
             f"- 报告生成：2026-09-15 v2（显式登记制 + caveats 节）", ""]
    summary = {}
    n_pass = n_total = 0

    # ---------------- 主验证项 ----------------
    lines.append("## 主验证项")
    lines.append("| 验证项 | 内容 | 判决 | 验收线 |")
    lines.append("|---|---|---|---|")
    for vid, title, rel, criterion in ITEMS:
        p = os.path.join(C.RESULTS_DIR, rel)
        if not os.path.isfile(p):
            lines.append(f"| {vid} | {title} | 未运行 | {criterion} |")
            summary[vid] = {"status": "not_run"}
            continue
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        v = j.get("verdict", "?")
        summary[vid] = {"status": v, "detail_file": rel}
        n_total += 1
        n_pass += (v == "PASS")
        lines.append(f"| {vid} | {title} | {v} | {criterion} |")

    # ---------------- 敏感性对比（显式登记） ----------------
    lines += ["", "## 敏感性对比（Level 3 鲁棒性）", "",
              "| 条件组 | 判决 | 性质 | 说明 | 来源 |",
              "|---|---|---|---|---|"]
    registered_paths = set()
    for tag, rel, role, note in COMPARE_ITEMS:
        p = os.path.join(C.RESULTS_DIR, rel)
        if not os.path.isfile(p):
            lines.append(f"| {tag} | 未运行 | {role} | {note} |  |")
            summary[tag] = {"status": "not_run", "role": role}
            continue
        registered_paths.add(os.path.abspath(p))
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        lines += _cmp_row(tag, j, role, note, rel)
        summary[tag] = {"status": j.get("verdict"), "role": role,
                        "detail_file": rel}

    # ---------------- 未登记对比（警示区） ----------------
    leftovers = [p for p in glob.glob(os.path.join(C.RESULTS_DIR, "*", "COMPARE_result.json"))
                 if os.path.abspath(p) not in registered_paths
                 and os.path.basename(os.path.dirname(p)) not in EXCLUDED_COMPARES]
    if leftovers:
        lines += ["", "### ⚠ 未登记的对比结果（未纳入上表，请人工归类或删除）", "",
                  "| 路径 | 判决 |", "|---|---|"]
        for p in sorted(leftovers):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    j = json.load(f)
                v = j.get("verdict", "?")
            except Exception:
                v = "读取失败"
            rel = os.path.relpath(p, C.RESULTS_DIR)
            lines.append(f"| `{rel}` | {v} |")
            summary[f"unregistered:{rel}"] = {"status": v, "role": "未登记"}
    if EXCLUDED_COMPARES:
        lines += ["", "### 已排除的对比（旁证，不作判决依据）", ""]
        for tag, reason in EXCLUDED_COMPARES.items():
            lines.append(f"- **{tag}**：{reason}")

    # ---------------- 已知问题与 caveats ----------------
    lines += ["", "## 已知问题与 caveats", ""]
    for name, desc in KNOWN_ISSUES:
        lines.append(f"- **{name}**：{desc}")
    summary["known_issues"] = {k: v for k, v in KNOWN_ISSUES}

    # ---------------- 统计 ----------------
    lines += ["", f"**主验证通过：{n_pass}/{n_total}**", "",
              "判决含义：全 PASS -> 达到 Level 2（计算方法可信）；",
              "敏感性判决依据全 PASS -> Level 3（结果鲁棒）。",
              "V5 为 FAIL：rate_max 高度敏感，结论以 caveat 形式保留（见上节），",
              "不重跑 production（生产即 adaptive，且引擎重跑可逐位复现）。",
              "任一项 FAIL 时，先读对应 detail JSON 定位，再决定是否修复重跑。"]

    report_path = os.path.join(C.RESULTS_DIR, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(C.RESULTS_DIR, "SUMMARY.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n".join(lines))
    print(f"\n[07] 报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
