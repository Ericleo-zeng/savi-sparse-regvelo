#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""06_compare_conditions.py — 敏感性条件对比（V5/V6/V7/V8 通用）。

对比同一细胞子集上不同条件的 L3 输出，回答评价报告的核心问题：
"生物学结论是否独立于参数/实现细节？"

判据（速度场层面）：
    - v 逐细胞 cosine：median > 0.99 -> 方向高度一致
    - v 相对 L2：median < 0.05 -> 幅度差异可忽略
    - 结论方向是否翻转，需在这些输出上重跑 downstream（sparse_regvelo_downstream.py）
      后人工核对 ΔP protective / ΔP IFN 符号

用法：
    python 06_compare_conditions.py \
        --ref results/V5_rate_max/l3_adaptive \
        --test results/V5_rate_max/l3_r20 results/V5_rate_max/l3_r100 \
        --tag V5_rate_max

输出：
    results/{tag}/COMPARE_result.json

2026-09-15 修订（V5 教训）：
    1. 未指定 --tag 时，默认目录名 = {ref}__vs__{test1_test2_...}，
       防止同一 ref 的多组对比互相覆盖 COMPARE_result.json；
    2. 结果 JSON 记录 generated_at / cmd / pass_criteria，便于存档溯源；
       目标文件已存在时打印覆盖警告；
    3. 对比前自动检查 ref/test 的 l3_metadata.json 键差异并提示
       （best-effort，不硬失败）——防止 T0.5 型"双方来自不同拟合数据"
       的混淆在无人察觉的情况下进入判决；
    4. 修复 shape 部分不匹配的边缘 bug：原守卫在 u_end 报错时会误判
       通过并继续算 v cosine。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

import config as C

COS_PASS = 0.99
REL_PASS = 0.05


def _load_l3_meta(d: str) -> dict | None:
    """best-effort 读取 L3 输出目录的 l3_metadata.json；缺失或损坏返回 None。"""
    p = os.path.join(d, "l3_metadata.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _ensure_complete_l3(d: str) -> None:
    """L3 输出完整性检查：l3_metadata.json 是 run_l3_only 跑完全程后才写的
    天然完成标记。缺失 = 崩溃/半成品输出，拒绝对比（2026-09-15 OOM 事件教训：
    崩溃目录里的全零 memmap 曾被当作正常结果算出 garbage 判决）。"""
    marker = os.path.join(d, "l3_metadata.json")
    if not os.path.exists(marker):
        raise SystemExit(f"[FATAL] {d} 缺少 l3_metadata.json：疑似崩溃/半成品 "
                         f"L3 输出，拒绝对比。请重跑 L3 或移除该目录。")
    meta = _load_l3_meta(d) or {}
    if "N" in meta:
        v = np.load(os.path.join(d, "v.npy"), mmap_mode="r")
        if v.shape[0] != meta["N"]:
            raise SystemExit(f"[FATAL] {d}: v.npy 行数 {v.shape[0]} 与 "
                             f"l3_metadata N={meta['N']} 不符，输出损坏。")


def main() -> int:
    ap = argparse.ArgumentParser(description="敏感性条件对比")
    ap.add_argument("--ref", required=True, help="参照条件 L3 输出目录")
    ap.add_argument("--test", nargs="+", required=True, help="待对比 L3 输出目录")
    ap.add_argument("--tag", type=str, default=None,
                    help="结果文件名标签（默认 {ref}__vs__{test名拼接}，防止覆盖）")
    args = ap.parse_args()

    _ensure_complete_l3(args.ref)
    ref = {n: np.load(os.path.join(args.ref, f"{n}.npy"), mmap_mode="r")
           for n in ("u_end", "s_end", "v")}
    ref_meta = _load_l3_meta(args.ref)
    result = {
        "ref": args.ref,
        "tests": {},
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "cmd": " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
        "pass_criteria": {"v_cosine_median": COS_PASS,
                          "v_rel_l2_median": REL_PASS},
    }
    all_pass = True

    for d in args.test:
        name = os.path.basename(d.rstrip("/"))
        _ensure_complete_l3(d)
        entry = {}
        has_error = False

        # ---- 配置差异提示（T0.5 教训：明示双方来源差异，不硬失败）
        tst_meta = _load_l3_meta(d)
        if ref_meta and tst_meta:
            diff_keys = sorted(k for k in set(ref_meta) | set(tst_meta)
                               if ref_meta.get(k) != tst_meta.get(k))
            if diff_keys:
                print(f"[CMP] 提示 {name}: 与 ref 的 l3_metadata 键差异: "
                      f"{diff_keys}（确认是否为本次实验的预期变量）")

        for arr in ("u_end", "s_end", "v"):
            tst = np.load(os.path.join(d, f"{arr}.npy"), mmap_mode="r")
            if tst.shape != ref[arr].shape:
                entry[arr] = {"error": f"shape mismatch {tst.shape} vs "
                                       f"{ref[arr].shape}（细胞子集必须一致）"}
                all_pass = False
                has_error = True
                continue
            rel = C.rel_l2_per_cell(np.asarray(tst), np.asarray(ref[arr]))
            entry[f"{arr}_rel_l2"] = C.summarize(rel)

        if not has_error:
            cos = C.cosine_per_cell(
                np.asarray(np.load(os.path.join(d, "v.npy"), mmap_mode="r")),
                np.asarray(ref["v"]))
            entry["v_cosine"] = C.summarize(cos)
            entry["v_cosine_min"] = float(cos.min())
            ok = (entry["v_cosine"]["median"] > COS_PASS
                  and entry["v_rel_l2"]["median"] < REL_PASS)
            entry["pass"] = bool(ok)
            all_pass &= ok
            print(f"[CMP] {name}: v cosine median={entry['v_cosine']['median']:.5f} "
                  f"min={entry['v_cosine_min']:.5f} | v rel_l2 median="
                  f"{entry['v_rel_l2']['median']:.3e} -> {'PASS' if ok else 'FAIL'}")
        result["tests"][name] = entry

    result["verdict"] = "PASS" if all_pass else "FAIL"

    # ---- 输出目录：默认含 test 名，防同 ref 多组对比互相覆盖
    if args.tag:
        tag = args.tag
    else:
        ref_base = os.path.basename(args.ref.rstrip("/"))
        tests_base = "_".join(os.path.basename(d.rstrip("/")) for d in args.test)
        tag = f"{ref_base}__vs__{tests_base}"
    out_dir = os.path.join(C.RESULTS_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "COMPARE_result.json")
    if os.path.exists(out_json):
        print(f"[CMP] 警告: {out_json} 已存在，将被覆盖"
              f"（用 --tag 区分不同对比组）")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[CMP] 判决: {result['verdict']}  ->  {out_json}")
    print("[CMP] 提醒：速度场一致 ≠ 结论一致，请在这些输出上重跑 downstream "
          "确认 ΔP 方向未翻转")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
