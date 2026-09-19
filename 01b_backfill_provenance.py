#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""01b_backfill_provenance.py — 回填 production checkpoint 缺失的 provenance。

背景（2026-09-14 盘点实锤）：
- l2_checkpoint_adaptive/metadata.json 记录 rate_max=20.0（用户传入默认值），
  但 logs_l2_adaptive 中真实 adaptive 值为 107.30583953857422（p99.5×1.2）。
- L3 实际 batch_size=1024（由 route_counts 74 batches × 1024 ≈ 74984 推断），
  checkpoint 记录的 2048 是 L2 时的参数，L3 运行参数未回写。

本脚本把找回的真实值回填进 checkpoint metadata.json（保留原记录不删除，
新增 recovered 字段并注明来源）。只改 metadata.json，不动任何数组。

用法（默认参数即盘点找回的值，直接运行即可）：
    python 01b_backfill_provenance.py
    python 01b_backfill_provenance.py --dry-run   # 只打印不写入
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np

import config as C


def main() -> int:
    ap = argparse.ArgumentParser(description="回填 production checkpoint provenance")
    ap.add_argument("--checkpoint", default=os.path.join(
        C.ENGINE_OUTPUT_DIR, "l2_checkpoint_adaptive"))
    ap.add_argument("--rate-max-actual", type=float, default=107.30583953857422)
    ap.add_argument("--beta-p995", type=float, default=89.4215316772461)
    ap.add_argument("--gamma-p995", type=float, default=89.4215316772461)
    ap.add_argument("--l3-batch-size", type=int, default=1024,
                    help="由 route_counts 总批次数推断的 L3 实际 batch_size")
    ap.add_argument("--recovered-from", default="logs_l2_adaptive/*.json (JsonLog)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta_p = os.path.join(args.checkpoint, "metadata.json")
    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("[回填前] metadata.json:")
    print(json.dumps(meta, indent=2))

    # ---- 诊断：beta 与 gamma 是否可疑地相同（p995 全精度相等的疑点）
    beta = np.load(os.path.join(args.checkpoint, "beta.npy"))
    gamma = np.load(os.path.join(args.checkpoint, "gamma.npy"))
    identical = bool(np.array_equal(beta, gamma))
    b_p995 = float(np.percentile(beta, 99.5))
    g_p995 = float(np.percentile(gamma, 99.5))
    print(f"\n[诊断] beta vs gamma: 完全相同={identical}")
    print(f"[诊断] checkpoint 内实际 p99.5: beta={b_p995:.4f}, gamma={g_p995:.4f}")
    print(f"[诊断] 日志记录 probe p99.5: beta={args.beta_p995:.4f}, gamma={args.gamma_p995:.4f}")
    if identical:
        print("[诊断] !! beta.npy 与 gamma.npy 完全相同——L2 拟合存在参数耦合 bug，必须调查！")
    elif abs(b_p995 - args.beta_p995) / max(b_p995, 1e-12) > 0.5:
        print("[诊断] 注意：checkpoint p99.5 与 probe p99.5 差异较大属正常"
              "（probe 轮 rate_max=1000，精化轮才用 adaptive 边界截断）")

    # ---- 回填
    updates = {
        "rate_max_requested": meta.get("rate_max"),       # 保留原记录（20.0=默认值）
        "rate_max_actual": args.rate_max_actual,
        "rate_max_mode": "adaptive",
        "beta_p995": args.beta_p995,
        "gamma_p995": args.gamma_p995,
        "l3_actual_batch_size": args.l3_batch_size,
        "provenance_recovered_from": args.recovered_from,
        "provenance_backfill_date": time.strftime("%Y-%m-%d"),
        "provenance_note": "原 rate_max=20.0 为代码 bug（记录用户传入值而非 "
                           "adaptive 实际值）；实际值为 107.3058...，"
                           "由 L2 运行日志找回。",
    }
    meta.update(updates)

    print("\n[回填内容]:")
    print(json.dumps(updates, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("\n[dry-run] 未写入。")
        return 0

    shutil.copy2(meta_p, meta_p + f".bak_{time.strftime('%Y%m%d_%H%M%S')}")
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n[完成] 已回填: {meta_p}（原文件已备份 .bak）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
