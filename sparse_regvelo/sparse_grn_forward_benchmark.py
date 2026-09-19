"""sparse_grn_forward_benchmark.py — Phase 0 稀疏 GRN 前向基准（SPEC §3.2）。

任务：
1. 合成稀疏 W（默认 G=17714, nnz=194843 随机稀疏；--small 时 G=512, nnz=3000），
   转 CSR（common.make_csr，索引转 int32），验证并记录 csr_bytes（全量 ≈1.6 MB，
   对照 dense FP32 1.255 GB）。
2. 可微前向 alpha = clamp(softplus(torch.sparse.mm(W_csr, s.T) + b).T, 0, 50)。
3. batch 梯度：alpha.square().mean() 反传，断言 W_csr.grad 为 CSR 稀疏张量。
4. batch 扫描，记录 peak_memory_gb / spmm_time_sec（中位）/ effective_bandwidth_gbps
   （访存口径：gather nnz×B×4B×2 + 写回 B×G×4B×2）。
5. 红线（仅 CUDA）：batch=8192 时 peak < 5 GB，违反 exit code 2；CPU 记 "skipped_cpu"。

红线合规：全程无 dense (G,G) 可学习矩阵；FP32 状态；GRN 一律 CSR。
"""
from __future__ import annotations

import argparse
import statistics
import sys

import numpy as np
import torch

from common import (
    DEVICE,
    JsonLog,
    csr_bytes,
    make_csr,
    now,
    peak_memory_gb,
    reset_peak_memory,
    synth_params,
)

ALPHA_CAP = 50.0
FP32 = 4  # bytes


# ---------------------------------------------------------------- 数据合成
def synth_random_csr(G: int, nnz: int, seed: int = 0) -> torch.Tensor:
    """随机稀疏 GRN：nnz 条唯一边 (regulator -> target)，权重 ~ N(0, 1)。

    直接以 COO 构造（不经过 dense (G,G)），转 CSR（FP32，索引 int32），放到 DEVICE。
    """
    rng = np.random.default_rng(seed)
    flat = rng.choice(G * G, size=min(nnz, G * G), replace=False)
    rows = torch.from_numpy(flat // G).to(torch.int64)
    cols = torch.from_numpy(flat % G).to(torch.int64)
    vals = torch.from_numpy(rng.standard_normal(flat.size).astype(np.float32))
    W_coo = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, (G, G)
    ).coalesce()
    W_csr = make_csr(W_coo)  # -> CSR float32 on DEVICE
    # 索引转 int32：csr_bytes 口径 = nnz*4(values) + nnz*4(col) + (G+1)*4(crow)
    W_csr = torch.sparse_csr_tensor(
        W_csr.crow_indices().to(torch.int32),
        W_csr.col_indices().to(torch.int32),
        W_csr.values().to(torch.float32),
        size=W_csr.shape,
        device=W_csr.device,
    )
    return W_csr


def synth_spliced(B: int, G: int, seed: int = 2) -> torch.Tensor:
    """合成 spliced 丰度 s (B, G) FP32，非负，量级 0~5。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.rand(B, G, generator=g) * 5.0
    return s.to(DEVICE)


# ---------------------------------------------------------------- 可微前向
def grn_forward(W_csr: torch.Tensor, s: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """alpha = clamp(softplus(W_csr @ s.T + b).T, 0, 50)。s: (B,G) -> alpha: (B,G)。"""
    z = torch.sparse.mm(W_csr, s.T) + bias.unsqueeze(1)
    return torch.clamp(torch.nn.functional.softplus(z).T, 0.0, ALPHA_CAP)


def _sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------- 基准
def benchmark_batch(W_csr, bias, B: int, G: int, nnz_actual: int,
                    repeat: int, warmup: int, seed: int) -> dict:
    s = synth_spliced(B, G, seed=seed + B)
    with torch.no_grad():
        for _ in range(warmup):
            grn_forward(W_csr, s, bias)
        _sync()
        reset_peak_memory()
        times = []
        for _ in range(repeat):
            t0 = now()
            grn_forward(W_csr, s, bias)
            _sync()
            times.append(now() - t0)
    peak = peak_memory_gb()
    med = statistics.median(times)
    # 访存口径：gather nnz×B×4B×2 + 写回 B×G×4B×2
    bytes_moved = nnz_actual * B * FP32 * 2 + B * G * FP32 * 2
    bw = bytes_moved / med / 1e9 if med > 0 else float("inf")
    del s
    return {
        "batch_size": B,
        "peak_memory_gb": round(peak, 4),
        "spmm_time_sec": round(med, 6),
        "effective_bandwidth_gbps": round(bw, 2),
        "bytes_moved": int(bytes_moved),
        "repeats": repeat,
    }


def grad_check(W_csr, bias, B: int, G: int, seed: int) -> bool:
    """alpha.square().mean() 反传，断言 W_csr.grad 为 CSR 稀疏张量且非零模式 ⊆ W 模式。"""
    s = synth_spliced(B, G, seed=seed + 7)
    alpha = grn_forward(W_csr, s, bias)
    loss = alpha.square().mean()
    loss.backward()
    g = W_csr.grad
    assert g is not None, "W_csr.grad is None — 稀疏反传失败"
    assert g.layout == torch.sparse_csr, f"W_csr.grad layout={g.layout}, 期望 sparse_csr"
    assert g._nnz() <= W_csr._nnz(), "grad 非零模式超出 W 模式"
    return True


# ---------------------------------------------------------------- CLI / 主流程
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 0 稀疏 GRN 前向基准（SPEC §3.2）")
    p.add_argument("--G", type=int, default=17714, help="基因数（全量默认 17714）")
    p.add_argument("--nnz", type=int, default=194843, help="GRN 边数（全量默认 194843）")
    p.add_argument("--batch-sizes", type=str, default="1024,2048,4096,8192",
                   help="逗号分隔的 batch 扫描档位")
    p.add_argument("--out-dir", type=str, default="logs", help="JSON 日志输出目录")
    p.add_argument("--small", action="store_true",
                   help="沙箱自检档：G=512, nnz=3000, batch={128,256}")
    p.add_argument("--repeat", type=int, default=10, help="每档计时重复次数（取中位）")
    p.add_argument("--warmup", type=int, default=3, help="每档预热次数")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.small:
        args.G, args.nnz = 512, 3000
        args.batch_sizes = "128,256"
    batches = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    assert batches, "batch-sizes 为空"

    log = JsonLog("sparse_grn_forward_benchmark", out_dir=args.out_dir)
    log.record("G", args.G)
    log.record("nnz_target", args.nnz)
    log.record("batch_sizes", batches)
    log.record("small_mode", bool(args.small))

    # 1) 合成 W -> CSR，验证存储
    W_csr = synth_random_csr(args.G, args.nnz, seed=args.seed)
    nnz_actual = W_csr._nnz()
    csr_b = csr_bytes(W_csr)
    dense_b = args.G * args.G * FP32  # 对照：dense FP32 (G,G)
    log.record("nnz_actual", nnz_actual)
    log.record("csr_bytes", csr_b)
    log.record("csr_mb", round(csr_b / 1e6, 3))
    log.record("dense_fp32_bytes", dense_b)
    log.record("dense_fp32_gb", round(dense_b / 1e9, 4))
    log.record("storage_ratio_dense_over_csr", round(dense_b / csr_b, 1))

    # bias（common 合成参数，仅取 bias）
    _, _, bias = synth_params(args.G, seed=args.seed + 1)
    bias = bias.to(DEVICE)

    # 2)+3) 可微前向 + batch 梯度断言（用最小 batch 做梯度检查，节省内存）
    W_csr = W_csr.detach().requires_grad_(True)
    grad_ok = grad_check(W_csr, bias, min(batches), args.G, seed=args.seed)
    log.record("grad_is_sparse", bool(grad_ok))
    log.record("grad_layout", str(W_csr.grad.layout))
    log.record("grad_nnz", int(W_csr.grad._nnz()))
    W_csr = W_csr.detach().requires_grad_(False)

    # 4) batch 扫描
    results = [
        benchmark_batch(W_csr, bias, B, args.G, nnz_actual,
                        repeat=args.repeat, warmup=args.warmup, seed=args.seed)
        for B in batches
    ]
    log.record("batches", results)

    # 5) 红线（仅 CUDA）：batch=8192 peak < 5 GB
    red_line_pass: object
    exit_code = 0
    if DEVICE == "cuda":
        entry = next((r for r in results if r["batch_size"] == 8192), None)
        if entry is not None:
            red_line_pass = bool(entry["peak_memory_gb"] < 5.0)
            if not red_line_pass:
                exit_code = 2
        else:
            red_line_pass = "no_8192_batch"
    else:
        red_line_pass = "skipped_cpu"
    log.record("red_line_pass", red_line_pass)
    log.record("red_line_rule", "cuda: peak@8192 < 5 GB else exit 2")

    path = log.finalize()
    print(f"[bench] device={DEVICE} G={args.G} nnz={nnz_actual} "
          f"csr={csr_b / 1e6:.2f}MB (dense {dense_b / 1e9:.3f}GB, "
          f"{dense_b / csr_b:.0f}x) grad_is_sparse={grad_ok}")
    for r in results:
        print(f"[bench] B={r['batch_size']:>5} peak={r['peak_memory_gb']:.4f}GB "
              f"t={r['spmm_time_sec'] * 1e3:.3f}ms bw={r['effective_bandwidth_gbps']:.2f}GB/s")
    print(f"[bench] red_line_pass={red_line_pass} -> {path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
