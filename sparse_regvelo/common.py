"""common.py — 后 RegVelo 稀疏闭式推断引擎公共基础模块（SPEC §2，接口冻结）。

提供：设备检测、JSON 运行日志、峰值显存/内存追踪、CSR 工具、合成 GRN/参数、
DOP853 高精度参考积分器、度量函数。所有子代理不得修改本文件接口。
"""
from __future__ import annotations

import json
import os
import time
import tracemalloc

import numpy as np
import torch

tracemalloc.start()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_device() -> str:
    return DEVICE


def now() -> float:
    return time.perf_counter()


# ---------------------------------------------------------------- 日志与显存
def reset_peak_memory() -> None:
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    else:
        global _CPU_PEAK_BASE
        _CPU_PEAK_BASE = tracemalloc.get_traced_memory()[0]


_CPU_PEAK_BASE = tracemalloc.get_traced_memory()[0]


def peak_memory_gb() -> float:
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    current, peak = tracemalloc.get_traced_memory()
    return max(peak - _CPU_PEAK_BASE, 0) / 1e9


class JsonLog:
    """每脚本一个 JSON 运行日志，finalize 自动附加 peak_memory_gb / wall_time_sec / device。"""

    def __init__(self, script: str, out_dir: str = "logs"):
        self.script = script
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.data = {"script": script, "device": DEVICE}
        self._t0 = time.perf_counter()

    def record(self, key: str, value) -> None:
        if isinstance(value, (torch.Tensor, np.ndarray, np.generic)):
            value = np.asarray(value).tolist()
        self.data[key] = value

    def finalize(self, extra: dict | None = None) -> str:
        if extra:
            for k, v in extra.items():
                self.record(k, v)
        self.data["wall_time_sec"] = round(time.perf_counter() - self._t0, 4)
        self.data["peak_memory_gb"] = round(peak_memory_gb(), 4)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.out_dir, f"{self.script}_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        return path


# ---------------------------------------------------------------- CSR 工具
def make_csr(W, device: str | None = None) -> torch.Tensor:
    """dense/coo/scipy-csr -> CSR float32。行=regulator, 列=target（与 RegVelo skeleton 约定一致）。"""
    import scipy.sparse
    
    device = device or DEVICE
    
    # 已经是 torch sparse_csr
    if isinstance(W, torch.Tensor) and W.layout == torch.sparse_csr:
        return W.to(device=device, dtype=torch.float32)
    
    # 已经是 torch sparse_coo（可能 sparse_dim!=2，先 coalesce 再转）
    if isinstance(W, torch.Tensor) and W.is_sparse:
        return W.coalesce().to_sparse_csr().to(device=device, dtype=torch.float32)
    
    # torch dense
    if isinstance(W, torch.Tensor) and W.layout == torch.strided:
        return W.to_sparse_csr().to(device=device, dtype=torch.float32)
    
    # scipy sparse（.npz 加载后会进入这里）
    if isinstance(W, scipy.sparse.spmatrix):
        if not isinstance(W, scipy.sparse.csr_matrix):
            W = W.tocsr()
        return torch.sparse_csr_tensor(
            torch.from_numpy(W.indptr).to(torch.int64),
            torch.from_numpy(W.indices).to(torch.int64),
            torch.from_numpy(W.data).to(torch.float32),
            size=W.shape,
            device=device,
        )
    
    # numpy dense
    if isinstance(W, np.ndarray):
        W_t = torch.from_numpy(W).to(torch.float32)
        return W_t.to_sparse_csr().to(device=device, dtype=torch.float32)
    
    raise TypeError(f"Unsupported W type: {type(W)}")
def csr_bytes(W_csr: torch.Tensor) -> int:
    """values(FP32) + col_indices(int32/64) + row_pointers 的总字节数。"""
    v = W_csr.values()
    ci = W_csr.col_indices()
    rp = W_csr.crow_indices()
    return v.numel() * v.element_size() + ci.numel() * ci.element_size() + rp.numel() * rp.element_size()


# ---------------------------------------------------------------- 合成数据
def synth_grn(G: int = 256, n_tf: int = 11, edges_per_tf: int | None = None,
              n_feedback: int = 6, seed: int = 0) -> torch.Tensor:
    """星形前馈 GRN（TF -> target），附加 n_feedback 条反馈边（target -> TF）。

    返回 dense 0/1 (G, G)，行=regulator、列=target，供测试转 CSR。
    """
    rng = np.random.default_rng(seed)
    W = np.zeros((G, G), dtype=np.float32)
    n_tf = min(n_tf, G)
    tfs = rng.choice(G, size=n_tf, replace=False)
    targets = np.setdiff1d(np.arange(G), tfs)
    if edges_per_tf is None:
        edges_per_tf = max(1, len(targets) // n_tf)
    for tf in tfs:
        chosen = rng.choice(targets, size=min(edges_per_tf, len(targets)), replace=False)
        W[tf, chosen] = 1.0
    # 反馈边：随机 target -> TF（用于 R-A1 风险压测）
    if n_feedback > 0 and len(targets) > 0:
        fb_src = rng.choice(targets, size=n_feedback, replace=True)
        fb_dst = rng.choice(tfs, size=n_feedback, replace=True)
        W[fb_src, fb_dst] = 1.0
    return torch.from_numpy(W)


def synth_params(G: int = 256, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """beta, gamma ~ U(0.5, 2.5)，bias ~ U(-1, 1)。返回 (beta, gamma, bias)，各 (G,) FP32。"""
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.5, 2.5, size=G).astype(np.float32)
    gamma = rng.uniform(0.5, 2.5, size=G).astype(np.float32)
    bias = rng.uniform(-1.0, 1.0, size=G).astype(np.float32)
    return (torch.from_numpy(beta), torch.from_numpy(gamma), torch.from_numpy(bias))


def _alpha_np(s: np.ndarray, W: np.ndarray, bias: np.ndarray) -> np.ndarray:
    z = s @ W.T + bias  # (N,G)
    sp = np.logaddexp(0.0, z)  # softplus
    return np.clip(sp, 0.0, 50.0)


def simulate_reference(u0: torch.Tensor, s0: torch.Tensor, W_csr: torch.Tensor,
                       bias: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor,
                       t_span: tuple[float, float] = (0.0, 10.0),
                       n_times: int = 41, rtol: float = 1e-9, atol: float = 1e-11) -> dict:
    """DOP853 高精度参考积分（扮演 RegVelo 耦合动力学 ground truth）。

    du/dt = clamp(softplus(W s + b), 0, 50) - beta * u
    ds/dt = beta * u - gamma * s

    u0/s0: (N, G)。返回 dict(t=(n_times,), u=(n_times,N,G), s=(n_times,N,G), solver="DOP853")。
    """
    from scipy.integrate import solve_ivp

    u0n = np.asarray(u0, dtype=np.float64)
    s0n = np.asarray(s0, dtype=np.float64)
    N, G = u0n.shape
    Wn = W_csr.to_dense().cpu().numpy().astype(np.float64)
    bn = bias.cpu().numpy().astype(np.float64)
    betan = beta.cpu().numpy().astype(np.float64)
    gamman = gamma.cpu().numpy().astype(np.float64)

    def rhs(t, y):
        u = y[: N * G].reshape(N, G)
        s = y[N * G:].reshape(N, G)
        a = _alpha_np(s, Wn, bn)
        du = a - betan[None, :] * u
        ds = betan[None, :] * u - gamman[None, :] * s
        return np.concatenate([du.ravel(), ds.ravel()])

    y0 = np.concatenate([u0n.ravel(), s0n.ravel()])
    t_eval = np.linspace(t_span[0], t_span[1], n_times)
    sol = solve_ivp(rhs, t_span, y0, method="DOP853", t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"DOP853 reference failed: {sol.message}")
    u = sol.y[: N * G].reshape(N, G, -1).transpose(2, 0, 1)
    s = sol.y[N * G:].reshape(N, G, -1).transpose(2, 0, 1)
    return {"t": sol.t, "u": u, "s": s, "solver": "DOP853"}


# ---------------------------------------------------------------- 度量
def velocity_cosine(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """逐细胞 cosine(v1, v2)，v: (N, G)。返回 (N,)。"""
    n1 = v1.norm(dim=-1, keepdim=True).clamp_min(eps)
    n2 = v2.norm(dim=-1, keepdim=True).clamp_min(eps)
    return ((v1 / n1) * (v2 / n2)).sum(dim=-1)


def rel_l2(a, b, eps: float = 1e-12) -> float:
    """||a-b|| / ||b||。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), eps))
