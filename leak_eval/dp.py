"""差分隐私训练（DP-SGD）与 RDP 隐私会计。

DP-SGD 流程（Dwork 2014 / Abadi 2016）：
  1. 对 batch 内每条样本单独求梯度 g_i
  2. 按 L2 范数裁剪：g_i <- g_i * min(1, C / ||g_i||)
  3. 求和后加高斯噪声 N(0, (σC)^2)，再除以 batch 大小
  4. 用 RDP 会计记录累计隐私消耗 (ε, δ)

逐样本梯度优先用 torch.func 的函数式变换向量化实现，
不兼容时回退到逐样本前向/反向循环（模型很小，成本可接受）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap

# RDP 阶数（Mironov 2017 常用取值，整数 + 少量分数阶）
_DEFAULT_ORDERS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256]


# ---------------------------------------------------------------------------
# RDP 隐私会计（Poisson 下采样高斯机制的标准上界近似）
# ---------------------------------------------------------------------------

def compute_rdp(steps: int, sampling_rate: float, noise_multiplier: float,
                orders: list[float] | None = None) -> list[float]:
    """返回每个阶数 α 对应的累计 RDP。

    采用 Mironov (2017) 常用的子采样近似：RDP(α) ≈ steps · q² · α / (2σ²)，
    q 较小时是紧上界近似，适合本工具量级的横向对比（非生产级保证）。
    """
    if orders is None:
        orders = _DEFAULT_ORDERS
    if noise_multiplier <= 0:
        return [float("inf")] * len(orders)
    per_step = sampling_rate ** 2 / (2.0 * noise_multiplier ** 2)
    return [steps * per_step * alpha for alpha in orders]


def rdp_to_dp(rdp: list[float], orders: list[float], delta: float) -> float:
    """RDP -> (ε, δ) 转换（Mironov 2017 命题 3 的标准界）：
    ε(α) = RDP(α) + log(1/δ) / (α − 1)，对所有阶数取最小。
    """
    eps = float("inf")
    log_delta = math.log(1.0 / delta)
    for r, alpha in zip(rdp, orders):
        if not math.isfinite(r) or alpha <= 1:
            continue
        bound = r + log_delta / (alpha - 1)
        eps = min(eps, bound)
    return max(0.0, eps)


def epsilon_after(steps: int, sampling_rate: float, noise_multiplier: float,
                  delta: float) -> float:
    orders = _DEFAULT_ORDERS
    return rdp_to_dp(compute_rdp(steps, sampling_rate, noise_multiplier, orders), orders, delta)


def solve_noise_multiplier(steps: int, sampling_rate: float, target_epsilon: float,
                           delta: float) -> float:
    """二分搜索满足目标 ε 的最小噪声倍率 σ。"""
    lo, hi = 0.05, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        eps = epsilon_after(steps, sampling_rate, mid, delta)
        if eps > target_epsilon:
            lo = mid          # 噪声不够，加大
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------------------
# 逐样本梯度
# ---------------------------------------------------------------------------

def _per_example_grads_vmap(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    """函数式 vmap 路径：一次调用得到 batch 内每条样本的梯度字典。

    randomness='different' 让每条样本在 dropout 上独立采样（与逐样本反向语义一致）。
    """
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def compute_loss(params, x_one, y_one):
        logits, _ = functional_call(model, (params, buffers), (x_one.unsqueeze(0),))
        return nn.functional.cross_entropy(
            logits.squeeze(0), y_one, ignore_index=-100, reduction="mean",
        )

    per_sample_grad = vmap(
        grad(compute_loss), in_dims=(None, 0, 0), randomness="different",
    )(params, x, y)
    return per_sample_grad


def _per_example_grads_loop(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    """回退路径：逐样本前向/反向，按参数名堆叠。"""
    grads: dict[str, list[torch.Tensor]] = {n: [] for n, _ in model.named_parameters()}
    for i in range(x.size(0)):
        model.zero_grad(set_to_none=True)
        logits, loss = model(x[i:i + 1], y[i:i + 1])
        loss.backward()
        for n, p in model.named_parameters():
            grads[n].append(p.grad.detach().clone())
    return {n: torch.stack(g, dim=0) for n, g in grads.items()}


def dpsgd_step(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor,
               y: torch.Tensor, max_grad_norm: float, noise_multiplier: float,
               generator: torch.Generator | None = None, use_vmap: bool = True) -> dict:
    """执行一步 DP-SGD，返回本步诊断信息（裁剪率、实际噪声系数）。"""
    device = x.device
    batch_size = x.size(0)

    if use_vmap:
        try:
            per_grad = _per_example_grads_vmap(model, x, y)
        except Exception:
            per_grad = _per_example_grads_loop(model, x, y)
    else:
        per_grad = _per_example_grads_loop(model, x, y)

    # 逐样本全局范数裁剪
    flat_norms = torch.zeros(batch_size, device=device)
    for g in per_grad.values():
        flat_norms += g.reshape(batch_size, -1).pow(2).sum(dim=-1)
    norms = flat_norms.sqrt()
    clip_factor = (max_grad_norm / (norms + 1e-6)).clamp(max=1.0)
    clipped_fraction = (norms > max_grad_norm).float().mean().item()

    sums: dict[str, torch.Tensor] = {}
    for n, g in per_grad.items():
        g_clipped = g * clip_factor.view((batch_size,) + (1,) * (g.ndim - 1))
        summed = g_clipped.sum(dim=0)
        noise = torch.normal(
            mean=0.0, std=noise_multiplier * max_grad_norm,
            size=summed.shape, device=device, generator=generator,
        )
        sums[n] = (summed + noise) / batch_size

    optimizer.zero_grad(set_to_none=True)
    for n, p in model.named_parameters():
        p.grad = sums[n]
    optimizer.step()
    return {
        "mean_grad_norm": norms.mean().item(),
        "clipped_fraction": clipped_fraction,
    }


def dpsgd_batch_step(model: nn.Module, optimizer: torch.optim.Optimizer,
                     x: torch.Tensor, y: torch.Tensor, max_grad_norm: float,
                     noise_multiplier: float, microbatch_size: int,
                     generator: torch.Generator | None = None,
                     use_vmap: bool = True) -> dict:
    """标准 DP-SGD 整批步：微批内逐样本裁剪并累积，整批只加一次噪声、step 一次。

    这样每个外层 batch 对应一次噪声化梯度释放，RDP 会计按外层步数计数才成立；
    若对每个微批分别加噪并 step，实际噪声释放次数会被低估、ε 被高估（过度乐观）。
    """
    device = x.device
    total = x.size(0)
    accum: dict[str, torch.Tensor] | None = None
    clip_rates, norm_sums = [], 0.0
    n_seen = 0

    for mb in range(0, total, microbatch_size):
        xb, yb = x[mb:mb + microbatch_size], y[mb:mb + microbatch_size]
        bs = xb.size(0)
        if use_vmap:
            try:
                per_grad = _per_example_grads_vmap(model, xb, yb)
            except Exception:
                per_grad = _per_example_grads_loop(model, xb, yb)
        else:
            per_grad = _per_example_grads_loop(model, xb, yb)

        flat_norms = torch.zeros(bs, device=device)
        for g in per_grad.values():
            flat_norms += g.reshape(bs, -1).pow(2).sum(dim=-1)
        norms = flat_norms.sqrt()
        clip_factor = (max_grad_norm / (norms + 1e-6)).clamp(max=1.0)
        clip_rates.append((norms > max_grad_norm).float().mean().item())
        norm_sums += norms.sum().item()
        n_seen += bs

        for n, g in per_grad.items():
            g_clipped = (g * clip_factor.view((bs,) + (1,) * (g.ndim - 1))).sum(dim=0)
            if accum is None:
                accum = {}
            accum[n] = g_clipped if n not in accum else accum[n] + g_clipped

    # 整批一次高斯噪声，平均化后作为本步梯度
    optimizer.zero_grad(set_to_none=True)
    for n, p in model.named_parameters():
        summed = accum[n]
        noise = torch.normal(
            mean=0.0, std=noise_multiplier * max_grad_norm,
            size=summed.shape, device=device, generator=generator,
        )
        p.grad = (summed + noise) / total
    optimizer.step()
    return {
        "mean_grad_norm": norm_sums / max(1, n_seen),
        "clipped_fraction": sum(clip_rates) / max(1, len(clip_rates)),
    }
