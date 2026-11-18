"""迷你 GPT（字符级），规模约 1M 参数，CPU 可训。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 384
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # is_causal 走 flash/mem-efficient 后端，无需显式 mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.attn_drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # 权重绑定，省参数
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"序列长度 {T} 超过 block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, *, greedy: bool = True,
                 temperature: float = 1.0, top_k: int | None = None,
                 eos_id: int | None = None) -> torch.Tensor:
        self.eval()
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            if greedy:
                nxt = probs.argmax(dim=-1, keepdim=True)
            else:
                nxt = torch.multinomial(probs, num_samples=1)
            if eos_id is not None:
                # 已结束的序列继续填充 EOS，便于保持 batch 形状
                nxt = torch.where(finished.unsqueeze(1),
                                  torch.full_like(nxt, eos_id), nxt)
                finished |= (nxt.squeeze(1) == eos_id)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and bool(finished.all()):
                break
        return idx

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, lr: float, weight_decay: float):
        decay, nodecay = set(), set()
        param_names = {n for n, _ in self.named_parameters()}
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                name = f"{mn}.{pn}" if mn else pn
                if name not in param_names:
                    continue  # 权重绑定的别名参数跳过
                if pn.endswith("bias") or isinstance(m, nn.LayerNorm) or pn == "pos_emb.weight":
                    nodecay.add(name)
                else:
                    decay.add(name)
        params = dict(self.named_parameters())
        return torch.optim.AdamW(
            [
                {"params": [params[n] for n in sorted(decay)], "weight_decay": weight_decay},
                {"params": [params[n] for n in sorted(nodecay)], "weight_decay": 0.0},
            ],
            lr=lr, betas=(0.9, 0.95),
        )


def cosine_lr(step: int, max_steps: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
