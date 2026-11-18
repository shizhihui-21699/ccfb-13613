"""训练管线：同一模型骨架在六个对照条件下分别训练。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import TokenBlockDataset
from .dp import dpsgd_batch_step, epsilon_after, solve_noise_multiplier
from .model import GPTConfig, MiniGPT, cosine_lr
from .tokenizer import BPETokenizer


def _condition_settings(condition: str, cfg: ExperimentConfig) -> dict:
    t = cfg.train
    settings = {
        "max_steps": t.max_steps,
        "weight_decay": t.weight_decay,
        "dropout": cfg.model.dropout,
        "use_dp": False,
    }
    if condition == "early_stop":
        settings["max_steps"] = t.early_stop_steps
    elif condition == "regularized":
        settings["weight_decay"] = t.reg_weight_decay
        settings["dropout"] = t.reg_dropout
    elif condition == "dp_sgd":
        settings["use_dp"] = True
    return settings


def train_condition(condition: str, cfg: ExperimentConfig,
                    tokenizer: BPETokenizer, corpus_path: Path,
                    out_dir: Path) -> dict:
    """训练单个条件，保存权重并返回训练曲线/超参等元信息。"""
    torch.manual_seed(cfg.data.seed)
    gen = torch.Generator().manual_seed(cfg.data.seed)
    device = torch.device(cfg.device)

    settings = _condition_settings(condition, cfg)
    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.model.block_size,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        dropout=settings["dropout"],
    )
    model = MiniGPT(model_cfg).to(device)
    optimizer = model.configure_optimizers(cfg.train.lr, settings["weight_decay"])

    dataset = TokenBlockDataset(corpus_path, tokenizer, cfg.model.block_size, cfg.device)
    n_blocks = dataset.n_blocks
    sampling_rate = min(1.0, cfg.train.batch_size / max(1, n_blocks))

    # DP 噪声倍率：若配置值不满足目标 ε，按目标反解一个更保守的 σ
    dp_info = {}
    noise_mult = cfg.train.dp_noise_multiplier
    if settings["use_dp"]:
        eps_at_configured = epsilon_after(
            settings["max_steps"], sampling_rate, noise_mult, cfg.train.dp_delta,
        )
        if eps_at_configured > cfg.train.dp_target_epsilon:
            noise_mult = solve_noise_multiplier(
                settings["max_steps"], sampling_rate,
                cfg.train.dp_target_epsilon, cfg.train.dp_delta,
            )
        dp_info = {
            "sampling_rate": sampling_rate,
            "noise_multiplier": noise_mult,
            "max_grad_norm": cfg.train.dp_max_grad_norm,
            "target_epsilon": cfg.train.dp_target_epsilon,
            "delta": cfg.train.dp_delta,
        }

    history = []
    clip_rates = []
    t0 = time.time()
    # CPU 上 SDPA 无 vmap 批处理规则，逐样本循环反而更稳更快；MPS/CUDA 可尝试 vmap
    use_vmap = cfg.device in ("mps", "cuda")
    for step in range(settings["max_steps"]):
        lr = cosine_lr(step, settings["max_steps"], cfg.train.warmup_steps, cfg.train.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = dataset.get_batch(cfg.train.batch_size, gen)

        if settings["use_dp"]:
            # 整批一次噪声化：微批只做逐样本裁剪与梯度累积，保证会计口径与外层步数一致
            diag = dpsgd_batch_step(
                model, optimizer, x, y,
                cfg.train.dp_max_grad_norm, noise_mult,
                microbatch_size=cfg.train.dp_microbatch_size,
                generator=gen, use_vmap=use_vmap,
            )
            clip_rates.append(diag["clipped_fraction"])
            with torch.no_grad():
                logits, _ = model(x, y)
                loss_val = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        else:
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            loss_val = loss.item()

        if step % cfg.train.eval_interval == 0 or step == settings["max_steps"] - 1:
            history.append({"step": step, "train_loss": loss_val, "lr": lr})

    elapsed = time.time() - t0
    if settings["use_dp"]:
        dp_info["epsilon_spent"] = epsilon_after(
            settings["max_steps"], sampling_rate, noise_mult, cfg.train.dp_delta,
        )
        dp_info["mean_clipped_fraction"] = sum(clip_rates) / max(1, len(clip_rates))

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    meta = {
        "condition": condition,
        "settings": settings,
        "n_blocks": n_blocks,
        "elapsed_sec": elapsed,
        "n_parameters": model.num_parameters(),
        "history": history,
        "dp": dp_info,
    }
    (out_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return meta


def load_model(ckpt_dir: Path, cfg: ExperimentConfig,
               tokenizer: BPETokenizer, dropout_override: float | None = None) -> MiniGPT:
    meta = json.loads((ckpt_dir / "train_meta.json").read_text(encoding="utf-8"))
    dropout = dropout_override if dropout_override is not None else meta["settings"]["dropout"]
    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.model.block_size,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        dropout=dropout,
    )
    model = MiniGPT(model_cfg)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu"))
    model.to(torch.device(cfg.device))
    model.eval()
    return model
