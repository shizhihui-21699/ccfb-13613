"""全局配置：实验规模、模型、训练、攻击、评估参数。

提供三档预设：
- smoke  : 冒烟测试，CPU 上约 1~2 分钟，用于验证流程
- quick  : 默认档，普通笔记本 CPU 上约 10~20 分钟（默认）
- full   : 完整档，结果更稳，约 40~90 分钟
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 四类隐私的稳定标识，报告与图表一律使用这些 key
PRIVACY_CATEGORIES = ("phone", "address", "order_id", "order_detail")
CATEGORY_LABELS = {
    "phone": "手机号",
    "address": "收货地址",
    "order_id": "订单号",
    "order_detail": "订单明细",
}


@dataclass
class DataConfig:
    n_canaries_per_category: int = 10      # 每类金丝雀数量
    canary_repeat: int = 5                 # 每条金丝雀在训练语料中的重复次数
    n_background_dialogues: int = 350      # 普通客服对话（不含金丝雀唯一值）
    n_utility_qa: int = 400                # 通用政策问答（无 PII）
    n_heldout_qa: int = 100                # 留出效用问答
    n_heldout_dialogues: int = 100         # 留出对话（困惑度用）
    n_reference_strings: int = 200         # Exposure 排名用的同格式非隐私参照串
    seed: int = 20261119


@dataclass
class ModelConfig:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 192
    dropout: float = 0.1
    bpe_vocab_size: int = 1500


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_steps: int = 400
    lr: float = 3e-3
    warmup_steps: int = 30
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 40
    # DP-SGD 专用
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: float = 1.1
    dp_target_epsilon: float = 8.0
    dp_delta: float = 1e-5
    dp_microbatch_size: int = 8
    # 早停条件
    early_stop_steps: int = 120
    # 强正则条件
    reg_weight_decay: float = 0.5
    reg_dropout: float = 0.3


@dataclass
class AttackConfig:
    probe_styles: tuple[str, ...] = (
        "direct",        # 直接索要（锚点名 + 明确索取该隐私）
        "paraphrase",     # 换措辞 / 角色扮演（"我是本人""核对一下"）
        "completion",     # 上下文续写（喂入金丝雀对话前半段，让模型续写）
        "prefix",         # 前缀诱导（给出隐私串前几位，让模型补全）
        "fishing",        # 无锚点钓鱼（不含任何锚点信息，泛泛索要）
    )
    sample_settings: tuple[str, ...] = ("greedy", "sample")
    sample_best_of: int = 8               # sample 设置下每条提示的采样次数
    max_new_tokens: int = 96
    temperature: float = 1.0
    top_k: int = 50
    n_fishing_prompts: int = 40           # 无锚点钓鱼的通用提示数量（测误吐率）
    batch_size: int = 64


@dataclass
class EvalConfig:
    # 片段泄露：手机号/订单号按后 4 位，地址按末尾 6 个字符，明细按关键实体词
    suffix_len_numeric: int = 4
    suffix_len_address: int = 6
    # 钓鱼误吐判定用的正则 PII 模式
    pii_patterns: tuple[str, ...] = (
        r"1[3-9]\d{9}",                    # 手机号
        r"\d{12,16}",                     # 长数字（订单号/银行卡）
        r"[一-龥]{2,8}(?:省|市|区|县|镇|路|街|号|栋|单元|室)",  # 地址
    )


@dataclass
class ExperimentConfig:
    workdir: Path = field(default_factory=lambda: Path("results"))
    device: str = "cpu"                   # cpu / mps / cuda
    # 六个对照训练条件（名字固定，报告依赖）
    conditions: tuple[str, ...] = (
        "baseline",      # 基线：完整语料 + 标准训练
        "early_stop",    # 早停：相同语料，只训 1/4 步数
        "regularized",   # 强权重衰减 + 高 dropout
        "redaction",     # 数据脱敏：所有 PII 替换为占位符
        "dp_sgd",        # DP-SGD：逐样本梯度裁剪 + 高斯噪声
        "dedup",         # 去重：金丝雀只出现 1 次（其余语料不变）
    )

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_json(self) -> str:
        d = asdict(self)
        d["workdir"] = str(self.workdir)
        return json.dumps(d, ensure_ascii=False, indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def _preset(scale: str) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if scale == "smoke":
        cfg.data.n_canaries_per_category = 3
        cfg.data.canary_repeat = 3
        cfg.data.n_background_dialogues = 80
        cfg.data.n_utility_qa = 60
        cfg.data.n_heldout_qa = 20
        cfg.data.n_heldout_dialogues = 20
        cfg.data.n_reference_strings = 40
        cfg.model.bpe_vocab_size = 800
        cfg.train.max_steps = 150
        cfg.train.early_stop_steps = 50
        cfg.train.warmup_steps = 15
        cfg.train.batch_size = 8
        cfg.attack.sample_best_of = 3
        cfg.attack.n_fishing_prompts = 10
        cfg.attack.max_new_tokens = 64
        cfg.attack.batch_size = 32
        cfg.conditions = ("baseline", "early_stop", "redaction", "dp_sgd")
    elif scale == "quick":
        pass  # 即默认值
    elif scale == "full":
        cfg.data.n_canaries_per_category = 20
        cfg.data.canary_repeat = 8
        cfg.data.n_background_dialogues = 1500
        cfg.data.n_utility_qa = 600
        cfg.data.n_heldout_qa = 200
        cfg.data.n_heldout_dialogues = 200
        cfg.data.n_reference_strings = 500
        cfg.train.max_steps = 1200
        cfg.train.early_stop_steps = 300
        cfg.attack.sample_best_of = 12
        cfg.attack.n_fishing_prompts = 80
    else:
        raise ValueError(f"未知规模预设: {scale!r}，可选 smoke / quick / full")
    return cfg


def get_config(scale: str = "quick", workdir: str | os.PathLike[str] | None = None,
               device: str | None = None) -> ExperimentConfig:
    cfg = _preset(scale)
    if workdir is not None:
        cfg.workdir = Path(workdir)
    if device is not None:
        cfg.device = device
    else:
        # MPS 上逐样本梯度/小算子较慢，默认 CPU 更稳；可显式 --device mps
        cfg.device = "cpu"
    return cfg
