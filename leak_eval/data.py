"""训练数据装配：语料切分成定长 token 块，按 batch 采样。"""

from __future__ import annotations

from pathlib import Path

import torch

from .tokenizer import BPETokenizer


class TokenBlockDataset:
    """把语料按文档（空行分隔）编码、文档末尾插入 EOS，再切成定长训练块。

    EOS 让模型学会"回答结束就停"，避免生成退化为无限重复；
    切块跨文档边界是可接受的（边界处带 EOS 标记）。
    """

    def __init__(self, corpus_path: str | Path, tokenizer: BPETokenizer,
                 block_size: int, device: str = "cpu"):
        text = Path(corpus_path).read_text(encoding="utf-8")
        ids: list[int] = []
        for doc in text.split("\n\n"):
            doc = doc.strip()
            if doc:
                ids.extend(tokenizer.encode_with_special(doc, add_eos=True))
        self.device = device
        self.block_size = block_size
        # 丢掉最后不足一个块的尾巴
        n_blocks = max(1, (len(ids) - 1) // block_size)
        total = n_blocks * block_size
        data = torch.tensor(ids[: total + 1], dtype=torch.long)
        self.x = data[:-1].view(n_blocks, block_size)
        self.y = data[1:].view(n_blocks, block_size)
        self.n_blocks = n_blocks

    def get_batch(self, batch_size: int, generator: torch.Generator | None = None):
        idx = torch.randint(0, self.n_blocks, (batch_size,), generator=generator)
        return self.x[idx].to(self.device), self.y[idx].to(self.device)


@torch.no_grad()
def document_perplexity(model, tokenizer: BPETokenizer, documents: list[str],
                        device: str, max_chars: int = 600) -> float:
    """对一组文档计算平均字符级困惑度（指数化平均负对数似然）。"""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for doc in documents:
        doc = doc[:max_chars]
        ids = torch.tensor([tokenizer.encode_with_special(doc)], dtype=torch.long, device=device)
        if ids.size(1) < 2:
            continue
        x, y = ids[:, :-1], ids[:, 1:]
        logits, _ = model(x)
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum",
        )
        total_nll += nll.item()
        total_tokens += y.numel()
    if total_tokens == 0:
        return float("nan")
    import math
    return math.exp(total_nll / total_tokens)
