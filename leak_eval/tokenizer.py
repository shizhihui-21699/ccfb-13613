"""字节对编码（BPE）分词器，纯 Python 实现、无外部依赖。

字符级模型在数字密集的客服语料上容易退化为逐字符数字自循环；
BPE 把常见数字串/词片段合并为完整 token，从根上消除该退化，
同时 Exposure 仍可在 token 序列上计算（Carlini 原文即 token 级）。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
_SPECIALS = [PAD, BOS, EOS, UNK]


class BPETokenizer:
    def __init__(self, vocab: list[str], merges: list[tuple[str, str]]):
        # vocab: 基础字节字符 + 特殊 token + 合并 token
        self.vocab = list(vocab)
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.merges = list(merges)
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    # ------------------------------------------------------------------ #
    # 训练
    # ------------------------------------------------------------------ #
    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 1400) -> "BPETokenizer":
        # 基础词表：语料中出现过的单字符（按字节 utf-8 编码，中文按字符保留）
        base_chars = sorted({ch for t in texts for ch in t})
        vocab = list(_SPECIALS) + base_chars
        # 每个文档表示为字符 tuple；用 Counter 统计相邻对
        word_counter = Counter(texts)
        splits: dict[str, list[str]] = {w: list(w) for w in word_counter}
        pair_counts: Counter = Counter()
        for w, f in word_counter.items():
            toks = splits[w]
            for a, b in zip(toks, toks[1:]):
                pair_counts[(a, b)] += f

        merges: list[tuple[str, str]] = []
        while len(vocab) < vocab_size and pair_counts:
            pair, _ = pair_counts.most_common(1)[0]
            if pair_counts[pair] <= 1:
                break
            merged = pair[0] + pair[1]
            merges.append(pair)
            vocab.append(merged)
            cls._apply_merge(splits, word_counter, pair_counts, pair, merged)
        return cls(vocab, merges)

    @staticmethod
    def _apply_merge(splits, word_counter, pair_counts, pair, merged):
        a, b = pair
        pattern = a + b
        for w in list(word_counter.keys()):
            if pattern not in w:
                continue
            old = splits[w]
            new: list[str] = []
            i = 0
            while i < len(old):
                if i < len(old) - 1 and old[i] == a and old[i + 1] == b:
                    new.append(merged)
                    i += 2
                else:
                    new.append(old[i])
                    i += 1
            splits[w] = new
        # 重新统计该词影响到的 pair（语料小，全量重算简单可靠）
        pair_counts.clear()
        for w, f in word_counter.items():
            toks = splits[w]
            for x, y in zip(toks, toks[1:]):
                pair_counts[(x, y)] += f

    # ------------------------------------------------------------------ #
    # 编解码
    # ------------------------------------------------------------------ #
    def _encode_word(self, word: str) -> list[str]:
        toks = list(word)
        while len(toks) >= 2:
            best_rank, best_idx = None, -1
            for i, pair in enumerate(zip(toks, toks[1:])):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_idx = rank, i
            if best_idx < 0:
                break
            toks[best_idx:best_idx + 2] = [toks[best_idx] + toks[best_idx + 1]]
        return toks

    def encode(self, text: str) -> list[int]:
        ids = []
        # 以空白/换行切词，保留分隔符作为独立 token
        for token in self._tokenize(text):
            if token in self.stoi:
                ids.append(self.stoi[token])
                continue
            for piece in self._encode_word(token):
                ids.append(self.stoi.get(piece, self.unk_id))
        return ids

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        out, buf = [], []
        for ch in text:
            if ch.isspace():
                if buf:
                    out.append("".join(buf)); buf = []
                out.append(ch)
            else:
                buf.append(ch)
        if buf:
            out.append("".join(buf))
        return out

    def decode(self, ids: list[int]) -> str:
        # UNK 没有可还原的字面量，解码时跳过，避免污染输出文本
        return "".join(self.itos_get(i) for i in ids
                       if i not in (self.pad_id, self.unk_id))

    def itos_get(self, i: int) -> str:
        return self.vocab[i] if 0 <= i < len(self.vocab) else UNK

    def encode_with_special(self, text: str, add_eos: bool = True) -> list[int]:
        ids = [self.bos_id] + self.encode(text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"vocab": self.vocab, "merges": self.merges}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["vocab"], [tuple(m) for m in d["merges"]])
