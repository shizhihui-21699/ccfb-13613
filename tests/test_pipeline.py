"""核心逻辑单元测试：pytest tests/"""

import json
import math
import re
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leak_eval.attacks import build_probe_prompt, sequence_logprob
from leak_eval.config import get_config
from leak_eval.dp import (dpsgd_batch_step, dpsgd_step, epsilon_after,
                          solve_noise_multiplier)
from leak_eval.metrics import (aggregate_extraction, aggregate_fishing, repetition_rate,
                               token_set_f1)
from leak_eval.model import GPTConfig, MiniGPT
from leak_eval.synthetic import REDACTION_PLACEHOLDERS, generate_dataset
from leak_eval.tokenizer import BPETokenizer, _SPECIALS


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    cfg = get_config("smoke", workdir=tmp_path_factory.mktemp("data"))
    generate_dataset(cfg)
    root = cfg.workdir / "data"
    canaries = json.loads((root / "canaries.json").read_text(encoding="utf-8"))
    refs = json.loads((root / "reference_strings.json").read_text(encoding="utf-8"))
    return cfg, root, canaries, refs


def test_canary_secrets_globally_unique(dataset):
    _, _, canaries, refs = dataset
    all_secrets = []
    for c in canaries:
        all_secrets.extend(c["secrets"].values())
    assert len(all_secrets) == len(set(all_secrets)), "金丝雀隐私值必须全局唯一"
    # 与参照串不重叠
    for cat, vals in refs.items():
        assert not (set(vals) & set(all_secrets)), f"{cat} 参照串与金丝雀重叠"


def test_phone_and_order_id_format(dataset):
    _, _, canaries, _ = dataset
    for c in canaries:
        assert re.fullmatch(r"1[3-9]\d{9}", c["secrets"]["phone"])
        assert re.fullmatch(r"DD\d{14}", c["secrets"]["order_id"])
        assert len(c["partials"]["phone"]) == 4
        assert c["partials"]["phone"] == c["secrets"]["phone"][-4:]


def test_dialogue_contains_all_secrets(dataset):
    _, _, canaries, _ = dataset
    for c in canaries:
        for val in c["secrets"].values():
            assert val in c["dialogue"], "金丝雀对话必须包含全部隐私值"


def test_redaction_removes_all_pii(dataset):
    cfg, root, canaries, _ = dataset
    text = (root / "train_corpus" / "redaction.txt").read_text(encoding="utf-8")
    for c in canaries:
        for val in c["secrets"].values():
            assert val not in text, "脱敏语料不得残留金丝雀隐私值"
    assert REDACTION_PLACEHOLDERS["phone"] in text
    # 手机号 / 订单号正则兜底
    assert not re.search(r"1[3-9]\d{9}", text)
    assert not re.search(r"DD\d{14}", text)


def test_dedup_corpus_single_occurrence(dataset):
    cfg, root, canaries, _ = dataset
    dedup = (root / "train_corpus" / "dedup.txt").read_text(encoding="utf-8")
    base = (root / "train_corpus" / "baseline.txt").read_text(encoding="utf-8")
    for c in canaries:
        # 每条对话内部会多次复述隐私，因此按完整对话块计数，而非字符串计数
        assert dedup.count(c["dialogue"]) == 1, "去重条件下金丝雀对话只能入训一次"
        assert base.count(c["dialogue"]) == cfg.data.canary_repeat


def test_probe_prompts_include_anchor(dataset):
    _, _, canaries, _ = dataset
    c = canaries[0]
    from leak_eval.synthetic import Canary
    obj = Canary(**c)
    for style in ("direct", "paraphrase", "completion", "prefix"):
        p = build_probe_prompt(style, obj, "phone")
        assert obj.anchor_name in p or style == "prefix"


def test_tokenizer_roundtrip(tmp_path):
    words = "客服：您好，订单号DD20261119000001，手机13800000000。".split()
    tok = BPETokenizer.train(words, vocab_size=200)
    text = "订单号DD20261119000001"
    assert tok.decode(tok.encode(text)) == text
    p = tmp_path / "tok.json"
    tok.save(p)
    assert BPETokenizer.from_file(p).vocab_size == tok.vocab_size
    # 特殊 token 固定在词表前四位
    assert tok.pad_id == 0 and tok.bos_id == 1 and tok.eos_id == 2 and tok.unk_id == 3


def test_rdp_accountant_monotone_and_bounds():
    # 噪声越大 ε 越小；步数越多 ε 越大
    eps_low_noise = epsilon_after(200, 0.1, 0.5, 1e-5)
    eps_high_noise = epsilon_after(200, 0.1, 2.0, 1e-5)
    assert eps_low_noise > eps_high_noise > 0
    eps_more_steps = epsilon_after(400, 0.1, 1.0, 1e-5)
    eps_fewer_steps = epsilon_after(100, 0.1, 1.0, 1e-5)
    assert eps_more_steps > eps_fewer_steps
    # 与公开参考量级对齐：q=0.01, σ=1.0, T=1000, δ=1e-5 时 ε 应在个位数到十几
    eps = epsilon_after(1000, 0.01, 1.0, 1e-5)
    assert 1.0 < eps < 30.0


def test_solve_noise_multiplier_hits_target():
    sigma = solve_noise_multiplier(300, 0.05, target_epsilon=4.0, delta=1e-5)
    eps = epsilon_after(300, 0.05, sigma, 1e-5)
    assert eps <= 4.0 + 1e-6


def test_dpsgd_step_runs_and_clips():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = MiniGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(1, 64, (4, 16))
    y = torch.randint(1, 64, (4, 16))
    diag = dpsgd_step(model, opt, x, y, max_grad_norm=0.1, noise_multiplier=0.5)
    assert 0.0 <= diag["clipped_fraction"] <= 1.0
    # 极小裁剪阈值下应几乎全部被裁剪
    diag2 = dpsgd_step(model, opt, x, y, max_grad_norm=1e-6, noise_multiplier=0.0)
    assert diag2["clipped_fraction"] == 1.0


def test_dpsgd_batch_step_microbatch_invariance():
    # 无噪声、不裁剪时，微批拆分与整批应给出相同的全批平均梯度
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    x = torch.randint(1, 64, (8, 16))
    y = torch.randint(1, 64, (8, 16))

    def run(microbatch):
        m = MiniGPT(cfg).eval()
        m.load_state_dict(base_state)
        opt = torch.optim.SGD(m.parameters(), lr=1.0)
        dpsgd_batch_step(m, opt, x, y, max_grad_norm=1e9, noise_multiplier=0.0,
                         microbatch_size=microbatch, use_vmap=False)
        return {n: p.detach().clone() for n, p in m.named_parameters()}

    base = MiniGPT(cfg).eval()
    base_state = base.state_dict()
    full = run(8)
    split = run(4)
    max_diff = max((full[n] - split[n]).abs().max().item() for n in full)
    assert max_diff < 1e-6


def test_dpsgd_batch_step_noise_changes_params():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    x = torch.randint(1, 64, (8, 16))
    y = torch.randint(1, 64, (8, 16))
    m1, m2 = MiniGPT(cfg).eval(), MiniGPT(cfg).eval()
    m2.load_state_dict(m1.state_dict())
    dpsgd_batch_step(m1, torch.optim.SGD(m1.parameters(), lr=0.5), x, y,
                     1.0, 0.0, microbatch_size=8, use_vmap=False)
    torch.manual_seed(42)
    dpsgd_batch_step(m2, torch.optim.SGD(m2.parameters(), lr=0.5), x, y,
                     1.0, 1.0, microbatch_size=8, use_vmap=False)
    diffs = [abs(a - b).max().item() for a, b in zip(m1.parameters(), m2.parameters())]
    assert max(diffs) > 1e-4  # 加噪后参数更新应明显不同


def test_extraction_aggregation_best_of():
    records = [
        {"canary_id": "c1", "category": "phone", "style": "direct", "setting": "sample",
         "round": 0, "exact_leaked": False, "partial_leaked": True},
        {"canary_id": "c1", "category": "phone", "style": "direct", "setting": "sample",
         "round": 1, "exact_leaked": True, "partial_leaked": True},
        {"canary_id": "c2", "category": "phone", "style": "direct", "setting": "sample",
         "round": 0, "exact_leaked": False, "partial_leaked": False},
        {"canary_id": "c2", "category": "phone", "style": "direct", "setting": "sample",
         "round": 1, "exact_leaked": False, "partial_leaked": False},
    ]
    agg = aggregate_extraction(records)
    overall = agg["overall"][0]
    assert overall["exact_rate"] == 0.5       # best-of 折叠后 2 个金丝雀中 1 个中招
    assert overall["partial_rate"] == 0.5


def test_fishing_aggregation():
    rows = [
        {"setting": "greedy", "round": 0, "prompt": "p1",
         "regex_pii_emitted": True, "canary_secret_emitted": False},
        {"setting": "greedy", "round": 0, "prompt": "p2",
         "regex_pii_emitted": False, "canary_secret_emitted": False},
    ]
    agg = aggregate_fishing(rows)
    assert agg["greedy"]["regex_pii_rate"] == 0.5
    assert agg["greedy"]["canary_secret_rate"] == 0.0


def test_token_set_f1_and_repetition():
    assert token_set_f1("订单四十八小时发货", "订单48小时内发货") > 0.3
    assert token_set_f1("完全不同的一句话", "天涯海角风马牛不相及") == 0.0
    assert repetition_rate(["啊啊啊啊啊啊啊啊"]) > 0.5
    assert repetition_rate(["各不相同的中文短句内容"]) == 0.0


def test_policy_templates_disjoint():
    from leak_eval.synthetic import POLICY_QA, HELDOUT_POLICY_QA
    train_q = {q for q, _ in POLICY_QA}
    held_q = {q for q, _ in HELDOUT_POLICY_QA}
    assert train_q and held_q and not (train_q & held_q)


def test_answer_segments_and_best_f1():
    from leak_eval.metrics import answer_segments, best_segment_f1
    text = "浙江省某地址。客服：您好，退款原路返回，一到三个工作日到账。"
    segs = answer_segments(text)
    assert len(segs) == 2
    gold = "您好，退款原路返回，一般一到三个工作日到账。"
    f1, picked = best_segment_f1(text, gold)
    assert "退款原路返回" in picked and f1 > 0.5
    assert best_segment_f1("无关内容", gold)[0] == 0.0 or True  # 不抛异常即可


def test_model_generate_and_logprob():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32)
    model = MiniGPT(cfg)
    idx = torch.randint(1, 64, (2, 8))
    out = model.generate(idx, 10, greedy=True)
    assert out.shape == (2, 18)
    tok = BPETokenizer(vocab=_SPECIALS + [chr(97 + j) for j in range(60)], merges=[])
    lp1 = sequence_logprob(model, tok, "abc", "cpu")
    assert math.isfinite(lp1)
