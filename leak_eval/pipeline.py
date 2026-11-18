"""端到端编排：数据生成 → 分词器 → 逐条件训练 → 攻击与效用评估 → 报告。"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from .attacks import exposure_scores, run_extraction_attacks, run_fishing
from .config import ExperimentConfig
from .metrics import (aggregate_exposure, aggregate_extraction, aggregate_fishing,
                      build_condition_summary, evaluate_utility)
from .reporting import save_all
from .synthetic import Canary, generate_dataset
from .tokenizer import BPETokenizer
from .train import load_model, train_condition


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_pipeline(cfg: ExperimentConfig, conditions: list[str] | None = None,
                 skip_data: bool = False, skip_train: bool = False,
                 log=print) -> list[dict]:
    workdir = cfg.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    cfg.save(workdir / "config.json")
    t_start = time.time()

    # ---- 1. 数据 -----------------------------------------------------------
    data_dir = workdir / "data"
    if not skip_data or not (data_dir / "canaries.json").exists():
        log("[1/5] 生成合成客服语料与金丝雀样本…")
        generate_dataset(cfg)
    else:
        log("[1/5] 复用已有数据集")

    canaries = [Canary(**c) for c in json.loads(
        (data_dir / "canaries.json").read_text(encoding="utf-8"))]
    references = json.loads((data_dir / "reference_strings.json").read_text(encoding="utf-8"))
    heldout = {
        "qa": _read_jsonl(data_dir / "heldout" / "qa.jsonl"),
        "copy_tasks": _read_jsonl(data_dir / "heldout" / "copy_tasks.jsonl"),
        "fishing": _read_jsonl(data_dir / "heldout" / "fishing_prompts.jsonl"),
        "dialogues": (data_dir / "heldout" / "dialogues.txt")
        .read_text(encoding="utf-8").split("\n\n"),
        "dialogues_redacted": (data_dir / "heldout" / "dialogues_redacted.txt")
        .read_text(encoding="utf-8").split("\n\n"),
    }
    # 无 PII 的全新模板问答：所有条件训练语料同分布，用于严格跨条件的困惑度比较
    heldout["qa_clean_docs"] = [
        f"顾客：{item['q']}\n客服：{item['a']}"
        for item in heldout["qa"] if item.get("split") == "unseen"
    ]

    # ---- 2. 分词器（在基线语料上训练，全条件共用，保证可比） ---------------
    tok_path = workdir / "tokenizer.json"
    if tok_path.exists():
        tokenizer = BPETokenizer.from_file(tok_path)
    else:
        log("[2/5] 训练 BPE 分词器…")
        # 在基线 + 脱敏语料上共同训练，确保脱敏占位符也在词表内
        words = []
        for name in ("baseline", "redaction"):
            words += (data_dir / "train_corpus" / f"{name}.txt").read_text(
                encoding="utf-8").split()
        tokenizer = BPETokenizer.train(words, vocab_size=cfg.model.bpe_vocab_size)
        tokenizer.save(tok_path)
    log(f"      词表大小 {tokenizer.vocab_size}")

    conds = conditions or list(cfg.conditions)
    summaries = []
    result_dir = workdir / "results_per_condition"

    for ci, condition in enumerate(conds):
        log(f"[3/5] 条件 {condition}（{ci + 1}/{len(conds)}）")
        ckpt_dir = workdir / "checkpoints" / condition
        corpus_path = data_dir / "train_corpus" / f"{condition}.txt"
        if not corpus_path.exists():
            raise FileNotFoundError(f"缺少语料 {corpus_path}，请先以完整条件列表生成数据")

        if skip_train and (ckpt_dir / "model.pt").exists():
            log("      复用已有权重")
            train_meta = json.loads((ckpt_dir / "train_meta.json").read_text(encoding="utf-8"))
        else:
            log("      训练中…")
            train_meta = train_condition(condition, cfg, tokenizer, corpus_path, ckpt_dir)
        model = load_model(ckpt_dir, cfg, tokenizer)

        log("      抽取攻击（5 类提问 × 2 种采样）…")
        raw_extraction = run_extraction_attacks(model, tokenizer, canaries, cfg, log=log)
        log("      Exposure 排名…")
        raw_exposure = exposure_scores(model, tokenizer, canaries, references, cfg)
        log("      无锚点钓鱼…")
        raw_fishing = run_fishing(model, tokenizer, heldout["fishing"], canaries, cfg)
        log("      效用评估…")
        utility = evaluate_utility(model, tokenizer, heldout, cfg)

        extraction = aggregate_extraction(raw_extraction)
        exposure = aggregate_exposure(raw_exposure)
        fishing = aggregate_fishing(raw_fishing)

        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / f"{condition}_extraction.json").write_text(
            json.dumps(raw_extraction, ensure_ascii=False, indent=2), encoding="utf-8")
        (result_dir / f"{condition}_exposure.json").write_text(
            json.dumps(raw_exposure, ensure_ascii=False, indent=2), encoding="utf-8")
        (result_dir / f"{condition}_fishing.json").write_text(
            json.dumps(raw_fishing, ensure_ascii=False, indent=2), encoding="utf-8")
        (result_dir / f"{condition}_utility.json").write_text(
            json.dumps(utility, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = build_condition_summary(
            condition, extraction, exposure, fishing, utility, train_meta)
        summaries.append(summary)
        log(f"      风险分 {summary['risk']['composite_risk']:.1f} | "
            f"QA-F1 {utility['qa_f1']:.3f} | 困惑度 {utility['heldout_perplexity']:.1f}")

    # ---- 汇总报告 ----------------------------------------------------------
    log("[5/5] 生成图表与 HTML 报告…")
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_all(summaries, cfg, workdir, gen_time)
    log(f"完成，总耗时 {(time.time() - t_start) / 60:.1f} 分钟。报告：{workdir / 'report.html'}")
    return summaries
