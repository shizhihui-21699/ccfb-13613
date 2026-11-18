#!/usr/bin/env python3
"""LeakEval 命令行入口。

示例：
  python scripts/run_eval.py --scale smoke                 # 冒烟（1~2 分钟）
  python scripts/run_eval.py --scale quick --out results   # 默认完整评估
  python scripts/run_eval.py --scale full --device mps     # 大规模 + GPU
  python scripts/run_eval.py --conditions baseline dp_sgd  # 只跑部分条件
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leak_eval.config import get_config
from leak_eval.pipeline import run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="训练数据隐私泄露评估工具")
    ap.add_argument("--scale", choices=["smoke", "quick", "full"], default="quick",
                    help="实验规模：smoke 冒烟 / quick 默认 / full 完整")
    ap.add_argument("--out", default="results", help="输出目录（默认 results/）")
    ap.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"],
                    help="计算设备，默认 cpu（MPS 逐样本梯度较慢）")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="只跑指定条件，如 baseline redaction dp_sgd")
    ap.add_argument("--skip-data", action="store_true", help="复用已有数据集")
    ap.add_argument("--skip-train", action="store_true", help="复用已训练权重")
    args = ap.parse_args()

    cfg = get_config(args.scale, workdir=args.out, device=args.device)
    run_pipeline(cfg, conditions=args.conditions,
                 skip_data=args.skip_data, skip_train=args.skip_train)


if __name__ == "__main__":
    main()
