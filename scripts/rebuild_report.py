#!/usr/bin/env python3
"""从已保存的 condition_summaries.json 离线重建 HTML/CSV 报告（不重训模型）。

用法：
  python scripts/rebuild_report.py --run results/quick
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leak_eval.config import get_config
from leak_eval.reporting import save_all


def main() -> None:
    ap = argparse.ArgumentParser(description="离线重建评估报告")
    ap.add_argument("--run", required=True, help="评估输出目录，如 results/quick")
    ap.add_argument("--scale", default="quick", help="配置档位（用于读取元信息）")
    args = ap.parse_args()

    run_dir = Path(args.run)
    summaries = json.loads((run_dir / "condition_summaries.json").read_text(encoding="utf-8"))
    cfg = get_config(args.scale, workdir=run_dir)
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "（离线重建）"
    save_all(summaries, cfg, run_dir, gen_time)
    print(f"报告已重建：{run_dir / 'report.html'}")


if __name__ == "__main__":
    main()
