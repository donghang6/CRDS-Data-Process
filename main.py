"""CRDS 数据处理 — 完整五步流水线

运行方式:
    python main.py                              # 处理全部数据
    python main.py O2/9386.2076                  # 仅处理 O2 下 9386.2076 跃迁
    python main.py O2/9386.2076/100Torr          # 仅处理 O2/9386.2076 的 100Torr
    python main.py O2/9386.2076 O2_N2/9386.2076  # 同时处理多个目标
"""

import sys

from crds_process.pipeline import CRDSPipeline

if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    pipeline = CRDSPipeline(targets=targets)
    pipeline.run()
