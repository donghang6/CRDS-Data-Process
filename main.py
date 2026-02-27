"""CRDS 数据处理 — 完整四步流水线

运行方式:
    python main.py
"""

from crds_process.pipeline import CRDSPipeline

if __name__ == "__main__":
    pipeline = CRDSPipeline()
    pipeline.run()
