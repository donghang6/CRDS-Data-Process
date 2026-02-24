"""测试脚本: 自动遍历 data/raw/{跃迁波数}/{压力}/ 处理原始衰荡数据

目录结构:
    data/raw/{跃迁波数}/{压力}/*.txt   ← 原始数据
    output/results/ringdown/{跃迁波数}/{压力}/  ← 处理结果

运行: python scripts/process_raw.py
"""

from crds_process.preprocessing import batch_preprocess_ringdown

if __name__ == "__main__":
    batch_preprocess_ringdown()

