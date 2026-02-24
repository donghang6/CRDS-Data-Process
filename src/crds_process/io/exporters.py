"""结果导出模块

将处理结果保存为 CSV / JSON 等格式。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_spectrum_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    """将吸收光谱数据保存为 CSV

    Parameters
    ----------
    df : pd.DataFrame
        包含波数和吸收系数的数据
    output_path : str or Path
        输出路径

    Returns
    -------
    Path
        保存的文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_fit_results(results: dict, output_path: str | Path) -> Path:
    """将拟合结果保存为 JSON

    Parameters
    ----------
    results : dict
        拟合结果字典
    output_path : str or Path
        输出路径

    Returns
    -------
    Path
        保存的文件路径
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 处理 numpy 类型序列化
    def _default(obj):
        import numpy as np
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_default)

    return output_path

