"""CRDS 日志配置

提供双通道日志: 终端输出 + 文件保存。

用法:
    from crds_process.log import setup_logging, logger

    # 初始化 (在流水线入口调用一次)
    setup_logging()            # 默认保存到 output/logs/
    setup_logging("my.log")    # 指定日志文件

    # 使用
    logger.info("处理完成")
    logger.warning("数据异常")
    logger.error("拟合失败")
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_LOG_DIR = _PROJECT_ROOT / "output" / "logs"

# 全局 logger
logger = logging.getLogger("crds")

_initialized = False


def setup_logging(
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    log_dir: Path | None = None,
) -> Path:
    """初始化日志系统: 终端 + 文件双通道输出

    Parameters
    ----------
    log_file : str or Path, optional
        日志文件名或完整路径。
        若仅为文件名，保存到 log_dir 下。
        若为 None，自动生成带时间戳的文件名。
    level : int
        日志级别 (默认 INFO)
    log_dir : Path, optional
        日志目录 (默认 output/logs/)

    Returns
    -------
    Path
        日志文件的完整路径
    """
    global _initialized

    # 避免重复初始化
    if _initialized:
        # 返回已有的文件 handler 路径
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                return Path(h.baseFilename)
        return _DEFAULT_LOG_DIR / "crds.log"

    log_dir = log_dir or _DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    # 确定日志文件路径
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"crds_{timestamp}.log"
    else:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = log_dir / log_path

    logger.setLevel(level)

    # 清除可能存在的旧 handler
    logger.handlers.clear()

    # ── 格式 ──
    # 终端: 简洁格式 (与之前 print 输出风格一致)
    console_fmt = logging.Formatter("%(message)s")
    # 文件: 详细格式 (带时间戳和级别)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── 终端 Handler ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # ── 文件 Handler ──
    file_handler = logging.FileHandler(
        str(log_path), mode="w", encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # 防止日志向上传播到 root logger 导致重复输出
    logger.propagate = False

    _initialized = True

    logger.info(f"日志已初始化: {log_path}")

    return log_path

