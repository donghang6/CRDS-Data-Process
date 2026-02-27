"""原始数据预处理器

将原始衰荡数据经过读取、多组连接检查、波数间隔过滤、衰荡时间统计等步骤，
输出干净的衰荡时间结果和可视化图表。

用法:
    from crds_process.preprocessing import RawDataProcessor

    proc = RawDataProcessor("data/raw/9386.2076", output_dir="output/test_results")
    results = proc.run()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crds_process.io.readers import ScanData, load_scan_directory, scans_to_dataframe
from crds_process.log import logger
from crds_process.ringdown.processing import RingdownResult, process_all_scans


@dataclass
class PreprocessResult:
    """预处理结果"""
    scans_raw: list[ScanData]                # 原始扫描数据
    scans_cleaned: list[ScanData]            # 清洗后的扫描数据
    ringdown_results: list[RingdownResult]   # 衰荡时间统计结果
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)   # 汇总表
    ringdown_df: pd.DataFrame = field(default_factory=pd.DataFrame)  # 衰荡时间表
    n_removed_spacing: int = 0               # 间隔异常剔除数


class RawDataProcessor:
    """原始衰荡数据预处理器

    处理步骤:
        1. 读取原始数据文件
        2. 波数间隔检查，剔除异常点
        3. 衰荡时间离群值过滤与统计

    Parameters
    ----------
    data_dir : str or Path
        原始数据目录
    output_dir : str or Path, optional
        输出目录（CSV + 图表），None 则不保存
    filter_method : str
        衰荡时间过滤方法 ("sigma_clip" 或 "iqr")
    sigma : float
        sigma-clip 阈值
    iqr_factor : float
        IQR 因子
    min_events : int
        每个波数点最少衰荡事件数
    spacing_sigma : float
        间隔异常检测的 sigma 倍数
    spacing_lower_ratio : float
        间隔过小阈值 = 中位数 × ratio
    verbose : bool
        是否打印处理过程
    """

    def __init__(
        self,
        data_dir: str | Path,
        output_dir: str | Path | None = None,
        filter_method: str = "sigma_clip",
        sigma: float = 3.0,
        iqr_factor: float = 1.5,
        min_events: int = 5,
        spacing_sigma: float = 3.0,
        spacing_lower_ratio: float = 0.1,
        verbose: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.filter_method = filter_method
        self.sigma = sigma
        self.iqr_factor = iqr_factor
        self.min_events = min_events
        self.spacing_sigma = spacing_sigma
        self.spacing_lower_ratio = spacing_lower_ratio
        self.verbose = verbose

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)

    # ==============================================================
    # Step 1: 读取原始数据
    # ==============================================================
    def read_raw(self) -> list[ScanData]:
        """读取原始数据目录下所有扫描文件"""
        self._log(f"[Step 1] 读取原始数据: {self.data_dir}")

        scans = load_scan_directory(self.data_dir)
        self._log(f"  文件数: {len(scans)}")

        if not scans:
            raise FileNotFoundError(f"未在 {self.data_dir} 找到数据文件")

        wn_min = min(s.meta.wavenumber for s in scans)
        wn_max = max(s.meta.wavenumber for s in scans)
        self._log(f"  波数范围: {wn_min:.5f} ~ {wn_max:.5f} cm-1")


        return scans


    # ==============================================================
    # Step 2: 波数间隔检查
    # ==============================================================
    def check_spacing(self, scans: list[ScanData]) -> tuple[list[ScanData], int]:
        """检查并剔除波数间隔异常的点

        Returns
        -------
        tuple[list[ScanData], int]
            (过滤后的列表, 剔除点数)
        """
        self._log(f"[Step 1.5] 波数间隔检查")

        wn = np.array([s.meta.wavenumber for s in scans])
        sp = np.diff(wn)
        median_sp = np.median(sp)
        std_sp = np.std(sp)

        upper = median_sp + self.spacing_sigma * std_sp
        lower = median_sp * self.spacing_lower_ratio

        large = np.where(sp > upper)[0]
        small = np.where(sp < lower)[0]

        self._log(f"  点数: {len(wn)}, 中位间隔: {median_sp:.5f}, std: {std_sp:.5f}")
        self._log(f"  阈值: upper={upper:.5f}, lower={lower:.5f}")
        self._log(f"  异常大: {len(large)}, 异常小: {len(small)}")

        remove: set[int] = set()
        for idx in large:
            remove.add(int(idx))
            remove.add(int(idx) + 1)
        for idx in small:
            remove.add(int(idx) + 1)

        if remove:
            filtered = [s for i, s in enumerate(scans) if i not in remove]
            self._log(f"  剔除 {len(remove)} 点, 剩余 {len(filtered)}")
        else:
            filtered = scans
            self._log(f"  无异常, 全部保留")

        return filtered, len(remove)

    # ==============================================================
    # Step 3: 衰荡时间过滤与统计
    # ==============================================================
    def process_ringdown(self, scans: list[ScanData]) -> tuple[list[RingdownResult], pd.DataFrame]:
        """衰荡时间离群值过滤 + 统计

        Returns
        -------
        tuple[list[RingdownResult], pd.DataFrame]
            (处理结果列表, 汇总 DataFrame)
        """
        self._log(f"[Step 3] 衰荡时间过滤与统计 (method={self.filter_method}, sigma={self.sigma})")

        results = process_all_scans(
            scans,
            filter_method=self.filter_method,
            sigma=self.sigma,
            iqr_factor=self.iqr_factor,
            min_events=self.min_events,
        )
        self._log(f"  {len(scans)} 输入 -> {len(results)} 有效")

        df = pd.DataFrame([{
            "wavenumber": r.wavenumber,
            "tau_mean": round(r.tau_mean, 5),
            "tau_std": round(r.tau_std, 5),
            "temperature": round(r.temperature, 3),
            "pressure": round(r.pressure, 3),
        } for r in results])

        if not df.empty:
            self._log(f"  tau range: {df['tau_mean'].min():.4f} ~ {df['tau_mean'].max():.4f} us")

        if self.output_dir:
            df.to_csv(self.output_dir / "ringdown_results.csv", index=False)
            self._plot_ringdown(scans, results, df)

        return results, df

    def _plot_ringdown(self, scans, results, df):
        wn = df["wavenumber"].values
        tau = df["tau_mean"].values
        tau_std = df["tau_std"].values

        # 衰荡时间光谱（1-sigma 带）
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(wn, tau, "-", color="steelblue", lw=1)
        ax.fill_between(wn, tau - tau_std, tau + tau_std,
                         alpha=0.2, color="steelblue", label="1-sigma band")
        ax.set_xlabel("Wavenumber (cm-1)")
        ax.set_ylabel("Ring-down time (us)")
        ax.set_title("Ring-down time spectrum (1-sigma band)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "tau_band.png", dpi=150)
        plt.close(fig)

        self._log(f"  图表已保存: tau_band.png")

    # ==============================================================
    # run: 执行全部预处理步骤
    # ==============================================================
    def run(self) -> PreprocessResult:
        """执行完整的预处理流程

        Returns
        -------
        PreprocessResult
            包含各阶段结果的数据对象
        """
        # Step 1
        scans_raw = self.read_raw()

        # Step 2
        scans, n_spacing = self.check_spacing(scans_raw)

        # Step 3
        results, df = self.process_ringdown(scans)

        result = PreprocessResult(
            scans_raw=scans_raw,
            scans_cleaned=scans,
            ringdown_results=results,
            summary_df=scans_to_dataframe(scans_raw),
            ringdown_df=df,
            n_removed_spacing=n_spacing,
        )

        self._log(f"\n[Done] {len(scans_raw)} -> {len(scans)} -> {len(results)} "
                   f"(间隔剔除 {n_spacing})")

        return result


# ==================================================================
# 项目级常量
# ==================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
RESULT_ROOT = PROJECT_ROOT / "output" / "results" / "ringdown"


# ==================================================================
# 自动发现 & 批量处理
# ==================================================================
def discover_tasks(raw_root: Path | None = None) -> list[tuple[str, str, Path]]:
    """自动发现 raw_root/{跃迁波数}/{压力}/ 下的数据集

    Parameters
    ----------
    raw_root : Path, optional
        原始数据根目录，默认为 RAW_ROOT

    Returns
    -------
    list[tuple[str, str, Path]]
        [(跃迁波数, 压力, 数据目录), ...]
    """
    root = raw_root or RAW_ROOT
    tasks: list[tuple[str, str, Path]] = []
    for transition_dir in sorted(root.iterdir()):
        if not transition_dir.is_dir() or transition_dir.name.startswith("."):
            continue
        for pressure_dir in sorted(transition_dir.iterdir()):
            if not pressure_dir.is_dir() or pressure_dir.name.startswith("."):
                continue
            if not list(pressure_dir.glob("*.txt")):
                continue
            tasks.append((transition_dir.name, pressure_dir.name, pressure_dir))
    return tasks


def batch_preprocess_ringdown(
    raw_root: Path | None = None,
    result_root: Path | None = None,
    filter_method: str = "sigma_clip",
    sigma: float = 3.0,
    min_events: int = 5,
) -> list[PreprocessResult]:
    """批量处理所有自动发现的数据集

    Parameters
    ----------
    raw_root : Path, optional
        原始数据根目录，默认为 RAW_ROOT
    result_root : Path, optional
        结果输出根目录，默认为 RESULT_ROOT
    filter_method : str
        衰荡时间过滤方法
    sigma : float
        sigma-clip 阈值
    min_events : int
        每个波数点最少衰荡事件数

    Returns
    -------
    list[PreprocessResult]
    """
    raw = raw_root or RAW_ROOT
    out = result_root or RESULT_ROOT

    tasks = discover_tasks(raw)
    if not tasks:
        logger.error(f"未在 {raw} 下找到 {{跃迁波数}}/{{压力}}/*.txt 数据")
        return []

    logger.info(f"{'#' * 60}")
    logger.info(f"  CRDS 原始数据批量预处理")
    logger.info(f"  数据根目录: {raw}")
    logger.info(f"  结果根目录: {out}")
    logger.info(f"  发现 {len(tasks)} 个数据集:")
    for transition, pressure, _ in tasks:
        logger.info(f"    {transition}/{pressure}/")
    logger.info(f"{'#' * 60}\n")

    all_results: list[PreprocessResult] = []

    for i, (transition, pressure, data_dir) in enumerate(tasks, 1):
        output_dir = out / transition / pressure
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  [{i}/{len(tasks)}] {transition} / {pressure}")
        logger.info(f"  数据: {data_dir}")
        logger.info(f"  输出: {output_dir}")
        logger.info(f"{'=' * 60}")

        proc = RawDataProcessor(
            data_dir=data_dir,
            output_dir=output_dir,
            filter_method=filter_method,
            sigma=sigma,
            min_events=min_events,
        )
        result = proc.run()
        all_results.append(result)

        logger.info(f"\nRingdown preview (first 5):")
        logger.info(result.ringdown_df.head().to_string(index=False))

    # 汇总
    logger.info(f"\n\n{'#' * 60}")
    logger.info("  全部处理完成! 输出文件:")
    logger.info(f"{'#' * 60}")
    for transition, pressure, _ in tasks:
        d = out / transition / pressure
        logger.info(f"\n  {transition}/{pressure}/")
        for f in sorted(d.glob("*")):
            logger.info(f"    {f.name:<40s}  {f.stat().st_size:>8,} bytes")

    return all_results


