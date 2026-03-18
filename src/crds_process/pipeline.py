"""CRDS 五步处理流水线

Step 1: 衰荡时间处理       (raw → ringdown_results.csv)
Step 2: 去除标准具          (ringdown_results → tau_etalon_corrected.csv)
Step 3: MATS 单光谱拟合    (etalon_corrected → 各压力独立拟合)
Step 4: 筛选 + 多光谱联合拟合  (剔除线强离群点 → 联合拟合 → 最终参数)
Step 5: 线性回归提取 N₂ 展宽  (O₂+N₂ 单光谱 + 纯 O₂ 联合拟合 → γ₀_N₂ 等)

目录约定:
    data/raw/{跃迁波数}/{压力}/*.txt                       ← 原始数据
    output/results/ringdown/{跃迁波数}/{压力}/              ← Step 1 输出
    output/results/etalon/{跃迁波数}/{压力}/                ← Step 2 输出
    output/results/mats/{跃迁波数}/{压力}/                  ← Step 3 输出
    output/results/mats_multi/{跃迁波数}/                   ← Step 4 输出
    output/results/final/                                   ← 最终汇总
    output/results/final/O2_N2/{跃迁}/linear_regression_*   ← Step 5 输出

用法:
    # 默认参数
    pipeline = CRDSPipeline()
    pipeline.run()

    # 自定义参数
    pipeline = CRDSPipeline(
        raw_root="data/raw",
        lineprofile="SDVP",
        sw_sigma=2.0,
    )
    pipeline.run()

    # 指定多光谱联合拟合使用的压力
    pipeline = CRDSPipeline(
        multi_fit_pressures={
            "O2/9386.2076": ["100Torr", "200Torr", "300Torr"],
        },
    )
    pipeline.step4_multi_fit()

    # 跳过 Step 1, 从已有 ringdown 结果开始执行 Step 2~5
    pipeline.run_from_ringdown()

    # 仅提取 N2 展宽 (跳过纯 O2 联合拟合)
    pipeline.run_n2_only()

    # 单独执行某一步
    pipeline.step1_ringdown()
    pipeline.step2_etalon()
    pipeline.step3_mats()
    pipeline.step4_multi_fit()
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from crds_process.gas_config import parse_gas_dir
from crds_process.log import logger, setup_logging


# ==================================================================
# 默认路径 (基于项目根目录)
# ==================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_PATHS = {
    "raw": _PROJECT_ROOT / "data" / "raw",
    "ringdown": _PROJECT_ROOT / "output" / "results" / "ringdown",
    "etalon": _PROJECT_ROOT / "output" / "results" / "etalon",
    "mats": _PROJECT_ROOT / "output" / "results" / "mats",
    "mats_multi": _PROJECT_ROOT / "output" / "results" / "mats_multi",
    "final": _PROJECT_ROOT / "output" / "results" / "final",
}

_O2_REMEASURE_PLAN_CSV = (
    _PROJECT_ROOT / "data" / "reference" / "o2_remeasure_pressure_plan.csv"
)


def _format_progress_time(seconds: float) -> str:
    """将秒数格式化为便于阅读的时长字符串。"""
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _print_progress_bar(
    prefix: str,
    current: int,
    total: int,
    start_time: float,
    *,
    finished: bool = False,
) -> None:
    """在终端输出简单进度条；非 TTY 环境下降低刷新频率。"""
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    fraction = current / total
    elapsed = max(time.time() - start_time, 0.0)
    rate = current / elapsed if elapsed > 0 and current > 0 else 0.0
    eta = (total - current) / rate if rate > 0 else float("inf")

    if not sys.stdout.isatty() and not finished:
        stride = max(total // 10, 1)
        if current not in (0, total) and current % stride != 0:
            return

    bar_width = 24
    filled = min(int(bar_width * fraction), bar_width)
    if filled >= bar_width:
        bar = "=" * bar_width
    else:
        head = "=" * filled
        tail = "." * max(bar_width - filled - 1, 0)
        bar = f"{head}>{tail}" if current > 0 else "." * bar_width

    line = (
        f"{prefix} [{bar}] {current}/{total} "
        f"({fraction * 100:5.1f}%) elapsed {_format_progress_time(elapsed)}"
    )
    if current < total and np.isfinite(eta):
        line += f" eta {_format_progress_time(eta)}"

    end = "\n" if finished or not sys.stdout.isatty() else "\r"
    print(line, end=end, flush=True)


# ==================================================================
# 子进程 worker 函数 (模块级，可被 pickle 序列化)
# ==================================================================
def _worker_ringdown(
    data_dir: Path, output_dir: Path,
    filter_method: str, sigma: float, min_events: int,
) -> str:
    """子进程: 处理单个 (transition/pressure) 的衰荡数据"""
    from crds_process.preprocessing import RawDataProcessor
    label = f"{data_dir.parent.name}/{data_dir.name}"
    try:
        proc = RawDataProcessor(
            data_dir=data_dir, output_dir=output_dir,
            filter_method=filter_method, sigma=sigma,
            min_events=min_events, verbose=False,
        )
        proc.run()
        return f"  ✓ {label}"
    except Exception as e:
        return f"  ✗ {label}: {e}"


def _worker_etalon(csv_path: Path, output_dir: Path, label: str,
                   gas_type: str = "O2") -> str:
    """子进程: 处理单个 (transition/pressure) 的标准具去除

    O2_N2 混合气体总压较高 (400~650 Torr)，HITRAN 模拟的吸收线翼区
    延展更远，默认 1% 阈值会导致排除区域过大。因此对 O2_N2 使用：
      - 更高的排除阈值 (根据压力自适应)
      - 排除区域最大宽度限制 (0.4 cm⁻¹)
    """
    import matplotlib
    matplotlib.use("Agg")
    from crds_process.baseline.etalon import EtalonRemover, HitranAbsorptionDetector
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        wn = df["wavenumber"].values
        tau = df["tau_mean"].values
        temp = float(df["temperature"].mean())
        pres = float(df["pressure"].mean())

        if gas_type == "O2_N2":
            # O2_N2: 根据总压自适应提高排除阈值
            # P ≤ 200 Torr: 基础 0.01; P > 200: 每增 100 Torr 翻倍, 上限 0.20
            if pres <= 200:
                ratio = 0.01
            else:
                excess = (pres - 200) / 100.0
                ratio = min(0.01 * (2.0 ** excess), 0.20)
            detector = HitranAbsorptionDetector(
                threshold_ratio=ratio, margin=0.03,
            )
            remover = EtalonRemover(
                hitran_detector=detector,
                auto_snr_threshold=2.0,             # 更敏感地检测弱标准具分量
                residual_improvement_threshold=0.02, # 2% 改善即接受新分量
                n_iter=7,                            # 更多迭代次数提高精度
            )
        else:
            remover = EtalonRemover()

        output_dir.mkdir(parents=True, exist_ok=True)
        result = remover.fit(wn, tau, temperature=temp, pressure_torr=pres)

        # O2_N2: 检查排除区域并收缩过宽的区域后重新拟合
        if gas_type == "O2_N2" and result.exclude_regions:
            max_width = 0.4
            need_refit = False
            trimmed = []
            for lo, hi in result.exclude_regions:
                w = hi - lo
                if w > max_width:
                    center = (lo + hi) / 2.0
                    lo = center - max_width / 2.0
                    hi = center + max_width / 2.0
                    need_refit = True
                trimmed.append([lo, hi])
            if need_refit:
                remover_trim = EtalonRemover(
                    exclude_regions=trimmed,
                    auto_snr_threshold=2.0,
                    residual_improvement_threshold=0.02,
                    n_iter=7,
                )
                result = remover_trim.fit(wn, tau)

        title = f"Etalon Removal — {label}"
        result.plot(title=title, save_path=output_dir / "etalon_removal.png")
        result.save_csv(df, output_dir / "tau_etalon_corrected.csv")
        return f"  ✓ {label}  (成功={result.model_result.success})"
    except Exception as e:
        return f"  ✗ {label}: {e}"


def _worker_mats(
    csv_path: Path, output_dir: Path, label: str,
    fitter_kwargs: dict,
) -> str:
    """子进程: 处理单个 (transition/pressure) 的 MATS 拟合"""
    import matplotlib
    matplotlib.use("Agg")
    from crds_process.spectral.mats_wrapper import MATSFitter
    try:
        fitter = MATSFitter(**fitter_kwargs)
        dataset_name = label.replace("/", "_").replace(" ", "") or "crds"
        output_dir.mkdir(parents=True, exist_ok=True)
        result = fitter.fit(csv_path, output_dir, dataset_name=dataset_name)
        if result.summary_df is not None and not result.summary_df.empty:
            fitter.plot_result(
                result, output_dir, title=f"MATS Fit — {label}",
            )
        return f"  ✓ {label}"
    except Exception as e:
        return f"  ✗ {label}: {e}"


def _worker_trial_multi_fit(
    etalon_csvs: list[Path],
    labels: list[str],
    fitter_kwargs: dict,
    transition: str,
    combo: tuple[str, ...],
) -> tuple[tuple[str, ...], float, float]:
    """子进程: 对一个压力组合执行试拟合，返回 (combo, QF, sw)。

    轻量版本: 在临时目录中拟合，不保存最终输出，仅获取 QF 和线强。
    拟合失败时返回 (combo, -1.0, 0.0)。
    """
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    from crds_process.spectral.mats_wrapper import MATSFitter

    if len(etalon_csvs) < 2:
        return combo, -1.0, 0.0

    try:
        fitter = MATSFitter(**fitter_kwargs)
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = fitter.fit_multi(
                etalon_csvs=etalon_csvs,
                labels=labels,
                output_dir=Path(tmp_dir),
                dataset_name=f"{transition}_trial",
            )
            if not result:
                return combo, -1.0, 0.0
            qf = result.qf
            # 提取目标跃迁的拟合线强 (选 nu 与 transition 最接近的行)
            sw = 0.0
            if not result.param_linelist.empty:
                fitted = result.param_linelist[
                    result.param_linelist["sw_vary"] == True]
                if not fitted.empty:
                    try:
                        nu_target = float(transition)
                        idx_closest = (fitted["nu"] - nu_target).abs().idxmin()
                        row = fitted.loc[idx_closest]
                    except (ValueError, KeyError):
                        row = fitted.iloc[0]
                    scale = row.get("sw_scale_factor", 1.0)
                    sw = row["sw"] * scale
            return combo, qf, sw
    except Exception:
        return combo, -1.0, 0.0


class CRDSPipeline:
    """CRDS 五步处理流水线

    Parameters
    ----------
    raw_root : Path or str, optional
        原始数据目录
    ringdown_root : Path or str, optional
        Step 1 输出目录
    etalon_root : Path or str, optional
        Step 2 输出目录
    mats_root : Path or str, optional
        Step 3 输出目录
    mats_multi_root : Path or str, optional
        Step 4 输出目录
    final_root : Path or str, optional
        最终汇总输出目录
    lineprofile : str
        MATS 线形 (默认 "SDVP")
    sw_sigma : float
        Step 4 线强筛选阈值 (默认 2.0)
    molecule : int
        HITRAN 分子编号 (默认 7 = O₂)
    isotopologue : int
        HITRAN 同位素编号 (默认 1)
    molefraction : dict
        摩尔分数 (默认 {7: 1.0})
    diluent : str
        稀释气体 (默认 "O2")
    baseline_order : int
        基线多项式阶数 (默认 1)
    fit_intensity : float
        拟合线强阈值 (默认 1e-30)
    threshold_intensity : float
        线强筛选阈值 (默认 1e-35)
    refit_threshold : float
        参数相对误差阈值 (默认 0.5 即 50%)，超过则固定该参数并重新拟合
    targets : list[str], optional
        指定要处理的数据子集，格式为 "气体类型/跃迁" 或 "气体类型/跃迁/压力"。
        例如: ["O2/9386.2076", "O2_N2/9386.2076/500Torr"]
        默认 None 表示处理全部数据。
    multi_fit_pressures : dict[str, list[str]], optional
        指定参与多光谱联合拟合 (Step 4) 的压力列表，跳过自动线强筛选。
        键为 "气体类型/跃迁"，值为压力目录名列表。
        例如: {"O2/9386.2076": ["100Torr", "200Torr", "300Torr"]}
        未指定的跃迁仍使用默认的 MAD 自动筛选。
        默认 None 表示全部使用自动筛选。
    auto_optimize_pressures : bool
        是否自动搜索最优压力组合 (默认 False)。
        启用后，Step 4 会枚举所有压力组合 (最少 min_multi_pressures 个，
        最多全部)，逐一执行多光谱联合拟合，选取 QF 最大的组合作为最终结果。
    min_multi_pressures : int
        自动搜索时每次组合的最少压力数 (默认 3)。
    fit_transitions : list[float], optional
        仅拟合指定跃迁波数(吸收线)，单位 cm⁻¹。
        若提供，则 Step 3/4 的 HITRAN 线表仅保留这些跃迁。
        例如: [9403.163069]
    remeasure_rel_threshold : float, optional
        建议重测报告的统一相对偏差阈值。若提供，则同时覆盖 O2 和 O2_N2。
    remeasure_rel_threshold_o2 : float, optional
        建议重测报告中，纯 O2 的相对偏差阈值 (默认 0.05 = 5%)
    remeasure_rel_threshold_o2n2 : float, optional
        建议重测报告中，O2_N2 的相对偏差阈值 (默认 0.10 = 10%)
    remeasure_sigma_threshold : float
        建议重测报告中，偏差相对联合不确定度的阈值 (默认 3σ)
    type_a_mc_samples : int
        Monte Carlo Type A 抽样次数 (默认 100)
    type_a_mc_seed : int
        Monte Carlo Type A 随机种子 (默认 12345)
    type_a_mc_wave_error_khz : float
        Monte Carlo 中施加的 x 轴频率噪声，单位 kHz (默认 4000 = 4 MHz)
    """

    # 干空气近似组成，用于由 HITRAN air/self 展宽反推 N2 展宽
    _AIR_O2_FRACTION = 0.21
    _AIR_N2_FRACTION = 0.79

    def __init__(
        self,
        raw_root: Path | str | None = None,
        ringdown_root: Path | str | None = None,
        etalon_root: Path | str | None = None,
        mats_root: Path | str | None = None,
        mats_multi_root: Path | str | None = None,
        final_root: Path | str | None = None,
        lineprofile: str = "SDVP",
        sw_sigma: float = 2.0,
        molecule: int = 7,
        isotopologue: int = 1,
        molefraction: dict | None = None,
        diluent: str = "O2",
        baseline_order: int = 1,
        fit_intensity: float = 1e-30,
        threshold_intensity: float = 1e-35,
        refit_threshold: float = 0.5,
        max_workers: int | None = None,
        targets: list[str] | None = None,
        multi_fit_pressures: dict[str, list[str]] | None = None,
        auto_optimize_pressures: bool = False,
        min_multi_pressures: int = 3,
        fit_transitions: list[float] | None = None,
        remeasure_rel_threshold: float | None = None,
        remeasure_rel_threshold_o2: float | None = None,
        remeasure_rel_threshold_o2n2: float | None = None,
        remeasure_sigma_threshold: float = 3.0,
        type_a_mc_samples: int = 100,
        type_a_mc_seed: int = 12345,
        type_a_mc_wave_error_khz: float = 4000.0,
    ):
        # ── 路径 ──
        self.raw_root = Path(raw_root) if raw_root else _DEFAULT_PATHS["raw"]
        self.ringdown_root = Path(ringdown_root) if ringdown_root else _DEFAULT_PATHS["ringdown"]
        self.etalon_root = Path(etalon_root) if etalon_root else _DEFAULT_PATHS["etalon"]
        self.mats_root = Path(mats_root) if mats_root else _DEFAULT_PATHS["mats"]
        self.mats_multi_root = Path(mats_multi_root) if mats_multi_root else _DEFAULT_PATHS["mats_multi"]
        self.final_root = Path(final_root) if final_root else _DEFAULT_PATHS["final"]
        self.type_a_mc_root = self.final_root.parent / "type_a_mc"

        # ── 并行 ──
        self.max_workers = max_workers or min(os.cpu_count() or 1, 6)

        # ── 目标过滤 ──
        self.targets = targets  # e.g. ["O2/9386.2076", "O2_N2/9386.2076/500Torr"]
        self._o2_remeasure_plan_cache: dict[str, set[str]] | None = None
        self._hitran_reference_cache: pd.DataFrame | None | bool = False
        self._measurement_window_cache: dict[tuple[str, float], dict] = {}
        self._measured_transition_cache: dict[str, np.ndarray] = {}

        # ── 多光谱联合拟合压力指定 ──
        self.multi_fit_pressures = multi_fit_pressures  # e.g. {"O2/9386.2076": ["100Torr", "200Torr"]}

        # ── 自动搜索最优压力组合 ──
        self.auto_optimize_pressures = auto_optimize_pressures
        self.min_multi_pressures = max(min_multi_pressures, 2)
        if fit_transitions:
            parsed_nu = []
            for nu in fit_transitions:
                try:
                    parsed_nu.append(float(nu))
                except (TypeError, ValueError):
                    logger.warning(f"  忽略无效的 fit_transitions 值: {nu}")
            self.fit_transitions = sorted(set(parsed_nu)) if parsed_nu else None
        else:
            self.fit_transitions = None

        # ── 建议重测报告阈值 ──
        self.remeasure_rel_threshold_o2 = (
            max(float(remeasure_rel_threshold_o2), 0.0)
            if remeasure_rel_threshold_o2 is not None else 0.05
        )
        self.remeasure_rel_threshold_o2n2 = (
            max(float(remeasure_rel_threshold_o2n2), 0.0)
            if remeasure_rel_threshold_o2n2 is not None else 0.10
        )
        if remeasure_rel_threshold is not None:
            shared_rel = max(float(remeasure_rel_threshold), 0.0)
            if remeasure_rel_threshold_o2 is None:
                self.remeasure_rel_threshold_o2 = shared_rel
            if remeasure_rel_threshold_o2n2 is None:
                self.remeasure_rel_threshold_o2n2 = shared_rel
        self.remeasure_sigma_threshold = max(float(remeasure_sigma_threshold), 0.0)

        # ── Monte Carlo Type A ──
        self.type_a_mc_samples = max(int(type_a_mc_samples), 1)
        self.type_a_mc_seed = int(type_a_mc_seed)
        self.type_a_mc_wave_error_khz = max(float(type_a_mc_wave_error_khz), 0.0)

        # ── MATS 拟合参数 ──
        self.lineprofile = lineprofile
        self.sw_sigma = sw_sigma
        self.molecule = molecule
        self.isotopologue = isotopologue
        self.molefraction = molefraction or {7: 1.0}
        self.diluent = diluent
        self.baseline_order = baseline_order
        self.fit_intensity = fit_intensity
        self.threshold_intensity = threshold_intensity
        self.refit_threshold = refit_threshold


    # ==============================================================
    # 原始数据命名规范检测
    # ==============================================================
    # 已知气体类型
    _KNOWN_GAS_TYPES = {"O2", "O2_N2"}
    # 跃迁波数格式: 纯数字或浮点数 (如 9386.2076)
    _RE_TRANSITION = re.compile(r"^\d+(\.\d+)?$")
    # 纯 O₂ 压力目录: {数字}Torr
    _RE_PRESSURE_O2 = re.compile(r"^\d+Torr$")
    # O₂+N₂ 压力目录: O2 {数字}Torr N2 {数字}Torr (允许 {数字} 前后有空格)
    _RE_PRESSURE_MIX = re.compile(
        r"^O2\s+\d+\s*Torr\s+N2\s+\d+\s*Torr$", re.IGNORECASE
    )
    # 数据文件名: {序号} {波数} {14位时间戳}.txt
    _RE_DATAFILE = re.compile(r"^\s*\d+\s+[\d.]+\s+\d{14}\.txt$")

    def validate_raw_data(self) -> bool:
        """检测 data/raw 下原始数据目录和文件命名是否符合规范。

        规范:
            data/raw/{气体类型}/{跃迁波数}/{压力目录}/
            - 气体类型: O2, O2_N2
            - 跃迁波数: 浮点数 (如 9386.2076)
            - 压力目录:
                O2:    "{数字}Torr"   (如 100Torr)
                O2_N2: "O2 {数字}Torr N2 {数字}Torr"
            - 数据文件: "{序号} {波数} {14位时间戳}.txt"

        Returns
        -------
        bool
            True 表示全部通过, False 表示有警告或错误
        """
        raw = self.raw_root
        if not raw.exists():
            logger.error(f"  原始数据目录不存在: {raw}")
            return False

        all_ok = True
        total_files = 0
        bad_files = 0

        logger.info("\n" + "─" * 60)
        logger.info("  原始数据命名规范检测")
        logger.info("─" * 60)

        # 第一层: 气体类型
        for gas_dir in sorted(raw.iterdir()):
            if not gas_dir.is_dir() or gas_dir.name.startswith("."):
                continue
            gas_type = gas_dir.name

            if gas_type not in self._KNOWN_GAS_TYPES:
                logger.warning(f"  ⚠ 未知气体类型目录: {gas_type}/")
                logger.warning(f"    已知类型: {', '.join(sorted(self._KNOWN_GAS_TYPES))}")
                all_ok = False
                continue

            # 第二层: 跃迁波数
            for trans_dir in sorted(gas_dir.iterdir()):
                if not trans_dir.is_dir() or trans_dir.name.startswith("."):
                    continue
                transition = trans_dir.name

                if not self._RE_TRANSITION.match(transition):
                    logger.warning(
                        f"  ⚠ 跃迁目录名不是有效波数: "
                        f"{gas_type}/{transition}/")
                    logger.warning(
                        f"    应为纯数字或浮点数 (如 9386.2076)")
                    all_ok = False
                    continue

                # 第三层: 压力目录
                for pres_dir in sorted(trans_dir.iterdir()):
                    if not pres_dir.is_dir() or pres_dir.name.startswith("."):
                        continue
                    pressure = pres_dir.name
                    tag = f"{gas_type}/{transition}/{pressure}"

                    # 检查压力目录命名
                    if gas_type == "O2":
                        if not self._RE_PRESSURE_O2.match(pressure):
                            logger.warning(
                                f"  ⚠ 压力目录命名不规范: {tag}/")
                            logger.warning(
                                f"    纯 O₂ 应为 '{{数字}}Torr' "
                                f"(如 100Torr)")
                            all_ok = False
                    elif gas_type == "O2_N2":
                        if not self._RE_PRESSURE_MIX.match(pressure):
                            logger.warning(
                                f"  ⚠ 压力目录命名不规范: {tag}/")
                            logger.warning(
                                f"    O₂+N₂ 应为 "
                                f"'O2 {{数字}}Torr N2 {{数字}}Torr'")
                            all_ok = False

                    # 检查数据文件命名
                    txt_files = list(pres_dir.glob("*.txt"))
                    if not txt_files:
                        logger.warning(f"  ⚠ 目录为空 (无 .txt 文件): {tag}/")
                        all_ok = False
                        continue

                    total_files += len(txt_files)
                    for f in txt_files:
                        if not self._RE_DATAFILE.match(f.name):
                            if bad_files < 10:  # 只显示前 10 个
                                logger.warning(
                                    f"  ⚠ 文件名不规范: {tag}/{f.name}")
                                logger.warning(
                                    f"    应为 '{{序号}} {{波数}} "
                                    f"{{YYYYMMDDHHmmss}}.txt'")
                            bad_files += 1
                            all_ok = False

        if bad_files > 10:
            logger.warning(
                f"  ... 还有 {bad_files - 10} 个文件名不规范 (已省略)")

        # 汇总
        if all_ok:
            logger.info(f"  ✓ 全部通过! 共 {total_files} 个数据文件")
        else:
            logger.warning(
                f"  ✗ 发现命名问题 "
                f"(共 {total_files} 个文件, {bad_files} 个文件名不规范)")
            logger.warning(
                f"  提示: 命名不规范可能导致解析失败, 建议修正后再运行流水线")

        logger.info("─" * 60)
        return all_ok

    # ==============================================================
    # 目标过滤
    # ==============================================================
    def _filter_tasks(
        self,
        tasks: list[tuple[str, str, str, Path]],
    ) -> list[tuple[str, str, str, Path]]:
        """根据 self.targets 过滤任务列表。

        targets 条目格式:
          - "O2"                      → 匹配该气体类型下所有数据
          - "O2/9386.2076"            → 匹配该气体 + 跃迁下所有压力
          - "O2/9386.2076/100Torr"    → 精确匹配气体 + 跃迁 + 压力

        Parameters
        ----------
        tasks : list of (gas_type, transition, pressure, path)

        Returns
        -------
        list of (gas_type, transition, pressure, path)
        """
        if not self.targets:
            return tasks

        filtered = []
        for gas_type, transition, pressure, path in tasks:
            for t in self.targets:
                parts = t.strip("/").split("/")
                if len(parts) == 1:
                    # "O2" → match gas_type only
                    if gas_type == parts[0]:
                        filtered.append((gas_type, transition, pressure, path))
                        break
                elif len(parts) == 2:
                    # "O2/9386.2076" → match gas_type + transition
                    if gas_type == parts[0] and transition == parts[1]:
                        filtered.append((gas_type, transition, pressure, path))
                        break
                elif len(parts) >= 3:
                    # "O2/9386.2076/100Torr" → exact match
                    if (gas_type == parts[0] and transition == parts[1]
                            and pressure == parts[2]):
                        filtered.append((gas_type, transition, pressure, path))
                        break
        return filtered

    def _target_gas_types(self) -> set[str] | None:
        """从 targets 中提取涉及的气体类型集合，None 表示全部"""
        if not self.targets:
            return None
        gas_types = set()
        for t in self.targets:
            parts = t.strip("/").split("/")
            gas_types.add(parts[0])
        return gas_types

    def _target_transitions(self, gas_type: str) -> set[str] | None:
        """从 targets 中提取某气体类型下涉及的跃迁集合，None 表示全部"""
        if not self.targets:
            return None
        transitions = set()
        for t in self.targets:
            parts = t.strip("/").split("/")
            if parts[0] == gas_type:
                if len(parts) >= 2:
                    transitions.add(parts[1])
                else:
                    return None  # "O2" → all transitions
        return transitions if transitions else None

    def _target_pressures(self, gas_type: str, transition: str) -> set[str] | None:
        """从 targets 中提取某跃迁下涉及的压力集合，None 表示全部。"""
        if not self.targets:
            return None

        matched_transition = False
        pressures: set[str] = set()
        for t in self.targets:
            parts = t.strip("/").split("/")
            if not parts or parts[0] != gas_type:
                continue
            if len(parts) == 1:
                return None
            if len(parts) >= 2 and parts[1] == transition:
                matched_transition = True
                if len(parts) == 2:
                    return None
                pressures.add(parts[2])

        if matched_transition:
            return pressures
        return set()

    @staticmethod
    def _match_transition(dir_name: str, target_trans: set[str]) -> bool:
        """检查目录名是否匹配目标跃迁集合中的某一项。

        支持精确匹配和前缀匹配:
          - "9386.207642" in {"9386.207642"}  → True  (精确)
          - "9386.207642".startswith("9386.2076")  → True  (前缀)
        """
        for t in target_trans:
            if dir_name == t or dir_name.startswith(t):
                return True
        return False

    def _filter_tasks_by_specified_pressures(
        self,
        tasks: list[tuple[str, str, str, Path]],
        stage_name: str,
    ) -> list[tuple[str, str, str, Path]]:
        """按 --pressures 过滤任务压力，并提前提示缺失压力。

        仅当调用方显式启用时使用（例如从 Step 1 结果续跑的场景）；
        未在 --pressures 指定的跃迁保持原样。
        """
        if not self.multi_fit_pressures:
            return tasks

        # 先汇总可用压力，便于输出清晰的缺失提示
        available_map: dict[tuple[str, str], set[str]] = {}
        for gas_type, transition, pressure, _ in tasks:
            available_map.setdefault((gas_type, transition), set()).add(pressure)

        for (gas_type, transition), available in sorted(available_map.items()):
            specified = self._lookup_multi_fit_pressures(gas_type, transition)
            if specified is None:
                continue
            missing = [p for p in specified if p not in available]
            if missing:
                logger.warning(
                    f"  [{stage_name}] [{gas_type}/{transition}] "
                    f"以下指定压力在当前阶段输入中未找到: {', '.join(missing)}"
                )
                logger.warning(
                    f"    可用压力: {', '.join(sorted(available))}"
                )

        filtered: list[tuple[str, str, str, Path]] = []
        for gas_type, transition, pressure, path in tasks:
            specified = self._lookup_multi_fit_pressures(gas_type, transition)
            if specified is None or pressure in specified:
                filtered.append((gas_type, transition, pressure, path))

        return filtered

    @staticmethod
    def _filter_tasks_by_gas_types(
        tasks: list[tuple[str, str, str, Path]],
        allowed_gas_types: set[str] | None,
    ) -> list[tuple[str, str, str, Path]]:
        """按气体类型过滤任务列表。"""
        if not allowed_gas_types:
            return tasks
        return [
            (gas_type, transition, pressure, path)
            for gas_type, transition, pressure, path in tasks
            if gas_type in allowed_gas_types
        ]

    # ==============================================================
    # Step 1: 衰荡时间处理
    # ==============================================================
    def step1_ringdown(
        self,
        allowed_gas_types: set[str] | None = None,
    ) -> None:
        """Step 1: 原始衰荡数据 → ringdown_results.csv (多进程并行)"""
        from crds_process.preprocessing import discover_tasks

        logger.info("\n" + "=" * 60)
        logger.info("  Step 1 / 4 — 衰荡时间处理")
        logger.info("=" * 60)

        tasks = discover_tasks(self.raw_root)
        tasks = self._filter_tasks(tasks)
        tasks = self._filter_tasks_by_gas_types(tasks, allowed_gas_types)
        if not tasks:
            logger.error(f"  未在 {self.raw_root} 下找到数据")
            return

        logger.info(f"  发现 {len(tasks)} 个数据集, "
                     f"使用 {self.max_workers} 进程并行处理")

        futures = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            for gas_type, transition, pressure, data_dir in tasks:
                output_dir = self.ringdown_root / gas_type / transition / pressure
                fut = pool.submit(
                    _worker_ringdown,
                    data_dir, output_dir,
                    "sigma_clip", 3.0, 5,
                )
                futures.append((f"{gas_type}/{transition}/{pressure}", fut))

            for label, fut in futures:
                msg = fut.result()
                logger.info(msg)

    # ==============================================================
    # Step 2: 去除标准具
    # ==============================================================
    def step2_etalon(
        self,
        restrict_to_multi_fit_pressures: bool = False,
        allowed_gas_types: set[str] | None = None,
    ) -> None:
        """Step 2: ringdown_results → tau_etalon_corrected.csv (多进程并行)"""
        from crds_process.baseline.etalon import EtalonBatchProcessor

        logger.info("\n" + "=" * 60)
        logger.info("  Step 2 / 4 — 去除标准具效应")
        logger.info("=" * 60)

        # 遍历所有气体类型子目录
        all_tasks = []
        for gas_dir in sorted(self.ringdown_root.iterdir()):
            if not gas_dir.is_dir() or gas_dir.name.startswith("."):
                continue
            gas_type = gas_dir.name
            proc = EtalonBatchProcessor(
                ringdown_root=gas_dir,
                etalon_root=self.etalon_root / gas_type,
            )
            for t, p, csv in proc.discover():
                all_tasks.append((gas_type, t, p, csv))

        all_tasks = self._filter_tasks(all_tasks)
        all_tasks = self._filter_tasks_by_gas_types(
            all_tasks, allowed_gas_types)
        if restrict_to_multi_fit_pressures:
            all_tasks = self._filter_tasks_by_specified_pressures(
                all_tasks, stage_name="Step 2")
        if not all_tasks:
            logger.error(f"  未在 {self.ringdown_root} 下找到数据 "
                         f"(目标/压力过滤后)")
            return

        logger.info(f"  发现 {len(all_tasks)} 个数据集, "
                     f"使用 {self.max_workers} 进程并行处理")

        futures = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            for gas_type, transition, pressure, csv_path in all_tasks:
                output_dir = self.etalon_root / gas_type / transition / pressure
                label = f"{gas_type}/{transition}/{pressure}"
                fut = pool.submit(
                    _worker_etalon, csv_path, output_dir, label, gas_type,
                )
                futures.append((label, fut))

            for label, fut in futures:
                msg = fut.result()
                logger.info(msg)

    # ==============================================================
    # Step 3: MATS 单光谱拟合 + 汇总
    # ==============================================================
    def _base_fitter_kwargs(self) -> dict:
        """返回 MATSFitter 构造基础参数 (不含气体类型相关参数)"""
        return dict(
            molecule=self.molecule,
            isotopologue=self.isotopologue,
            lineprofile=self.lineprofile,
            baseline_order=self.baseline_order,
            fit_intensity=self.fit_intensity,
            threshold_intensity=self.threshold_intensity,
            refit_threshold=self.refit_threshold,
            allowed_nu=self._get_allowed_nu(),
        )

    def _get_allowed_nu(self) -> list[float] | None:
        """构建允许拟合的跃迁波数列表。

        优先级:
          1. 若显式指定 fit_transitions，直接使用该列表
          2. 否则从原始数据目录中提取所有跃迁波数

        返回值将传给 HitranLinelistBuilder.build()，
        使 HITRAN 线表仅保留这些目标谱线。
        """
        if self.fit_transitions:
            return self.fit_transitions

        if not self.raw_root.exists():
            return None
        nu_set: set[float] = set()
        for gas_dir in self.raw_root.iterdir():
            if not gas_dir.is_dir() or gas_dir.name.startswith("."):
                continue
            for t_dir in gas_dir.iterdir():
                if not t_dir.is_dir() or t_dir.name.startswith("."):
                    continue
                try:
                    nu_set.add(float(t_dir.name))
                except ValueError:
                    continue
        return sorted(nu_set) if nu_set else None

    def _fitter_kwargs_for_gas(self, gas_type: str,
                               pressure_label: str = "") -> dict:
        """根据气体类型构建完整的 MATSFitter 参数"""
        from crds_process.gas_config import parse_gas_dir
        kw = self._base_fitter_kwargs()
        gc = parse_gas_dir(pressure_label, gas_type)
        kw.update(gc.to_fitter_kwargs())
        return kw

    def step3_mats(
        self,
        restrict_to_multi_fit_pressures: bool = False,
        allowed_gas_types: set[str] | None = None,
    ) -> None:
        """Step 3: etalon_corrected → MATS 拟合 (多进程并行, 支持多气体类型)"""
        from crds_process.spectral.mats_wrapper import MATSBatchProcessor

        logger.info("\n" + "=" * 60)
        logger.info(f"  Step 3 / 4 — MATS 光谱拟合 (线形: {self.lineprofile})")
        logger.info("=" * 60)

        # 遍历所有气体类型子目录
        all_tasks = []
        for gas_dir in sorted(self.etalon_root.iterdir()):
            if not gas_dir.is_dir() or gas_dir.name.startswith("."):
                continue
            gas_type = gas_dir.name
            proc = MATSBatchProcessor(
                etalon_root=gas_dir,
                mats_root=self.mats_root / gas_type,
            )
            for t, p, csv in proc.discover():
                all_tasks.append((gas_type, t, p, csv))

        all_tasks = self._filter_tasks(all_tasks)
        all_tasks = self._filter_tasks_by_gas_types(
            all_tasks, allowed_gas_types)
        if restrict_to_multi_fit_pressures:
            all_tasks = self._filter_tasks_by_specified_pressures(
                all_tasks, stage_name="Step 3")
        if not all_tasks:
            logger.error(f"  未在 {self.etalon_root} 下找到数据 "
                         f"(目标/压力过滤后)")
            return

        logger.info(f"  发现 {len(all_tasks)} 个数据集, "
                     f"使用 {self.max_workers} 进程并行处理")

        futures = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            for gas_type, transition, pressure, csv_path in all_tasks:
                output_dir = self.mats_root / gas_type / transition / pressure
                label = f"{gas_type}/{transition}/{pressure}"
                # 每个任务根据气体类型构建独立的 fitter 参数
                fitter_kw = self._fitter_kwargs_for_gas(gas_type, pressure)
                fut = pool.submit(
                    _worker_mats, csv_path, output_dir, label, fitter_kw,
                )
                futures.append((label, fut))

            for label, fut in futures:
                msg = fut.result()
                logger.info(msg)

        # 汇总
        self._collect_final_summary()

    # ==============================================================
    # Step 4: 筛选线强离群点 + 多光谱联合拟合 (仅纯 O₂)
    # ==============================================================
    def step4_multi_fit(self) -> None:
        """Step 4: 根据 Step 3 单光谱结果筛除线强离群点，
        用剩余光谱做多光谱联合拟合。

        仅对纯 O₂ 数据执行联合拟合；
        O₂+N₂ 混合气的 N₂ 展宽由 Step 5 线性回归提取。
        """
        if not self.mats_root.exists():
            logger.error(f"   Step 3 结果不存在: {self.mats_root}")
            logger.error(f"         请先运行 Step 3")
            return

        logger.info("\n" + "=" * 60)
        logger.info(f"  Step 4 / 5 — 筛选 + 多光谱联合拟合 (σ={self.sw_sigma})")
        logger.info("=" * 60)

        for gas_dir in sorted(self.mats_root.iterdir()):
            if not gas_dir.is_dir() or gas_dir.name.startswith("."):
                continue
            gas_type = gas_dir.name

            # 目标过滤: 跳过不在 targets 中的气体类型
            target_gas = self._target_gas_types()
            if target_gas and gas_type not in target_gas:
                continue

            # O₂+N₂ 混合气: 跳过联合拟合, N₂ 展宽由 Step 5 线性回归提取
            if gas_type == "O2_N2":
                logger.info(f"\n  [{gas_type}] 跳过多光谱联合拟合"
                            f" (N₂ 展宽将在 Step 5 通过线性回归提取)")
                continue

            target_trans = self._target_transitions(gas_type)
            for t_dir in sorted(gas_dir.iterdir()):
                if not t_dir.is_dir() or t_dir.name.startswith("."):
                    continue
                if target_trans and not self._match_transition(
                        t_dir.name, target_trans):
                    continue
                self._process_transition_multi_fit(
                    gas_type, t_dir.name)

    def _process_transition_multi_fit(self, gas_type: str,
                                      transition: str) -> None:
        """对单个 (气体类型, 跃迁) 执行线强筛选 + 多光谱联合拟合

        仅用于纯 O₂ 数据。

        三种模式 (优先级从高到低):
          1. multi_fit_pressures 指定 → 直接使用指定压力
          2. auto_optimize_pressures → 枚举组合，选 QF 最大
          3. 默认 → MAD 线强筛选后拟合
        """
        t_dir = self.mats_root / gas_type / transition
        tag = f"{gas_type}/{transition}"

        # ---- 1. 收集 Step 3 各压力点的拟合线强 ----
        diluent_col = "gamma0_O2"
        records = self._collect_sw_records(t_dir, diluent_col)
        if not records:
            logger.info(f"\n  [{tag}] 未找到 Step 3 拟合结果，跳过")
            return

        # ---- 2. 确定参与联合拟合的压力 ----
        specified_pressures = self._lookup_multi_fit_pressures(
            gas_type, transition)

        if specified_pressures is not None:
            # 模式 1: 用户指定压力
            sw_df = pd.DataFrame(records)
            sw_df["keep"] = sw_df["pressure"].isin(specified_pressures)

            available = set(sw_df["pressure"].tolist())
            missing = [p for p in specified_pressures if p not in available]
            if missing:
                logger.warning(f"  [{tag}] 以下指定压力在 Step 3 结果中未找到: "
                               f"{', '.join(missing)}")
                logger.warning(f"    可用压力: {', '.join(sorted(available))}")

            kept = sw_df[sw_df["keep"]]
            logger.info(f"\n  [{tag}] 使用手动指定的 {len(kept)} 个压力点"
                        f" (跳过自动筛选)")
            logger.info(f"  指定压力: {', '.join(specified_pressures)}")
            if len(kept) < 2:
                logger.warning(f"   有效压力点 < 2，无法联合拟合，跳过")
                return

            self._do_multi_fit_and_save(
                gas_type, transition, sw_df, kept)

        elif self.auto_optimize_pressures:
            # 模式 2: 自动搜索最优压力组合
            self._optimize_pressure_combination(
                gas_type, transition, records)

        else:
            # 模式 3: 默认 MAD 筛选
            sw_df = self._screen_sw(records, tag)
            kept = sw_df[sw_df["keep"]]
            if len(kept) < 2:
                logger.warning(f"   保留点数 < 2，无法联合拟合，跳过")
                return

            self._do_multi_fit_and_save(
                gas_type, transition, sw_df, kept)

    def _do_multi_fit_and_save(
        self, gas_type: str, transition: str,
        sw_df: pd.DataFrame, kept: pd.DataFrame,
    ) -> None:
        """执行多光谱联合拟合并保存结果 (核心拟合流程)"""
        tag = f"{gas_type}/{transition}"

        etalon_csvs, labels = self._collect_etalon_csvs(
            gas_type, transition, kept)
        if len(etalon_csvs) < 2:
            logger.warning(f"   有效 etalon CSV < 2，跳过联合拟合")
            return

        logger.info(f"\n  开始多光谱联合拟合 ({len(etalon_csvs)} 条光谱)...")
        first_pressure = kept.iloc[0]["pressure"]
        fitter_kw = self._fitter_kwargs_for_gas(gas_type, first_pressure)
        base_kw = self._base_fitter_kwargs()
        base_kw.update(fitter_kw)

        from crds_process.spectral.mats_wrapper import MATSFitter
        fitter = MATSFitter(**base_kw)

        multi_out = self.mats_multi_root / gas_type / transition
        multi_out.mkdir(parents=True, exist_ok=True)

        try:
            result = fitter.fit_multi(
                etalon_csvs=etalon_csvs,
                labels=labels,
                output_dir=multi_out,
                dataset_name=f"{transition}_multi",
            )
            if (result and result.summary_df is not None
                    and not result.summary_df.empty):
                fitter.plot_result(
                    result, multi_out,
                    title=f"Multi-spectrum Fit — {tag} "
                          f"({len(etalon_csvs)} spectra)",
                )

            sw_df.to_csv(multi_out / "sw_screening.csv", index=False)
            self._save_multi_fit_summary(
                result, multi_out, gas_type, transition, labels)

        except Exception as e:
            logger.error(f"  多光谱联合拟合失败: {e}")
            logger.exception("  详细错误信息:")

    def _optimize_pressure_combination(
        self, gas_type: str, transition: str,
        records: list[dict],
    ) -> None:
        """枚举所有压力组合 (≥ min_multi_pressures 个)，
        使用多进程并行拟合，选取 QF 最大的组合作为最终结果。
        """
        tag = f"{gas_type}/{transition}"
        # 去重并保持顺序 (一个压力下可能有多条 sw_vary=True 的谱线)
        all_pressures = list(dict.fromkeys(r["pressure"] for r in records))
        n = len(all_pressures)
        min_k = self.min_multi_pressures

        if n < min_k:
            logger.warning(f"  [{tag}] 可用压力 ({n}) < 最少要求 ({min_k})，跳过")
            return

        # 生成所有组合: C(n, min_k), C(n, min_k+1), ..., C(n, n)
        all_combos: list[tuple[str, ...]] = []
        for k in range(min_k, n + 1):
            all_combos.extend(combinations(all_pressures, k))

        total = len(all_combos)
        n_workers = min(self.max_workers, total)
        logger.info(f"\n  [{tag}] 自动搜索最优压力组合")
        logger.info(f"  可用压力: {', '.join(all_pressures)}")
        logger.info(f"  组合数: {total} (最少 {min_k} 个, 最多 {n} 个)")
        logger.info(f"  并行进程: {n_workers}")
        logger.info(f"  {'─' * 60}")

        # 预构建 fitter 参数 (所有组合共享)
        first_pressure = all_pressures[0]
        fitter_kw = self._fitter_kwargs_for_gas(gas_type, first_pressure)
        base_kw = self._base_fitter_kwargs()
        base_kw.update(fitter_kw)

        # 预构建每个组合的 etalon CSV 路径和 labels
        combo_tasks: list[tuple[tuple[str, ...], list[Path], list[str]]] = []
        for combo in all_combos:
            etalon_csvs: list[Path] = []
            labels: list[str] = []
            for p in combo:
                csv_path = (self.etalon_root / gas_type / transition
                            / p / "tau_etalon_corrected.csv")
                if csv_path.exists():
                    etalon_csvs.append(csv_path)
                    labels.append(p)
            combo_tasks.append((combo, etalon_csvs, labels))

        # 多进程并行拟合
        results_log: list[tuple[tuple[str, ...], float, float]] = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = []
            for combo, etalon_csvs, labels in combo_tasks:
                fut = pool.submit(
                    _worker_trial_multi_fit,
                    etalon_csvs, labels, base_kw, transition, combo,
                )
                futures.append((combo, fut))

            for idx, (combo, fut) in enumerate(futures, 1):
                combo_result, qf, sw = fut.result()
                results_log.append((combo_result, qf, sw))
                combo_str = ", ".join(combo_result)
                status = f"QF = {qf:.2f}" if qf >= 0 else "失败"
                sw_str = f", S = {sw:.4e}" if sw > 0 else ""
                logger.info(f"  [{idx}/{total}] {combo_str}  →  {status}{sw_str}")

        # 找最优
        best_qf = -1.0
        best_combo: tuple[str, ...] | None = None
        for combo, qf, sw in results_log:
            if qf > best_qf:
                best_qf = qf
                best_combo = combo

        # 打印汇总排名
        logger.info(f"\n  {'═' * 60}")
        logger.info(f"  [{tag}] 压力组合搜索结果 (按 QF 降序)")
        logger.info(f"  {'═' * 60}")
        ranked = sorted(results_log, key=lambda x: x[1], reverse=True)
        for rank, (combo, qf, sw) in enumerate(ranked, 1):
            marker = " ← 最优" if combo == best_combo else ""
            qf_str = f"{qf:.2f}" if qf >= 0 else "失败"
            sw_str = f"  S={sw:.4e}" if sw > 0 else ""
            logger.info(f"  #{rank:<3d} QF={qf_str:<10s}{sw_str}"
                        f"  {', '.join(combo)}{marker}")
        logger.info(f"  {'═' * 60}")

        if best_combo is None or best_qf < 0:
            logger.warning(f"  [{tag}] 所有组合均拟合失败，跳过")
            return

        # 用最优组合执行正式拟合 (保存结果和图表)
        logger.info(f"\n  ★ 最优组合: {', '.join(best_combo)}  "
                    f"(QF = {best_qf:.2f})")
        logger.info(f"  正在用最优组合执行正式拟合...")

        sw_df = pd.DataFrame(records)
        sw_df["keep"] = sw_df["pressure"].isin(best_combo)
        kept = sw_df[sw_df["keep"]]

        # 保存搜索结果日志
        multi_out = self.mats_multi_root / gas_type / transition
        multi_out.mkdir(parents=True, exist_ok=True)
        search_rows = []
        for combo, qf, sw in ranked:
            search_rows.append({
                "pressures": "+".join(combo),
                "n_pressures": len(combo),
                "QF": qf,
                "sw": sw,
                "is_best": combo == best_combo,
            })
        pd.DataFrame(search_rows).to_csv(
            multi_out / "pressure_optimization.csv", index=False)

        self._do_multi_fit_and_save(
            gas_type, transition, sw_df, kept)

    # ==============================================================
    # Step 5: 线性回归提取 N₂ 展宽
    # ==============================================================
    def step5_linear_regression(self) -> None:
        """Step 5: 利用 O₂+N₂ 单光谱拟合结果 + 纯 O₂ 联合拟合结果，
        通过线性回归提取 N₂ 展宽 / 位移参数。

        物理基础:
            γ₀_total = γ₀_O₂ · x_O₂ + γ₀_N₂ · x_N₂

        固定 γ₀_O₂ (来自纯 O₂ 多光谱联合拟合)，
        对各混合比的 γ₀_air 做加权线性回归提取 γ₀_N₂。
        """
        from crds_process.spectral.linear_regression import N2BroadeningExtractor

        logger.info("\n" + "=" * 60)
        logger.info("  Step 5 / 5 — 线性回归提取 N₂ 展宽参数")
        logger.info("=" * 60)


        # 遍历 O₂+N₂ 的各个跃迁
        o2n2_dir = self.final_root / "O2_N2"
        if not o2n2_dir.exists():
            logger.warning("  未找到 O₂+N₂ 数据，跳过 Step 5")
            return

        # 目标跃迁过滤 (同时接受 O2_N2/xxx 和 O2/xxx 指定)
        target_trans_n2 = self._target_transitions("O2_N2")
        target_trans_o2 = self._target_transitions("O2")
        if target_trans_n2 is not None and target_trans_o2 is not None:
            target_trans = target_trans_n2 | target_trans_o2
        elif target_trans_n2 is not None:
            target_trans = target_trans_n2
        elif target_trans_o2 is not None:
            target_trans = target_trans_o2
        else:
            target_trans = None  # 无过滤，全部处理

        n_processed = 0
        for t_dir in sorted(o2n2_dir.iterdir()):
            if not t_dir.is_dir() or t_dir.name.startswith("."):
                continue
            transition = t_dir.name

            # 跳过不在目标列表中的跃迁
            if target_trans is not None and transition not in target_trans:
                continue

            # 检查所需文件
            mix_stats = t_dir / "fit_summary_statistics.csv"
            o2_multi = self.final_root / "O2" / transition / "multi_fit_result.csv"

            if not mix_stats.exists():
                logger.warning(f"  [{transition}] 未找到 O₂+N₂ 单光谱统计表: "
                               f"{mix_stats}")
                continue
            if not o2_multi.exists():
                logger.warning(f"  [{transition}] 未找到该跃迁对应的纯 O₂ "
                               f"多光谱联合拟合结果，跳过 N₂ 线性回归")
                logger.warning(f"    缺少: {o2_multi}")
                logger.warning(f"    ⚠ 请先对该跃迁运行纯 O₂ 的 Step 4 联合拟合")
                continue

            n_processed += 1
            logger.info(f"\n  [{transition}] 线性回归提取 N₂ 参数...")
            logger.info(f"    O₂+N₂ 数据: {mix_stats}")
            logger.info(f"    纯 O₂ 参考: {o2_multi}")

            specified_pressures = self._lookup_multi_fit_pressures(
                "O2_N2", transition)
            optimize_pressures = (
                self.auto_optimize_pressures and specified_pressures is None
            )
            if specified_pressures is not None:
                logger.info("    Step 5 压力: 使用手动指定压力 "
                            + ", ".join(specified_pressures))
            elif optimize_pressures:
                logger.info("    Step 5 压力: 自动搜索最优组合 "
                            f"(按 gamma0_N2 的 R², 最少 "
                            f"{max(self.min_multi_pressures, 3)} 个压力)")
            else:
                logger.info("    Step 5 压力: 使用全部可用压力")

            output_dir = t_dir  # 直接保存在 final/O2_N2/{transition}/ 下
            extractor = N2BroadeningExtractor(
                transition=transition,
                outlier_sigma=self.sw_sigma,
            )

            try:
                results = extractor.run(
                    mix_stats_csv=mix_stats,
                    o2_multi_csv=o2_multi,
                    output_dir=output_dir,
                    allowed_pressures=specified_pressures,
                    optimize_pressures=optimize_pressures,
                    min_pressures=max(self.min_multi_pressures, 3),
                )

                if results:
                    logger.info(f"\n  {'─' * 60}")
                    logger.info(f"  线性回归结果 ({transition}):")
                    logger.info(f"  {'─' * 60}")
                    for name, r in results.items():
                        logger.info(r.summary())
                    logger.info(f"  {'─' * 60}")
                else:
                    logger.warning(f"  [{transition}] 线性回归无有效结果")

            except Exception as e:
                logger.error(f"  [{transition}] 线性回归失败: {e}")
                logger.exception("  详细错误信息:")

        if n_processed == 0:
            logger.info("  未检测到符合条件的 O₂+N₂ 跃迁，跳过线性回归")

    # ==============================================================
    # 完整五步流水线
    # ==============================================================
    def run(self) -> None:
        """执行完整的 CRDS 五步处理流水线"""
        log_path = setup_logging()
        t0 = time.time()

        logger.info("=" * 60)
        logger.info("  CRDS 完整处理流水线")
        logger.info("  Step 1: 衰荡时间处理")
        logger.info("  Step 2: 去除标准具效应")
        logger.info("  Step 3: MATS 单光谱拟合 (各压力独立)")
        logger.info("  Step 4: 筛选 + 多光谱联合拟合 (纯 O₂)")
        logger.info("  Step 5: 线性回归提取 N₂ 展宽")
        logger.info("=" * 60)
        logger.info(f"  线形:     {self.lineprofile}")
        logger.info(f"  线强筛选: {self.sw_sigma}σ (Step 4)")
        logger.info(f"  并行进程: {self.max_workers}")
        logger.info(f"  原始数据: {self.raw_root}")
        if self.targets:
            logger.info(f"  指定目标: {', '.join(self.targets)}")
        else:
            logger.info(f"  指定目标: 全部")
        if self.fit_transitions:
            logger.info("  指定拟合跃迁: "
                        + ", ".join(f"{nu:.6f}" for nu in self.fit_transitions))
        if self.multi_fit_pressures:
            for key, pressures in self.multi_fit_pressures.items():
                logger.info(f"  联合拟合压力 [{key}]: {', '.join(pressures)}")
        if self.auto_optimize_pressures:
            logger.info(f"  自动搜索最优压力组合: 开启 "
                        f"(最少 {self.min_multi_pressures} 个压力)")
        logger.info(f"  日志文件: {log_path}")
        logger.info("=" * 60)

        # 原始数据命名检测
        self.validate_raw_data()

        self.step1_ringdown()
        self.step2_etalon()
        self.step3_mats()
        self.step4_multi_fit()
        self.step5_linear_regression()
        self._build_master_table()

        elapsed = time.time() - t0
        logger.info(f"\n{'#' * 60}")
        logger.info(f"  全部完成! 耗时 {elapsed:.1f} s")
        logger.info(f"{'#' * 60}")

    def run_n2_only(self) -> None:
        """仅处理 O2_N2 数据并提取 N2 展宽。

        执行 Step 1~3 和 Step 5，跳过纯 O2 的 Step 4 联合拟合。
        Step 5 仍依赖已有的纯 O2 多光谱联合拟合结果。
        """
        log_path = setup_logging()
        t0 = time.time()

        logger.info("=" * 60)
        logger.info("  CRDS 处理流水线 (仅提取 N₂ 展宽)")
        logger.info("  Step 1: 衰荡时间处理 (仅 O₂+N₂)")
        logger.info("  Step 2: 去除标准具效应 (仅 O₂+N₂)")
        logger.info("  Step 3: MATS 单光谱拟合 (仅 O₂+N₂)")
        logger.info("  ── 跳过 Step 4: 纯 O₂ 多光谱联合拟合")
        logger.info("  Step 5: 线性回归提取 N₂ 展宽")
        logger.info("=" * 60)
        logger.info(f"  线形:     {self.lineprofile}")
        logger.info(f"  线强筛选: {self.sw_sigma}σ (Step 5)")
        logger.info(f"  并行进程: {self.max_workers}")
        logger.info(f"  原始数据: {self.raw_root}")
        if self.targets:
            logger.info(f"  指定目标: {', '.join(self.targets)}")
        else:
            logger.info("  指定目标: 全部 O₂+N₂")
        if self.fit_transitions:
            logger.info("  指定拟合跃迁: "
                        + ", ".join(f"{nu:.6f}" for nu in self.fit_transitions))
        logger.info(f"  日志文件: {log_path}")
        logger.info("=" * 60)

        self.validate_raw_data()
        self.step1_ringdown(allowed_gas_types={"O2_N2"})
        self.step2_etalon(allowed_gas_types={"O2_N2"})
        self.step3_mats(allowed_gas_types={"O2_N2"})
        self.step5_linear_regression()
        self._build_master_table()

        elapsed = time.time() - t0
        logger.info(f"\n{'#' * 60}")
        logger.info(f"  全部完成! 耗时 {elapsed:.1f} s")
        logger.info(f"{'#' * 60}")

    def run_from_ringdown(self) -> None:
        """从已有的 Step 1 结果开始执行 Step 2~5

        跳过 Step 1 (衰荡时间处理)，直接使用 ringdown_root 下
        已有的 ringdown_results.csv 执行后续步骤。

        适用于:
          - 原始数据未变化，仅需重新做去标准具和后续拟合
          - Step 1 已耗时完成，不希望重复计算
        """
        log_path = setup_logging()
        t0 = time.time()

        logger.info("=" * 60)
        logger.info("  CRDS 处理流水线 (从 Step 1 结果开始)")
        logger.info("  ── 跳过 Step 1: 衰荡时间处理")
        logger.info("  Step 2: 去除标准具效应")
        logger.info("  Step 3: MATS 单光谱拟合 (各压力独立)")
        logger.info("  Step 4: 筛选 + 多光谱联合拟合 (纯 O₂)")
        logger.info("  Step 5: 线性回归提取 N₂ 展宽")
        logger.info("=" * 60)
        logger.info(f"  线形:     {self.lineprofile}")
        logger.info(f"  线强筛选: {self.sw_sigma}σ (Step 4)")
        logger.info(f"  并行进程: {self.max_workers}")
        logger.info(f"  Ringdown 数据: {self.ringdown_root}")
        if self.targets:
            logger.info(f"  指定目标: {', '.join(self.targets)}")
        else:
            logger.info(f"  指定目标: 全部")
        if self.fit_transitions:
            logger.info("  指定拟合跃迁: "
                        + ", ".join(f"{nu:.6f}" for nu in self.fit_transitions))
        if self.multi_fit_pressures:
            for key, pressures in self.multi_fit_pressures.items():
                logger.info(f"  联合拟合压力 [{key}]: {', '.join(pressures)}")
        if self.auto_optimize_pressures:
            logger.info(f"  自动搜索最优压力组合: 开启 "
                        f"(最少 {self.min_multi_pressures} 个压力)")
        logger.info(f"  日志文件: {log_path}")
        logger.info("=" * 60)

        # 检查 ringdown 数据是否存在
        if not self.ringdown_root.exists():
            logger.error(f"  Step 1 结果目录不存在: {self.ringdown_root}")
            logger.error("  请先运行完整流水线或先执行 Step 1")
            return

        self.step2_etalon(restrict_to_multi_fit_pressures=True)
        self.step3_mats(restrict_to_multi_fit_pressures=True)
        self.step4_multi_fit()
        self.step5_linear_regression()
        self._build_master_table()

        elapsed = time.time() - t0
        logger.info(f"\n{'#' * 60}")
        logger.info(f"  全部完成! 耗时 {elapsed:.1f} s")
        logger.info(f"{'#' * 60}")

    def run_n2_only_from_ringdown(self) -> None:
        """从已有 Step 1 结果开始，仅提取 N2 展宽。"""
        log_path = setup_logging()
        t0 = time.time()

        logger.info("=" * 60)
        logger.info("  CRDS 处理流水线 (从 Step 1 结果开始, 仅提取 N₂ 展宽)")
        logger.info("  ── 跳过 Step 1: 衰荡时间处理")
        logger.info("  Step 2: 去除标准具效应 (仅 O₂+N₂)")
        logger.info("  Step 3: MATS 单光谱拟合 (仅 O₂+N₂)")
        logger.info("  ── 跳过 Step 4: 纯 O₂ 多光谱联合拟合")
        logger.info("  Step 5: 线性回归提取 N₂ 展宽")
        logger.info("=" * 60)
        logger.info(f"  线形:     {self.lineprofile}")
        logger.info(f"  线强筛选: {self.sw_sigma}σ (Step 5)")
        logger.info(f"  并行进程: {self.max_workers}")
        logger.info(f"  Ringdown 数据: {self.ringdown_root}")
        if self.targets:
            logger.info(f"  指定目标: {', '.join(self.targets)}")
        else:
            logger.info("  指定目标: 全部 O₂+N₂")
        if self.fit_transitions:
            logger.info("  指定拟合跃迁: "
                        + ", ".join(f"{nu:.6f}" for nu in self.fit_transitions))
        if self.multi_fit_pressures:
            for key, pressures in self.multi_fit_pressures.items():
                logger.info(f"  联合拟合压力 [{key}]: {', '.join(pressures)}")
        logger.info(f"  日志文件: {log_path}")
        logger.info("=" * 60)

        if not self.ringdown_root.exists():
            logger.error(f"  Step 1 结果目录不存在: {self.ringdown_root}")
            logger.error("  请先运行完整流水线或先执行 Step 1")
            return

        self.step2_etalon(
            restrict_to_multi_fit_pressures=True,
            allowed_gas_types={"O2_N2"},
        )
        self.step3_mats(
            restrict_to_multi_fit_pressures=True,
            allowed_gas_types={"O2_N2"},
        )
        self.step5_linear_regression()
        self._build_master_table()

        elapsed = time.time() - t0
        logger.info(f"\n{'#' * 60}")
        logger.info(f"  全部完成! 耗时 {elapsed:.1f} s")
        logger.info(f"{'#' * 60}")

    def run_from_etalon(self) -> None:
        """从已有的去除标准具数据开始执行 Step 3~5

        跳过 Step 1 (衰荡时间处理) 和 Step 2 (去除标准具)，
        直接使用 etalon_root 下已有的 tau_etalon_corrected.csv
        执行后续的光谱拟合、联合拟合和线性回归。

        适用于:
          - 已手动调整过去除标准具后的数据，只需重新拟合
          - 只修改了拟合参数，不需要重新处理原始数据
        """
        log_path = setup_logging()
        t0 = time.time()

        logger.info("=" * 60)
        logger.info("  CRDS 处理流水线 (从去除标准具后数据开始)")
        logger.info("  ── 跳过 Step 1: 衰荡时间处理")
        logger.info("  ── 跳过 Step 2: 去除标准具效应")
        logger.info("  Step 3: MATS 单光谱拟合 (各压力独立)")
        logger.info("  Step 4: 筛选 + 多光谱联合拟合 (纯 O₂)")
        logger.info("  Step 5: 线性回归提取 N₂ 展宽")
        logger.info("=" * 60)
        logger.info(f"  线形:     {self.lineprofile}")
        logger.info(f"  线强筛选: {self.sw_sigma}σ (Step 4)")
        logger.info(f"  并行进程: {self.max_workers}")
        logger.info(f"  Etalon 数据: {self.etalon_root}")
        if self.targets:
            logger.info(f"  指定目标: {', '.join(self.targets)}")
        else:
            logger.info(f"  指定目标: 全部")
        if self.fit_transitions:
            logger.info("  指定拟合跃迁: "
                        + ", ".join(f"{nu:.6f}" for nu in self.fit_transitions))
        if self.multi_fit_pressures:
            for key, pressures in self.multi_fit_pressures.items():
                logger.info(f"  联合拟合压力 [{key}]: {', '.join(pressures)}")
        if self.auto_optimize_pressures:
            logger.info(f"  自动搜索最优压力组合: 开启 "
                        f"(最少 {self.min_multi_pressures} 个压力)")
        logger.info(f"  日志文件: {log_path}")
        logger.info("=" * 60)

        # 检查 etalon 数据是否存在
        if not self.etalon_root.exists():
            logger.error(f"  去除标准具数据目录不存在: {self.etalon_root}")
            logger.error(f"  请先运行完整流水线生成 etalon 数据")
            return

        self.step3_mats(restrict_to_multi_fit_pressures=True)
        self.step4_multi_fit()
        self.step5_linear_regression()
        self._build_master_table()

        elapsed = time.time() - t0
        logger.info(f"\n{'#' * 60}")
        logger.info(f"  全部完成! 耗时 {elapsed:.1f} s")
        logger.info(f"{'#' * 60}")

    def run_n2_only_from_etalon(self) -> None:
        """从已有去除标准具数据开始，仅提取 N2 展宽。"""
        log_path = setup_logging()
        t0 = time.time()

        logger.info("=" * 60)
        logger.info("  CRDS 处理流水线 (从去除标准具后数据开始, 仅提取 N₂ 展宽)")
        logger.info("  ── 跳过 Step 1: 衰荡时间处理")
        logger.info("  ── 跳过 Step 2: 去除标准具效应")
        logger.info("  Step 3: MATS 单光谱拟合 (仅 O₂+N₂)")
        logger.info("  ── 跳过 Step 4: 纯 O₂ 多光谱联合拟合")
        logger.info("  Step 5: 线性回归提取 N₂ 展宽")
        logger.info("=" * 60)
        logger.info(f"  线形:     {self.lineprofile}")
        logger.info(f"  线强筛选: {self.sw_sigma}σ (Step 5)")
        logger.info(f"  并行进程: {self.max_workers}")
        logger.info(f"  Etalon 数据: {self.etalon_root}")
        if self.targets:
            logger.info(f"  指定目标: {', '.join(self.targets)}")
        else:
            logger.info("  指定目标: 全部 O₂+N₂")
        if self.fit_transitions:
            logger.info("  指定拟合跃迁: "
                        + ", ".join(f"{nu:.6f}" for nu in self.fit_transitions))
        if self.multi_fit_pressures:
            for key, pressures in self.multi_fit_pressures.items():
                logger.info(f"  联合拟合压力 [{key}]: {', '.join(pressures)}")
        logger.info(f"  日志文件: {log_path}")
        logger.info("=" * 60)

        if not self.etalon_root.exists():
            logger.error(f"  去除标准具数据目录不存在: {self.etalon_root}")
            logger.error("  请先运行完整流水线生成 etalon 数据")
            return

        self.step3_mats(
            restrict_to_multi_fit_pressures=True,
            allowed_gas_types={"O2_N2"},
        )
        self.step5_linear_regression()
        self._build_master_table()

        elapsed = time.time() - t0
        logger.info(f"\n{'#' * 60}")
        logger.info(f"  全部完成! 耗时 {elapsed:.1f} s")
        logger.info(f"{'#' * 60}")

    # ==============================================================
    # Monte Carlo Type A
    # ==============================================================
    def run_type_a_monte_carlo(self) -> None:
        """基于已有纯 O2 多光谱联合拟合结果执行 Monte Carlo Type A 分析。"""
        log_path = setup_logging()
        t0 = time.time()

        logger.info("=" * 60)
        logger.info("  Monte Carlo Type A 误差分析")
        logger.info("=" * 60)
        logger.info("  说明: 读取已有纯 O2 多光谱联合拟合结果")
        logger.info(f"  线形:     {self.lineprofile}")
        logger.info(f"  抽样次数: {self.type_a_mc_samples}")
        logger.info(f"  随机种子: {self.type_a_mc_seed}")
        logger.info(f"  x 轴噪声: {self.type_a_mc_wave_error_khz / 1000.0:.3f} MHz")
        logger.info(f"  输出目录: {self.type_a_mc_root}")
        if self.targets:
            logger.info(f"  指定目标: {', '.join(self.targets)}")
        else:
            logger.info("  指定目标: 全部纯 O2 已拟合跃迁")
        logger.info(f"  日志文件: {log_path}")
        logger.info("=" * 60)

        target_trans = self._target_transitions("O2")
        if target_trans is None:
            o2_root = self.final_root / "O2"
            transitions = [
                p.name for p in sorted(o2_root.iterdir())
                if p.is_dir() and not p.name.startswith(".")
            ] if o2_root.exists() else []
        else:
            transitions = sorted(target_trans)

        if not transitions:
            logger.warning("  未找到可用于 Type A 的纯 O2 跃迁")
            return

        non_o2_targets = self._target_gas_types()
        if non_o2_targets and non_o2_targets - {"O2"}:
            logger.warning("  Type A Monte Carlo 当前只支持纯 O2 多光谱联合拟合结果")
            logger.warning("  非 O2 目标将被忽略")

        processed = 0
        transition_progress_t0 = time.time()
        _print_progress_bar("  跃迁进度", 0, len(transitions), transition_progress_t0)
        for idx, transition in enumerate(transitions, start=1):
            try:
                logger.info(f"\n  [{idx}/{len(transitions)}] 开始 O2/{transition}")
                self._run_transition_type_a_monte_carlo("O2", transition)
                processed += 1
            except Exception as exc:
                logger.error(f"  [O2/{transition}] Monte Carlo Type A 失败: {exc}")
                logger.exception("  详细错误信息:")
            finally:
                _print_progress_bar(
                    "  跃迁进度",
                    idx,
                    len(transitions),
                    transition_progress_t0,
                    finished=(idx == len(transitions)),
                )

        elapsed = time.time() - t0
        if processed == 0:
            logger.warning("  没有成功完成任何跃迁的 Type A 分析")
        logger.info(f"\n{'#' * 60}")
        logger.info(f"  Monte Carlo Type A 完成! 耗时 {elapsed:.1f} s")
        logger.info(f"{'#' * 60}")

    @staticmethod
    def _pressure_combo_slug(labels: list[str]) -> str:
        """将压力组合转换为稳定目录名。"""
        return "__".join(label.replace(" ", "_") for label in labels)

    @staticmethod
    def _wave_error_cm1_from_khz(freq_khz: float) -> float:
        """将频率噪声 (kHz) 转为波数噪声 (cm⁻1)。"""
        c_cm_s = 2.99792458e10
        return float(freq_khz) * 1e3 / c_cm_s

    @staticmethod
    def _select_target_line(param_df: pd.DataFrame, transition: str) -> pd.Series:
        """从参数线表中选出与目标跃迁最接近的拟合行。"""
        if param_df.empty:
            raise ValueError("参数线表为空")
        try:
            nu_target = float(transition)
        except ValueError as exc:
            raise ValueError(f"无效跃迁波数: {transition}") from exc

        candidates = param_df
        if "sw_vary" in param_df.columns:
            fitted = param_df[param_df["sw_vary"] == True]
            if not fitted.empty:
                candidates = fitted
        if "nu" not in candidates.columns:
            raise KeyError("参数线表缺少 nu 列")
        idx = (pd.to_numeric(candidates["nu"], errors="coerce") - nu_target).abs().idxmin()
        return candidates.loc[idx]

    @classmethod
    def _target_param_value(cls, row: pd.Series, param: str) -> float:
        """从目标行中读取参数值，线强自动恢复为物理单位。"""
        if param == "sw":
            scale = cls._safe_float(row.get("sw_scale_factor", 1.0))
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            return cls._safe_float(row.get("sw")) * scale
        return cls._safe_float(row.get(param))

    @classmethod
    def _target_param_error(cls, row: pd.Series, param: str) -> float:
        """从目标行中读取参数误差，线强自动恢复为物理单位。"""
        err_col = f"{param}_err"
        if param == "sw":
            scale = cls._safe_float(row.get("sw_scale_factor", 1.0))
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            return cls._safe_float(row.get(err_col)) * scale
        return cls._safe_float(row.get(err_col))

    @staticmethod
    def _extract_baseline_terms(row: pd.Series) -> list[float]:
        """从 baseline_paramlist 行中提取多项式系数。"""
        cols = [
            c for c in row.index
            if c.startswith("baseline_")
            and not c.endswith("_err")
            and not c.endswith("_vary")
        ]
        cols = sorted(cols, key=lambda name: name.split("_", 1)[1])
        terms = [float(row[c]) for c in cols]
        return terms if terms else [0.0]

    def _resolve_type_a_pressures(self, gas_type: str, transition: str) -> list[str]:
        """确定 Monte Carlo Type A 要复用的正式联合拟合压力组合。"""
        final_csv = self.final_root / gas_type / transition / "multi_fit_result.csv"
        if not final_csv.exists():
            raise FileNotFoundError(f"未找到正式多光谱联合拟合结果: {final_csv}")

        final_df = pd.read_csv(final_csv)
        if final_df.empty or "pressures" not in final_df.columns:
            raise ValueError(f"{final_csv} 缺少 pressures 列")

        pressures_raw = str(final_df.iloc[0]["pressures"]).strip()
        if not pressures_raw:
            raise ValueError(f"{final_csv} 中 pressures 为空")

        final_labels = [p.strip() for p in pressures_raw.split("+") if p.strip()]
        specified = self._lookup_multi_fit_pressures(gas_type, transition)
        if specified is None:
            return final_labels

        if set(specified) != set(final_labels):
            raise ValueError(
                "当前 Type A Monte Carlo 仅支持基于已存在的正式联合拟合压力组合。"
                f" 正式结果使用: {', '.join(final_labels)};"
                f" 当前命令指定: {', '.join(specified)}"
            )
        return [p for p in final_labels if p in specified]

    def _load_type_a_reference(self, gas_type: str, transition: str) -> dict:
        """加载 Monte Carlo Type A 参考拟合结果。"""
        multi_dir = self.mats_multi_root / gas_type / transition
        if not multi_dir.exists():
            raise FileNotFoundError(f"未找到多光谱联合拟合目录: {multi_dir}")

        labels = self._resolve_type_a_pressures(gas_type, transition)
        combo_slug = self._pressure_combo_slug(labels)

        param_files = sorted(multi_dir.glob("*Parameter_LineList*.csv"))
        base_files = sorted(multi_dir.glob("*baseline_paramlist*.csv"))
        summary_candidates = [
            p for p in sorted(multi_dir.glob("*.csv"))
            if "Parameter_LineList" not in p.name
            and "baseline_paramlist" not in p.name
            and "_linelist" not in p.name
            and "pressure_optimization" not in p.name
            and "sw_screening" not in p.name
            and p.name != "multi_fit_result.csv"
        ]

        if not param_files:
            raise FileNotFoundError(f"未找到 Parameter_LineList: {multi_dir}")
        if not base_files:
            raise FileNotFoundError(f"未找到 baseline_paramlist: {multi_dir}")
        if not summary_candidates:
            raise FileNotFoundError(f"未找到多光谱 summary CSV: {multi_dir}")

        param_df = pd.read_csv(param_files[0], index_col=0)
        base_df = pd.read_csv(base_files[0])
        summary_df = pd.read_csv(summary_candidates[0])

        target_row = self._select_target_line(param_df, transition)
        sim_param_df = param_df.copy()
        if "sw_scale_factor" in sim_param_df.columns:
            scale = pd.to_numeric(
                sim_param_df["sw_scale_factor"], errors="coerce"
            ).fillna(1.0)
            sim_param_df["sw"] = pd.to_numeric(
                sim_param_df["sw"], errors="coerce"
            ).fillna(0.0) * scale
            if "sw_err" in sim_param_df.columns:
                sim_param_df["sw_err"] = pd.to_numeric(
                    sim_param_df["sw_err"], errors="coerce"
                ).fillna(0.0) * scale
            sim_param_df["sw_scale_factor"] = 1.0

        reference_info: dict[str, dict] = {}
        if "Spectrum Number" not in summary_df.columns:
            raise KeyError("多光谱 summary 缺少 Spectrum Number 列")

        for spec_num, grp in summary_df.groupby("Spectrum Number"):
            spectrum_name = str(grp["Spectrum Name"].iloc[0]) if "Spectrum Name" in grp.columns else ""
            label = next((item for item in labels if item in spectrum_name), None)
            if label is None:
                continue

            etalon_csv = self.etalon_root / gas_type / transition / label / "tau_etalon_corrected.csv"
            if not etalon_csv.exists():
                raise FileNotFoundError(f"缺少去标准具数据: {etalon_csv}")
            etalon_df = pd.read_csv(etalon_csv)

            base_row = base_df[base_df["Spectrum Number"] == spec_num]
            if base_row.empty:
                raise ValueError(f"Spectrum Number {spec_num} 在 baseline_paramlist 中不存在")
            base_row = base_row.iloc[0]

            residual_std = float(np.std(grp["Residuals (ppm/cm)"].values))
            model_max = float(np.nanmax(np.abs(grp["Model (ppm/cm)"].values)))
            if not np.isfinite(model_max) or model_max <= 0:
                model_max = float(np.nanmax(np.abs(grp["Alpha (ppm/cm)"].values)))
            snr = model_max / residual_std if residual_std > 0 else 1e12

            reference_info[label] = {
                "spectrum_number": int(spec_num),
                "etalon_csv": etalon_csv,
                "wavenumbers": etalon_df["wavenumber"].astype(float).tolist(),
                "pressure_torr": float(grp["Pressure (Torr)"].mean()),
                "temperature_c": float(grp["Temperature (C)"].mean()),
                "residual_std": residual_std,
                "model_max": model_max,
                "snr": float(max(snr, 1.0)),
                "x_shift": float(base_row.get("x_shift", 0.0)),
                "baseline_terms": self._extract_baseline_terms(base_row),
            }

        missing = [label for label in labels if label not in reference_info]
        if missing:
            raise ValueError(
                "以下正式联合拟合压力无法在 summary 中建立映射: "
                + ", ".join(missing)
            )

        final_csv = self.final_root / gas_type / transition / "multi_fit_result.csv"
        final_df = pd.read_csv(final_csv)
        summary_row = final_df.iloc[0].to_dict() if not final_df.empty else {}

        return {
            "labels": labels,
            "combo_slug": combo_slug,
            "param_df": param_df,
            "sim_param_df": sim_param_df,
            "baseline_df": base_df,
            "summary_df": summary_df,
            "target_row": target_row,
            "reference_info": reference_info,
            "summary_row": summary_row,
            "param_file": param_files[0],
            "baseline_file": base_files[0],
            "summary_file": summary_candidates[0],
        }

    def _fit_type_a_mc_sample(
        self,
        fitter,
        spectra: list,
        param_linelist: pd.DataFrame,
        dataset_name: str,
        workdir: Path,
    ):
        """对一组模拟多光谱执行一次联合拟合。"""
        from crds_process.spectral.mats_wrapper import MATSFitResult, _import_mats

        _import_mats()
        from MATS import Dataset, Generate_FitParam_File, Fit_DataSet

        saved_dir = os.getcwd()
        os.chdir(str(workdir))
        try:
            ds = Dataset(spectra, dataset_name, param_linelist)
            base_linelist = ds.generate_baseline_paramlist()

            param_save = f"{dataset_name}_Parameter_LineList"
            base_save = f"{dataset_name}_baseline_paramlist"

            fitparam = Generate_FitParam_File(
                ds, param_linelist, base_linelist,
                lineprofile=self.lineprofile,
                linemixing=False,
                threshold_intensity=self.threshold_intensity,
                fit_intensity=self.fit_intensity,
                sim_window=5,
                param_linelist_savename=param_save,
                base_linelist_savename=base_save,
                nu_constrain=True, sw_constrain=True,
                gamma0_constrain=True, delta0_constrain=True,
                aw_constrain=True, as_constrain=True,
                nuVC_constrain=True, eta_constrain=True,
                linemixing_constrain=True,
            )
            fitparam.generate_fit_param_linelist_from_linelist(
                vary_nu={self.molecule: {self.isotopologue: False}},
                vary_sw={self.molecule: {self.isotopologue: True}},
                vary_gamma0={self.molecule: {self.isotopologue: True}},
                vary_n_gamma0={self.molecule: {self.isotopologue: True}},
                vary_delta0={self.molecule: {self.isotopologue: True}},
                vary_n_delta0={self.molecule: {self.isotopologue: False}},
                vary_aw={self.molecule: {self.isotopologue: True}},
                vary_n_gamma2={self.molecule: {self.isotopologue: False}},
                vary_as={self.molecule: {self.isotopologue: True}},
                vary_n_delta2={self.molecule: {self.isotopologue: False}},
                vary_nuVC={self.molecule: {self.isotopologue: False}},
                vary_n_nuVC={self.molecule: {self.isotopologue: False}},
                vary_eta={},
                vary_linemixing={self.molecule: {self.isotopologue: False}},
            )
            fitparam.generate_fit_baseline_linelist(
                vary_baseline=True,
                vary_pressure=False,
                vary_temperature=False,
                vary_molefraction={self.molecule: False},
                vary_xshift=True,
            )

            fit = Fit_DataSet(
                ds,
                fitparam.base_linelist_savename,
                fitparam.param_linelist_savename,
                minimum_parameter_fit_intensity=self.fit_intensity,
                weight_spectra=True,
            )
            params = fit.generate_params()
            fitter._apply_param_constraints(params)
            result = fit.fit_data(params, wing_cutoff=25)

            unreliable = fitter._find_unreliable_params(result)
            if unreliable:
                params2 = fit.generate_params()
                fitter._apply_param_constraints(params2)
                for pname, pval in unreliable.items():
                    if pname in params2:
                        params2[pname].set(value=pval, vary=False)
                result = fit.fit_data(params2, wing_cutoff=25)

            fit.residual_analysis(result, indv_resid_plot=False)
            fit.update_params(result)
            ds.generate_summary_file(save_file=True)
            summary, updated_params, updated_baseline, res_std = (
                fitter._read_fit_outputs(
                    dataset_name,
                    fitparam.param_linelist_savename,
                    fitparam.base_linelist_savename,
                )
            )
            return MATSFitResult(
                fit_result=result,
                param_linelist=updated_params,
                baseline_linelist=updated_baseline,
                summary_df=summary,
                residual_std=res_std,
                qf=float(ds.average_QF()) if hasattr(ds, "average_QF") else 0.0,
            )
        finally:
            os.chdir(saved_dir)

    def _run_transition_type_a_monte_carlo(self, gas_type: str, transition: str) -> None:
        """对单个跃迁执行 Monte Carlo Type A 分析。"""
        from crds_process.spectral.mats_wrapper import MATSFitter, _import_mats

        if gas_type != "O2":
            raise ValueError("当前仅支持纯 O2 的多光谱联合拟合 Type A 分析")

        ref = self._load_type_a_reference(gas_type, transition)
        output_dir = self.type_a_mc_root / gas_type / transition / ref["combo_slug"]
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n  [O2/{transition}] Monte Carlo Type A")
        logger.info(f"    压力组合: {', '.join(ref['labels'])}")
        logger.info(f"    抽样次数: {self.type_a_mc_samples}")

        fitter = MATSFitter(**self._fitter_kwargs_for_gas(gas_type, ref["labels"][0]))
        MATS = _import_mats()
        simulate_spectrum = MATS.simulate_spectrum
        Spectrum = MATS.Spectrum

        wave_error_cm1 = self._wave_error_cm1_from_khz(self.type_a_mc_wave_error_khz)

        # 保存参考输入信息
        input_rows = []
        for label in ref["labels"]:
            meta = ref["reference_info"][label]
            baseline_terms = meta["baseline_terms"]
            row = {
                "pressure": label,
                "pressure_torr": meta["pressure_torr"],
                "temperature_c": meta["temperature_c"],
                "n_points": len(meta["wavenumbers"]),
                "residual_std_ppm_cm": meta["residual_std"],
                "model_max_ppm_cm": meta["model_max"],
                "snr": meta["snr"],
                "x_shift_cm-1": meta["x_shift"],
            }
            for idx, coef in enumerate(baseline_terms):
                row[f"baseline_term_{idx}"] = coef
            input_rows.append(row)
        pd.DataFrame(input_rows).to_csv(output_dir / "mc_input_spectra.csv", index=False)

        ref_row = ref["target_row"].copy()
        ref_row["sw"] = self._target_param_value(ref["target_row"], "sw")
        ref_row["sw_err"] = self._target_param_error(ref["target_row"], "sw")
        ref_row["sw_scale_factor"] = 1.0
        pd.DataFrame([ref_row]).to_csv(output_dir / "reference_target_line.csv", index=False)

        settings_df = pd.DataFrame([{
            "gas_type": gas_type,
            "transition": transition,
            "pressures": "+".join(ref["labels"]),
            "lineprofile": self.lineprofile,
            "mc_samples": self.type_a_mc_samples,
            "mc_seed": self.type_a_mc_seed,
            "wave_error_khz": self.type_a_mc_wave_error_khz,
            "wave_error_cm-1": wave_error_cm1,
            "reference_residual_std": ref["summary_row"].get("residual_std", np.nan),
            "reference_QF": ref["summary_row"].get("QF", np.nan),
        }])
        settings_df.to_csv(output_dir / "mc_settings.csv", index=False)

        sample_rows: list[dict] = []
        sample_progress_t0 = time.time()
        _print_progress_bar(
            f"    样本进度 O2/{transition}",
            0,
            self.type_a_mc_samples,
            sample_progress_t0,
        )
        for sample_idx in range(self.type_a_mc_samples):
            sample_seed = self.type_a_mc_seed + sample_idx
            np.random.seed(sample_seed)

            try:
                with tempfile.TemporaryDirectory(
                    prefix=f"type_a_mc_{transition}_",
                    dir=output_dir,
                ) as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    spectra = []
                    for label in ref["labels"]:
                        meta = ref["reference_info"][label]
                        sim_name = tmp_path / f"{transition}_mc_{sample_idx:04d}_{label.replace(' ', '_')}"
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message="Mean of empty slice",
                                category=RuntimeWarning,
                            )
                            warnings.filterwarnings(
                                "ignore",
                                message="invalid value encountered in scalar divide",
                                category=RuntimeWarning,
                            )
                            simulate_spectrum(
                                ref["sim_param_df"].copy(),
                                wavenumbers=list(meta["wavenumbers"]),
                                wave_error=wave_error_cm1,
                                SNR=meta["snr"],
                                baseline_terms=meta["baseline_terms"],
                                temperature=meta["temperature_c"],
                                pressure=meta["pressure_torr"],
                                filename=str(sim_name),
                                molefraction=fitter.molefraction.copy(),
                                diluent=fitter.diluent,
                                Diluent=copy.deepcopy(fitter.Diluent),
                                nominal_temperature=296,
                                x_shift=meta["x_shift"],
                                IntensityThreshold=self.threshold_intensity,
                                num_segments=1,
                            )
                        sim_csv = Path(str(sim_name) + ".csv")
                        sim_df = pd.read_csv(sim_csv)
                        sim_df["Noise (%)"] = max(100.0 / max(meta["snr"], 1.0), 1e-12)
                        sim_df.to_csv(sim_csv, index=False)
                        spec = Spectrum(
                            str(sim_name),
                            molefraction=fitter.molefraction.copy(),
                            natural_abundance=True,
                            diluent=fitter.diluent,
                            Diluent=copy.deepcopy(fitter.Diluent),
                            input_freq=False,
                            input_tau=False,
                            pressure_column="Pressure (Torr)",
                            temperature_column="Temperature (C)",
                            frequency_column="Wavenumber + Noise (cm-1)",
                            tau_column="Alpha + Noise (ppm/cm)",
                            tau_stats_column="Noise (%)",
                            segment_column="Segment Number",
                            nominal_temperature=296,
                            x_shift=meta["x_shift"],
                            baseline_order=max(len(meta["baseline_terms"]) - 1, 0),
                        )
                        spectra.append(spec)

                    mc_result = self._fit_type_a_mc_sample(
                        fitter=fitter,
                        spectra=spectra,
                        param_linelist=ref["sim_param_df"].copy(),
                        dataset_name=f"{transition}_mc_{sample_idx:04d}",
                        workdir=tmp_path,
                    )
                    mc_row = self._select_target_line(
                        mc_result.param_linelist, transition)
                    sample_rec = {
                        "sample": sample_idx,
                        "seed": sample_seed,
                        "success": True,
                        "residual_std": mc_result.residual_std,
                        "QF": mc_result.qf,
                    }
                    for param in [
                        "sw", "gamma0_O2", "n_gamma0_O2",
                        "SD_gamma_O2", "delta0_O2",
                        "SD_delta_O2", "nuVC_O2",
                    ]:
                        sample_rec[param] = self._target_param_value(mc_row, param)
                    sample_rows.append(sample_rec)
            except Exception as exc:
                sample_rows.append({
                    "sample": sample_idx,
                    "seed": sample_seed,
                    "success": False,
                    "error": str(exc),
                })
            finally:
                current = sample_idx + 1
                _print_progress_bar(
                    f"    样本进度 O2/{transition}",
                    current,
                    self.type_a_mc_samples,
                    sample_progress_t0,
                    finished=(current == self.type_a_mc_samples),
                )

        samples_df = pd.DataFrame(sample_rows)
        samples_df.to_csv(output_dir / "mc_samples.csv", index=False)

        success_df = samples_df[samples_df["success"] == True].copy()
        summary_rows = []
        reference_row = ref["target_row"]
        for param in [
            "sw", "gamma0_O2", "n_gamma0_O2",
            "SD_gamma_O2", "delta0_O2",
            "SD_delta_O2", "nuVC_O2",
        ]:
            if param not in success_df.columns:
                continue
            values = pd.to_numeric(success_df[param], errors="coerce").dropna()
            if values.empty:
                continue
            fit_value = self._target_param_value(reference_row, param)
            fit_err = self._target_param_error(reference_row, param)
            mc_mean = float(values.mean())
            mc_std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary_rows.append({
                "parameter": param,
                "fit_value": fit_value,
                "fit_stderr": fit_err,
                "mc_mean": mc_mean,
                "mc_std": mc_std,
                "mc_rel_pct": (
                    abs(mc_std / fit_value) * 100.0
                    if np.isfinite(fit_value) and fit_value != 0 else np.nan
                ),
                "mc_to_fit_ratio": (
                    mc_std / fit_err
                    if np.isfinite(fit_err) and fit_err > 0 else np.nan
                ),
                "n_success": int(len(values)),
                "n_requested": int(self.type_a_mc_samples),
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "mc_summary.csv", index=False)

        logger.info(f"    成功样本: {len(success_df)}/{self.type_a_mc_samples}")
        logger.info(f"    输出目录: {output_dir}")
        if not summary_df.empty:
            logger.info("    关键结果:")
            for _, row in summary_df.iterrows():
                logger.info(
                    f"      {row['parameter']}: "
                    f"fit err = {row['fit_stderr']:.6e}, "
                    f"MC std = {row['mc_std']:.6e}, "
                    f"ratio = {row['mc_to_fit_ratio']:.3f}"
                )

    @staticmethod
    def _as_bool(value) -> bool:
        """稳健解析布尔值。"""
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if pd.isna(value):
            return False
        return str(value).strip().lower() in {"true", "1", "yes", "y"}

    @staticmethod
    def _safe_float(value) -> float:
        """将任意标量安全转换为 float，失败返回 NaN。"""
        try:
            result = float(value)
        except (TypeError, ValueError):
            return np.nan
        return result if np.isfinite(result) else np.nan

    @staticmethod
    def _closest_transition_row(
        df: pd.DataFrame,
        transition: str,
        nu_col: str = "nu_HITRAN",
    ) -> pd.Series | None:
        """返回与目标跃迁最接近的一行。"""
        if df.empty:
            return None
        if nu_col not in df.columns:
            return df.iloc[0]
        try:
            nu_target = float(transition)
        except (TypeError, ValueError):
            return df.iloc[0]

        nu_vals = pd.to_numeric(df[nu_col], errors="coerce")
        if nu_vals.notna().any():
            idx = (nu_vals - nu_target).abs().idxmin()
            return df.loc[idx]
        return df.iloc[0]

    def _select_target_line_rows(
        self,
        df: pd.DataFrame,
        transition: str,
    ) -> pd.DataFrame:
        """每个压力仅保留最接近目标跃迁的一行。"""
        if df.empty or "pressure" not in df.columns or "nu_HITRAN" not in df.columns:
            return df.copy()

        try:
            nu_target = float(transition)
        except (TypeError, ValueError):
            return df.copy()

        work = df.copy()
        work["_pressure"] = work["pressure"].astype(str)
        work["_nu_dist"] = (
            pd.to_numeric(work["nu_HITRAN"], errors="coerce") - nu_target
        ).abs()
        if not np.isfinite(work["_nu_dist"]).any():
            return df.copy()

        idx = work.groupby("_pressure")["_nu_dist"].idxmin()
        selected = work.loc[idx].copy()
        selected = selected.drop(columns=["_pressure", "_nu_dist"])
        return selected.reset_index(drop=True)

    def _filter_report_pressures(
        self,
        df: pd.DataFrame,
        gas_type: str,
        transition: str,
    ) -> pd.DataFrame:
        """按 targets 中指定的压力过滤报告数据。"""
        if df.empty or "pressure" not in df.columns:
            return df.copy()

        specified = self._target_pressures(gas_type, transition)
        if specified is None:
            return df.copy()
        if not specified:
            return df.iloc[0:0].copy()

        available = df["pressure"].astype(str).tolist()
        missing = [p for p in sorted(specified) if p not in set(available)]
        if missing:
            logger.warning(
                f"  [重测报告] [{gas_type}/{transition}] 以下指定压力未找到: "
                f"{', '.join(missing)}"
            )
            logger.warning(f"    可用压力: {', '.join(available)}")

        return df[df["pressure"].astype(str).isin(specified)].copy()

    def _evaluate_remeasure_metric(
        self,
        observed: float,
        expected: float,
        observed_err: float = np.nan,
        expected_err: float = np.nan,
        rel_threshold: float | None = None,
    ) -> tuple[bool, float, float]:
        """判断单个指标是否超出建议重测阈值。"""
        obs = self._safe_float(observed)
        exp = self._safe_float(expected)
        obs_err = self._safe_float(observed_err)
        exp_err = self._safe_float(expected_err)
        rel_limit = (
            self.remeasure_rel_threshold_o2
            if rel_threshold is None else max(float(rel_threshold), 0.0)
        )

        if not np.isfinite(obs) or not np.isfinite(exp):
            return False, np.nan, np.nan

        diff = abs(obs - exp)
        rel_diff_pct = np.nan
        if exp != 0:
            rel_diff_pct = diff / abs(exp) * 100.0

        sigma_ratio = np.nan
        err_terms = []
        if np.isfinite(obs_err) and obs_err > 0:
            err_terms.append(obs_err**2)
        if np.isfinite(exp_err) and exp_err > 0:
            err_terms.append(exp_err**2)
        if err_terms:
            combined_err = float(np.sqrt(sum(err_terms)))
            if combined_err > 0:
                sigma_ratio = diff / combined_err

        flagged = False
        if np.isfinite(rel_diff_pct):
            flagged |= rel_diff_pct > rel_limit * 100.0
        if np.isfinite(sigma_ratio):
            flagged |= sigma_ratio > self.remeasure_sigma_threshold

        return flagged, rel_diff_pct, sigma_ratio

    @staticmethod
    def _o2_sw_peer_reference(
        stat_df: pd.DataFrame,
        row_index,
    ) -> tuple[float, float]:
        """用同一跃迁下其他压力点的线强作为 O2 自比较参考。"""
        if "sw" not in stat_df.columns:
            return np.nan, np.nan

        peers = stat_df.loc[stat_df.index != row_index].copy()
        if peers.empty:
            return np.nan, np.nan

        sw_vals = pd.to_numeric(peers["sw"], errors="coerce")
        sw_vals = sw_vals[np.isfinite(sw_vals)]
        if sw_vals.empty:
            return np.nan, np.nan

        sw_ref = float(np.median(sw_vals))

        err_terms: list[float] = []
        if "sw_err" in peers.columns:
            sw_err = pd.to_numeric(peers["sw_err"], errors="coerce")
            sw_err = sw_err[np.isfinite(sw_err) & (sw_err > 0)]
            if not sw_err.empty:
                err_terms.append(float(np.median(sw_err)))

        if len(sw_vals) >= 2:
            peer_std = float(np.std(sw_vals.values, ddof=1))
            if np.isfinite(peer_std) and peer_std > 0:
                err_terms.append(peer_std)

        sw_ref_err = max(err_terms) if err_terms else np.nan
        return sw_ref, sw_ref_err

    @staticmethod
    def _pressure_sort_value(label: str) -> tuple[float, float, str]:
        """为压力标签生成稳定排序键。"""
        label = str(label)
        nums = re.findall(r"(\d+(?:\.\d+)?)", label)
        if not nums:
            return (float("inf"), float("inf"), label)
        first = float(nums[0])
        second = float(nums[1]) if len(nums) > 1 else 0.0
        return (first, second, label)

    @staticmethod
    def _pressure_filename(label: str, suffix: str = ".pdf") -> str:
        """将压力标签转换为稳定的报告文件名。"""
        safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(label).strip())
        safe = re.sub(r"_+", "_", safe).strip("_")
        return (safe or "pressure") + suffix

    @staticmethod
    def _build_measurement_groups(
        transitions: list[str],
        merge_gap: float = 1.0,
        single_margin: float = 1.0,
        merged_margin: float = 2.0,
    ) -> list[dict]:
        """将间隔较近的跃迁合并为一次测量，并给出推荐范围。"""
        parsed: list[tuple[float, str]] = []
        for text in transitions:
            try:
                parsed.append((float(text), str(text).strip()))
            except (TypeError, ValueError):
                continue

        if not parsed:
            return []

        parsed.sort(key=lambda item: item[0])
        groups: list[list[tuple[float, str]]] = [[parsed[0]]]
        for item in parsed[1:]:
            if item[0] - groups[-1][-1][0] < merge_gap:
                groups[-1].append(item)
            else:
                groups.append([item])

        measurement_groups: list[dict] = []
        for idx, group in enumerate(groups, start=1):
            left = group[0][0]
            right = group[-1][0]
            margin = merged_margin if len(group) > 1 else single_margin
            measurement_groups.append({
                "index": idx,
                "transition_label": " + ".join(item[1] for item in group),
                "transition_count": len(group),
                "start": left - margin,
                "end": right + margin,
                "group_min": left,
                "group_max": right,
            })

        return measurement_groups

    def _write_pressure_pdf(
        self,
        out_path: Path,
        gas_type: str,
        pressure: str,
        transitions: list[str],
    ) -> None:
        """导出一个压力对应的重测跃迁 PDF 清单。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        measurement_groups = self._build_measurement_groups(transitions)
        lines_per_page = 28
        if not transitions:
            transitions = ["(empty)"]
        if not measurement_groups:
            measurement_groups = [{
                "index": 1,
                "transition_label": "(empty)",
                "transition_count": 0,
                "start": np.nan,
                "end": np.nan,
                "group_min": np.nan,
                "group_max": np.nan,
            }]

        total_pages = max(
            1,
            (len(measurement_groups) + lines_per_page - 1) // lines_per_page,
        )

        with PdfPages(out_path) as pdf:
            for page_idx in range(total_pages):
                start = page_idx * lines_per_page
                end = start + lines_per_page
                page_groups = measurement_groups[start:end]

                fig = plt.figure(figsize=(8.27, 11.69))
                ax = fig.add_axes([0.06, 0.05, 0.88, 0.9])
                ax.axis("off")

                ax.text(
                    0.0,
                    0.98,
                    "Remeasure Transition List",
                    fontsize=18,
                    fontweight="bold",
                    va="top",
                    ha="left",
                )
                ax.text(
                    0.0,
                    0.94,
                    f"Gas: {gas_type}",
                    fontsize=12,
                    va="top",
                    ha="left",
                )
                ax.text(
                    0.0,
                    0.91,
                    f"Pressure: {pressure}",
                    fontsize=12,
                    va="top",
                    ha="left",
                )
                ax.text(
                    0.0,
                    0.88,
                    f"Transitions: {len(transitions)}",
                    fontsize=12,
                    va="top",
                    ha="left",
                )
                ax.text(
                    0.0,
                    0.85,
                    f"Measurements: {len(measurement_groups)}",
                    fontsize=11,
                    va="top",
                    ha="left",
                )
                ax.text(
                    0.0,
                    0.82,
                    (
                        "Rule: single line -> extend 1.000 cm^-1 on both sides; "
                        "gap < 1.000 cm^-1 -> merge, then extend 2.000 cm^-1 on both sides"
                    ),
                    fontsize=11,
                    va="top",
                    ha="left",
                )
                ax.text(
                    0.02,
                    0.78,
                    "#   Transition(s)                 Recommended Range (cm^-1)     Lines",
                    fontsize=10,
                    family="monospace",
                    fontweight="bold",
                    va="top",
                    ha="left",
                )

                y = 0.74
                step = 0.024
                for group in page_groups:
                    if np.isfinite(group["start"]) and np.isfinite(group["end"]):
                        range_text = f"{group['start']:.3f}-{group['end']:.3f}"
                    else:
                        range_text = "N/A"
                    if np.isfinite(group["transition_count"]):
                        count_text = f"{int(group['transition_count']):>2} line(s)"
                    else:
                        count_text = "N/A"
                    line_text = (
                        f"{int(group['index']):>2}. {group['transition_label']:<28} "
                        f"{range_text:<28} {count_text}"
                    )
                    ax.text(
                        0.02,
                        y,
                        line_text,
                        fontsize=9.5,
                        family="monospace",
                        va="top",
                        ha="left",
                    )
                    y -= step

                ax.text(
                    1.0,
                    0.02,
                    f"Page {page_idx + 1}/{total_pages}",
                    fontsize=10,
                    va="bottom",
                    ha="right",
                )

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    @staticmethod
    def _is_missing_master_value(value) -> bool:
        """判断主表中的参数值是否为空。"""
        if pd.isna(value):
            return True
        if isinstance(value, str) and not value.strip():
            return True
        return False

    @staticmethod
    def _issue_tokens(value) -> list[str]:
        """将 fit_issue 字段拆分为问题列表。"""
        if pd.isna(value):
            return []
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return []
        return [item for item in text.split(";") if item]

    @staticmethod
    def _transition_plan_key(value) -> str:
        """将跃迁波数规整到压力计划表使用的显示精度。"""
        text = str(value).strip()
        if not text:
            return ""
        try:
            return f"{float(text):.2f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            return text

    @staticmethod
    def _normalize_pressure_label(value) -> str:
        """将压力值规整为目录中使用的标签格式。"""
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return ""
        if re.search(r"[A-Za-z]", text):
            return text
        try:
            number = float(text)
        except (TypeError, ValueError):
            return text
        if number.is_integer():
            return f"{int(number)}Torr"
        return f"{number:g}Torr"

    def _load_o2_remeasure_plan(self) -> dict[str, set[str]]:
        """读取纯 O2 重测压力计划表。"""
        if self._o2_remeasure_plan_cache is not None:
            return self._o2_remeasure_plan_cache

        plan: dict[str, set[str]] = {}
        if not _O2_REMEASURE_PLAN_CSV.exists():
            logger.warning(
                f"  建议重测报告缺少纯 O2 压力计划表: {_O2_REMEASURE_PLAN_CSV}"
            )
            self._o2_remeasure_plan_cache = plan
            return plan

        try:
            plan_df = pd.read_csv(_O2_REMEASURE_PLAN_CSV, dtype=str)
        except Exception as exc:
            logger.warning(f"  读取纯 O2 压力计划表失败: {exc}")
            self._o2_remeasure_plan_cache = plan
            return plan

        pressure_cols = [
            col for col in plan_df.columns
            if col.lower().startswith("pressure")
        ]
        for _, row in plan_df.iterrows():
            transition_key = self._transition_plan_key(row.get("transition", ""))
            if not transition_key:
                continue
            allowed = {
                self._normalize_pressure_label(row.get(col, ""))
                for col in pressure_cols
            }
            allowed = {item for item in allowed if item}
            if allowed:
                plan[transition_key] = allowed

        self._o2_remeasure_plan_cache = plan
        return plan

    def _filter_o2_remeasure_plan_pressures(
        self,
        stat_df: pd.DataFrame,
        transition: str,
    ) -> pd.DataFrame:
        """纯 O2 建议重测仅在压力计划表定义的压力集合内判断。"""
        if stat_df.empty or "pressure" not in stat_df.columns:
            return stat_df

        plan = self._load_o2_remeasure_plan()
        transition_key = self._transition_plan_key(transition)
        allowed = plan.get(transition_key)
        if not allowed:
            logger.warning(
                f"  [重测报告] [O2/{transition}] 未在纯 O2 压力计划表中找到，"
                "跳过该跃迁的压力点重测判断"
            )
            return stat_df.iloc[0:0].copy()

        return stat_df[stat_df["pressure"].astype(str).isin(allowed)].copy()

    def _collect_missing_master_params(self) -> dict[tuple[str, str], list[str]]:
        """从 spectral_parameters.csv 收集漏测的核心参数。"""
        master_csv = self.final_root / self._MASTER_TABLE_NAME
        if not master_csv.exists():
            logger.warning(f"  建议重测报告缺少主表: {master_csv}")
            return {}

        try:
            master_df = pd.read_csv(master_csv, dtype={"nu": str})
        except Exception as exc:
            logger.warning(f"  读取主表失败，跳过漏测参数检查: {exc}")
            return {}

        if master_df.empty or "nu" not in master_df.columns:
            return {}

        missing_map: dict[tuple[str, str], list[str]] = {}
        target_gases = self._target_gas_types()
        target_trans_o2 = self._target_transitions("O2")
        target_trans_o2n2 = self._target_transitions("O2_N2")

        for _, row in master_df.iterrows():
            transition = str(row.get("nu", "")).strip()
            if not transition:
                continue

            if target_gases is None or "O2" in target_gases:
                if target_trans_o2 is None or transition in target_trans_o2:
                    missing_params: list[str] = []
                    for param in ["sw", "gamma0_O2"]:
                        if param in row.index and self._is_missing_master_value(row.get(param)):
                            missing_params.append(param)
                    if missing_params:
                        missing_map[("O2", transition)] = missing_params

            if target_gases is None or "O2_N2" in target_gases:
                if target_trans_o2n2 is None or transition in target_trans_o2n2:
                    missing_params = []
                    if ("gamma0_N2" in row.index
                            and self._is_missing_master_value(row.get("gamma0_N2"))):
                        missing_params.append("gamma0_N2")
                    if missing_params:
                        missing_map[("O2_N2", transition)] = missing_params

        return missing_map

    def _build_remeasure_rows_o2(self) -> list[dict]:
        """生成纯 O2 建议重测点列表。"""
        rows: list[dict] = []
        o2_root = self.final_root / "O2"
        if not o2_root.exists():
            return rows

        target_trans = self._target_transitions("O2")
        for t_dir in sorted(o2_root.iterdir()):
            if not t_dir.is_dir() or t_dir.name.startswith("."):
                continue
            transition = t_dir.name
            if target_trans is not None and transition not in target_trans:
                continue

            stat_csv = t_dir / "fit_summary_statistics.csv"
            if not stat_csv.exists():
                logger.warning(f"  [O2/{transition}] 缺少统计表: {stat_csv}")
                continue

            try:
                stat_df = pd.read_csv(stat_csv)
            except Exception as exc:
                logger.warning(f"  [O2/{transition}] 读取统计表失败: {exc}")
                continue
            if stat_df.empty:
                continue

            stat_df = self._select_target_line_rows(stat_df, transition)
            stat_df = self._filter_report_pressures(stat_df, "O2", transition)
            stat_df = self._filter_o2_remeasure_plan_pressures(stat_df, transition)
            if stat_df.empty:
                continue

            has_fit_valid = "fit_valid" in stat_df.columns
            has_fit_issue = "fit_issue" in stat_df.columns

            for idx, row in stat_df.iterrows():
                raw_issue_tokens = (
                    self._issue_tokens(row.get("fit_issue"))
                    if has_fit_issue else []
                )
                # 纯 O2 重测只看线强相关问题，不把 gamma0_O2 问题计入。
                fit_issue_tokens = [
                    token for token in raw_issue_tokens
                    if token in {"missing_sw_err", "sw_near_lower_bound"}
                ]
                if has_fit_issue:
                    fit_valid = len(fit_issue_tokens) == 0
                else:
                    fit_valid = (
                        self._as_bool(row.get("fit_valid"))
                        if has_fit_valid else True
                    )
                fit_issue = ";".join(fit_issue_tokens)

                failed_metrics: list[str] = []
                sw_ref = np.nan
                sw_ref_err = np.nan
                sw_rel = np.nan
                sw_sigma = np.nan

                if not fit_valid:
                    failed_metrics.append("fit_valid")

                sw_ref, sw_ref_err = self._o2_sw_peer_reference(stat_df, idx)
                sw_flagged, sw_rel, sw_sigma = self._evaluate_remeasure_metric(
                    observed=row.get("sw"),
                    expected=sw_ref,
                    observed_err=row.get("sw_err"),
                    expected_err=sw_ref_err,
                    rel_threshold=self.remeasure_rel_threshold_o2,
                )
                if sw_flagged:
                    failed_metrics.append("sw")

                if not failed_metrics:
                    continue

                rows.append({
                    "gas_type": "O2",
                    "transition": transition,
                    "pressure": str(row.get("pressure", "")),
                    "recommend_remeasure": True,
                    "failed_metrics": ";".join(sorted(set(failed_metrics))),
                    "fit_valid": fit_valid,
                    "fit_issue": fit_issue,
                    "residual_std": self._safe_float(row.get("residual_std")),
                    "sw_value": self._safe_float(row.get("sw")),
                    "sw_err": self._safe_float(row.get("sw_err")),
                    "sw_ref": sw_ref,
                    "sw_ref_err": sw_ref_err,
                    "sw_rel_diff_pct": sw_rel,
                    "sw_sigma_ratio": sw_sigma,
                    "gamma0_O2_value": self._safe_float(row.get("gamma0_O2")),
                    "gamma0_O2_err": self._safe_float(row.get("gamma0_O2_err")),
                    "gamma0_O2_ref": np.nan,
                    "gamma0_O2_ref_err": np.nan,
                    "gamma0_O2_rel_diff_pct": np.nan,
                    "gamma0_O2_sigma_ratio": np.nan,
                    "gamma0_air_value": np.nan,
                    "gamma0_air_err": np.nan,
                    "gamma0_air_model": np.nan,
                    "gamma0_air_model_err": np.nan,
                    "gamma0_air_rel_diff_pct": np.nan,
                    "gamma0_air_sigma_ratio": np.nan,
                    "gamma0_N2_ref": np.nan,
                    "gamma0_N2_ref_err": np.nan,
                    "x_O2": 1.0,
                    "x_N2": 0.0,
                })

        return rows

    def _build_remeasure_rows_o2n2(self) -> list[dict]:
        """生成 O2_N2 建议重测点列表。"""
        rows: list[dict] = []
        mix_root = self.final_root / "O2_N2"
        if not mix_root.exists():
            return rows

        target_trans = self._target_transitions("O2_N2")
        for t_dir in sorted(mix_root.iterdir()):
            if not t_dir.is_dir() or t_dir.name.startswith("."):
                continue
            transition = t_dir.name
            if target_trans is not None and transition not in target_trans:
                continue

            stat_csv = t_dir / "fit_summary_statistics.csv"
            if not stat_csv.exists():
                logger.warning(f"  [O2_N2/{transition}] 缺少统计表: {stat_csv}")
                continue

            try:
                stat_df = pd.read_csv(stat_csv)
            except Exception as exc:
                logger.warning(f"  [O2_N2/{transition}] 读取统计表失败: {exc}")
                continue
            if stat_df.empty:
                continue

            stat_df = self._select_target_line_rows(stat_df, transition)
            stat_df = self._filter_report_pressures(stat_df, "O2_N2", transition)
            if stat_df.empty:
                continue

            has_fit_valid = "fit_valid" in stat_df.columns
            has_fit_issue = "fit_issue" in stat_df.columns
            o2_ref_row = None
            o2_multi_csv = self.final_root / "O2" / transition / "multi_fit_result.csv"
            if o2_multi_csv.exists():
                try:
                    o2_df = pd.read_csv(o2_multi_csv)
                    o2_ref_row = self._closest_transition_row(o2_df, transition)
                except Exception as exc:
                    logger.warning(f"  [O2_N2/{transition}] 读取纯 O2 参考失败: {exc}")
            else:
                logger.warning(f"  [O2_N2/{transition}] 缺少纯 O2 参考: {o2_multi_csv}")

            sw_ref = np.nan
            sw_ref_err = np.nan
            if o2_ref_row is not None:
                sw_ref = self._safe_float(o2_ref_row.get("sw"))
                sw_ref_err = self._safe_float(o2_ref_row.get("sw_err"))

            for _, row in stat_df.iterrows():
                fit_valid = (
                    self._as_bool(row.get("fit_valid"))
                    if has_fit_valid else True
                )
                fit_issue = str(row.get("fit_issue", "")).strip() if has_fit_issue else ""
                if fit_issue.lower() == "nan":
                    fit_issue = ""

                failed_metrics: list[str] = []
                if not fit_valid:
                    failed_metrics.append("fit_valid")

                x_o2 = np.nan
                x_n2 = np.nan
                sw_rel = np.nan
                sw_sigma = np.nan

                pressure_label = str(row.get("pressure", ""))
                try:
                    gas_cfg = parse_gas_dir(pressure_label, "O2_N2")
                    x_o2 = gas_cfg.o2_fraction
                    x_n2 = gas_cfg.n2_fraction
                except Exception as exc:
                    logger.warning(
                        f"  [O2_N2/{transition}] 无法解析压力标签 '{pressure_label}': {exc}"
                    )

                sw_flagged, sw_rel, sw_sigma = self._evaluate_remeasure_metric(
                    observed=row.get("sw"),
                    expected=sw_ref,
                    observed_err=row.get("sw_err"),
                    expected_err=sw_ref_err,
                    rel_threshold=self.remeasure_rel_threshold_o2n2,
                )
                if sw_flagged:
                    failed_metrics.append("sw")

                if not failed_metrics:
                    continue

                rows.append({
                    "gas_type": "O2_N2",
                    "transition": transition,
                    "pressure": pressure_label,
                    "recommend_remeasure": True,
                    "failed_metrics": ";".join(sorted(set(failed_metrics))),
                    "fit_valid": fit_valid,
                    "fit_issue": fit_issue,
                    "residual_std": self._safe_float(row.get("residual_std")),
                    "sw_value": self._safe_float(row.get("sw")),
                    "sw_err": self._safe_float(row.get("sw_err")),
                    "sw_ref": sw_ref,
                    "sw_ref_err": sw_ref_err,
                    "sw_rel_diff_pct": sw_rel,
                    "sw_sigma_ratio": sw_sigma,
                    "gamma0_O2_value": np.nan,
                    "gamma0_O2_err": np.nan,
                    "gamma0_O2_ref": np.nan,
                    "gamma0_O2_ref_err": np.nan,
                    "gamma0_O2_rel_diff_pct": np.nan,
                    "gamma0_O2_sigma_ratio": np.nan,
                    "gamma0_air_value": self._safe_float(row.get("gamma0_air")),
                    "gamma0_air_err": self._safe_float(row.get("gamma0_air_err")),
                    "gamma0_air_model": np.nan,
                    "gamma0_air_model_err": np.nan,
                    "gamma0_air_rel_diff_pct": np.nan,
                    "gamma0_air_sigma_ratio": np.nan,
                    "gamma0_N2_ref": np.nan,
                    "gamma0_N2_ref_err": np.nan,
                    "x_O2": x_o2,
                    "x_N2": x_n2,
                })

        return rows

    def generate_remeasure_report(self) -> None:
        """读取已有最终结果，生成建议重测点报告。"""
        log_path = setup_logging()
        point_out_path = self.final_root / "remeasure_candidates.csv"
        transition_out_path = self.final_root / "remeasure_transitions.csv"
        transition_o2_out_path = self.final_root / "remeasure_transitions_O2.csv"
        transition_o2n2_out_path = self.final_root / "remeasure_transitions_O2_N2.csv"
        pressure_out_path = self.final_root / "remeasure_pressures.csv"
        pressure_o2_out_path = self.final_root / "remeasure_pressures_O2.csv"
        pressure_o2n2_out_path = self.final_root / "remeasure_pressures_O2_N2.csv"
        pressure_lists_root = self.final_root / "remeasure_pressure_lists"

        logger.info("=" * 60)
        logger.info("  建议重测点报告")
        logger.info("=" * 60)
        logger.info("  数据来源: 已有 final/ 结果")
        logger.info(
            "  相对偏差阈值: "
            f"O2={self.remeasure_rel_threshold_o2 * 100:.1f}%, "
            f"O2_N2={self.remeasure_rel_threshold_o2n2 * 100:.1f}%"
        )
        logger.info(f"  sigma 阈值: {self.remeasure_sigma_threshold:.1f}σ")
        logger.info(f"  纯 O2 压力计划表: {_O2_REMEASURE_PLAN_CSV}")
        if self.targets:
            logger.info(f"  指定目标: {', '.join(self.targets)}")
        else:
            logger.info("  指定目标: 全部")
        logger.info(f"  日志文件: {log_path}")
        logger.info("=" * 60)

        rows: list[dict] = []
        target_gases = self._target_gas_types()
        if target_gases is None or "O2" in target_gases:
            rows.extend(self._build_remeasure_rows_o2())
        if target_gases is None or "O2_N2" in target_gases:
            rows.extend(self._build_remeasure_rows_o2n2())

        columns = [
            "gas_type",
            "transition",
            "pressure",
            "recommend_remeasure",
            "failed_metrics",
            "fit_valid",
            "fit_issue",
            "residual_std",
            "sw_value",
            "sw_err",
            "sw_ref",
            "sw_ref_err",
            "sw_rel_diff_pct",
            "sw_sigma_ratio",
            "gamma0_O2_value",
            "gamma0_O2_err",
            "gamma0_O2_ref",
            "gamma0_O2_ref_err",
            "gamma0_O2_rel_diff_pct",
            "gamma0_O2_sigma_ratio",
            "gamma0_air_value",
            "gamma0_air_err",
            "gamma0_air_model",
            "gamma0_air_model_err",
            "gamma0_air_rel_diff_pct",
            "gamma0_air_sigma_ratio",
            "gamma0_N2_ref",
            "gamma0_N2_ref_err",
            "x_O2",
            "x_N2",
        ]

        report_df = pd.DataFrame(rows, columns=columns)
        if not report_df.empty:
            report_df["_transition_num"] = pd.to_numeric(
                report_df["transition"], errors="coerce")
            pressure_sort = report_df["pressure"].map(self._pressure_sort_value)
            report_df["_pressure_1"] = pressure_sort.map(lambda item: item[0])
            report_df["_pressure_2"] = pressure_sort.map(lambda item: item[1])
            report_df = report_df.sort_values(
                ["gas_type", "_transition_num", "transition",
                 "_pressure_1", "_pressure_2", "pressure"],
                na_position="last",
            ).drop(columns=["_transition_num", "_pressure_1", "_pressure_2"])

        self.final_root.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(point_out_path, index=False)

        pressure_columns = [
            "gas_type",
            "pressure",
            "recommend_remeasure",
            "n_transitions",
            "transitions",
            "failed_metrics",
            "has_invalid_fit",
            "worst_sw_rel_diff_pct",
            "worst_sw_sigma_ratio",
            "worst_gamma0_air_rel_diff_pct",
            "worst_gamma0_air_sigma_ratio",
        ]
        if report_df.empty:
            pressure_df = pd.DataFrame(columns=pressure_columns)
        else:
            pressure_rows: list[dict] = []
            for (gas_type, pressure), sub in report_df.groupby(
                ["gas_type", "pressure"], sort=False
            ):
                transitions = sorted(
                    sub["transition"].astype(str).tolist(),
                    key=lambda item: (
                        pd.to_numeric(pd.Series([item]), errors="coerce").iloc[0]
                        if pd.notna(pd.to_numeric(pd.Series([item]), errors="coerce").iloc[0])
                        else float("inf"),
                        item,
                    ),
                )
                metric_union = sorted({
                    item
                    for value in sub["failed_metrics"].astype(str)
                    for item in value.split(";")
                    if item and item.lower() != "nan"
                })
                pressure_rows.append({
                    "gas_type": gas_type,
                    "pressure": pressure,
                    "recommend_remeasure": True,
                    "n_transitions": int(len(transitions)),
                    "transitions": ";".join(transitions),
                    "failed_metrics": ";".join(metric_union),
                    "has_invalid_fit": bool(
                        (~sub["fit_valid"].map(self._as_bool)).any()
                    ),
                    "worst_sw_rel_diff_pct": pd.to_numeric(
                        sub["sw_rel_diff_pct"], errors="coerce").max(),
                    "worst_sw_sigma_ratio": pd.to_numeric(
                        sub["sw_sigma_ratio"], errors="coerce").max(),
                    "worst_gamma0_air_rel_diff_pct": pd.to_numeric(
                        sub["gamma0_air_rel_diff_pct"], errors="coerce").max(),
                    "worst_gamma0_air_sigma_ratio": pd.to_numeric(
                        sub["gamma0_air_sigma_ratio"], errors="coerce").max(),
                })

            pressure_df = pd.DataFrame(pressure_rows, columns=pressure_columns)
            pressure_sort = pressure_df["pressure"].map(self._pressure_sort_value)
            pressure_df["_pressure_1"] = pressure_sort.map(lambda item: item[0])
            pressure_df["_pressure_2"] = pressure_sort.map(lambda item: item[1])
            pressure_df = pressure_df.sort_values(
                ["gas_type", "_pressure_1", "_pressure_2", "pressure"],
                na_position="last",
            ).drop(columns=["_pressure_1", "_pressure_2"])

        pressure_df.to_csv(pressure_out_path, index=False)
        pressure_df[
            pressure_df["gas_type"] == "O2"
        ].to_csv(pressure_o2_out_path, index=False)
        pressure_df[
            pressure_df["gas_type"] == "O2_N2"
        ].to_csv(pressure_o2n2_out_path, index=False)

        if pressure_lists_root.exists():
            shutil.rmtree(pressure_lists_root)
        pressure_lists_root.mkdir(parents=True, exist_ok=True)
        for _, row in pressure_df.iterrows():
            gas_type = str(row.get("gas_type", "")).strip()
            pressure = str(row.get("pressure", "")).strip()
            transitions_text = str(row.get("transitions", "")).strip()
            if not gas_type or not pressure:
                continue

            out_dir = pressure_lists_root / gas_type
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / self._pressure_filename(pressure)

            transitions = [
                item.strip() for item in transitions_text.split(";")
                if item.strip()
            ]
            self._write_pressure_pdf(out_path, gas_type, pressure, transitions)

        transition_columns = [
            "gas_type",
            "transition",
            "recommend_remeasure",
            "n_bad_points",
            "bad_pressures",
            "failed_metrics",
            "has_invalid_fit",
            "has_missing_params",
            "missing_params",
            "n_missing_params",
            "worst_sw_rel_diff_pct",
            "worst_sw_sigma_ratio",
            "worst_gamma0_O2_rel_diff_pct",
            "worst_gamma0_O2_sigma_ratio",
            "worst_gamma0_air_rel_diff_pct",
            "worst_gamma0_air_sigma_ratio",
        ]
        transition_rows: list[dict] = []
        if not report_df.empty:
            for (gas_type, transition), sub in report_df.groupby(
                ["gas_type", "transition"], sort=False
            ):
                metric_union = sorted({
                    item
                    for value in sub["failed_metrics"].astype(str)
                    for item in value.split(";")
                    if item and item.lower() != "nan"
                })
                bad_pressures = [
                    str(p) for p in sub["pressure"].astype(str).tolist()
                    if p and p.lower() != "nan"
                ]
                transition_rows.append({
                    "gas_type": gas_type,
                    "transition": transition,
                    "recommend_remeasure": True,
                    "n_bad_points": int(len(sub)),
                    "bad_pressures": ";".join(bad_pressures),
                    "failed_metrics": ";".join(metric_union),
                    "has_invalid_fit": bool(
                        (~sub["fit_valid"].map(self._as_bool)).any()
                    ),
                    "has_missing_params": False,
                    "missing_params": "",
                    "n_missing_params": 0,
                    "worst_sw_rel_diff_pct": pd.to_numeric(
                        sub["sw_rel_diff_pct"], errors="coerce").max(),
                    "worst_sw_sigma_ratio": pd.to_numeric(
                        sub["sw_sigma_ratio"], errors="coerce").max(),
                    "worst_gamma0_O2_rel_diff_pct": pd.to_numeric(
                        sub["gamma0_O2_rel_diff_pct"], errors="coerce").max(),
                    "worst_gamma0_O2_sigma_ratio": pd.to_numeric(
                        sub["gamma0_O2_sigma_ratio"], errors="coerce").max(),
                    "worst_gamma0_air_rel_diff_pct": pd.to_numeric(
                        sub["gamma0_air_rel_diff_pct"], errors="coerce").max(),
                    "worst_gamma0_air_sigma_ratio": pd.to_numeric(
                        sub["gamma0_air_sigma_ratio"], errors="coerce").max(),
                })

        missing_master_map = self._collect_missing_master_params()
        transition_index = {
            (row["gas_type"], row["transition"]): row
            for row in transition_rows
        }
        for key, missing_params in missing_master_map.items():
            rec = transition_index.get(key)
            if rec is None:
                rec = {
                    "gas_type": key[0],
                    "transition": key[1],
                    "recommend_remeasure": True,
                    "n_bad_points": 0,
                    "bad_pressures": "",
                    "failed_metrics": "",
                    "has_invalid_fit": False,
                    "has_missing_params": False,
                    "missing_params": "",
                    "n_missing_params": 0,
                    "worst_sw_rel_diff_pct": np.nan,
                    "worst_sw_sigma_ratio": np.nan,
                    "worst_gamma0_O2_rel_diff_pct": np.nan,
                    "worst_gamma0_O2_sigma_ratio": np.nan,
                    "worst_gamma0_air_rel_diff_pct": np.nan,
                    "worst_gamma0_air_sigma_ratio": np.nan,
                }
                transition_rows.append(rec)
                transition_index[key] = rec

            rec["recommend_remeasure"] = True
            rec["has_missing_params"] = True
            rec["missing_params"] = ";".join(missing_params)
            rec["n_missing_params"] = len(missing_params)

        if transition_rows:
            transition_df = pd.DataFrame(transition_rows, columns=transition_columns)
            transition_df["_transition_num"] = pd.to_numeric(
                transition_df["transition"], errors="coerce")
            transition_df = transition_df.sort_values(
                ["gas_type", "_transition_num", "transition"],
                na_position="last",
            ).drop(columns="_transition_num")
        else:
            transition_df = pd.DataFrame(columns=transition_columns)

        transition_df.to_csv(transition_out_path, index=False)
        transition_df[
            transition_df["gas_type"] == "O2"
        ].to_csv(transition_o2_out_path, index=False)
        transition_df[
            transition_df["gas_type"] == "O2_N2"
        ].to_csv(transition_o2n2_out_path, index=False)

        logger.info(f"  压力点明细已保存: {point_out_path}")
        logger.info(f"  压力汇总已保存:   {pressure_out_path}")
        logger.info(f"  纯 O2 压力清单:   {pressure_o2_out_path}")
        logger.info(f"  O2_N2 压力清单:  {pressure_o2n2_out_path}")
        logger.info(f"  压力 PDF 清单:   {pressure_lists_root}")
        logger.info(f"  跃迁汇总已保存:   {transition_out_path}")
        logger.info(f"  纯 O2 跃迁清单:   {transition_o2_out_path}")
        logger.info(f"  O2_N2 跃迁清单:  {transition_o2n2_out_path}")
        if report_df.empty and transition_df.empty:
            logger.info("  未发现建议重测点")
            return

        logger.info(f"  共发现 {len(report_df)} 个建议重测点")
        for gas_type, sub in report_df.groupby("gas_type"):
            logger.info(f"    {gas_type}: {len(sub)} 个")
        if not pressure_df.empty:
            logger.info(f"  共涉及 {len(pressure_df)} 个需重测压力")

        logger.info(f"  共涉及 {len(transition_df)} 个建议重测跃迁")
        missing_transition_count = int(
            transition_df["has_missing_params"].fillna(False).astype(bool).sum()
        ) if not transition_df.empty else 0
        if missing_transition_count > 0:
            logger.info(f"  其中主表参数漏测跃迁: {missing_transition_count} 个")
        for _, row in transition_df.iterrows():
            parts: list[str] = []
            n_bad_points = int(row.get("n_bad_points", 0) or 0)
            bad_pressures = str(row.get("bad_pressures", "")).strip()
            if n_bad_points > 0:
                parts.append(f"{n_bad_points} 个压力点: {bad_pressures}")
            if self._as_bool(row.get("has_missing_params")):
                parts.append(f"漏测参数: {row.get('missing_params', '')}")

            metrics = str(row.get("failed_metrics", "")).strip()
            summary = "  ".join(parts)
            if metrics and metrics.lower() != "nan":
                summary += f"  [{metrics}]"
            logger.info(
                f"  - {row['gas_type']}/{row['transition']}"
                + (f"  ({summary})" if summary else "")
            )

    _MASTER_TABLE_NAME = "spectral_parameters.csv"

    # HITRAN 参考数据路径
    _HITRAN_CSV = _PROJECT_ROOT / "data" / "hitran" / "O2_9000_10000_sw_ge_1e-29.csv"

    def _load_hitran_reference(self) -> pd.DataFrame | None:
        """加载 HITRAN 参考数据 CSV，返回 DataFrame (nu 为索引列)。

        若文件不存在则返回 None。
        """
        if isinstance(self._hitran_reference_cache, pd.DataFrame):
            return self._hitran_reference_cache
        if self._hitran_reference_cache is None:
            return None

        csv_path = self._HITRAN_CSV
        if not csv_path.exists():
            logger.warning(f"  HITRAN 参考数据文件不存在: {csv_path}")
            self._hitran_reference_cache = None
            return None
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"  已加载 HITRAN 参考数据: {csv_path.name}"
                        f" ({len(df)} 条吸收线)")
            self._hitran_reference_cache = df
            return df
        except Exception as e:
            logger.warning(f"  读取 HITRAN 参考数据失败: {e}")
            self._hitran_reference_cache = None
            return None

    @staticmethod
    def _match_hitran_line(nu_transition: float, hitran_df: pd.DataFrame,
                           tol: float = 0.01) -> pd.Series | None:
        """根据跃迁波数在 HITRAN 数据中查找最近匹配。

        Parameters
        ----------
        nu_transition : float
            跃迁波数 (cm⁻¹)
        hitran_df : pd.DataFrame
            HITRAN 参考数据
        tol : float
            最大允许偏差 (cm⁻¹)，默认 0.01

        Returns
        -------
        pd.Series or None
            匹配的 HITRAN 行，或 None (无匹配)
        """
        diffs = np.abs(hitran_df["nu"].values - nu_transition)
        idx_min = int(np.argmin(diffs))
        if diffs[idx_min] <= tol:
            return hitran_df.iloc[idx_min]
        return None

    def _get_measured_transition_nu(self, gas_type: str) -> np.ndarray:
        """收集指定气体类型下已测跃迁波数，并按升序返回。"""
        cached = self._measured_transition_cache.get(gas_type)
        if cached is not None:
            return cached

        nu_set: set[float] = set()
        for root in [self.final_root / gas_type, self.raw_root / gas_type]:
            if not root.exists():
                continue
            for t_dir in root.iterdir():
                if not t_dir.is_dir() or t_dir.name.startswith("."):
                    continue
                try:
                    nu_set.add(float(t_dir.name))
                except ValueError:
                    continue

        values = np.array(sorted(nu_set), dtype=float) if nu_set else np.array([], dtype=float)
        self._measured_transition_cache[gas_type] = values
        return values

    def _recommend_measurement_window(
        self,
        gas_type: str,
        transition: str,
        width: float = 2.0,
        tol: float = 0.01,
    ) -> dict:
        """为指定跃迁推荐固定宽度的测量窗口。

        规则:
        1. 窗口宽度固定为 width cm^-1
        2. 目标跃迁必须落在窗口内
        3. 仅在当前气体类型的已测跃迁集合内比较相邻线
        4. 优先保持目标跃迁位于窗口中央区域
        5. 在该前提下优先选包含跃迁数最少的
        6. 若并列，则优先目标线更接近窗口中心
        """
        cache_key = (f"{gas_type}:{transition}", float(width))
        cached = self._measurement_window_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            nu_target = float(transition)
        except (TypeError, ValueError):
            rec = {
                "start": np.nan,
                "end": np.nan,
                "n_lines": np.nan,
                "is_isolated": False,
                "shift": np.nan,
                "line_positions": [],
            }
            self._measurement_window_cache[cache_key] = rec
            return rec

        default_start = nu_target - width / 2.0
        default_end = nu_target + width / 2.0
        nu_vals = self._get_measured_transition_nu(gas_type)
        if nu_vals.size == 0:
            rec = {
                "start": default_start,
                "end": default_end,
                "n_lines": np.nan,
                "is_isolated": False,
                "shift": 0.0,
                "line_positions": [nu_target],
            }
            self._measurement_window_cache[cache_key] = rec
            return rec

        start_min = nu_target - width
        start_max = nu_target
        candidate_starts = {
            start_min,
            nu_target - width / 2.0,
            start_max,
        }
        nearby = nu_vals[
            (nu_vals >= start_min - tol) &
            (nu_vals <= nu_target + width + tol)
        ]
        for value in nearby:
            candidate_starts.add(float(value))
            candidate_starts.add(float(value) - width)

        candidate_records: list[tuple[tuple[float, float, float, float], dict]] = []
        max_center_shift = width / 4.0
        for raw_start in sorted(candidate_starts):
            start = min(max(raw_start, start_min), start_max)
            end = start + width
            if not (start - tol <= nu_target <= end + tol):
                continue

            inside = nu_vals[(nu_vals >= start - tol) & (nu_vals <= end + tol)]
            line_count = int(len(inside))
            center_penalty = abs((start + end) / 2.0 - nu_target)
            shift = start - (nu_target - width / 2.0)
            score = (
                line_count,
                center_penalty,
                abs(shift),
                start,
            )
            candidate_records.append((
                score,
                {
                    "start": float(start),
                    "end": float(end),
                    "n_lines": line_count,
                    "is_isolated": line_count == 1,
                    "shift": float(shift),
                    "line_positions": [float(x) for x in inside],
                },
            ))

        if candidate_records:
            preferred = [
                item for item in candidate_records
                if abs(item[1]["shift"]) <= max_center_shift + tol
            ]
            _, best_rec = min(preferred or candidate_records, key=lambda item: item[0])
        else:
            best_rec = {
                "start": default_start,
                "end": default_end,
                "n_lines": int(((nu_vals >= default_start - tol) & (nu_vals <= default_end + tol)).sum()),
                "is_isolated": False,
                "shift": 0.0,
                "line_positions": [
                    float(x) for x in nu_vals[
                        (nu_vals >= default_start - tol) &
                        (nu_vals <= default_end + tol)
                    ]
                ],
            }

        self._measurement_window_cache[cache_key] = best_rec
        return best_rec

    @classmethod
    def _estimate_hitran_n2_broadening(
        cls,
        gamma_air: float,
        gamma_self: float,
    ) -> float | None:
        """由 HITRAN air/self 展宽近似反推 N2 展宽。

        采用干空气二组分近似:
            gamma_air = x_O2 * gamma_O2 + x_N2 * gamma_N2

        其中对 O2 谱线取 gamma_O2 = gamma_self。
        """
        if cls._AIR_N2_FRACTION <= 0:
            return None
        return (
            gamma_air - cls._AIR_O2_FRACTION * gamma_self
        ) / cls._AIR_N2_FRACTION

    def _build_master_table(self) -> None:
        """将 O₂ 多光谱联合拟合和 N₂ 线性回归的所有参数
        合并为一张主表，以跃迁波数为第一列。

        同时将 HITRAN 参考值 (线强、自展宽、空气展宽等) 写在对应参数旁边，
        方便对比。

        若表格已存在，先备份为 spectral_parameters_YYYYMMDD_HHMMSS.csv，
        再写入新表。
        """
        logger.info("\n" + "=" * 60)
        logger.info("  汇总: 构建参数主表")
        logger.info("=" * 60)

        # ── 加载 HITRAN 参考数据 ──
        hitran_df = self._load_hitran_reference()

        rows: list[dict] = []

        # 收集所有跃迁 (从 final/O2 和 final/O2_N2 中发现)
        transitions: set[str] = set()
        for sub in ["O2", "O2_N2"]:
            d = self.final_root / sub
            if d.exists():
                for t_dir in d.iterdir():
                    if (t_dir.is_dir()
                            and not t_dir.name.startswith(".")
                            and not t_dir.name.startswith("merged_")):
                        transitions.add(t_dir.name)

        for transition in sorted(transitions):
            rec: dict = {"nu": transition}

            # ── HITRAN 参考值 ──
            if hitran_df is not None:
                try:
                    nu_val = float(transition)
                    hitran_row = self._match_hitran_line(nu_val, hitran_df)
                    if hitran_row is not None:
                        rec["sw_HITRAN"] = hitran_row["sw"]
                        rec["gamma_self_HITRAN"] = hitran_row["gamma_self"]
                        rec["gamma_air_HITRAN"] = hitran_row["gamma_air"]
                        rec["gamma0_N2_HITRAN"] = self._estimate_hitran_n2_broadening(
                            gamma_air=float(hitran_row["gamma_air"]),
                            gamma_self=float(hitran_row["gamma_self"]),
                        )
                        rec["n_air_HITRAN"] = hitran_row["n_air"]
                        rec["delta_air_HITRAN"] = hitran_row["delta_air"]
                        rec["elower_HITRAN"] = hitran_row["elower"]
                except (ValueError, KeyError):
                    pass

            # ── O₂ 多光谱联合拟合 ──
            o2_csv = self.final_root / "O2" / transition / "multi_fit_result.csv"
            if o2_csv.exists():
                try:
                    o2_df = pd.read_csv(o2_csv)
                    if not o2_df.empty:
                        # 匹配目标跃迁: 选 nu_HITRAN 与 transition 波数最接近的行
                        nu_target = float(transition)
                        if "nu_HITRAN" in o2_df.columns:
                            idx_closest = (o2_df["nu_HITRAN"] - nu_target).abs().idxmin()
                            row = o2_df.loc[idx_closest]
                        else:
                            row = o2_df.iloc[0]
                        rec["n_spectra_O2"] = int(row.get("n_spectra", 0))
                        rec["QF_O2"] = row.get("QF", 0)
                        rec["residual_std_O2"] = row.get("residual_std", 0)
                        # 核心参数
                        for param in ["sw", "gamma0_O2", "n_gamma0_O2",
                                      "SD_gamma_O2", "delta0_O2", "SD_delta_O2"]:
                            rec[param] = row.get(param, "")
                            rec[f"{param}_err"] = row.get(f"{param}_err", "")
                except Exception as e:
                    logger.warning(f"  [{transition}] 读取 O₂ 结果失败: {e}")

            # ── N₂ 线性回归 ──
            n2_csv = (self.final_root / "O2_N2" / transition
                      / "linear_regression_n2.csv")
            if n2_csv.exists():
                try:
                    n2_df = pd.read_csv(n2_csv)
                    for _, lr_row in n2_df.iterrows():
                        param_base = lr_row["parameter"]  # gamma0, SD_gamma, ...
                        if param_base != "gamma0":
                            continue
                        col_n2 = f"{param_base}_N2"
                        rec[col_n2] = lr_row["value_N2"]
                        rec[f"{col_n2}_err"] = lr_row["uncertainty_N2"]
                        rec[f"{col_n2}_R2"] = lr_row["R_squared"]
                        rec[f"{col_n2}_npts"] = int(lr_row["n_points"])
                except Exception as e:
                    logger.warning(f"  [{transition}] 读取 N₂ 结果失败: {e}")

            rows.append(rec)

        if not rows:
            logger.warning("  未找到任何拟合结果，跳过主表构建")
            return

        master_df = pd.DataFrame(rows)
        if "gamma0_N2" not in master_df.columns:
            master_df["gamma0_N2"] = np.nan

        # 列排序: nu → 各参数组 (拟合值 + HITRAN 参考 紧邻排列)
        desired_order = ["nu"]
        # ── 线强: sw, sw_err, sw_HITRAN ──
        desired_order.extend(["sw", "sw_err", "sw_HITRAN"])
        # ── O₂ 自展宽: gamma0_O2, gamma0_O2_err, gamma_self_HITRAN ──
        desired_order.extend(["gamma0_O2", "gamma0_O2_err", "gamma_self_HITRAN"])
        # ── n_air: n_gamma0_O2, n_gamma0_O2_err, n_air_HITRAN ──
        desired_order.extend(["n_gamma0_O2", "n_gamma0_O2_err", "n_air_HITRAN"])
        # ── SD_gamma_O2 ──
        desired_order.extend(["SD_gamma_O2", "SD_gamma_O2_err"])
        # ── delta0_O2, delta0_O2_err, delta_air_HITRAN ──
        desired_order.extend(["delta0_O2", "delta0_O2_err", "delta_air_HITRAN"])
        # ── SD_delta_O2 ──
        desired_order.extend(["SD_delta_O2", "SD_delta_O2_err"])
        # ── gamma_air_HITRAN (空气展宽参考) ──
        desired_order.append("gamma_air_HITRAN")
        # ── N₂ 展宽: 拟合值 + HITRAN 反推值 ──
        desired_order.extend([
            "gamma0_N2", "gamma0_N2_HITRAN",
            "gamma0_N2_err", "gamma0_N2_R2", "gamma0_N2_npts",
        ])
        # ── 辅助列 ──
        desired_order.extend(["elower_HITRAN",
                              "n_spectra_O2", "QF_O2", "residual_std_O2"])

        # 主表中仅保留 N2 展宽，不包含其他 N2 参数
        drop_cols = [
            "SD_gamma_N2", "SD_gamma_N2_err", "SD_gamma_N2_R2", "SD_gamma_N2_npts",
            "delta0_N2", "delta0_N2_err", "delta0_N2_R2", "delta0_N2_npts",
            "SD_delta_N2", "SD_delta_N2_err", "SD_delta_N2_R2", "SD_delta_N2_npts",
        ]
        master_df = master_df.drop(columns=drop_cols, errors="ignore")

        # 只保留实际存在的列，按 desired_order 排序，其余追加在末尾
        existing = list(master_df.columns)
        ordered = [c for c in desired_order if c in existing]
        remaining = [c for c in existing if c not in ordered]
        master_df = master_df[ordered + remaining]

        # ── 保存 (备份旧表) ──
        out_path = self.final_root / self._MASTER_TABLE_NAME
        if out_path.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = out_path.with_name(
                f"{out_path.stem}_{ts}{out_path.suffix}")
            shutil.copy2(out_path, backup)
            logger.info(f"  已备份旧表: {backup.name}")

        self.final_root.mkdir(parents=True, exist_ok=True)
        master_df.to_csv(out_path, index=False)
        logger.info(f"  参数主表已保存: {out_path}")


        # ── 绘制与 HITRAN 对比图 ──
        self._plot_hitran_comparison(master_df)

    # ==============================================================
    # 对比图: 线强 & 自展宽 vs HITRAN
    # ==============================================================
    _FIGURES_ROOT = _PROJECT_ROOT / "output" / "figures"

    def _plot_hitran_comparison(self, master_df: pd.DataFrame) -> None:
        """绘制拟合结果与 HITRAN 参考数据的对比图

        Panel 1: 线强 (sw) vs HITRAN — 上方主图 + 下方残差
        Panel 2: O₂ 自展宽 (gamma0_O2) vs HITRAN — 上方主图 + 下方残差

        同时保存 PNG 和 PDF。
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter

        # 筛选有拟合数据 且 有 HITRAN 匹配的行
        needed = ["sw", "sw_err", "sw_HITRAN",
                  "gamma0_O2", "gamma0_O2_err", "gamma_self_HITRAN"]
        df = master_df.dropna(subset=needed).copy()
        if df.empty:
            logger.warning("  无有效数据用于绘制 HITRAN 对比图")
            return

        nu = df["nu"].astype(float).values
        sw_fit = df["sw"].astype(float).values
        sw_err = df["sw_err"].astype(float).values
        sw_hitran = df["sw_HITRAN"].astype(float).values
        gamma_fit = df["gamma0_O2"].astype(float).values
        gamma_err = df["gamma0_O2_err"].astype(float).values
        gamma_hitran = df["gamma_self_HITRAN"].astype(float).values

        # 残差 (相对百分比)
        sw_res_pct = (sw_fit - sw_hitran) / sw_hitran * 100
        gamma_res_pct = np.where(
            gamma_hitran != 0,
            (gamma_fit - gamma_hitran) / gamma_hitran * 100,
            0.0,
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                                 gridspec_kw={"height_ratios": [3, 1],
                                              "hspace": 0.08, "wspace": 0.30},
                                 constrained_layout=True)

        # ─────── Panel 1: 线强 (sw) ───────
        ax_sw = axes[0, 0]
        ax_sw_res = axes[1, 0]

        ax_sw.errorbar(nu, sw_fit, yerr=sw_err, fmt="o", ms=5, capsize=3,
                       color="C0", label="This work", zorder=3)
        ax_sw.scatter(nu, sw_hitran, marker="^", s=50, color="C3",
                      label="HITRAN", zorder=2)
        ax_sw.set_ylabel("Line intensity  S  (cm/molecule)", fontsize=11)
        ax_sw.set_title("Line Intensity Comparison", fontsize=12, fontweight="bold")
        ax_sw.legend(fontsize=10, loc="best")
        ax_sw.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax_sw.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax_sw.set_xticklabels([])
        ax_sw.grid(True, alpha=0.3)

        # 残差
        ax_sw_res.bar(nu, sw_res_pct, width=1.5, color="C0", alpha=0.7)
        ax_sw_res.axhline(0, color="k", lw=0.8)
        ax_sw_res.set_xlabel("Wavenumber  (cm⁻¹)", fontsize=11)
        ax_sw_res.set_ylabel("Residual  (%)", fontsize=10)
        ax_sw_res.grid(True, alpha=0.3)

        # ─────── Panel 2: 自展宽 (gamma0_O2) ───────
        ax_gam = axes[0, 1]
        ax_gam_res = axes[1, 1]

        ax_gam.errorbar(nu, gamma_fit, yerr=gamma_err, fmt="s", ms=5,
                        capsize=3, color="C0", label="This work", zorder=3)
        ax_gam.scatter(nu, gamma_hitran, marker="^", s=50, color="C3",
                       label="HITRAN", zorder=2)
        ax_gam.set_ylabel(r"$\gamma_{\rm self}$  (cm⁻¹/atm)", fontsize=11)
        ax_gam.set_title("Self-broadening Comparison", fontsize=12,
                         fontweight="bold")
        ax_gam.legend(fontsize=10, loc="best")
        ax_gam.set_xticklabels([])
        ax_gam.grid(True, alpha=0.3)

        # 残差
        ax_gam_res.bar(nu, gamma_res_pct, width=1.5, color="C0", alpha=0.7)
        ax_gam_res.axhline(0, color="k", lw=0.8)
        ax_gam_res.set_xlabel("Wavenumber  (cm⁻¹)", fontsize=11)
        ax_gam_res.set_ylabel("Residual  (%)", fontsize=10)
        ax_gam_res.grid(True, alpha=0.3)

        fig.suptitle("Fitted Parameters vs HITRAN Reference  —  O₂ A-band",
                     fontsize=13, fontweight="bold")

        # ── 保存 ──
        self._FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
        png_path = self._FIGURES_ROOT / "hitran_comparison.png"
        pdf_path = self._FIGURES_ROOT / "hitran_comparison.pdf"
        fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
        fig.savefig(str(pdf_path), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  HITRAN 对比图已保存:")
        logger.info(f"    PNG: {png_path}")
        logger.info(f"    PDF: {pdf_path}")

    # ==============================================================
    # Step 4 辅助方法
    # ==============================================================

    @staticmethod
    def _collect_sw_records(t_dir: Path,
                            gamma_col: str = "gamma0_O2") -> list[dict]:
        """收集某个跃迁下各压力点的拟合线强"""
        records: list[dict] = []
        for p_dir in sorted(t_dir.iterdir()):
            if not p_dir.is_dir() or p_dir.name.startswith("."):
                continue
            param_files = list(p_dir.glob("*Parameter_LineList*.csv"))
            if not param_files:
                continue
            param_df = pd.read_csv(param_files[0], index_col=0)
            fitted = param_df[param_df["sw_vary"] == True]
            if fitted.empty:
                continue
            for _, row in fitted.iterrows():
                sw_real = row["sw"] * row.get("sw_scale_factor", 1.0)
                rec = {"pressure": p_dir.name, "sw": sw_real, "sw_raw": row["sw"], "gamma0": row.get(gamma_col, 0)}
                # 自适应列名: gamma0_O2 或 gamma0_air
                records.append(rec)
        return records

    def _lookup_multi_fit_pressures(
        self, gas_type: str, transition: str,
    ) -> list[str] | None:
        """从 multi_fit_pressures 中查找匹配的压力列表。

        支持两种匹配方式:
          1. 精确匹配:  键 == "O2/9386.207642"  (与目录名完全一致)
          2. 前缀匹配:  键 == "O2/9386.2076"    (目录名以键中的波数开头)

        Returns
        -------
        list[str] | None
            匹配到的压力列表；未指定或无匹配时返回 None。
        """
        if not self.multi_fit_pressures:
            return None

        tag = f"{gas_type}/{transition}"

        # 1. 精确匹配
        if tag in self.multi_fit_pressures:
            return self.multi_fit_pressures[tag]

        # 2. 前缀匹配: 遍历所有键，比较 gas_type 和 transition 前缀
        for key, pressures in self.multi_fit_pressures.items():
            parts = key.strip("/").split("/")
            if len(parts) != 2:
                continue
            key_gas, key_trans = parts
            if key_gas != gas_type:
                continue
            # transition 以 key_trans 开头 (如 "9386.207642".startswith("9386.2076"))
            if transition.startswith(key_trans):
                return pressures

        return None

    def _screen_sw(self, records: list[dict], transition: str) -> pd.DataFrame:
        """基于 MAD 筛选线强离群点

        双重保护:
          1. MAD σ 筛选: 偏差 > sw_sigma 倍 MAD 则标记为离群
          2. 相对偏差保护: 即使 MAD σ 很大，若相对偏差 < 5% 仍保留
             (MAD 在样本数少时过于敏感)
        """
        sw_df = pd.DataFrame(records)
        sw_values = sw_df["sw"].values
        median_sw = float(np.median(sw_values))
        mad_sw = float(np.median(np.abs(sw_values - median_sw)))
        if mad_sw < 1e-35:
            mad_sw = float(np.std(sw_values))

        sw_df["deviation"] = (
            np.abs(sw_df["sw"] - median_sw) / (mad_sw if mad_sw > 0 else 1)
        )
        sw_df["rel_deviation"] = (
            np.abs(sw_df["sw"] - median_sw) / median_sw if median_sw > 0 else 0
        )
        # 双重条件: MAD σ ≤ 阈值 OR 相对偏差 < 5%
        rel_threshold = 0.05  # 5%
        sw_df["keep"] = (
            (sw_df["deviation"] <= self.sw_sigma)
            | (sw_df["rel_deviation"] < rel_threshold)
        )

        # 打印筛选结果
        logger.info(f"\n  [{transition}] 线强筛选 (中位数={median_sw:.4e}, "
              f"MAD={mad_sw:.4e}, 阈值={self.sw_sigma}σ)")
        logger.info(f"  {'压力':<10s} {'S (cm/molec)':<14s} {'偏差/MAD':<10s} "
                     f"{'相对偏差':<10s} {'状态':<8s}")
        logger.info(f"  {'─' * 60}")
        for _, r in sw_df.iterrows():
            status = "✓ 保留" if r["keep"] else "✗ 剔除"
            logger.info(f"  {r['pressure']:<10s} {r['sw']:.4e}   "
                  f"{r['deviation']:>6.2f}σ     "
                  f"{r['rel_deviation']*100:>5.1f}%     {status}")

        kept = sw_df[sw_df["keep"]]
        removed = sw_df[~sw_df["keep"]]
        if len(removed) > 0:
            logger.info(f"\n  剔除 {len(removed)} 个离群点: "
                  f"{', '.join(removed['pressure'].tolist())}")
        else:
            logger.info(f"\n  无离群点，全部 {len(kept)} 个压力点保留")

        return sw_df

    def _collect_etalon_csvs(self, gas_type: str, transition: str,
                             kept: pd.DataFrame) -> tuple[list[Path], list[str]]:
        """收集保留压力点的 etalon CSV 路径"""
        etalon_csvs: list[Path] = []
        labels: list[str] = []
        for _, r in kept.iterrows():
            csv_path = (self.etalon_root / gas_type / transition
                        / r["pressure"] / "tau_etalon_corrected.csv")
            if csv_path.exists():
                etalon_csvs.append(csv_path)
                labels.append(r["pressure"])
            else:
                logger.warning(f"   找不到: {csv_path}")
        return etalon_csvs, labels

    # ==============================================================
    # 结果汇总方法
    # ==============================================================
    def _save_multi_fit_summary(self, result, output_dir: Path,
                                gas_type: str, transition: str,
                                labels: list[str]) -> None:
        """保存多光谱联合拟合的最终统计 CSV (仅纯 O₂)"""
        if result is None or result.param_linelist.empty:
            return

        param_df = result.param_linelist
        fitted = param_df[param_df["sw_vary"] == True]
        if fitted.empty:
            return

        diluents_to_output = ["O2"]

        rows: list[dict] = []
        for _, row in fitted.iterrows():
            scale = row.get("sw_scale_factor", 1.0)
            rec = {
                "gas_type": gas_type,
                "transition": transition,
                "n_spectra": len(labels),
                "pressures": "+".join(labels),
                "nu_HITRAN": row["nu"],
                "sw": row["sw"] * scale,
                "sw_err": row.get("sw_err", 0) * scale,
                "residual_std": result.residual_std,
                "QF": result.qf,
            }
            for dil in diluents_to_output:
                rec[f"gamma0_{dil}"] = row.get(f"gamma0_{dil}", 0)
                rec[f"gamma0_{dil}_err"] = row.get(f"gamma0_{dil}_err", 0)
                rec[f"n_gamma0_{dil}"] = row.get(f"n_gamma0_{dil}", 0)
                rec[f"n_gamma0_{dil}_err"] = row.get(f"n_gamma0_{dil}_err", 0)
                rec[f"SD_gamma_{dil}"] = row.get(f"SD_gamma_{dil}", 0)
                rec[f"SD_gamma_{dil}_err"] = row.get(f"SD_gamma_{dil}_err", 0)
                rec[f"delta0_{dil}"] = row.get(f"delta0_{dil}", 0)
                rec[f"delta0_{dil}_err"] = row.get(f"delta0_{dil}_err", 0)
                rec[f"SD_delta_{dil}"] = row.get(f"SD_delta_{dil}", 0)
                rec[f"SD_delta_{dil}_err"] = row.get(f"SD_delta_{dil}_err", 0)
            rows.append(rec)

        final_df = pd.DataFrame(rows)
        out_path = output_dir / "multi_fit_result.csv"
        final_df.to_csv(out_path, index=False)

        final_dir = self.final_root / gas_type / transition
        final_dir.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(final_dir / "multi_fit_result.csv", index=False)

        tag = f"{gas_type}/{transition}"
        logger.info(f"\n  {'#' * 60}")
        logger.info(f"  多光谱联合拟合最终结果 ({tag})")
        logger.info(f"  光谱数: {len(labels)}, 压力: {', '.join(labels)}")
        logger.info(f"  {'#' * 60}")
        for _, r in final_df.iterrows():
            logger.info(f"  ν = {r['nu_HITRAN']:.6f} cm⁻¹")
            logger.info(f"    S          = {r['sw']:.6e} ± {r['sw_err']:.2e}")
            for dil in diluents_to_output:
                logger.info(f"    γ₀_{dil:<3s}  = {r[f'gamma0_{dil}']:.6f}"
                             f" ± {r[f'gamma0_{dil}_err']:.6f} cm⁻¹/atm")
                logger.info(f"    n_γ₀_{dil:<3s}= {r[f'n_gamma0_{dil}']:.4f}"
                             f" ± {r[f'n_gamma0_{dil}_err']:.4f}")
                logger.info(f"    SD_γ_{dil:<3s} = {r[f'SD_gamma_{dil}']:.6f}"
                             f" ± {r[f'SD_gamma_{dil}_err']:.6f}")
                logger.info(f"    δ₀_{dil:<3s}  = {r[f'delta0_{dil}']:.6f}"
                             f" ± {r[f'delta0_{dil}_err']:.6f} cm⁻¹/atm")
            logger.info(f"    Res. σ     = {r['residual_std']:.4e}")
            logger.info(f"    QF         = {r['QF']:.1f}")
        logger.info(f"\n  结果已保存: {out_path}")
        logger.info(f"  汇总目录:   {final_dir / 'multi_fit_result.csv'}")

    def _collect_final_summary(self) -> None:
        """从 MATS 单光谱拟合结果中提取关键参数并保存汇总表"""
        self.final_root.mkdir(parents=True, exist_ok=True)

        all_rows: list[dict] = []

        if not self.mats_root.exists():
            return

        for gas_dir in sorted(self.mats_root.iterdir()):
            if not gas_dir.is_dir() or gas_dir.name.startswith("."):
                continue
            gas_type = gas_dir.name
            # Step 3 单光谱: O₂ 用 "O2", O₂+N₂ 用 "air" 近似
            diluents = ["O2"] if gas_type == "O2" else ["air"]

            for t_dir in sorted(gas_dir.iterdir()):
                if not t_dir.is_dir() or t_dir.name.startswith("."):
                    continue
                transition = t_dir.name
                for p_dir in sorted(t_dir.iterdir()):
                    if not p_dir.is_dir() or p_dir.name.startswith("."):
                        continue
                    pressure_label = p_dir.name

                    param_files = list(p_dir.glob("*Parameter_LineList*.csv"))
                    if not param_files:
                        continue

                    param_df = pd.read_csv(param_files[0], index_col=0)

                    x_shift = self._read_x_shift(p_dir)
                    residual_std = self._read_residual_std(p_dir)

                    for _, row in param_df.iterrows():
                        nu = row.get("nu", 0)
                        sw = row.get("sw", 0)
                        sw_scale = row.get("sw_scale_factor", 1.0)
                        if sw * sw_scale < 1e-35:
                            continue
                        issues = self._single_fit_issues_from_row(row, diluents)
                        rec = {
                            "gas_type": gas_type,
                            "transition": transition,
                            "pressure_label": pressure_label,
                            "nu": nu,
                            "sw": sw * sw_scale,
                            "sw_raw": sw,
                            "sw_scale_factor": sw_scale,
                            "x_shift": x_shift,
                            "residual_std": residual_std,
                            "sw_vary": row.get("sw_vary", False),
                            "fit_valid": len(issues) == 0,
                            "fit_issue": ";".join(issues),
                        }
                        for dil in diluents:
                            rec[f"gamma0_{dil}"] = row.get(f"gamma0_{dil}", 0)
                            rec[f"gamma0_{dil}_err"] = row.get(f"gamma0_{dil}_err", 0)
                            rec[f"n_gamma0_{dil}"] = row.get(f"n_gamma0_{dil}", 0)
                            rec[f"delta0_{dil}"] = row.get(f"delta0_{dil}", 0)
                            rec[f"delta0_{dil}_err"] = row.get(f"delta0_{dil}_err", 0)
                            rec[f"SD_gamma_{dil}"] = row.get(f"SD_gamma_{dil}", 0)
                            rec[f"SD_gamma_{dil}_err"] = row.get(f"SD_gamma_{dil}_err", 0)
                            rec[f"SD_delta_{dil}"] = row.get(f"SD_delta_{dil}", 0)
                        all_rows.append(rec)

        if not all_rows:
            logger.warning("\n  未找到任何有效拟合结果，跳过汇总")
            return

        df = pd.DataFrame(all_rows)
        if "nu" in df.columns:
            df = df.sort_values(["gas_type", "nu"]).reset_index(drop=True)

        summary_path = self.final_root / "spectral_parameters_summary.csv"
        df.to_csv(summary_path, index=False)

        for (gas_type, transition), sub in df.groupby(
                ["gas_type", "transition"]):
            trans_dir = self.final_root / str(gas_type) / str(transition)
            trans_dir.mkdir(parents=True, exist_ok=True)
            sub.to_csv(trans_dir / "fitted_parameters.csv", index=False)

        self._generate_fit_statistics()

        logger.info(f"\n  单光谱拟合汇总表: {summary_path}")
        logger.info(f"  详细目录: {self.final_root}")

    def _generate_fit_statistics(self) -> None:
        """从 MATS 拟合结果中提取被拟合目标线的完整参数，生成统计 CSV"""
        if not self.mats_root.exists():
            return

        for gas_dir in sorted(self.mats_root.iterdir()):
            if not gas_dir.is_dir() or gas_dir.name.startswith("."):
                continue
            gas_type = gas_dir.name
            # Step 3 单光谱: O₂ 用 "O2", O₂+N₂ 用 "air" 近似
            diluents = ["O2"] if gas_type == "O2" else ["air"]

            for t_dir in sorted(gas_dir.iterdir()):
                if not t_dir.is_dir() or t_dir.name.startswith("."):
                    continue
                transition = t_dir.name
                rows: list[dict] = []

                for p_dir in sorted(t_dir.iterdir()):
                    if not p_dir.is_dir() or p_dir.name.startswith("."):
                        continue
                    pressure_label = p_dir.name

                    param_files = list(p_dir.glob("*Parameter_LineList*.csv"))
                    if not param_files:
                        continue
                    param_df = pd.read_csv(param_files[0], index_col=0)

                    x_shift = self._read_x_shift(p_dir)
                    residual_std = self._read_residual_std(p_dir)

                    fitted = param_df[param_df["sw_vary"] == True]
                    for _, row in fitted.iterrows():
                        scale = row.get("sw_scale_factor", 1.0)
                        issues = self._single_fit_issues_from_row(row, diluents)
                        rec = {
                            "pressure": pressure_label,
                            "nu_HITRAN": row["nu"],
                            "sw": row["sw"] * scale,
                            "sw_err": row.get("sw_err", 0) * scale,
                            "x_shift": x_shift,
                            "residual_std": residual_std,
                            "fit_valid": len(issues) == 0,
                            "fit_issue": ";".join(issues),
                        }
                        for dil in diluents:
                            rec[f"gamma0_{dil}"] = row.get(f"gamma0_{dil}", 0)
                            rec[f"gamma0_{dil}_err"] = row.get(f"gamma0_{dil}_err", 0)
                            rec[f"SD_gamma_{dil}"] = row.get(f"SD_gamma_{dil}", 0)
                            rec[f"SD_gamma_{dil}_err"] = row.get(f"SD_gamma_{dil}_err", 0)
                            rec[f"delta0_{dil}"] = row.get(f"delta0_{dil}", 0)
                            rec[f"delta0_{dil}_err"] = row.get(f"delta0_{dil}_err", 0)
                            rec[f"SD_delta_{dil}"] = row.get(f"SD_delta_{dil}", 0)
                            rec[f"SD_delta_{dil}_err"] = row.get(f"SD_delta_{dil}_err", 0)
                        rows.append(rec)

                if not rows:
                    continue

                stat_df = pd.DataFrame(rows)
                # 尝试从压力标签中提取数字排序
                nums = stat_df["pressure"].str.extract(r"(\d+)")
                if not nums.empty and nums[0].notna().any():
                    stat_df["_p"] = nums[0].astype(float)
                    stat_df = (stat_df.sort_values("_p")
                               .drop(columns="_p").reset_index(drop=True))

                out_dir = self.final_root / gas_type / transition
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "fit_summary_statistics.csv"
                stat_df.to_csv(out_path, index=False)

    # ==============================================================
    # 通用 I/O 辅助方法
    # ==============================================================
    @staticmethod
    def _read_x_shift(p_dir: Path) -> float:
        """从 baseline_paramlist CSV 中读取 x_shift"""
        baseline_files = list(p_dir.glob("*baseline_paramlist*.csv"))
        if not baseline_files:
            return 0.0
        try:
            bl_df = pd.read_csv(baseline_files[0])
            if "x_shift" in bl_df.columns:
                return float(bl_df["x_shift"].iloc[0])
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _read_residual_std(p_dir: Path) -> float:
        """从 MATS 输出 CSV 中读取残差标准差"""
        for csv_file in sorted(p_dir.glob("*.csv")):
            if any(kw in csv_file.name for kw in [
                "Parameter_LineList", "baseline_paramlist",
                "_linelist", "_spectrum",
            ]):
                continue
            try:
                sdf = pd.read_csv(csv_file)
                res_col = next(
                    (c for c in sdf.columns if "Residual" in c), None)
                if res_col:
                    return float(np.std(sdf[res_col].values))
            except Exception:
                pass
        return 0.0

    @staticmethod
    def _single_fit_issues_from_row(
        row: pd.Series,
        diluents: list[str],
    ) -> list[str]:
        """统一判断单谱目标线是否足够可靠，可进入后续统计/回归。"""
        issues: list[str] = []

        sw_val = pd.to_numeric(row.get("sw"), errors="coerce")
        sw_err = pd.to_numeric(row.get("sw_err"), errors="coerce")
        if not np.isfinite(sw_err) or sw_err <= 0:
            issues.append("missing_sw_err")
        if np.isfinite(sw_val) and sw_val <= 1.05:
            issues.append("sw_near_lower_bound")

        for dil in diluents:
            gamma_col = f"gamma0_{dil}"
            gamma_err_col = f"{gamma_col}_err"
            gamma_val = pd.to_numeric(row.get(gamma_col), errors="coerce")
            gamma_err = pd.to_numeric(row.get(gamma_err_col), errors="coerce")
            if not np.isfinite(gamma_err) or gamma_err <= 0:
                issues.append(f"missing_{gamma_err_col}")
            if np.isfinite(gamma_val) and gamma_val <= 0.0055:
                issues.append(f"{gamma_col}_near_lower_bound")

        return sorted(set(issues))


# ==================================================================
# 便捷函数 (向后兼容)
# ==================================================================
def run_pipeline(**kwargs) -> None:
    """便捷函数: 创建 CRDSPipeline 并执行完整流水线

    支持的关键字参数同 CRDSPipeline.__init__
    """
    pipeline = CRDSPipeline(**kwargs)
    pipeline.run()
