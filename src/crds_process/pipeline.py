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

    # 单独执行某一步
    pipeline.step1_ringdown()
    pipeline.step2_etalon()
    pipeline.step3_mats()
    pipeline.step4_multi_fit()
"""

from __future__ import annotations

import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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
    """

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
    ):
        # ── 路径 ──
        self.raw_root = Path(raw_root) if raw_root else _DEFAULT_PATHS["raw"]
        self.ringdown_root = Path(ringdown_root) if ringdown_root else _DEFAULT_PATHS["ringdown"]
        self.etalon_root = Path(etalon_root) if etalon_root else _DEFAULT_PATHS["etalon"]
        self.mats_root = Path(mats_root) if mats_root else _DEFAULT_PATHS["mats"]
        self.mats_multi_root = Path(mats_multi_root) if mats_multi_root else _DEFAULT_PATHS["mats_multi"]
        self.final_root = Path(final_root) if final_root else _DEFAULT_PATHS["final"]

        # ── 并行 ──
        self.max_workers = max_workers or min(os.cpu_count() or 1, 6)

        # ── 目标过滤 ──
        self.targets = targets  # e.g. ["O2/9386.2076", "O2_N2/9386.2076/500Torr"]

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

    # ==============================================================
    # Step 1: 衰荡时间处理
    # ==============================================================
    def step1_ringdown(self) -> None:
        """Step 1: 原始衰荡数据 → ringdown_results.csv (多进程并行)"""
        from crds_process.preprocessing import discover_tasks

        logger.info("\n" + "=" * 60)
        logger.info("  Step 1 / 4 — 衰荡时间处理")
        logger.info("=" * 60)

        tasks = discover_tasks(self.raw_root)
        tasks = self._filter_tasks(tasks)
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
    def step2_etalon(self) -> None:
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
        if not all_tasks:
            logger.error(f"  未在 {self.ringdown_root} 下找到数据")
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
        )

    def _fitter_kwargs_for_gas(self, gas_type: str,
                               pressure_label: str = "") -> dict:
        """根据气体类型构建完整的 MATSFitter 参数"""
        from crds_process.gas_config import parse_gas_dir
        kw = self._base_fitter_kwargs()
        gc = parse_gas_dir(pressure_label, gas_type)
        kw.update(gc.to_fitter_kwargs())
        return kw

    def step3_mats(self) -> None:
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
        if not all_tasks:
            logger.error(f"  未在 {self.etalon_root} 下找到数据")
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
                if target_trans and t_dir.name not in target_trans:
                    continue
                self._process_transition_multi_fit(
                    gas_type, t_dir.name)

    def _process_transition_multi_fit(self, gas_type: str,
                                      transition: str) -> None:
        """对单个 (气体类型, 跃迁) 执行线强筛选 + 多光谱联合拟合

        仅用于纯 O₂ 数据。
        """
        t_dir = self.mats_root / gas_type / transition
        tag = f"{gas_type}/{transition}"

        # ---- 1. 收集 Step 3 各压力点的拟合线强 ----
        diluent_col = "gamma0_O2"
        records = self._collect_sw_records(t_dir, diluent_col)
        if not records:
            logger.info(f"\n  [{tag}] 未找到 Step 3 拟合结果，跳过")
            return

        # ---- 2. MAD 筛选 ----
        sw_df = self._screen_sw(records, tag)
        kept = sw_df[sw_df["keep"]]
        if len(kept) < 2:
            logger.warning(f"   保留点数 < 2，无法联合拟合，跳过")
            return

        # ---- 3. 收集保留的 etalon CSV ----
        etalon_csvs, labels = self._collect_etalon_csvs(
            gas_type, transition, kept)
        if len(etalon_csvs) < 2:
            logger.warning(f"   有效 etalon CSV < 2，跳过联合拟合")
            return

        # ---- 4. 多光谱联合拟合 ----
        logger.info(f"\n  开始多光谱联合拟合 ({len(etalon_csvs)} 条光谱)...")
        # 使用第一个压力目录的名称确定气体参数
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

        for t_dir in sorted(o2n2_dir.iterdir()):
            if not t_dir.is_dir() or t_dir.name.startswith("."):
                continue
            transition = t_dir.name

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

            logger.info(f"\n  [{transition}] 线性回归提取 N₂ 参数...")
            logger.info(f"    O₂+N₂ 数据: {mix_stats}")
            logger.info(f"    纯 O₂ 参考: {o2_multi}")

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

    # ==============================================================
    # 汇总表: 将所有拟合参数合并为一张主表
    # ==============================================================
    _MASTER_TABLE_NAME = "spectral_parameters.csv"

    def _build_master_table(self) -> None:
        """将 O₂ 多光谱联合拟合和 N₂ 线性回归的所有参数
        合并为一张主表，以跃迁波数为第一列。

        若表格已存在，先备份为 spectral_parameters_YYYYMMDD_HHMMSS.csv，
        再写入新表。
        """
        logger.info("\n" + "=" * 60)
        logger.info("  汇总: 构建参数主表")
        logger.info("=" * 60)

        rows: list[dict] = []

        # 收集所有跃迁 (从 final/O2 和 final/O2_N2 中发现)
        transitions: set[str] = set()
        for sub in ["O2", "O2_N2"]:
            d = self.final_root / sub
            if d.exists():
                for t_dir in d.iterdir():
                    if t_dir.is_dir() and not t_dir.name.startswith("."):
                        transitions.add(t_dir.name)

        for transition in sorted(transitions):
            rec: dict = {"nu": transition}

            # ── O₂ 多光谱联合拟合 ──
            o2_csv = self.final_root / "O2" / transition / "multi_fit_result.csv"
            if o2_csv.exists():
                try:
                    o2_df = pd.read_csv(o2_csv)
                    if not o2_df.empty:
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

        # 列排序: nu → O₂ 参数 → N₂ 参数 → 辅助信息
        desired_order = ["nu"]
        # O₂ 参数 (值 + 误差 交替)
        o2_params = ["sw", "gamma0_O2", "n_gamma0_O2",
                     "SD_gamma_O2", "delta0_O2", "SD_delta_O2"]
        for p in o2_params:
            desired_order.append(p)
            desired_order.append(f"{p}_err")
        # N₂ 参数
        n2_params = ["gamma0_N2", "SD_gamma_N2", "delta0_N2", "SD_delta_N2"]
        for p in n2_params:
            desired_order.append(p)
            desired_order.append(f"{p}_err")
            desired_order.append(f"{p}_R2")
            desired_order.append(f"{p}_npts")
        # 辅助列
        desired_order.extend(["n_spectra_O2", "QF_O2", "residual_std_O2"])

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

        # 打印表格
        logger.info(f"\n  {'─' * 80}")
        logger.info(f"  参数主表 ({len(master_df)} 个跃迁)")
        logger.info(f"  {'─' * 80}")
        for _, row in master_df.iterrows():
            logger.info(f"\n  ν = {row['nu']} cm⁻¹")
            for col in master_df.columns:
                if col == "nu":
                    continue
                val = row[col]
                if pd.notna(val) and val != "":
                    logger.info(f"    {col:<25s} = {val}")
        logger.info(f"  {'─' * 80}")

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

        self._print_summary_table(df, summary_path)
        self._generate_fit_statistics()

    def _print_summary_table(self, df: pd.DataFrame, summary_path: Path) -> None:
        """打印单光谱拟合汇总表"""
        logger.info(f"\n{'#' * 60}")
        logger.info(f"  单光谱拟合结果汇总 ({self.lineprofile} 线形)")
        logger.info(f"{'#' * 60}")

        x_cols = ["gas_type", "transition", "pressure_label", "x_shift"]
        x_cols = [c for c in x_cols if c in df.columns]
        x_shifts = df[x_cols].drop_duplicates()
        for _, xs in x_shifts.iterrows():
            prefix = f"{xs.get('gas_type', '')}/" if "gas_type" in xs else ""
            logger.info(f"\n  [{prefix}{xs['transition']}/{xs['pressure_label']}] "
                  f"x_shift = {xs['x_shift']:.6f} cm⁻¹")

        fitted = df[df["sw_vary"] == True] if "sw_vary" in df.columns else df
        if fitted.empty:
            fitted = df

        # 找到展宽列名
        gamma_col = next((c for c in df.columns if c.startswith("gamma0_")
                          and not c.endswith("_err")), "gamma0")

        logger.info(f"\n  {'─' * 110}")
        logger.info(f"  {'气体':<6s} {'跃迁':<12s} {'压力':<28s} {'ν (cm⁻¹)':<14s} "
              f"{'S (cm/molec)':<14s} "
              f"{gamma_col:<18s} "
              f"{'Res. σ':<12s}")
        logger.info(f"  {'─' * 110}")

        for _, row in fitted.iterrows():
            gas = str(row.get("gas_type", ""))
            gamma_val = row.get(gamma_col, 0)
            gamma_err = row.get(f"{gamma_col}_err", 0)
            gamma_str = f"{gamma_val:.6f}"
            if gamma_err > 0:
                gamma_str += f" ± {gamma_err:.6f}"
            logger.info(
                f"  {gas:<6s} {row['transition']:<12s} "
                f"{str(row['pressure_label']):<28s} "
                f"{row['nu']:>12.6f}  "
                f"{row['sw']:>12.4e}  "
                f"{gamma_str:<18s} "
                f"{row['residual_std']:>10.4e}"
            )

        logger.info(f"  {'─' * 110}")
        logger.info(f"\n  汇总表: {summary_path}")
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
                        rec = {
                            "pressure": pressure_label,
                            "nu_HITRAN": row["nu"],
                            "sw": row["sw"] * scale,
                            "sw_err": row.get("sw_err", 0) * scale,
                            "x_shift": x_shift,
                            "residual_std": residual_std,
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
                logger.info(f"\n  统计表: {out_path}")

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


# ==================================================================
# 便捷函数 (向后兼容)
# ==================================================================
def run_pipeline(**kwargs) -> None:
    """便捷函数: 创建 CRDSPipeline 并执行完整流水线

    支持的关键字参数同 CRDSPipeline.__init__
    """
    pipeline = CRDSPipeline(**kwargs)
    pipeline.run()

