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

import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import combinations
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

    _MASTER_TABLE_NAME = "spectral_parameters.csv"

    # HITRAN 参考数据路径
    _HITRAN_CSV = _PROJECT_ROOT / "data" / "hitran" / "O2_9000_10000_sw_ge_1e-29.csv"

    def _load_hitran_reference(self) -> pd.DataFrame | None:
        """加载 HITRAN 参考数据 CSV，返回 DataFrame (nu 为索引列)。

        若文件不存在则返回 None。
        """
        csv_path = self._HITRAN_CSV
        if not csv_path.exists():
            logger.warning(f"  HITRAN 参考数据文件不存在: {csv_path}")
            return None
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"  已加载 HITRAN 参考数据: {csv_path.name}"
                        f" ({len(df)} 条吸收线)")
            return df
        except Exception as e:
            logger.warning(f"  读取 HITRAN 参考数据失败: {e}")
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
