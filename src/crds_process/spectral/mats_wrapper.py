"""MATS 光谱拟合封装 (Multi-spectrum Analysis Tool for Spectroscopy)

将去除标准具后的衰荡时间数据转换为 MATS 输入格式，
使用 Voigt / SDVP 线形拟合，提取光谱参数。

MATS 工作流:
    1. 准备 spectrum CSV (MATS 格式)
    2. 从 HITRAN 构建 param_linelist
    3. Spectrum → Dataset → Generate_FitParam_File → Fit_DataSet
    4. 拟合 + 残差分析

核心类:
    MATSSpectrumPreparer — 数据格式转换
    HitranLinelistBuilder — HITRAN → MATS 线表
    MATSFitter           — 拟合控制器
    MATSBatchProcessor   — 自动发现 & 批量处理

用法:
    from crds_process.spectral.mats_wrapper import batch_mats_fitting
    batch_mats_fitting()
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from crds_process.log import logger


# ==================================================================
# 项目级常量
# ==================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ETALON_ROOT = PROJECT_ROOT / "output" / "results" / "etalon"
MATS_ROOT = PROJECT_ROOT / "output" / "results" / "mats"
HITRAN_DIR = PROJECT_ROOT / "data" / "hitran"
_ETALON_CSV = "tau_etalon_corrected.csv"


# ==================================================================
# 静默导入 MATS/HAPI
# ==================================================================
def _import_mats():
    """静默导入 MATS"""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    try:
        import MATS
        return MATS
    except ImportError:
        raise ImportError(
            "MATS 未安装。请安装: pip install git+https://github.com/usnistgov/MATS.git"
        )
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout, sys.stderr = old_out, old_err


# ==================================================================
# 数据准备
# ==================================================================
class MATSSpectrumPreparer:
    """将 etalon corrected 数据转换为 MATS 所需的 CSV 格式

    MATS Spectrum 类期望的 CSV 列名:
        - pressure_column: 压力 (Torr)
        - temperature_column: 温度 (°C)
        - frequency_column: 波数 (cm⁻¹)
        - tau_column: 衰荡时间 (μs)
        - tau_stats_column: tau 标准差 (可选)
    """

    MATS_PRESSURE = "Cavity Pressure /Torr"
    MATS_TEMPERATURE = "Cavity Temperature Side 2 /C"
    MATS_FREQUENCY = "Total Frequency /MHz"
    MATS_TAU = "Mean tau/us"
    MATS_TAU_STATS = "Tau_stats"

    def __init__(
        self,
        pressure_col: str = "pressure",
        temperature_col: str = "temperature",
        wavenumber_col: str = "wavenumber",
        tau_col: str = "tau_mean_no_etalon",
        tau_stats_col: str | None = "tau_std",
    ):
        self.pressure_col = pressure_col
        self.temperature_col = temperature_col
        self.wavenumber_col = wavenumber_col
        self.tau_col = tau_col
        self.tau_stats_col = tau_stats_col

    def prepare(self, df: pd.DataFrame, save_stem: str) -> str:
        """转换并保存 MATS 格式 CSV

        Parameters
        ----------
        df : pd.DataFrame
            etalon corrected 数据
        save_stem : str
            不含 .csv 后缀的文件名 (MATS Spectrum 需要此格式)

        Returns
        -------
        str
            不含后缀的路径 (直接传给 MATS Spectrum)
        """
        save_stem = str(save_stem)
        if save_stem.endswith(".csv"):
            save_stem = save_stem[:-4]
        csv_path = save_stem + ".csv"

        mats_df = pd.DataFrame()
        mats_df[self.MATS_PRESSURE] = df[self.pressure_col]
        mats_df[self.MATS_TEMPERATURE] = df[self.temperature_col]
        mats_df[self.MATS_FREQUENCY] = df[self.wavenumber_col]
        mats_df[self.MATS_TAU] = df[self.tau_col]
        if self.tau_stats_col and self.tau_stats_col in df.columns:
            mats_df[self.MATS_TAU_STATS] = df[self.tau_stats_col]

        parent = Path(csv_path).parent
        if str(parent) != ".":
            parent.mkdir(parents=True, exist_ok=True)
        mats_df.to_csv(csv_path, index=False)
        return save_stem


# ==================================================================
# HITRAN → MATS 线表
# ==================================================================
class HitranLinelistBuilder:
    """从 HITRAN 数据构建 MATS param_linelist"""

    _MOL_PREFIX = {1: "H2O", 2: "CO2", 6: "CH4", 7: "O2"}

    def __init__(
        self,
        molecule: int = 7,
        isotopologue: int = 1,
        hitran_dir: Path | str | None = None,
        hitran_table: str | None = None,
    ):
        self.molecule = molecule
        self.isotopologue = isotopologue
        self.hitran_dir = str(hitran_dir or HITRAN_DIR)
        self.hitran_table = hitran_table
        self._hapi = None

    @property
    def mol_prefix(self) -> str:
        return self._MOL_PREFIX.get(self.molecule, f"M{self.molecule}")

    def _init_hapi(self):
        if self._hapi is not None:
            return
        MATS = _import_mats()
        self._hapi = MATS.hapi
        os.makedirs(self.hitran_dir, exist_ok=True)
        saved = os.getcwd()
        os.chdir(self.hitran_dir)
        try:
            old_out = sys.stdout
            sys.stdout = open(os.devnull, "w")
            try:
                self._hapi.db_begin(self.hitran_dir)
            finally:
                sys.stdout.close()
                sys.stdout = old_out
        finally:
            os.chdir(saved)

    def _resolve_table(self, wn_min: float, wn_max: float) -> str:
        self._init_hapi()
        hapi = self._hapi
        if self.hitran_table:
            return self.hitran_table
        for tname in hapi.LOCAL_TABLE_CACHE:
            if tname.startswith(self.mol_prefix):
                self.hitran_table = tname
                return tname
        tname = f"{self.mol_prefix}_iso{self.isotopologue}"
        saved = os.getcwd()
        os.chdir(self.hitran_dir)
        try:
            logger.info(f"  [HITRAN] 下载 {self.mol_prefix} "
                  f"({wn_min:.0f}-{wn_max:.0f} cm⁻¹)...")
            hapi.fetch(TableName=tname, M=self.molecule,
                       I=self.isotopologue, numin=wn_min, numax=wn_max)
        finally:
            os.chdir(saved)
        self.hitran_table = tname
        return tname

    def build(self, wn_min: float, wn_max: float,
              save_path: Path | str | None = None) -> pd.DataFrame:
        """构建 MATS 格式 param_linelist"""
        self._init_hapi()
        table = self._resolve_table(wn_min - 5, wn_max + 5)
        data = self._hapi.LOCAL_TABLE_CACHE[table]["data"]
        df = pd.DataFrame({
            "molec_id": data["molec_id"],
            "local_iso_id": data["local_iso_id"],
            "nu": data["nu"],
            "sw": data["sw"],
            "gamma0_air": data["gamma_air"],
            "n_gamma0_air": data["n_air"],
            "delta0_air": data["delta_air"],
            "elower": data["elower"],
            "gamma0_O2": data["gamma_self"],
            "n_gamma0_O2": data["n_air"],  # 近似使用 n_air
        })
        margin = 5.0
        df = df[(df["nu"] >= wn_min - margin) & (df["nu"] <= wn_max + margin)]
        df["sw_scale_factor"] = 1.0
        # 注意: n_gamma0_O2 已从 HITRAN 赋值，不在此列表中覆盖为 0
        # SD_gamma 设置非零初始值 (SDVP 线形需要)
        for col in [
            "n_delta0_air", "n_delta0_O2",
            "SD_delta_air", "n_gamma2_air", "n_delta2_air",
            "n_gamma2_O2", "n_delta2_O2",
            "nuVC_air", "nuVC_O2", "n_nuVC_air", "n_nuVC_O2",
            "eta_air", "eta_O2", "y_air", "n_y_air", "y_O2", "n_y_O2",
        ]:
            df[col] = 0.0
        # delta0_O2 初始猜测 (O₂ A-band 典型值 ~ -0.005 cm⁻¹/atm)
        df["delta0_O2"] = -0.005
        # SD_delta 初始猜测
        df["SD_delta_air"] = 0.05
        df["SD_delta_O2"] = 0.05
        # SD_gamma 初始猜测 ~0.10 (O₂ A-band 典型值)
        df["SD_gamma_air"] = 0.10
        df["SD_gamma_O2"] = 0.10
        df = df.reset_index(drop=True)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(str(save_path), index=False)
        return df


# ==================================================================
# 拟合结果
# ==================================================================
@dataclass
class MATSFitResult:
    """MATS 拟合结果"""
    fit_result: object = None
    param_linelist: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_linelist: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    residual_std: float = 0.0
    chi_squared: float = 0.0
    qf: float = 0.0

    def summary(self) -> str:
        lines = [
            f"MATS Fit Result:",
            f"  Residual σ = {self.residual_std:.6e}",
            f"  QF         = {self.qf:.4f}",
        ]
        if not self.param_linelist.empty:
            for _, row in self.param_linelist.iterrows():
                nu = row.get("nu", 0)
                sw = row.get("sw", 0)
                gamma0_air = row.get("gamma0_air", 0)
                gamma0_O2 = row.get("gamma0_O2", row.get("gamma0_self", 0))
                delta0_air = row.get("delta0_air", 0)
                delta0_O2 = row.get("delta0_O2", row.get("delta0_self", 0))
                sd_gamma = row.get("SD_gamma_O2", row.get("SD_gamma_self", row.get("SD_gamma_air", 0)))
                sd_delta = row.get("SD_delta_O2", row.get("SD_delta_self", row.get("SD_delta_air", 0)))
                lines.append(
                    f"  Line: ν={nu:.6f} cm⁻¹, "
                    f"S={sw:.4e}, "
                    f"γ₀_O2={gamma0_O2:.5f}, "
                    f"γ₀_air={gamma0_air:.5f}, "
                    f"δ₀_O2={delta0_O2:.6f}, "
                    f"δ₀_air={delta0_air:.6f}, "
                    f"SD_γ={sd_gamma:.4f}, "
                    f"SD_δ={sd_delta:.4f}"
                )
        return "\n".join(lines)


# ==================================================================
# MATS 拟合器
# ==================================================================
class MATSFitter:
    """MATS 光谱拟合控制器

    Parameters
    ----------
    molecule : int
        HITRAN 分子编号 (7 = O₂)
    isotopologue : int
        同位素编号
    molefraction : dict
        摩尔分数 {molec_id: fraction}
    diluent : str
        稀释气体 (纯 O₂ 用 "O2"，混合气用 "air")
    lineprofile : str
        线形 ("VP", "SDVP", "NGP", "SDNGP", "HTP")
    baseline_order : int
        基线多项式阶数
    etalons : dict
        残余标准具 {order: [amp, period, phase]}
    fit_intensity : float
        最低可浮动线强
    threshold_intensity : float
        最低模拟线强
    """

    def __init__(
        self,
        molecule: int = 7,
        isotopologue: int = 1,
        molefraction: dict | None = None,
        diluent: str = "O2",
        Diluent: dict | None = None,
        lineprofile: str = "SDVP",
        baseline_order: int = 1,
        etalons: dict | None = None,
        fit_intensity: float = 1e-30,
        threshold_intensity: float = 1e-35,
    ):
        self.molecule = molecule
        self.isotopologue = isotopologue
        self.molefraction = molefraction or {molecule: 1.0}
        self.diluent = diluent
        # 显式设置 Diluent (与参考脚本一致)
        # 纯 O₂: Diluent={'O2': {'composition':1, 'm': 31.9988}}
        # O₂+N₂: Diluent={'air': {'composition':1, 'm': 28.014}}
        if Diluent is not None:
            self.Diluent = Diluent
        else:
            if diluent == "O2":
                self.Diluent = {"O2": {"composition": 1, "m": 31.9988}}
            elif diluent == "self":
                self.Diluent = {"self": {"composition": 1, "m": 31.9988}}
            else:
                self.Diluent = {diluent: {"composition": 1, "m": 28.964}}
        self.lineprofile = lineprofile
        self.baseline_order = baseline_order
        self.etalons = etalons or {}
        self.fit_intensity = fit_intensity
        self.threshold_intensity = threshold_intensity
        self._preparer = MATSSpectrumPreparer()
        self._linelist_builder = HitranLinelistBuilder(
            molecule=molecule, isotopologue=isotopologue,
        )

    def fit(
        self,
        etalon_csv: Path | str,
        output_dir: Path | str,
        dataset_name: str = "crds_fit",
    ) -> MATSFitResult:
        """执行单光谱 MATS 拟合"""
        MATS = _import_mats()
        from MATS import Spectrum, Dataset, Generate_FitParam_File, Fit_DataSet

        etalon_csv = Path(etalon_csv)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: 读取数据
        df = pd.read_csv(etalon_csv)
        wn = df["wavenumber"].values
        wn_min, wn_max = float(wn.min()), float(wn.max())
        logger.info(f"  波数范围: {wn_min:.5f} ~ {wn_max:.5f} cm⁻¹, 点数: {len(wn)}")

        # MATS 使用 cwd 相对路径读写文件，切到 output_dir
        saved_dir = os.getcwd()
        os.chdir(str(output_dir))
        try:
            # Step 2: 准备 MATS 格式 spectrum CSV (相对路径)
            spec_name = f"{dataset_name}_spectrum"
            self._preparer.prepare(df, spec_name)

            # Step 3: 构建 param_linelist
            linelist_csv = f"{dataset_name}_linelist.csv"
            param_linelist = self._linelist_builder.build(
                wn_min, wn_max, save_path=linelist_csv,
            )
            logger.info(f"  线表: {len(param_linelist)} 条谱线")
            for _, row in param_linelist.iterrows():
                if row["sw"] > self.threshold_intensity:
                    logger.info(f"    ν={row['nu']:.6f}, S={row['sw']:.4e}, "
                          f"γ₀_air={row['gamma0_air']:.5f}")
            if len(param_linelist) == 0:
                logger.warning("  未找到谱线，跳过拟合")
                return MATSFitResult()

            # Step 4~8: MATS 拟合
            return self._run_mats_fit(
                MATS, spec_name, param_linelist, dataset_name,
                Spectrum, Dataset, Generate_FitParam_File, Fit_DataSet,
            )
        finally:
            os.chdir(saved_dir)

    def fit_multi(
        self,
        etalon_csvs: list[Path | str],
        labels: list[str],
        output_dir: Path | str,
        dataset_name: str = "crds_multi",
    ) -> MATSFitResult:
        """多光谱联合拟合

        将多个压力下的光谱同时放入一个 Dataset 拟合，
        线强 (sw)、���宽系数 (gamma0) 等参数在所有光谱间共享约束，
        而基线和 x_shift 对每条光谱独立。

        Parameters
        ----------
        etalon_csvs : list[Path | str]
            各压力的 etalon corrected CSV 路径列表
        labels : list[str]
            对应的标签 (如 ["100Torr", "150Torr", ...])
        output_dir : Path | str
            输出目录
        dataset_name : str
            数据集名

        Returns
        -------
        MATSFitResult
        """
        MATS = _import_mats()
        from MATS import Spectrum, Dataset, Generate_FitParam_File, Fit_DataSet

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 读取所有数据, 获取全局波数范围
        all_dfs = []
        wn_global_min, wn_global_max = 1e9, 0
        for csv_path in etalon_csvs:
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
            wn_global_min = min(wn_global_min, float(df["wavenumber"].min()))
            wn_global_max = max(wn_global_max, float(df["wavenumber"].max()))

        logger.info(f"  多光谱联合拟合: {len(etalon_csvs)} 条光谱")
        logger.info(f"  全局波数范围: {wn_global_min:.5f} ~ {wn_global_max:.5f} cm⁻¹")
        for i, (lbl, df) in enumerate(zip(labels, all_dfs)):
            logger.info(f"    [{i+1}] {lbl}: {len(df)} 点, "
                  f"P={df['pressure'].mean():.1f} Torr")

        saved_dir = os.getcwd()
        os.chdir(str(output_dir))
        try:
            # 构建 param_linelist (只需一份, 所有光谱共享)
            linelist_csv = f"{dataset_name}_linelist.csv"
            param_linelist = self._linelist_builder.build(
                wn_global_min, wn_global_max, save_path=linelist_csv,
            )
            logger.info(f"  线表: {len(param_linelist)} 条谱线")
            for _, row in param_linelist.iterrows():
                if row["sw"] > self.threshold_intensity:
                    logger.info(f"    ν={row['nu']:.6f}, S={row['sw']:.4e}")
            if len(param_linelist) == 0:
                logger.warning("  未找到谱线，跳过拟合")
                return MATSFitResult()

            # 为每个光谱创建 Spectrum 对象
            spectra = []
            for i, (lbl, df) in enumerate(zip(labels, all_dfs)):
                spec_name = f"{dataset_name}_{lbl}_spectrum"
                self._preparer.prepare(df, spec_name)

                spec_csv = pd.read_csv(spec_name + ".csv")
                has_stats = self._preparer.MATS_TAU_STATS in spec_csv.columns

                spec = Spectrum(
                    spec_name,
                    molefraction=self.molefraction,
                    natural_abundance=True,
                    diluent=self.diluent,
                    Diluent=self.Diluent,
                    input_freq=False,
                    input_tau=True,
                    pressure_column=self._preparer.MATS_PRESSURE,
                    temperature_column=self._preparer.MATS_TEMPERATURE,
                    frequency_column=self._preparer.MATS_FREQUENCY,
                    tau_column=self._preparer.MATS_TAU,
                    tau_stats_column=(self._preparer.MATS_TAU_STATS
                                     if has_stats else None),
                    etalons=self.etalons,
                    nominal_temperature=296,
                    baseline_order=self.baseline_order,
                )
                spectra.append(spec)
                logger.info(f"    Spectrum {lbl}: "
                      f"P={spec.pressure:.4f} atm, T={spec.temperature:.2f} K")

            # 联合 Dataset
            ds = Dataset(spectra, dataset_name, param_linelist)
            base_linelist = ds.generate_baseline_paramlist()

            param_save = f"{dataset_name}_Parameter_LineList"
            base_save = f"{dataset_name}_baseline_paramlist"

            # Generate_FitParam_File — constrain=True 使参数在光谱间约束
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

            param_file = fitparam.param_linelist_savename
            base_file = fitparam.base_linelist_savename

            # 拟合
            logger.info(f"  开始多光谱联合拟合 ({self.lineprofile} 线形, "
                  f"{len(spectra)} 光谱)...")
            fit = Fit_DataSet(
                ds, base_file, param_file,
                minimum_parameter_fit_intensity=self.fit_intensity,
                weight_spectra=False,
            )
            params = fit.generate_params()

            for param in params:
                if "SD_gamma" in param and params[param].vary:
                    params[param].set(value=0.10, min=0.01, max=0.25)
                elif "SD_delta" in param and params[param].vary:
                    params[param].set(value=0.05, min=0.0, max=0.25)
                elif "delta0" in param and params[param].vary:
                    params[param].set(min=-0.05, max=0.05)

            result = fit.fit_data(params, wing_cutoff=25)

            # 更新参数 & 生成结果
            fit.residual_analysis(result, indv_resid_plot=False)
            fit.update_params(result)
            summary = ds.generate_summary_file(save_file=True)

            res_col = next((c for c in summary.columns if "Residual" in c), None)
            residuals = summary[res_col].values if res_col else np.array([])
            res_std = float(np.std(residuals)) if len(residuals) > 0 else 0.0
            updated_params = pd.read_csv(param_file + ".csv", index_col=0)
            updated_baseline = pd.read_csv(base_file + ".csv")

            mats_result = MATSFitResult(
                fit_result=result,
                param_linelist=updated_params,
                baseline_linelist=updated_baseline,
                summary_df=summary,
                residual_std=res_std,
                qf=float(ds.average_QF()) if hasattr(ds, "average_QF") else 0.0,
            )
            logger.info(f"  多光谱联合拟合完成!")
            logger.info(f"  {mats_result.summary()}")
            return mats_result
        finally:
            os.chdir(saved_dir)

    def _run_mats_fit(
        self, MATS, spec_stem, param_linelist, dataset_name,
        Spectrum, Dataset, Generate_FitParam_File, Fit_DataSet,
    ) -> MATSFitResult:
        """在 output_dir 中执行 MATS 拟合流程 (参考 Fitting_Protocol_ABand)"""

        # 检查 tau_stats 列是否存在
        spec_csv = pd.read_csv(str(spec_stem) + ".csv")
        has_stats = self._preparer.MATS_TAU_STATS in spec_csv.columns

        # ---- Step 4: 创建 Spectrum ----
        # 关键: 显式传入 Diluent 和 diluent
        spec = Spectrum(
            str(spec_stem),
            molefraction=self.molefraction,
            natural_abundance=True,
            diluent=self.diluent,
            Diluent=self.Diluent,
            input_freq=False,
            input_tau=True,
            pressure_column=self._preparer.MATS_PRESSURE,
            temperature_column=self._preparer.MATS_TEMPERATURE,
            frequency_column=self._preparer.MATS_FREQUENCY,
            tau_column=self._preparer.MATS_TAU,
            tau_stats_column=self._preparer.MATS_TAU_STATS if has_stats else None,
            etalons=self.etalons,
            nominal_temperature=296,
            baseline_order=self.baseline_order,
        )
        logger.info(f"  Spectrum: P={spec.pressure:.4f} atm, T={spec.temperature:.2f} K")

        # ---- Step 5: Dataset ----
        ds = Dataset([spec], dataset_name, param_linelist)
        base_linelist = ds.generate_baseline_paramlist()

        param_save = f"{dataset_name}_Parameter_LineList"
        base_save = f"{dataset_name}_baseline_paramlist"

        # ---- Step 6: Generate_FitParam_File ----
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
            vary_n_gamma0={self.molecule: {self.isotopologue: False}},
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

        param_file = fitparam.param_linelist_savename
        base_file = fitparam.base_linelist_savename

        # ---- Step 7: 拟合 ----
        logger.info(f"  开始拟合 ({self.lineprofile} 线形)...")
        fit = Fit_DataSet(
            ds, base_file, param_file,
            minimum_parameter_fit_intensity=self.fit_intensity,
            weight_spectra=False,
        )
        params = fit.generate_params()

        # SD_gamma / SD_delta / delta0 约束 (参考 Fitting_Protocol_ABand)
        for param in params:
            if "SD_gamma" in param and params[param].vary:
                params[param].set(value=0.10, min=0.01, max=0.25)
            elif "SD_delta" in param and params[param].vary:
                params[param].set(value=0.05, min=0.0, max=0.25)
            elif "delta0" in param and params[param].vary:
                params[param].set(min=-0.05, max=0.05)

        result = fit.fit_data(params, wing_cutoff=25)

        # ---- Step 8: 更新参数 & 生成结果 ----
        fit.residual_analysis(result, indv_resid_plot=False)
        fit.update_params(result)
        summary = ds.generate_summary_file(save_file=True)

        # MATS summary 列名格式: 'Residuals (ppm/cm)', 'Alpha (ppm/cm)', etc.
        res_col = next((c for c in summary.columns if "Residual" in c), None)
        residuals = summary[res_col].values if res_col else np.array([])
        res_std = float(np.std(residuals)) if len(residuals) > 0 else 0.0
        updated_params = pd.read_csv(param_file + ".csv", index_col=0)
        updated_baseline = pd.read_csv(base_file + ".csv")

        mats_result = MATSFitResult(
            fit_result=result,
            param_linelist=updated_params,
            baseline_linelist=updated_baseline,
            summary_df=summary,
            residual_std=res_std,
            qf=float(ds.average_QF()) if hasattr(ds, "average_QF") else 0.0,
        )
        logger.info(f"  拟合完成!")
        logger.info(f"  {mats_result.summary()}")
        return mats_result

    def plot_result(
        self, result: MATSFitResult, output_dir: Path | str,
        title: str = "MATS Fit",
    ):
        """绘制拟合结果

        多光谱联合拟合时，按 Spectrum Number 分组绘制，
        避免不同光谱的基线值连线导致斜线问题。
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        summary = result.summary_df
        if summary.empty:
            return

        # MATS summary 列名查找
        def _find_col(keywords):
            for c in summary.columns:
                if all(k.lower() in c.lower() for k in keywords):
                    return c
            return None

        wn_col = _find_col(["Wavenumber"])
        alpha_col = _find_col(["Alpha"])
        model_col = _find_col(["Model"])
        res_col = _find_col(["Residual"])
        tau_col = _find_col(["Tau"])
        spec_col = _find_col(["Spectrum", "Number"])
        name_col = _find_col(["Spectrum", "Name"])

        if wn_col is None:
            logger.warning(f"  找不到波数列，可用: {list(summary.columns)}")
            return

        # 判断是否为多光谱
        is_multi = (spec_col is not None
                    and summary[spec_col].nunique() > 1)

        # 全局波数范围
        wn_all = summary[wn_col].values
        wn_min_global = float(wn_all.min())
        wn_max_global = float(wn_all.max())

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        # 读取各光谱的 x_shift (从 baseline_linelist)
        x_shifts = {}
        if result.baseline_linelist is not None and not result.baseline_linelist.empty:
            bl = result.baseline_linelist
            if "Spectrum Number" in bl.columns and "x_shift" in bl.columns:
                for _, row in bl.iterrows():
                    x_shifts[int(row["Spectrum Number"])] = float(row["x_shift"])

        def _shifted_wn(spec_id, wn_arr):
            """给波数加上 x_shift 校正"""
            xs = x_shifts.get(int(spec_id), 0.0)
            return wn_arr + xs

        # ── Panel 1: Alpha + Model ──
        ax = axes[0]
        all_residuals = []

        if is_multi:
            # 多光谱: 按 Spectrum Number 分组绘制
            groups = summary.groupby(spec_col)
            colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

            for i, (spec_id, grp) in enumerate(groups):
                wn = _shifted_wn(spec_id, grp[wn_col].values)
                color = colors[i]

                # 提取光谱标签
                label = f"Spec {spec_id}"
                if name_col and not grp[name_col].empty:
                    raw_name = str(grp[name_col].iloc[0])
                    # 从名称中提取压力标签 (如 "...100Torr_spectrum" → "100Torr")
                    import re
                    m = re.search(r'(\d+Torr)', raw_name)
                    if m:
                        label = m.group(1)

                if alpha_col:
                    ax.plot(wn, grp[alpha_col].values, ".",
                            ms=2, alpha=0.5, color=color, label=label)
                if model_col:
                    ax.plot(wn, grp[model_col].values, "-",
                            lw=1, color=color)

                if res_col:
                    all_residuals.extend(grp[res_col].values)
        else:
            # 单光谱
            wn = _shifted_wn(1, wn_all)
            if alpha_col:
                ax.plot(wn, summary[alpha_col].values, "b.",
                        ms=2, alpha=0.5, label="Data")
            if model_col:
                ax.plot(wn, summary[model_col].values, "r-",
                        lw=1, label="MATS Fit")

            if res_col:
                all_residuals = list(summary[res_col].values)

        ax.set_ylabel("α (ppm/cm)")
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=3, loc="upper right")
        ax.grid(True, alpha=0.3)

        # ── Panel 2: Residuals ──
        ax = axes[1]
        if res_col:
            if is_multi:
                for i, (spec_id, grp) in enumerate(groups):
                    wn = _shifted_wn(spec_id, grp[wn_col].values)
                    ax.plot(wn, grp[res_col].values,
                            ".", ms=2, alpha=0.4, color=colors[i])
            else:
                wn = _shifted_wn(1, wn_all)
                ax.plot(wn, summary[res_col].values,
                        ".", ms=2, color="tomato", alpha=0.5)
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.set_ylabel("Residual (ppm/cm)")
            res_std = float(np.std(all_residuals)) if all_residuals else 0
            ax.set_title(f"Residual (σ={res_std:.4e})")
        ax.grid(True, alpha=0.3)

        # ── Panel 3: Tau ──
        ax = axes[2]
        if tau_col and "Error" not in tau_col:
            if is_multi:
                for i, (spec_id, grp) in enumerate(groups):
                    wn = _shifted_wn(spec_id, grp[wn_col].values)
                    ax.plot(wn, grp[tau_col].values,
                            ".", ms=2, alpha=0.5, color=colors[i])
            else:
                wn = _shifted_wn(1, wn_all)
                ax.plot(wn, summary[tau_col].values, "b.",
                        ms=2, alpha=0.5)
            ax.set_ylabel("τ (μs)")
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        save_path = output_dir / "mats_fit.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info(f"  图表已保存: {save_path}")
        plt.close(fig)


# ==================================================================
# 批量处理器
# ==================================================================
class MATSBatchProcessor:
    """MATS 光谱拟合批量处理器"""

    def __init__(
        self,
        etalon_root: Path | None = None,
        mats_root: Path | None = None,
        fitter: MATSFitter | None = None,
    ):
        self.etalon_root = etalon_root or ETALON_ROOT
        self.mats_root = mats_root or MATS_ROOT
        self.fitter = fitter or MATSFitter()

    def discover(self) -> list[tuple[str, str, Path]]:
        tasks: list[tuple[str, str, Path]] = []
        if not self.etalon_root.exists():
            return tasks
        for t_dir in sorted(self.etalon_root.iterdir()):
            if not t_dir.is_dir() or t_dir.name.startswith("."):
                continue
            for p_dir in sorted(t_dir.iterdir()):
                if not p_dir.is_dir() or p_dir.name.startswith("."):
                    continue
                csv = p_dir / _ETALON_CSV
                if csv.exists():
                    tasks.append((t_dir.name, p_dir.name, csv))
        return tasks

    def process_one(self, csv_path: Path, output_dir: Path,
                    label: str = "") -> MATSFitResult | None:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = self.fitter.fit(
                csv_path, output_dir,
                dataset_name=label.replace("/", "_").replace(" ", "") or "crds",
            )
            if result.summary_df is not None and not result.summary_df.empty:
                self.fitter.plot_result(
                    result, output_dir,
                    title=f"MATS Fit — {label}" if label else "MATS Fit",
                )
            return result
        except Exception as e:
            logger.error(f"    拟合失败: {e}")
            logger.exception("    详细错误信息:")
            return None

    def run(self):
        tasks = self.discover()
        if not tasks:
            logger.error(f"未在 {self.etalon_root} 下找到 "
                  f"{{跃迁波数}}/{{压力}}/{_ETALON_CSV}")
            return

        logger.info(f"{'#' * 60}")
        logger.info(f"  CRDS MATS 光谱拟合")
        logger.info(f"  输入: {self.etalon_root}")
        logger.info(f"  输出: {self.mats_root}")
        logger.info(f"  发现 {len(tasks)} 个数据集:")
        for t, p, _ in tasks:
            logger.info(f"    {t}/{p}/")
        logger.info(f"{'#' * 60}")

        ok = 0
        for i, (transition, pressure, csv_path) in enumerate(tasks, 1):
            out_dir = self.mats_root / transition / pressure
            logger.info(f"\n{'=' * 60}")
            logger.info(f"  [{i}/{len(tasks)}] {transition} / {pressure}")
            logger.info(f"{'=' * 60}")
            if self.process_one(csv_path, out_dir, f"{transition}/{pressure}"):
                ok += 1

        logger.info(f"\n\n{'#' * 60}")
        logger.info(f"  全部完成! {ok}/{len(tasks)} 成功")
        logger.info(f"{'#' * 60}")
        for t, p, _ in tasks:
            d = self.mats_root / t / p
            if d.exists():
                logger.info(f"\n  {t}/{p}/")
                for f in sorted(d.glob("*")):
                    if not f.name.startswith("."):
                        logger.info(f"    {f.name:<50s} {f.stat().st_size:>10,} bytes")


# ==================================================================
# 便捷函数
# ==================================================================
def batch_mats_fitting(
    etalon_root: Path | None = None,
    mats_root: Path | None = None,
    **fitter_kwargs,
):
    """批量 MATS 拟合 (便捷入口)

    Parameters
    ----------
    etalon_root : Path, optional
        标准具去除结果目录
    mats_root : Path, optional
        MATS 输出目录
    **fitter_kwargs
        传递给 MATSFitter 的参数:
        - molecule: 分子 ID (默认 7 = O₂)
        - molefraction: 摩尔分数 (默认 {7: 1.0})
        - diluent: 稀释气名 (默认 "O2")
        - Diluent: 显式稀释气字典 (默认 {"O2": {"composition":1, "m":31.9988}})
        - lineprofile: 线形 (默认 "SDVP")
        - fit_intensity: 可浮动线强阈值 (默认 1e-30)
    """
    fitter = MATSFitter(**fitter_kwargs) if fitter_kwargs else None
    MATSBatchProcessor(
        etalon_root=etalon_root,
        mats_root=mats_root,
        fitter=fitter,
    ).run()

