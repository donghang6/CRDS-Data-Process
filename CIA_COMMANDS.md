# CIA 数据处理常用命令

下面命令默认在项目根目录运行：

```bash
cd /Users/donghang/Projects/CRDS-Data-Process
```

如果已经进入 `CRDS-Data-Process` conda 环境，可以把命令开头的
`conda run -n CRDS-Data-Process env PYTHONPATH=src` 去掉。

## 1. Ar 从原始 txt 重新运行 Step 1

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python main.py \
  --step1-dir '/Users/donghang/科研/实验数据/氧气连续吸收温度/原始数据初步处理/273K/Ar 500Torr' \
  --step1-output 'output/results/ringdown/CIA/273K/Ar 500Torr'
```

输出：

```text
output/results/ringdown/CIA/273K/Ar 500Torr/ringdown_results.csv
```

## 2. Ar 运行 CIA Step 2

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python main.py \
  --continuum \
  --from-ringdown \
  'CIA/273K/Ar 500Torr' \
  --continuum-step2-mode ar \
  --cia-fit-window 40 \
  --cia-fit-step 5 \
  --cia-fit-order 2 \
  --cia-fit-sigma 4 \
  --cia-fit-smooth 20
```

输出：

```text
output/results/continuum/CIA/273K/Ar 500Torr/continuum_step2_fit.csv
output/results/continuum/CIA/273K/Ar 500Torr/continuum_spectrum.png
```

## 3. Ar 的 lmfit 全局样条，knots-every 全局使用 15

只预览，不写入 CSV：

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/apply_lmfit_spline_region.py \
  'output/results/continuum/CIA/273K/Ar 500Torr/continuum_step2_fit.csv' \
  --column tau_fit_us \
  --knots-every 15 \
  --smooth-lambda 0.1
```

写入新列 `tau_fit_us_lmfit_spline`，不覆盖原来的 `tau_fit_us`：

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/apply_lmfit_spline_region.py \
  'output/results/continuum/CIA/273K/Ar 500Torr/continuum_step2_fit.csv' \
  --column tau_fit_us \
  --knots-every 15 \
  --smooth-lambda 0.1 \
  --apply
```

直接覆盖 `tau_fit_us`，并同步更新 loss 和 residual：

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/apply_lmfit_spline_region.py \
  'output/results/continuum/CIA/273K/Ar 500Torr/continuum_step2_fit.csv' \
  --column tau_fit_us \
  --knots-every 15 \
  --smooth-lambda 0.1 \
  --overwrite \
  --update-derived \
  --apply
```

可选：同时输出预览图。

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/apply_lmfit_spline_region.py \
  'output/results/continuum/CIA/273K/Ar 500Torr/continuum_step2_fit.csv' \
  --column tau_fit_us \
  --knots-every 15 \
  --smooth-lambda 0.1 \
  --plot 'output/results/continuum/CIA/273K/Ar 500Torr/lmfit_spline_knots15_preview.png'
```

## 4. O2 从原始 txt 重新运行 Step 1

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python main.py \
  --step1-dir '/Users/donghang/科研/实验数据/氧气连续吸收温度/原始数据初步处理/273K/O2 500Torr' \
  --step1-output 'output/results/ringdown/CIA/273K/O2 500Torr'
```

输出：

```text
output/results/ringdown/CIA/273K/O2 500Torr/ringdown_results.csv
```

## 5. O2 只做 HITRAN2024 扣除预览，不拟合

这一步用于先查看 `loss - HITRAN2024 O2` 后的结果。

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python main.py \
  --continuum \
  --from-ringdown \
  'CIA/273K/O2 500Torr' \
  --continuum-step2-mode o2-hitran
```

输出：

```text
output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_hitran_subtracted.csv
output/results/continuum/CIA/273K/O2 500Torr/continuum_spectrum.png
```

## 6. O2 运行 CIA Step 2，剔除 O2 吸收后只保留 CIA baseline

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python main.py \
  --continuum \
  --from-ringdown \
  'CIA/273K/O2 500Torr' \
  --continuum-step2-mode o2 \
  --cia-fit-window 8 \
  --cia-fit-step 1 \
  --cia-fit-order 2 \
  --cia-fit-sigma 2 \
  --cia-fit-smooth 2
```

输出：

```text
output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_fit.csv
output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_cia_baseline.csv
output/results/continuum/CIA/273K/O2 500Torr/continuum_spectrum.png
```

## 7. O2 的 lmfit 分区域样条

只预览，不写入 CSV：

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/apply_lmfit_spline_region.py \
  'output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_fit.csv' \
  --column tau_fit_us \
  --anchor-width 5 \
  --smooth-lambda 0.1 \
  --region 9100 9200 12 \
  --region 9200 9600 10 \
  --region 9600 9900 15
```

写入新列 `tau_fit_us_lmfit_spline`，不覆盖原来的 `tau_fit_us`：

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/apply_lmfit_spline_region.py \
  'output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_fit.csv' \
  --column tau_fit_us \
  --anchor-width 5 \
  --smooth-lambda 0.1 \
  --region 9100 9200 12 \
  --region 9200 9600 10 \
  --region 9600 9900 15 \
  --apply
```

直接覆盖 `tau_fit_us`，并同步更新 loss 和 residual：

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/apply_lmfit_spline_region.py \
  'output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_fit.csv' \
  --column tau_fit_us \
  --anchor-width 5 \
  --smooth-lambda 0.1 \
  --region 9100 9200 12 \
  --region 9200 9600 10 \
  --region 9600 9900 15 \
  --overwrite \
  --update-derived \
  --apply
```

如果只想对纯 CIA baseline 表做样条，可把输入文件改成：

```text
output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_cia_baseline.csv
```

## 8. 把两列整理数据插值到 9120-9820，步长 0.01

适用于无表头文件：第 1 列是波数，第 2 列是衰荡时间。

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/interpolate_processed_result_grid.py \
  '/path/to/整理好的数据.txt' \
  --no-header \
  --output '/path/to/整理好的数据_interp_9120_9820_step0p01.csv'
```

如果文件有表头，但仍然只想使用前两列：

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/interpolate_processed_result_grid.py \
  '/path/to/整理好的数据.csv' \
  --two-column \
  --output '/path/to/整理好的数据_interp_9120_9820_step0p01.csv'
```

明确写出范围和步长：

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/interpolate_processed_result_grid.py \
  '/path/to/整理好的数据.txt' \
  --no-header \
  --start 9120 \
  --end 9820 \
  --step 0.01 \
  --output '/path/to/整理好的数据_interp_9120_9820_step0p01.csv'
```

输出列：

```text
wavenumber,tau_us
```

## 9. 直接把 Ar 的处理结果 CSV 插值到 0.01 网格

默认会插值所有数值列：

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/interpolate_processed_result_grid.py \
  'output/results/continuum/CIA/273K/Ar 500Torr/continuum_step2_fit.csv' \
  --output 'output/results/continuum/CIA/273K/Ar 500Torr/continuum_step2_fit_interp_9120_9820_step0p01.csv'
```

如果只想输出波数和最终平滑后的衰荡时间：

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/interpolate_processed_result_grid.py \
  'output/results/continuum/CIA/273K/Ar 500Torr/continuum_step2_fit.csv' \
  --columns tau_fit_us_lmfit_spline \
  --drop-metadata \
  --output 'output/results/continuum/CIA/273K/Ar 500Torr/ar_tau_lmfit_interp_9120_9820_step0p01.csv'
```

如果输出文件已经存在，需要覆盖：

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/interpolate_processed_result_grid.py \
  'output/results/continuum/CIA/273K/Ar 500Torr/continuum_step2_fit.csv' \
  --columns tau_fit_us_lmfit_spline \
  --drop-metadata \
  --output 'output/results/continuum/CIA/273K/Ar 500Torr/ar_tau_lmfit_interp_9120_9820_step0p01.csv' \
  --overwrite-output
```

## 10. 直接把 O2 的处理结果 CSV 插值到 0.01 网格

插值 O2 的 Step 2 完整结果：

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/interpolate_processed_result_grid.py \
  'output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_fit.csv' \
  --output 'output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_fit_interp_9120_9820_step0p01.csv'
```

只输出 O2 的 CIA baseline loss 和对应的 tau：

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/interpolate_processed_result_grid.py \
  'output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_cia_baseline.csv' \
  --columns cia_baseline_loss_ppm_per_cm tau_fit_us \
  --drop-metadata \
  --output 'output/results/continuum/CIA/273K/O2 500Torr/o2_cia_baseline_interp_9120_9820_step0p01.csv'
```

如果已经对 O2 写入了 `tau_fit_us_lmfit_spline`，只输出该列：

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/interpolate_processed_result_grid.py \
  'output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_fit.csv' \
  --columns tau_fit_us_lmfit_spline \
  --drop-metadata \
  --output 'output/results/continuum/CIA/273K/O2 500Torr/o2_tau_lmfit_interp_9120_9820_step0p01.csv'
```

如果输出文件已经存在，需要覆盖：

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/interpolate_processed_result_grid.py \
  'output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_cia_baseline.csv' \
  --columns cia_baseline_loss_ppm_per_cm tau_fit_us \
  --drop-metadata \
  --output 'output/results/continuum/CIA/273K/O2 500Torr/o2_cia_baseline_interp_9120_9820_step0p01.csv' \
  --overwrite-output
```
