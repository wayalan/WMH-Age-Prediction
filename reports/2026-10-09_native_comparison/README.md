# MATLAB / Python native WMH 实际对照测试

测试日期：2026-10-09。报告使用现有 MRNE085 / MRNE066 数据。

MATLAB 基准为仓库既有原版 LST native 概率图；本次 Python 使用自己的既有 native FLAIR、脑掩膜和空间先验，重新运行 FAST 与当前 LGA。Python 不读取 MATLAB p0、组织图或参考病灶图作为分割输入。配准不在测试范围内。

## 结果

| 案例 | MATLAB 概率体积 ml | Python 概率体积 ml | 相对差 | Dice |
|---|---:|---:|---:|---:|
| MRNE085 | 16.656004 | 21.101417 | +26.69% | 0.811911 |
| MRNE066 | 2.424133 | 1.420344 | -41.41% | 0.558704 |

PDF：[WMH_native_comparison_2026-10-09.pdf](WMH_native_comparison_2026-10-09.pdf)，8 页。仓库发布静态报告、图像、指标与复现脚本；生成的本机网页运行文件不随仓库发布。

## 复核文件

- `data/metrics.csv`：主指标完整精度。
- `data/per_slice_metrics.csv`：352 行逐切片统计。
- `data/metrics.json`：环境、参数、原始体素一致性、全部输入/输出 SHA-256。
- `data/verification.json`：独立重算 Dice/Jaccard、切片总量对账、哈希复核。
- `data/report_quality_checks.json`：视觉与输出检查记录。
- `data/native/`：本地新 FAST 输出、Python native 概率图和二值图，不随仓库发布。
- `data/lst_lga_tested_snapshot.py`：被测核心代码快照。
- `images/figure_selections.json`：自动选片、显示窗宽与裁剪范围。
- `images/`：网页/PDF 使用的全部图像。定量图为网页原生图卡导出。

## 重跑

从项目目录执行，使用新的输出目录：

```bash
python3 reports/2026-10-09_native_comparison/run_comparison.py \
  --pipeline-root "$PWD" \
  --cases reports/2026-10-09_native_comparison/cases.json \
  --output /tmp/wmh_native_new_run
```

已有被测代码之后如发生变更，新运行不再代表本报告版本；核对代码 SHA-256 与快照。不要覆盖本次报告或现有受试者目录。

公开 JSON 的路径以项目根目录为基准。原始 MRI、组织图与病灶 NIfTI 保留在本地；完整重跑需要提供 `cases.json` 中指定的输入文件。定量图从已验证的本机图文报告导出，发布的 PNG 与 PDF 使用相同证据。

MRI 图只用于展示，不修改掩膜。概率体积为概率之和乘体素体积；所有二值指标与叠加均以严格 >0.5 阈值计算。主比较没有使用此前“共享 MATLAB 输入、Dice=1”的条件实验。
