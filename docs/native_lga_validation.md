# Native LGA 移植与验证

2026-10-08。当前目标以原版 MATLAB LST 的 native-space WMH 总量和分割为金标准，移植其数学与更新规则；不进行配准检验，不根据 example 拟合参数、体积系数或空间补丁。

## 已实现的原版规则

`pywmh/lst_lga.py` 恢复了原始 PVE 类内 min–max 标签、T1 强度分区、CSF 高信号病灶核心重标记、GM 邻域传播、单位宽度 FLAIR 众数直方图、belief 支持域、Gamma Newton 拟合、Gaussian 样本标准差、棋盘式顺序更新、仅更新新 frontier，以及原版停止与清理规则。默认 kappa=0.3、phi=1、max_iter=50 均来自原版，不是针对样例选择。

Python 核心只接收数组，不读取 MATLAB 文件、受试者 ID、参考分割、参考总量或 example 路径。原版的 2.4 重标记、95% GM 分位数、0.01 frontier 条件、18 个邻居清理条件等均保留源算法含义。

以下数值异常处理是通用扩展，未按样本调整：常数 PVE 类使用纯组织端点；不存在的 Gaussian 类贡献零权重；无法拟合的退化分布或无有效众数输入显式报错。正常、非退化输入使用原版数值规则。

## 相同分割输入的结果

通过独立验证适配器，从 LST 中间缓存读取 FLAIR、p0 和 native prior；运行 Python 后才读取原版病灶图作比较。生产代码不导入这个适配器。

| 数据 | Python native 概率体积 ml | 原版 MATLAB 概率体积 ml | Dice | 概率最大误差 |
|---|---:|---:|---:|---:|
| MRNE085 | 16.656004120789405 | 16.656004120789405 | 1.0 | 0 |
| MRNE066 | 2.4241334168007187 | 2.4241334168007187 | 1.0 | 0 |

两组概率图逐体素一致，0.5 阈值后的二值图也逐体素一致。这证明该分割核心可以复现已有原版输出；并不证明原始 T1/FLAIR 经 FAST/FSL 后与 SPM 前处理等价。

## 独立合成数据验证

`tests/generate_lst_reference_fixtures.py` 使用固定随机种子生成三组独立数据，包含 PVE、T1、FLAIR、病灶核心、部分高信号邻域及 exclusion。随机种子用于复现，不参与生产代码，未读取临床 example。

PVE 标签的参考计算使用安装的原版 `ps_LST_lga.m` 中初始化语句和 LST helper。概率图参考直接调用原版 `ps_LST_lga`；不运行配准或 SPM 前处理。

| 尺寸 | kappa | max_iter | phi | p0 最大误差 | 二值不一致体素 | 概率最大误差 |
|---|---:|---:|---:|---:|---:|---:|
| 20×22×24 | 0.30 | 50 | 1.00 | 0 | 0 | 2.96×10⁻⁸ |
| 21×23×25 | 0.17 | 12 | 0.65 | 0 | 0 | 2.23×10⁻⁸ |
| 18×20×22 | 0.40 | 25 | 1.20 | 0 | 0 | 0 |

三组 Dice 都为 1。非零概率误差来自 Python 输出 float32，而合成 MATLAB 参考输出为 float64，均小于 float32 量化精度。PVE 标签比较在 float64 下逐元素一致。

这些测试覆盖不同尺寸、奇数网格、参数和连续 frontier 增长。它们提供独立于 example 的算法证据，不能证明所有可能的数据都有效或所有 MRI 的完整前处理等价。

## 主程序接入

- FAST 增加 `-B`，保存其 bias-corrected T1；PVE 缓存需要包含 `T1_restore.nii.gz`。这是输出既有 FAST 校正结果，不改变注册函数或注册参数。
- 主程序将校正 T1 传给标签构建，避免原接口缺少必要 T1 信息。
- 分割产物显式保存为 float32 概率图和 uint8 二值图。
- 分割缓存绑定实际输入文件 SHA-256、算法版本、kappa、phi 和 max_iter；旧缓存不再仅凭文件存在被接受。
- 7 项自动测试通过，包括标签方向、直方图边界/众数并列、独立 Gamma MLE 对照、退化输入、零种子、缓存内容变化，以及不调用配准的主程序接入和输出类型检查。

未更改配准函数、配准参数、PV/DW 年龄模型或 example 数据。所有临床测试的新增产物均写入独立临时目录。

## 默认 FAST/FSL 输入的剩余差异

保持已有 native FLAIR、脑掩膜与空间先验，仅在临时目录重新执行 FAST 保存校正 T1，再运行新核心；未运行或评估配准，也未调 kappa。

| 数据 | 新 Python native 概率体积 ml | 原版 MATLAB ml | Dice |
|---|---:|---:|---:|
| MRNE085 | 21.1014 | 16.6560 | 0.8119 |
| MRNE066 | 1.4203 | 2.4241 | 0.5587 |

因此**默认 FAST/FSL 路径尚未达到完整 native 分割近乎一致的目标**。不能将相同输入下的 Dice=1 当作默认 T1/FLAIR 流程已达到 Dice=1。

固定原版 native 空间先验、只替换组织标签的诊断中：使用 FAST 校正 T1/PVE 和原版 FLAIR 构建 p0，MRNE085 为 16.3665 ml、Dice=0.9425；MRNE066 为 1.6848 ml、Dice=0.8132。组织标签输入本身仍会改变结果。

保留原版 p0，仅替换为现有 FSL FLAIR 时也出现明显差异；MRNE066 的全脑单位宽度直方图众数降为 4，产生严重异常增长。这是一个固定标签的反事实输入替换，**不是生产流程输出，也不能单独归因于 bias correction 或配准**。它说明 p0、FLAIR 和原版强度校正输入不能混用。

下一层需要按原版方式对齐组织分割及 FLAIR 校正的输入定义。恢复 200-bin 众数、改 kappa、加体积补偿或定制空间限制来消除以上例子的误差，都会偏离本次移植目标，因此未采用。

## 复现

自动测试：

```bash
python3 -m unittest discover -s tests -v
```

相同输入的 native 核心对照（路径可替换为任意 LST 个案）：

```bash
pip install -r requirements-validation.txt
python3 docs/validate_native_lga.py \
  --cache /path/to/LST_lga_rmFLAIR.mat \
  --reference /path/to/ples_lga_0.3_rmFLAIR.nii \
  --output /tmp/native_validation.json
```

生成独立合成数据（输出目录必须尚不存在）：

```bash
python3 tests/generate_lst_reference_fixtures.py \
  --spm-root /path/to/spm12 \
  --outdir /tmp/independent_native_lga
```

用 MATLAB 运行生成的 `run_reference.m`，再使用同一 `validate_native_lga.py` 比较每组缓存和参考图，并传入生成器列出的 kappa、max_iter、phi。

完整测量记录：`native_lga_validation_metrics.json`。最新独立前端测试与图像见 [2026-10-09 实际对照报告](../reports/2026-10-09_native_comparison/README.md)。
