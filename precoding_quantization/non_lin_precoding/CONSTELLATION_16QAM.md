# 16QAM 接收星座：GNN 与 IDE（标定的固定 β）、1/2/3 bit DAC 对比

本实验使用项目已有的 cell-free 仿真信道和 GNN checkpoint，构建从 Gray 16QAM 比特映射、预编码、DAC 量化、无线信道及 AWGN，到接收均衡和硬判决的完整符号链路。图中的散点来自保存的接收复数 IQ 样本。

## 实验设置

| 项目 | 默认设置 |
| --- | --- |
| AP / 用户数 | M=40；分别仿真 K=1、K=2 |
| 预编码 | GNN、IDE（标定的固定 β，`--baseline ide_cal`，默认，论文图所用）；可选 IDE（β_WF，`--baseline ide_wf`）、IDE（块 β，`--baseline ide_block`）或 WMMSE + DAC（`--baseline wmmse`），均为早期运行 |
| DAC 精度 | 每个 I/Q 分支 1、2、3 bit |
| 调制 | Gray 16QAM；I、Q 电平为 {−3,−1,1,3}/√10，平均符号能量为 1 |
| 功率 / SNR | 总发送功率 Pt=40；发送 SNR=20 dB |
| 噪声 | 复数 AWGN，E[\|n\|²]=Pt/10^(SNR/10)=0.4，每个实维方差 0.2 |
| 统计信道 | 每个 K 使用 Htest[4096:6144]，共 2048 个信道块 |
| 每块符号数 | 每用户 125 个符号；信道在块内固定 |
| 固定信道星座图 | 预先选定 Htest[4096]，另生成 32 块，共 4000 个符号/用户 |
| 随机种子 | 发送符号 1234；AWGN 基础种子 20260930，各 K 实际使用 `seed_noise+K` |
| 接收增益 | 与论文一致的无噪声整块有效增益，即 oracle 增益 |

默认 K=1 的 AWGN 实际种子为 20260931，K=2 为 20260932；各自记录在 `metadata.json` 的数据集条目中。同一 K 下，各个预编码和 DAC 精度共享信道、发送符号与同一份 AWGN。噪声根据总发送功率设定；所有方法使用相同的噪声方差。每个 125 符号块单独进行总功率归一化和增益计算。固定信道展示块追加在统计数据之后，不重复计入 2048 块的平均指标；固定信道在运行前选定，不根据 GNN 或基线的表现挑选。

### 信道与 DAC

信道来自已有数据集的 `Htest.npy`。论文模型在 100×100 m² 区域固定放置 40 个 AP，每个信道实现重新放置用户；大尺度衰落为 `−30−37log10(d)+ζ` dB，其中距离下限为 10 m，阴影项 ζ 的标准差为 8 dB，再叠加 Rayleigh 小尺度衰落。数据集使用统一比例缩放，使平均信道功率为 1，并保留 AP 和用户之间的相对增益差异。

GNN 使用已有继续训练运行的最佳 checkpoint，以确定性的 `argmax` 输出 DAC 电平。网络原本使用 Gaussian 符号训练，此处直接输入 16QAM 符号。

IDE（标定的固定 β，`--baseline ide_cal`）与论文 16QAM BER 图（`ber_16qam_20dB.py` 的 `ide_cal`）完全相同：每个信道实现先调用 `ide_baseline.calibrated_beta(...)`，在一段与数据无关的训练块（125 个 16-QAM 符号向量，`--seed-cal 777`，与 BER 脚本相同）上运行块 β 更新得到 β，再用 `beta_mode='fixed'` 固定该 β 处理数据，每个符号向量独立处理；固定信道展示的 32 块共用该信道的 β，保存在 `K{K}_b{b}_ide_cal.npz` 的 `ide_beta` 中。早期的 `--baseline ide_wf`：调用 `ide_baseline.ide(..., 'ide', beta_mode='wf')`，T=100、阻尼 0.95，β 固定为每个信道实现的维纳滤波因子 β_WF（式 (7)），从不更新；每个符号向量独立处理，与 GNN 同为逐符号处理（固定信道展示的 32 块共用同一 β_WF）。`--baseline ide_block` 则按式 (26) 每 10 次迭代更新 β，其分子分母在块内 125 个符号上求和，因此每块一个 β、需缓存整块。IDE 在除以 √Pt 的同一 DAC 格点上运行（σ²=1/SNR），输出的最后一次硬判决再映射回原 DAC 电平索引，由 `receive()` 检查。2048 个统计块上的 BER 与 `exp_results/ber_16qam_20dB.json` 中对应的 `ide_cal` / `ide_wf` / `ide_block` 一致。

WMMSE（`--baseline wmmse`）针对当前信道及 SNR 计算，比较 RZF、MRT、ZF 三种初值；K=1 时退化为 MRT。WMMSE 输出先按每个 AP 的线性预编码行范数缩放，再分别将 I/Q 映射到最近电平。

所有方案使用相同的 DAC 格点：1 bit 为 ±1/√2，2/3 bit 使用项目已有的 Lloyd–Max 电平。WMMSE 使用 `denorm=False`，不在 DAC 后恢复各 AP 的行范数；所有方案都用每块共享的标量系数满足 Pt。对于多 bit DAC，共同电平集合并不要求各 AP 实际平均功率严格相同。

### 接收与指标定义

信道矩阵 H 的形状为 `[AP, user]`，发送样本 y 的形状为 `[AP, symbol]`。接收模型为：

```text
r0 = H.T @ y
r  = r0 + n
g[k] = sum_t(r0[k,t] * conj(s[k,t])) / sum_t(abs(s[k,t])**2)
z[k,t] = r[k,t] / g[k]
```

这里的转置为 `H.T`，不取共轭。`rx_raw` 保存 r，`rx_equalized` 保存 z。图中展示均衡后的 z，并以真实发送符号区分点簇。

增益 g 使用该块真实发送符号和无噪声接收值，因此是论文采用的 oracle 增益，不是从独立、含噪导频估计的增益。本实验是符号速率、平坦块衰落的等效复基带仿真；没有加入脉冲整形、定时误差、载波频偏或 RF 硬件失真。

- **Monte Carlo BER**：对实际保存的含噪 IQ 进行 Gray 16QAM 硬判决，将错误比特数除以发送比特数。每用户每符号有 4 个比特。
- **SER**：错误符号数除以发送符号数。
- **RMS EVM**：`sqrt(sum(abs(z-s)**2) / sum(abs(s)**2))`，参考为真实发送符号；百分数表示时乘以 100。
- **解析 BER**：固定无噪声均衡样本，对 AWGN 引起的比特错误概率解析积分，单独记录为 `ber_analytic`，用于和 Monte Carlo 结果对照。

Monte Carlo BER 为 0 仅表示有限样本中没有观察到错误；不能据此断言真实 BER 为 0。解析 BER 与 Monte Carlo BER 的计算方式不同，读图和引用时应保留对应标签。

K=1、1 bit 时，WMMSE/MRT（仅 `--baseline wmmse` 运行）无法区分某些相位相同、幅度不同的 16QAM 符号，例如 `(1+j)/√10` 与 `(3+3j)/√10`。由此产生的接收点簇合并应保留在星座图中。双用户分别显示两个 UE，避免掩盖用户之间的接收质量差异。

## 运行

项目已有环境 `/home/user/Quantization/env/bin/python` 提供 NumPy、PyTorch 和 Matplotlib。脚本及输入文件通过其所在位置解析，下面的命令可在任意工作目录执行。

### 完整仿真

```bash
/home/user/Quantization/env/bin/python \
  /home/user/Quantization/precoding_quantization/non_lin_precoding/constellation_16qam.py \
  --baseline ide_cal \
  --output-dir /home/user/Quantization/precoding_quantization/non_lin_precoding/exp_results/constellation_16qam_20dB_ide_cal \
  --device cuda:0 --threads 16 \
  --users 1 2 --bits 1 2 3 --snr-db 20 \
  --n-channels 2048 --start 4096 --symbols 125 --snapshot-blocks 32 \
  --seed-symbols 1234 --seed-noise 20260930
```

命令使用项目可用的 CUDA GPU（GNN），IDE 在 CPU 上运行，线程数设为 16；未给出 `--output-dir` 时，IDE 默认写入 `constellation_16qam_20dB_<baseline>/`（`_ide_cal`、`_ide_wf`、`_ide_block`），WMMSE 默认写入最早的 `constellation_16qam_20dB/`；这些目录均已存在，脚本拒绝覆盖。各次运行的 GNN DAC 索引逐位一致，接收 IQ 仅有 float32 末位差异（≤1.7e-7，CPU 线程数不同）。无可用 CUDA GPU 时，将 `--device cuda:0` 改为 `--device cpu`。改变参数时请使用新的 `--output-dir`，以便保留已有结果。

### 从保存的 IQ 重画图

```bash
/home/user/Quantization/env/bin/python \
  /home/user/Quantization/precoding_quantization/non_lin_precoding/plot_constellation_16qam.py \
  --input-dir /home/user/Quantization/precoding_quantization/non_lin_precoding/exp_results/constellation_16qam_20dB_ide_cal
```

### 验证保存的链路数据

```bash
/home/user/Quantization/env/bin/python \
  /home/user/Quantization/precoding_quantization/non_lin_precoding/validate_constellation_16qam.py \
  --input-dir /home/user/Quantization/precoding_quantization/non_lin_precoding/exp_results/constellation_16qam_20dB_ide_cal
```

`validation.json` 记录验证结果。应核对 DAC 索引及电平重建、块发送功率、`rx_noiseless=H.T@tx_samples`、`rx_raw=rx_noiseless+noise`、均衡关系和保存的误码指标。

## 输出文件

IDE 标定 β 运行（论文图）的输出根目录为：

```text
/home/user/Quantization/precoding_quantization/non_lin_precoding/exp_results/constellation_16qam_20dB_ide_cal/
```

早期运行（IDE β_WF、IDE 块 β 和 WMMSE，目录分别为 `exp_results/constellation_16qam_20dB_ide_wf/`、`exp_results/constellation_16qam_20dB_ide_block/` 和 `exp_results/constellation_16qam_20dB/`，方案文件为 `K{K}_b{b}_<方案>.npz`，文件结构相同）已于 2026-09-30 删除。种子固定，需要时可用 `python constellation_16qam.py --baseline ide_wf`（或 `ide_block`、`wmmse`）重新生成。

| 文件 | 内容 |
| --- | --- |
| `shared_K1.npz`、`shared_K2.npz` | 每个 K 的公共信道、发送符号/比特、AWGN 与块索引 |
| `K{K}_b{b}_gnn.npz` | 对应 K、DAC 精度的 GNN 发送和接收样本及指标 |
| `K{K}_b{b}_ide_cal.npz` | 对应 K、DAC 精度的 IDE（标定的固定 β）发送和接收样本、指标及每块的 β（`ide_beta`） |
| `metadata.json` | 运行配置、种子、接收机定义、数据集与 checkpoint 路径及 SHA 摘要 |
| `metrics.csv`、`summary.json` | 按 `scope` 区分统计：`ensemble` 仅汇总前 `n_eval` 块，`snapshot` 仅汇总尾部展示块 |
| `validation.json` | 对保存数据执行的数值一致性验证 |
| `figures/constellation_K1.png`、`.pdf` | 固定信道的单用户对比 |
| `figures/constellation_K2.png`、`.pdf` | 固定信道的双用户对比，分别显示两个 UE |
| `figures/constellation_overview.png`、`.pdf` | 单、双用户总览 |
| `figures/constellation_ensemble.png`、`.pdf` | 多信道统计样本的接收星座展示 |

公共 NPZ 字段：

| 字段 | 含义 |
| --- | --- |
| `H` | 每块信道，维度为 `[block, AP, user]` |
| `tx_symbols` | 16QAM 发送符号，维度为 `[block, user, symbol]` |
| `tx_bits` | 发送符号对应的 Gray 比特 |
| `noise` | 所有方法共用的接收 AWGN，维度为 `[block, user, symbol]` |
| `channel_indices` | 各块对应的 Htest 信道索引 |
| `n_eval` | 用于统计的前缀块数；默认 2048 |
| `snapshot_channel_index` | 固定信道展示使用的 Htest 索引；默认 4096 |

方案 NPZ 字段：

| 字段 | 含义 |
| --- | --- |
| `tx_samples` | DAC 选择及块功率归一化后的发送 IQ，维度为 `[block, AP, symbol]` |
| `dac_indices`、`levels` | I/Q DAC 电平索引和电平集合 |
| `power_scale` | 每块共同的功率缩放系数 |
| `rx_noiseless` | 信道叠加后的无噪声接收 IQ |
| `rx_raw` | 实际加入 AWGN 后的原始接收 IQ |
| `rx_equalized` | 使用块增益均衡后的接收 IQ |
| `gain` | 每块、每用户的复数有效增益 |
| `bit_errors`、`symbol_errors` | 比特和符号错误计数 |
| `evm_rms` | 对真实发送符号计算的 RMS EVM |
| `ber_analytic` | 对 AWGN 解析积分得到的 BER |

前三个接收 IQ 数组的维度均为 `[block, user, symbol]`。默认各数组包含前 2048 个统计块，以及尾部 32 个固定信道展示块。尾部每块有独立的发送符号和 AWGN，并分别进行功率归一化与增益计算。

PNG 适合快速浏览，PDF 可用于论文排版。用于论文的图可复制到 `/home/user/Quantized-aware-training-and-deploy-study/Figure/constellation_16qam/`；实验脚本不修改 `main.tex`。

## 读取复数 IQ

以下示例读取双用户、1 bit GNN 的第一个统计块，以及尾部固定信道样本。NumPy 保留复数数组，可以直接使用 `.real` 和 `.imag`。

```python
from pathlib import Path
import numpy as np

root = Path(
    "/home/user/Quantization/precoding_quantization/non_lin_precoding/"
    "exp_results/constellation_16qam_20dB_ide_cal"
)

with np.load(root / "shared_K2.npz", allow_pickle=False) as shared, \
     np.load(root / "K2_b1_gnn.npz", allow_pickle=False) as result:
    n_eval = int(shared["n_eval"])
    block, ue = 0, 0  # Python 索引从 0 开始：UE1
    s = shared["tx_symbols"][block, ue]
    raw_iq = result["rx_raw"][block, ue]
    eq_iq = result["rx_equalized"][block, ue]
    print("H:", shared["H"].shape)
    print("tx_bits:", shared["tx_bits"].shape)
    print("rx_raw:", result["rx_raw"].shape)
    print("前 5 个原始 I/Q:", np.column_stack((raw_iq.real, raw_iq.imag))[:5])
    print("前 5 个均衡 I/Q:", np.column_stack((eq_iq.real, eq_iq.imag))[:5])
    evm = np.sqrt(np.sum(np.abs(eq_iq - s)**2) / np.sum(np.abs(s)**2))
    print("当前块、UE1 的 EVM (%):", 100 * evm)

    # 只使用尾部展示块，默认每用户合计 32×125=4000 个点。
    snapshot_iq = result["rx_equalized"][n_eval:, ue, :].reshape(-1)
    snapshot_tx = shared["tx_symbols"][n_eval:, ue, :].reshape(-1)
    snapshot_xy = np.column_stack((snapshot_iq.real, snapshot_iq.imag))

    # 重新构造第一个块的原始接收信号，核对复数链路。
    reconstructed = (
        shared["H"][block].T @ result["tx_samples"][block]
        + shared["noise"][block]
    )
    print("接收链路最大残差:", np.max(np.abs(reconstructed - result["rx_raw"][block])))
```

只需将文件名的 `gnn` 改为 `ide_cal`（早期运行中为 `ide_wf`、`ide_block` 或 `wmmse`），即可使用完全相同的公共输入比较另一方案。跨信道指标使用切片 `[:n_eval]`，对应汇总表的 `scope=ensemble`；固定信道展示及其指标使用 `[n_eval:]`，对应 `scope=snapshot`。比较总体性能时只选择 `ensemble` 行，避免合并两个范围。

## 与论文的关系和可复现性

主要依据是 [main.tex](/home/user/Quantized-aware-training-and-deploy-study/main.tex) 的 System Model、Simulation Setup 和 Uncoded BER with 16-QAM Signaling；实现复用 [train_sweep.py](train_sweep.py)、[ide_baseline.py](ide_baseline.py)、[wmmse_baseline.py](wmmse_baseline.py) 和 [ber_16qam_20dB.py](ber_16qam_20dB.py) 的模型及约定。

论文原有 16QAM BER 脚本针对 AWGN 计算解析概率；本实验额外生成并保存含噪接收 IQ，以便重画星座图、检查链路和计算实际样本的硬判决 BER。固定信道星座用于解释失真形态，跨信道统计用于比较总体接收质量，两者应结合阅读。

默认 `Htest[4096:6144]` 沿用论文评估切片，该切片也曾用于最佳 checkpoint 选择。因此这里的结果属于复现已有评估设置，不能宣称为独立于 checkpoint 选择的新测试集。若需要独立检验，可使用 `--start 6144 --n-channels 2048` 并指定新的输出目录；固定展示信道随起始索引一起改变。

`metadata.json` 中的数据集与 checkpoint SHA 摘要用于核对实际输入。复现时同时保留脚本、配置、公共 NPZ、方案 NPZ 和这些摘要；不同设备的浮点计算可能产生很小的数值差异。
