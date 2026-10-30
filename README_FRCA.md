# TimeEmb + 频域参照时域补充分支（研究原型）

这份工程以用户提供的 TimeEmb 代码为基础，新增 `TimeEmb_FRCA`。原始模型仍可通过 `--model TimeEmb` 使用。本次修改直接应用于这个 TimeEmb 仓库，作为名为 `2` 的提交；其父提交为 `2e8c3209db0bb69b76ffaf43f95bb00530ca34c0`（提交名 `1`）。来源与 SHA-256 记录在 `source_manifest.json`。

它实现的是讨论中的候选机制，不代表已经证实预测性能提升或学术原创性。

## 先看这几个文件

- `models/TimeEmb_FRCA.py`：全部新增模型代码，只有 `ReferenceAttention` 和 `Model` 两个类。
- `models/TimeEmb.py`：保留原版，用作基线。
- `scripts/run_frca_ablation.py`：匹配训练配置的对照实验入口。
- `tests/test_frca.py`：频域一致性、梯度、数值稳定性与短程优化检查。
- `utils/forecast_loss.py`：统一 FP32 / AMP 的原有混合损失。
- `VALIDATION.md`：本次实际执行的验证及其边界。

## 结构及公式

```text
X → 原 TimeEmb 归一化 → X_norm
                         ├→ FFT / 日历嵌入 / 频域滤波 / IFFT → S → 原 MLP → Y_f
                         │                                    │
                         └→ 共享时间片段嵌入 E(X_norm)         E(S)（默认截断 S 的梯度）
                                        │                     │
                                        └→ 频域参照注意力 ←────┘
                                                  ↓
                                             时域预测头 → ΔY_t
Y_norm = Y_f + η ⊙ ΔY_t → 使用同一组历史均值/方差反归一化 → 最终预测
```

`S` 是预测头之前、历史长度为 L 的表示，不是未来预测，不保证是周期或低频成分。时域注意力按变量独立计算，在历史时间片段之间建立联系。

设 `Zx = E(X_norm)`、`Zs = E(S)`，共享的 `E` 为无偏置线性片段嵌入。每个头 h：

```text
Q_h = split_h(Wq LayerNorm(Zx + position))
K_h = split_h(Wk LayerNorm(Zx + position))
V_h = split_h(Wv Zx) - beta_h * split_h(Wv Zs)
beta_h = sigmoid(theta_h)
A_h = softmax(Q_h K_h^T / sqrt(head_dim))
low_h = A_h V_h
high_h = V_h - low_h
```

训练时注意力有 dropout，`high` 始终用实际的同一次 `low` 相减。无 dropout 的公式才是严格的 `(I-A)V`。位置编码只进入 Q/K 和门控，不进入 V 的差分。

合并各头后，借用 FAiT 的两种门控函数：

```text
g_low = tanh(Linear(Zx + position))
p = softplus(Linear(Zx + position))
g_high = 2*p^2 / (p^2 + 0.3678)
z = Linear(g_low*low + g_high*high)
z = z + FFN(LayerNorm(z))
ΔY_t = Linear(flatten(z))
```

源码为避免半精度平方溢出使用了等价的 `hypot` 写法。没有额外的原始 X 残差连接绕过 V 差分。FFN 和预测头有偏置，因此即使 V 为零，整个时域分支也不保证为零。

## 与提供的 FAiT 源码的区别

用户确认采用“此前讨论的时域注意力”方案。提供的 FAiT 代码在 `Fre_Trans` 中将 `[3,B,C,D,T]` 的 Q/K/V 压缩为 `[3,B,C,d_model]`，之后的编码器在变量 C 之间计算注意力。其 `v-x` 中的 x 还经过输出投影、残差和 FFN，不是裸的 `AV`。

这里没有整段移植该编码器，而是根据反制注意力的公式，在历史片段之间显式计算 `AV` 与 `V-AV`。保留两种门控函数的思想；没有移植源码中的 `lf_gamma/hf_gamma`、DTFM 频域模块、额外 RevIN 或原始 X 的外部残差。因此 `plain` 是本工程的“无频域参照、FAiT 思想的时域分支”对照，不能写成“原版 FAiT 复现”。

参考：[TimeEmb 官方代码](https://github.com/showmeon/TimeEmb)、[FAiT 论文](https://arxiv.org/abs/2606.01306)。借用的模块与新增机制应分别说明、保留引用。

## 默认参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| 时域层数 | 1 | 第一版固定为一层，`e_layers` 不控制此分支 |
| `frca_dim` | 64 | 时域嵌入维度，与 TimeEmb 的 `d_model` 分开 |
| `frca_heads` | 4 | 因而有 4 个独立 beta |
| `frca_patch_len` / `frca_stride` | 16 / 8 | 时间片段长度/步长；末端不足时复制末值补齐 |
| `frca_beta_init` | 0.1 | `theta=log(0.1/0.9)`，训练更新 theta |
| `frca_beta_fixed` | -1 | -1 为学习；0 到 1 为固定系数对照 |
| `frca_eta_init` | 0.01 | 每个未来时间步一个可学习系数，可正可负，不经过 sigmoid |
| `frca_detach_reference` | 1 | 只截断 S→时域参照这条梯度；频域主路径仍训练 |
| `frca_dropout` | 0.1 | 时域分支 dropout |

头数必须整除时域维度。beta 是全局的每头参数，推理时不随样本动态变化。默认参数是起点，不能当作调优后的配置。

## 在你的 WSL 中运行

在你原来的 WSL 项目目录中运行：

```bash
cd '/home/f1348224707/code/TimeEmb-main (origin)'
# 激活你原来运行 TimeEmb 的环境，再按需安装依赖：
python -m pip install -r requirements.txt
```

现有 PyTorch >= 2.1 可以直接使用；这个命令没有强制升级已满足要求的 PyTorch。GPU 的 PyTorch 安装按你现有 CUDA 环境保持一致。

先执行短程检查（只训练 4 个 batch，评估 2 个 batch）：

```bash
python scripts/run_frca_ablation.py \
  --root_path '/home/f1348224707/code/FAiT-main/dataset/ETT-small' \
  --variants complement --seeds 2024 \
  --train_epochs 1 --batch_size 4 --smoke
```

正式的初步对照（移除 `--smoke`，完整训练/验证/测试）：

```bash
python scripts/run_frca_ablation.py \
  --root_path '/home/f1348224707/code/FAiT-main/dataset/ETT-small' \
  --variants baseline plain residual complement \
  --seeds 2024 2025 2026 --train_epochs 30 --batch_size 32
```

入口默认是 ETTh1 的 96→96 配置、同一个优化器与混合损失、学习率 0.001；这是控制变量的起点，不是 TimeEmb/FAiT 的各自最优调参复现。正式比较前应在验证集上给予基线和新增模型合理且一致的调参预算。

单独运行并改变 beta/head 等配置：

```bash
python -u run.py \
  --is_training 1 --model_id ETTh1_FRCA --model TimeEmb_FRCA \
  --data ETTh1 --root_path '/home/f1348224707/code/FAiT-main/dataset/ETT-small' \
  --data_path ETTh1.csv --features M --enc_in 7 \
  --seq_len 96 --pred_len 96 --d_model 512 \
  --use_hour_index 1 --use_day_index 0 \
  --frca_mode complement --frca_dim 64 --frca_heads 4 \
  --frca_beta_init 0.1 --frca_detach_reference 1 \
  --rec_lambda 0.25 --auxi_lambda 0.75 \
  --train_epochs 30 --batch_size 32 --learning_rate 0.001 --num_workers 0
```

ETT 数据加载器返回的 day_index 为 -1，所以本例关闭周嵌入。沿用 TimeEmb 原来的小时索引约定（预测起点对应的已知日历索引）。只使用历史数值作为模型输入。

## 对照含义

| variant | Q/K 内容 | V 内容 | 目的 |
|---|---|---|---|
| `baseline` | 无时域分支 | 无 | 原 TimeEmb |
| `plain` | X | X | 测试单纯增加分支的效果 |
| `residual` | X-S | X-S | 与“完整历史形成 Q/K”对比；仍保留相同的反制门控 |
| `complement` | X | X-beta*S | 完整候选机制 |
| `moving_average` | X | X-beta*MA(X) | 判断是否需要 TimeEmb 的参照 |
| `no_stopgrad` | X | X-beta*S | 判断截断参照梯度的影响 |
| `no_inverted` | X | X-beta*S | 使用标准 AV、不使用反制路径与门控 |

表中省略共享嵌入和投影。想比较普通 Transformer 处理残差，可直接在 `run.py` 中组合 `--frca_mode residual --frca_inverted 0`。

训练损失只对最终融合预测计算，没有人为给 S 增加“纯周期”标签，也没有用未来真实值构造 S。仅通过相减不能保证去冗余、去噪或严格分解；完整模型同时改变了模型容量，须通过匹配的分支对照分析机制收益。

## 训练记录与验证

- `checkpoints/<setting>/config.json`：完整参数。
- `mechanism_history.json`：每一轮的 beta 和 eta。
- `mechanism_best.json`：重新加载最佳验证集 checkpoint 后的系数。
- `results/<setting>/metrics.json`：MSE、MAE 与 `smoke_test` 标记。
- 实验名包含 seed 与配置摘要，防止不同消融覆盖同一个 checkpoint。修改任一训练/数据配置后属于新实验；仅测试旧 checkpoint 时应保留原参数，并改 `--is_training 0`。

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

适配时还修正了：原训练器 AMP 分支只用 MSE、非 AMP 分支用混合损失的不一致；NumPy 2 的 `np.Inf`；pandas 新版的 `drop` 参数；`--use_gpu 0` 的布尔解析；测试时忽略自定义 checkpoint 路径的问题。验证集改为固定顺序且保留末尾 batch，训练数据单独设随机种子，防止模型初始化消耗的随机数改变不同对照的数据顺序。没有宣称修复原工程的所有功能，本次集成验证集中于训练、验证、测试和模型梯度。
