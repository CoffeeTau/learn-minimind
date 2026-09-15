> 适用范围：本文依据当前仓库中的 `trainer/train_lora.py` 及其直接依赖编写。它解释当前代码能够确认的行为，不把其他 LoRA 框架的惯例当成这里已经实现的功能。

# 1. 核心地图：这段程序到底在做什么

一句话概括：

> 加载一个已经完成 SFT 的 MiniMind 基础模型，在部分线性层旁边增加小型 LoRA 分支，冻结基础参数，只训练新增参数，最后单独保存这些增量权重。

默认流程是：

```text
../out/full_sft_768.pth          ../dataset/lora_medical.jsonl
        基础模型权重                         医疗对话
                 \                         /
                  \                       /
                   ↓                     ↓
                   MiniMind + LoRA 分支
                            ↓
                   只更新 LoRA 参数
                            ↓
                ../out/lora_medical_768.pth
```

它在 MiniMind 训练路线中的位置是：

```text
Pretrain → Full SFT → LoRA → 组合推理或合并权重
                       ↑
                    当前阶段
```

LoRA 仍然是监督微调。它与 Full SFT 都使用对话数据和 next-token loss，关键区别是参数更新范围：

| 对比项 | Full SFT | 当前 LoRA 训练 |
| --- | --- | --- |
| 默认起点 | 预训练模型 | `full_sft` 模型 |
| 数据 | 对话数据 | 对话数据 |
| 监督目标 | assistant 回答 | assistant 回答 |
| 更新范围 | 模型主体参数 | 新增 LoRA 参数 |
| 小型独立产物 | 没有 | LoRA 适配器权重 |

## 1.1 五个文件各自负责什么

| 文件 | 负责 | 不负责 |
| --- | --- | --- |
| [`trainer/train_lora.py`](../trainer/train_lora.py) | 组织模型、数据、优化器、训练循环和保存 | 不定义 Transformer 和 LoRA 数学结构 |
| [`model/model_lora.py`](../model/model_lora.py) | 定义、注入、保存、加载、合并 LoRA | 不读取训练数据 |
| [`dataset/lm_dataset.py`](../dataset/lm_dataset.py) | 将对话变成 `input_ids` 和 `labels` | 不更新模型参数 |
| [`model/model_minimind.py`](../model/model_minimind.py) | Transformer 前向、logits 和语言模型 loss | 不决定优化器管理哪些参数 |
| [`trainer/trainer_utils.py`](../trainer/trainer_utils.py) | 模型初始化、学习率、DDP、checkpoint、跳批 | 不实现 LoRA 分支 |

真实调用主线是：

```text
命令行参数
  → 初始化设备、DDP、随机种子
  → MiniMindConfig
  → init_model 加载基础权重
  → apply_lora 注入低秩分支
  → 冻结非 LoRA 参数
  → SFTDataset / DataLoader
  → AdamW(lora_params)
  → train_epoch
  → forward / loss / backward / optimizer step
  → 保存适配器与 checkpoint
```

# 2. LoRA 机制：大矩阵旁边为什么能加一条小支路

## 2.1 从普通线性层开始

忽略 bias，普通线性层可以写成：

$$
y = Wx
$$

假设输入、输出维度都是 768，原权重 `W` 有：

$$
768 \times 768 = 589{,}824
$$

个参数。Full SFT 会直接更新这个大矩阵。

当前仓库的 LoRA 不修改 `W`，而是在旁边增加两个小矩阵：

$$
y = Wx + B(Ax)
$$

等价地，可以把有效权重理解为：

$$
W_{effective} = W + BA
$$

当 `rank=16` 时：

```text
x                         [*, 768]
  → A                     [16, 768]
中间低维表示               [*, 16]
  → B                     [768, 16]
LoRA 输出                 [*, 768]
```

一个 `768 → 768` LoRA 分支的参数量是：

$$
16 \times 768 + 768 \times 16 = 24{,}576
$$

它远少于原矩阵的 589,824 个参数。

> PyTorch 的 `nn.Linear.weight` 形状是 `[out_features, in_features]`。公式采用常见列向量写法，代码则按照 PyTorch 的批量张量规则完成同一变换。

## 2.2 当前代码怎样把 LoRA 接到原层上

`model/model_lora.py::apply_lora` 对目标线性层执行三步：

```python
lora = LoRA(module.in_features, module.out_features, rank=rank)
setattr(module, "lora", lora)
module.forward = forward_with_lora
```

新的 `forward` 返回：

```python
return original_forward(x) + lora(x)
```

因此：

- 原分支继续计算 `Wx`。
- LoRA 分支计算 `B(Ax)`。
- 两条分支输出形状相同，可以直接相加。
- `lora` 被注册成子模块，会出现在 `named_parameters()` 和 `state_dict()` 中。

函数把原层和当前 LoRA 层放进默认参数：

```python
def forward_with_lora(x, layer1=original_forward, layer2=lora):
```

这样每个新函数都会记住当前循环中的两个层，不会在循环结束后全部引用最后一个层。

## 2.3 为什么刚注入时模型行为不变

当前初始化方式是：

```text
A：小随机数
B：全 0
```

所以训练开始前：

$$
B(Ax) = 0
$$

模型仍然只表现出基础分支的结果 `Wx`。第一次反向传播时，`B` 可以获得有效梯度；由于此时 `B=0`，`A` 的梯度起初为 0。`B` 更新后，`A` 才会逐渐获得有效梯度。

**思考**

如果 `A` 和 `B` 都使用随机初始化，注入 LoRA 后、正式训练前，模型输出会怎样变化？

<details>
<summary><strong>参考分析</strong></summary>

此时 `B(Ax)` 通常不再为 0，相当于立刻向每个目标线性层加入随机扰动。模型起点不再严格等于原基础模型，可能破坏已经学到的能力。令其中一个矩阵为 0，可以让增量分支从“零影响”开始学习。

</details>

## 2.4 当前默认配置实际注入哪些层

当前代码不是按名称选择 `q_proj`，而是选择所有满足下面条件的层：

```python
isinstance(module, nn.Linear) and module.in_features == module.out_features
```

默认 `hidden_size=768`、8 个 Attention 头、4 个 KV 头时：

| 线性层 | 输入 → 输出 | 是否注入 |
| --- | ---: | --- |
| `q_proj` | 768 → 768 | 是 |
| `k_proj` | 768 → 384 | 否 |
| `v_proj` | 768 → 384 | 否 |
| `o_proj` | 768 → 768 | 是 |
| `gate_proj` | 768 → 2432 | 否 |
| `up_proj` | 768 → 2432 | 否 |
| `down_proj` | 2432 → 768 | 否 |
| `lm_head` | 768 → 6400 | 否 |

每个 Transformer Block 有两处，8 层共 16 处。当前环境实际实例化模型后得到：

| 项目 | 参数量 |
| --- | ---: |
| 注入前基础模型 | 63,912,192 |
| 新增 LoRA 参数 | 393,216 |
| 注入后总参数 | 64,305,408 |
| LoRA 占注入后总参数 | 约 0.61% |

LoRA 参数量的计算过程是：

$$
24{,}576 \times 16 = 393{,}216
$$

这组结论只适用于当前默认配置和当前筛选规则。修改层数、隐藏维度、Attention 配置或筛选方式后，目标层和参数量都可能变化。

**思考**

当前代码没有给 `k_proj`、`v_proj` 加 LoRA，是否说明 LoRA 原理禁止适配 K/V？

<details>
<summary><strong>参考分析</strong></summary>

不是。它们没有被选中，只是因为默认形状为 `768 → 384`，不满足当前代码的“输入输出维度相等”条件。其他 LoRA 实现完全可以按模块名称选择 K/V。这是实现策略，不是 LoRA 数学限制。

</details>

# 3. 数据流：一条对话怎样变成 loss

## 3.1 原始数据

默认数据文件是 `dataset/lora_medical.jsonl`。每一行包含一段对话：

```json
{"conversations": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}]}
```

`train_lora.py` 使用的不是专门的 LoRA Dataset，而是与 SFT 共用的 `SFTDataset`。

## 3.2 对话处理链

`dataset/lm_dataset.py::SFTDataset` 的数据变化如下：

```text
一行 JSONL
  → json.loads
conversations
  → 补齐 reasoning_content / tools / tool_calls
  → pre_processing_chat
  → tokenizer.apply_chat_template
完整对话字符串
  → tokenizer
token ids
  → 截断到 max_seq_len
  → padding 到 max_seq_len
input_ids [T]
  → generate_labels
labels [T]
```

普通对话可能以 20% 概率补充 system prompt；工具调用数据保持原样。空的思考标签也可能在后处理阶段被移除，因此随机种子不仅影响样本顺序，也可能影响部分文本预处理。

## 3.3 `input_ids` 与 `labels` 各自负责什么

模型需要阅读完整上下文，所以 system、user、assistant 都保留在 `input_ids` 中。

`labels` 最初全部是 `-100`，只有 assistant 回答区域会替换成真实 token id：

| 区域 | `input_ids` | `labels` | 是否贡献 loss |
| --- | --- | --- | --- |
| system | 真实 token | `-100` | 否 |
| user | 真实 token | `-100` | 否 |
| assistant 内容 | 真实 token | 真实 token | 是 |
| assistant 结束标记 | 真实 token | 真实 token | 是 |
| padding | PAD | `-100` | 否 |

`-100` 的含义不是“从输入中删除”，而是“在交叉熵中忽略这个目标位置”。

**思考**

用户问题位置的 `labels` 是 `-100`，模型还能利用用户问题回答吗？

<details>
<summary><strong>参考分析</strong></summary>

能。用户问题仍在 `input_ids` 中，会参与 Transformer 前向传播并成为回答的上下文。`-100` 只让这些位置不直接贡献监督损失，不会阻止模型读取它们。

</details>

## 3.4 next-token 错位在哪里发生

Dataset 返回的 `input_ids`、`labels` 长度相同。真正的错位发生在 `MiniMindForCausalLM.forward`：

```python
x = logits[..., :-1, :]
y = labels[..., 1:]
loss = F.cross_entropy(
    x.view(-1, x.size(-1)),
    y.view(-1),
    ignore_index=-100
)
```

模型实际学习的是：

```text
当前位置输入：<assistant>  北    京
下一个目标：      北       京   <eos>
```

`logits` 是对词表中所有 token 的未归一化分数，不是概率。`F.cross_entropy` 内部会完成数值稳定的 log-softmax 和负对数似然计算。

默认主要形状为：

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `input_ids` | `[B, 340]` | token id |
| `labels` | `[B, 340]` | assistant 目标或 `-100` |
| `hidden_states` | `[B, 340, 768]` | Transformer 隐状态 |
| `logits` | `[B, 340, 6400]` | 词表预测分数 |
| shift 后的 `x` | `[B, 339, 6400]` | 参与交叉熵的预测 |
| shift 后的 `y` | `[B, 339]` | 下一个 token 目标 |
| `res.loss` | 标量 | assistant 区域平均交叉熵 |

# 4. 训练流：梯度怎样到达 LoRA 参数

## 4.1 模型和优化器的创建顺序

`train_lora.py` 按以下顺序准备训练对象：

```text
init_model
  → 创建 MiniMindForCausalLM
  → 加载 ../out/{from_weight}_{hidden_size}.pth
  → 将模型移到 device

apply_lora
  → 添加 LoRA 子模块并替换目标层 forward

遍历 named_parameters
  → 名称含 lora：requires_grad=True，加入 lora_params
  → 其他参数：requires_grad=False

AdamW(lora_params)
  → 优化器只持有 LoRA 参数
```

更新范围受到双重限制：

1. 基础参数设置为 `requires_grad=False`。
2. AdamW 的参数列表只有 `lora_params`。

基础模型仍然参与前向传播。冻结意味着“不更新基础权重”，不意味着“跳过基础模型计算”。因此只训练约 0.61% 参数，不代表计算量和激活显存也只剩 0.61%。

## 4.2 一个 micro-batch 的完整顺序

核心训练代码可以压缩为：

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = (res.loss + res.aux_loss) / args.accumulation_steps

scaler.scale(loss).backward()

if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

各操作的责任边界是：

| 操作 | 作用 | 是否更新模型参数 |
| --- | --- | --- |
| `model(...)` | 前向传播并计算 loss | 否 |
| `backward()` | 计算并累积梯度 | 否 |
| `unscale_()` | FP16 时还原被放大的梯度 | 否 |
| `clip_grad_norm_()` | 限制 LoRA 梯度总范数 | 否 |
| `scaler.step(optimizer)` | 梯度有效时调用 AdamW | 是 |
| `scaler.update()` | 调整下一轮 FP16 缩放因子 | 否 |
| `zero_grad()` | 清空本轮梯度 | 否 |

先 `unscale_` 再裁剪很重要，否则裁剪面对的是人为放大的 FP16 梯度，阈值不再表示真实梯度范数。

## 4.3 `step` 不一定是一次参数更新

`train_epoch` 中的 `step` 是 DataLoader batch 计数，也就是 micro-batch 计数。

若 `accumulation_steps=4`：

```text
step 1：backward，暂不更新
step 2：backward，暂不更新
step 3：backward，暂不更新
step 4：backward → optimizer step → 清梯度
```

每个 loss 都先除以 4，使一个完整累积周期的梯度接近四个 micro-batch 的平均值：

$$
g = \frac{g_1 + g_2 + g_3 + g_4}{4}
$$

完整周期的近似有效全局 batch size 是：

$$
\text{batch\_size} \times \text{accumulation\_steps} \times \text{DDP 进程数}
$$

例如 `batch_size=8`、累积 4 步、两张卡时，约为 `8×4×2=64` 条样本一次更新。最后不足大小的 batch 或不足一个累积周期时，实际数量会更小。

日志中的：

```python
current_loss = loss.item() * args.accumulation_steps
```

只是在恢复当前 micro-batch 被除小前的 loss，不是最近 4 个 micro-batch loss 的平均值。梯度跨 step 累积，局部变量 `loss` 不会跨 step 自动累计。

**思考**

`accumulation_steps=4` 时，日志打印到了 `step=3`，能否据此认为模型已经更新了三次？

<details>
<summary><strong>参考分析</strong></summary>

不能。`step=3` 只表示已经处理三个 DataLoader batch，并完成三次 backward。正常情况下要到 `step=4` 才执行第一次 optimizer step。日志步数和参数更新次数不是同一个概念。

</details>

## 4.4 BF16、FP16 与 GradScaler

| 运行方式 | autocast | GradScaler |
| --- | --- | --- |
| CPU | 当前代码使用 `nullcontext` | 默认 BF16 参数下关闭 |
| CUDA + `bfloat16` | BF16 autocast | 关闭 |
| CUDA + `float16` | FP16 autocast | 开启 |

需要分开理解：

```text
autocast：选择部分算子的计算精度
GradScaler：管理 FP16 的 loss 和梯度缩放
```

BF16 默认不启用 GradScaler，但代码仍统一调用 `scaler.step(optimizer)`；禁用状态下它基本直通到优化器。真正更新参数的入口仍是 `step`，不是 `scaler.update()`。

## 4.5 学习率、MoE 与 DDP

- `get_lr` 使用无 warmup 的余弦衰减，从接近基础学习率逐渐下降到其 10%。
- 学习率在每个 micro-batch 更新，而不只在 optimizer step 更新。
- Dense 模型的 `res.aux_loss` 默认是 0；MoE 模型会加入专家负载均衡辅助损失。
- DDP 为每个进程保留完整模型副本，`DistributedSampler` 分配数据，反向传播时同步 LoRA 梯度。
- DDP 是数据并行，不是把模型的不同层拆到不同 GPU。
- 只有主进程打印、记录和保存，避免多个进程重复写同一文件。

# 5. 保存与续训：三个文件不要混淆

默认从 `trainer` 目录运行时，相关产物是：

| 文件 | 实际内容 | 用途 |
| --- | --- | --- |
| `../out/lora_medical_768.pth` | 只有各层 `lora.A.weight`、`lora.B.weight`，以 FP16 保存 | 与基础模型组合推理 |
| `../checkpoints/lora_medical_768.pth` | 包含基础参数和 LoRA 参数的完整 `state_dict` | 完整权重快照 |
| `../checkpoints/lora_medical_768_resume.pth` | 模型、optimizer、scaler、epoch、step、world size、实验 run id | 恢复训练 |

只有 `save_lora(...)` 写出的 `out` 文件是“仅 LoRA 权重”。`lm_checkpoint(...)` 写出的文件名虽然也含 `lora_medical`，普通 checkpoint 仍包含整个已注入 LoRA 的模型。

## 5.1 组合推理与合并权重不是一回事

组合推理保留两份权重：

```text
full_sft 基础权重 + lora_medical 适配器 → 推理
```

在仓库根目录可使用：

```bash
python eval_llm.py --weight full_sft --lora_weight lora_medical
```

合并权重则执行：

$$
W \leftarrow W + BA
$$

`model/model_lora.py::merge_lora` 通过下面的矩阵乘法完成增量合并：

```python
module.lora.B.weight @ module.lora.A.weight
```

推理使用的基础权重必须与训练 LoRA 时的基础权重一致。LoRA 学到的是相对于特定基础模型的增量，不是一套可以任意叠加到其他模型上的完整能力。

**思考**

如果训练时基于 `full_sft_768.pth`，推理时却把 LoRA 加到另一套同形状权重上，只要形状能加载，效果就一定正确吗？

<details>
<summary><strong>参考分析</strong></summary>

不一定。形状一致只能说明矩阵可以相加，不能保证语义一致。LoRA 增量是在特定基础权重附近学到的；更换基础模型后，同一个 `BA` 可能作用在完全不同的表示空间上，结果不可预期。

</details>

## 5.2 resume 怎样恢复到中间 step

恢复链路是：

```text
从 checkpoints 读取 resume 数据到 CPU
  → 按 from_weight 创建并加载基础模型
  → 重新注入相同结构的 LoRA
  → 创建 AdamW(lora_params)
  → 恢复 model / optimizer / scaler / epoch / step
  → SkipBatchSampler 跳过已完成 batch
```

模型隐藏维度、层数、Dense/MoE 类型和 LoRA 注入结构应与保存时一致。`strict=False` 只会放宽模型键检查，不会让任意结构的 optimizer 状态都能安全恢复。

如果续训 GPU 数量改变，`lm_checkpoint` 会按旧、新 world size 换算 `step`，尽量维持已经处理的数据量。

# 6. 简单实践 Demo

以下 Demo 都是本地只读检查，不训练模型、不连接外部服务，也不改写权重。

## 6.1 检查 LoRA 注入位置和参数量

在仓库根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -c '
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora

model = MiniMindForCausalLM(MiniMindConfig())
base = sum(p.numel() for p in model.parameters())
apply_lora(model)

targets = [name for name, module in model.named_modules()
           if hasattr(module, "lora")]
lora_count = sum(p.numel() for name, p in model.named_parameters()
                 if "lora" in name)

print("base:", base)
print("lora:", lora_count)
print("targets:", len(targets))
print(*targets, sep="\n")
'
```

当前环境的实际结果是：

```text
base: 63912192
lora: 393216
targets: 16
```

后续名称由 8 层中的 `q_proj`、`o_proj` 组成。

观察重点：

- 验证选择规则，而不是只相信层名推断。
- 验证新增参数量与手算一致。
- 该 Demo 只实例化随机模型结构，没有加载或评估训练权重。

## 6.2 检查一条数据的监督区域

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -c '
import random
from transformers import AutoTokenizer
from dataset.lm_dataset import SFTDataset

random.seed(0)
tokenizer = AutoTokenizer.from_pretrained("model")
dataset = SFTDataset(
    "dataset/lora_medical.jsonl",
    tokenizer,
    max_length=64
)
input_ids, labels = dataset[0]

print("samples:", len(dataset))
print("shape:", tuple(input_ids.shape), tuple(labels.shape))
print("supervised:", int((labels != -100).sum()))
print("ignored:", int((labels == -100).sum()))
'
```

当前环境实际得到：

```text
samples: 25276
shape: (64,) (64,)
supervised: 42
ignored: 22
```

观察重点：

- `input_ids` 和 `labels` 长度相同。
- 一部分标签是 assistant 监督 token，另一部分是 `-100`。
- 具体数量依赖数据内容、随机预处理、tokenizer 和 `max_length`，不能当作所有样本的固定比例。

# 7. 运行方式：先确认路径，再开始训练

本脚本中的 `../out`、`../dataset`、`../checkpoints` 都相对于当前工作目录，而不是相对于 `train_lora.py` 文件位置。因此默认应从 `trainer` 目录运行。

先检查输入：

```bash
cd trainer
ls ../out/full_sft_768.pth
ls ../dataset/lora_medical.jsonl
```

CPU 小配置适合观察流程，但不代表训练速度理想：

```bash
python train_lora.py \
  --device cpu \
  --epochs 1 \
  --batch_size 1 \
  --max_seq_len 64 \
  --num_workers 0 \
  --log_interval 1
```

单卡 CUDA 示例：

```bash
python train_lora.py \
  --device cuda:0 \
  --dtype bfloat16 \
  --epochs 1 \
  --batch_size 8 \
  --accumulation_steps 4
```

两张卡 DDP 示例：

```bash
torchrun --nproc_per_node 2 train_lora.py \
  --dtype bfloat16 \
  --batch_size 8 \
  --accumulation_steps 4
```

是否支持 BF16、实际显存占用和 DDP 行为需要在目标 GPU 环境中验证。本文的本地检查没有启动完整训练，也没有验证 CUDA/NCCL。

# 8. 当前实现的适用边界

## 8.1 它不是通用 LoRA 框架的全部功能

当前实现是便于学习的精简版本：

- 直接使用 `Wx + BAx`，没有常见的 `alpha / rank` 缩放。
- 没有 LoRA dropout。
- 按“线性层是否为方阵”选择目标，不支持命令行配置目标名称。
- 通过动态替换 `forward` 注入，因此脚本主动关闭 `torch.compile`。

## 8.2 训练循环有两个尾部边界

假设 `accumulation_steps=4`，epoch 最后只剩两个 micro-batch：

1. 每个 loss 仍除以 4，而不是实际剩余数量 2，所以尾部更新比“按两个 batch 求平均”更小。
2. 最后一个 DataLoader step 会因为 `step == iters` 先保存文件；剩余梯度随后才在循环外更新，所以刚保存的文件不包含最后这次参数更新。

这两点是当前代码顺序能够确认的行为。理解文档时应如实记录，不应把它们误写成所有梯度累积实现的标准方式。

**思考**

若 epoch 最后只剩两个 micro-batch，它们仍各自除以 4，那么最终累积梯度与这两个 batch 的平均梯度相差多少倍？

<details>
<summary><strong>参考分析</strong></summary>

当前代码得到 `(g1 + g2) / 4`，而按实际两个 batch 求平均应为 `(g1 + g2) / 2`。前者是后者的二分之一，因此尾部更新会偏小。

</details>

## 8.3 其他阅读提醒

- `warnings.filterwarnings('ignore')` 会隐藏警告，AMP 或硬件提示可能不易察觉。
- `--dtype` 没有限定 `choices`；只有精确字符串 `bfloat16` 进入 BF16 分支，其他字符串会落入 FP16 分支。
- `init_model` 在冻结前打印一次 `Trainable Params`，该数字不代表最终 AdamW 更新范围。
- 单独 LoRA 文件较小，但完整 checkpoint 仍可能接近完整模型大小。
- 若截断后完全没有 assistant 回答，`labels` 可能全是 `-100`，样本无法提供有效监督。
- 冻结基础权重会减少梯度和优化器状态，但完整模型前向及相关激活仍然存在。

## 8.4 证据边界

| 结论 | 证据状态 |
| --- | --- |
| 默认注入 `q_proj`、`o_proj`，共 16 处 | 当前源码确认，本地实例化验证 |
| 默认 LoRA 参数为 393,216 | 当前源码确认，本地实例化验证 |
| 一条 64-token 样本含监督与忽略位置 | 当前数据和 tokenizer 本地验证 |
| optimizer 只接收 `lora_params` | 当前源码确认 |
| 三类保存文件的字段区别 | 当前源码确认 |
| 完整 GPU 训练可成功结束 | 未在本次任务中验证 |
| 多卡 NCCL 与恢复训练实际可用 | 未在本次任务中验证 |
| 某种 GPU 的速度与峰值显存 | 需要目标硬件实测 |

# 9. 迁移到其他 LoRA 脚本时怎样判断

遇到新的 LoRA 代码，可以沿下面的顺序检查：

```text
1. 基础模型从哪里加载？
2. LoRA 插入哪些模块？
3. A、B 的形状和 rank 是什么？
4. 是否存在 alpha / rank 和 dropout？
5. 哪些参数 requires_grad=True？
6. 优化器实际持有哪些参数？
7. input_ids 与 labels 怎样生成？
8. 哪些 token 被 loss 忽略？
9. 一个 step 是 micro-batch 还是 optimizer step？
10. 保存的是适配器、完整权重还是续训状态？
```

不要只根据文件名或日志中的 “LoRA” 判断。最可靠的证据依次是：

```text
参数筛选代码
  → requires_grad
  → optimizer 参数列表
  → forward 真实调用
  → state_dict 保存筛选
```

# 10. 一页总结

```text
训练入口
  trainer/train_lora.py

基础权重
  ../out/full_sft_768.pth

训练数据
  SFT 对话 JSONL

监督目标
  assistant token；其他 labels 为 -100

LoRA 公式
  y = Wx + B(Ax)

当前默认目标
  8 层中的 q_proj、o_proj，共 16 处

当前默认可训练参数
  393,216，约占注入后总参数 0.61%

优化器
  AdamW，只持有 lora_params

一次参数更新
  forward → loss → backward → unscale → clip → step → zero_grad

混合精度
  默认 BF16 autocast；FP16 时启用 GradScaler

小型独立产物
  ../out/lora_medical_768.pth

续训产物
  ../checkpoints/lora_medical_768_resume.pth

关键边界
  无 LoRA 缩放和 dropout；按方阵筛选；compile 自动关闭；
  默认路径依赖 trainer 工作目录；尾部累积与保存顺序需留意
```
