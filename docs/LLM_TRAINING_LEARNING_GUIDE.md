# MiniMind 大模型训练全流程学习指南

## 学习目标

通过 MiniMind 项目建立一条完整的大模型训练认知链路：

```text
原始文本
  -> Tokenizer
  -> Dataset / DataLoader
  -> Transformer 前向传播
  -> logits 与 next-token loss
  -> 反向传播与梯度
  -> 优化器更新参数
  -> checkpoint
  -> SFT / 对齐训练
  -> 推理与部署
```

学习时先掌握最小闭环，再逐步加入混合精度、梯度累积、DDP、MoE 和断点续训等工程机制。

## 总体学习顺序

```text
0. Tokenizer              trainer/train_tokenizer.py
        ↓
1. 从零预训练             trainer/train_pretrain.py
        ↓
2. 监督微调               trainer/train_full_sft.py
        ↓
3. 参数高效微调           trainer/train_lora.py
        ↓
4. 偏好对齐               trainer/train_dpo.py
        ↓
5. 在线强化学习           trainer/train_ppo.py / trainer/train_grpo.py
        ↓
6. 模型蒸馏               trainer/train_distillation.py
        ↓
7. Agent 强化学习         trainer/train_agent.py
        ↓
8. 推理与服务             eval_llm.py / scripts/serve_openai_api.py
```

第一轮必须掌握的主线是：

```text
Tokenizer -> Pretrain -> SFT -> 推理
```

DPO、PPO、GRPO、蒸馏和 Agent RL 放在预训练、SFT 与推理闭环之后学习。

---

## 第一阶段：建立预训练全局图

入口文件：`trainer/train_pretrain.py`

第一遍主要阅读主函数，不急着深入每个函数的内部实现。先建立以下流程：

```text
命令行参数
  -> 初始化设备、DDP 和随机种子
  -> 创建 MiniMindConfig
  -> 加载 tokenizer 并初始化模型
  -> 创建 PretrainDataset
  -> DataLoader 组织 batch
  -> 创建 AdamW 优化器
  -> 恢复 checkpoint（可选）
  -> torch.compile / DDP 包装（可选）
  -> 逐 epoch 调用 train_epoch
  -> forward -> loss -> backward -> optimizer.step
  -> 保存模型权重和断点续训状态
```

### 第一阶段检查问题

完成第一遍阅读后，应能回答：

1. 数据路径从哪里传入，数据如何进入 `PretrainDataset`？
2. 文本如何转换为 `input_ids` 和 `labels`？
3. `input_ids` 和 `labels` 分别表示什么？
4. 模型在哪里执行前向传播并得到 loss？
5. 梯度在哪里计算？梯度累积和梯度裁剪在哪里发生？
6. 参数在哪里被优化器更新？
7. 模型权重和完整 checkpoint 分别保存在哪里？

---

## 第二阶段：跟踪一个样本的数据流

入口：`dataset/lm_dataset.py` 中的 `PretrainDataset`。

一条原始数据类似：

```json
{"text": "秦始皇是中国历史上的第一位皇帝……"}
```

数据处理过程：

```text
sample["text"]
  -> tokenizer(...).input_ids
  -> 在首尾添加 BOS 和 EOS
  -> 使用 PAD 补齐到 max_length
  -> 转换为 torch.long Tensor
  -> labels = input_ids.clone()
  -> labels 中 PAD 所在位置替换为 -100
```

注意：`PretrainDataset` 返回的 `labels` 并没有手动错开一位。真正的错位发生在 `MiniMindForCausalLM.forward` 中：

```python
x = logits[..., :-1, :]
y = labels[..., 1:]
```

因此模型实际执行的是：

```text
当前位置输入：BOS    我    喜欢    学习
下一词目标：  我    喜欢    学习    EOS
```

`labels` 中的 `-100` 会被交叉熵的 `ignore_index=-100` 忽略，因此 PAD 不参与 loss。

### 需要记录的 Tensor 形状

设：

- `B`：batch size
- `T`：sequence length
- `H`：hidden size
- `V`：vocabulary size

则主要形状为：

```text
input_ids:     [B, T]
labels:        [B, T]
hidden_states: [B, T, H]
logits:        [B, T, V]
shift_logits:  [B, T-1, V]
shift_labels:  [B, T-1]
loss:          标量
```

---

## 第三阶段：理解模型前向传播

入口文件：`model/model_minimind.py`。

推荐阅读顺序：

1. `MiniMindConfig`
2. `MiniMindForCausalLM.forward`
3. `MiniMindModel.forward`
4. `MiniMindBlock`
5. `Attention`
6. `FeedForward`
7. `RMSNorm`
8. RoPE 相关函数
9. 最后再学习 MoE

### 模型主数据流

```text
input_ids [B, T]
  -> Embedding
hidden_states [B, T, H]
  -> N 个 MiniMindBlock
hidden_states [B, T, H]
  -> RMSNorm
  -> lm_head
logits [B, T, V]
  -> shift logits / labels
  -> CrossEntropyLoss
loss 标量
```

### 单个 Transformer Block

```text
x
  -> RMSNorm -> Attention -> 与残差相加
  -> RMSNorm -> FeedForward -> 与残差相加
```

第一轮使用 `use_moe=0` 理解 Dense Transformer。掌握主线后，再研究专家路由、Top-K 专家选择和 `aux_loss`。

---

## 第四阶段：逐行理解训练循环

入口：`trainer/train_pretrain.py` 中的 `train_epoch`。

核心过程：

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = (res.loss + res.aux_loss) / args.accumulation_steps

scaler.scale(loss).backward()

if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

对应关系：

| 代码 | 含义 |
| --- | --- |
| `model(...)` | 前向传播 |
| `res.loss` | next-token 交叉熵 |
| `res.aux_loss` | MoE 路由辅助损失，Dense 模型通常为 0 |
| `/ accumulation_steps` | 为梯度累积缩放 loss |
| `backward()` | 通过反向传播计算并累积梯度 |
| `unscale_()` | FP16 时先将梯度恢复到真实尺度 |
| `clip_grad_norm_()` | 对梯度范数做裁剪 |
| `optimizer.step()` | 根据梯度更新模型参数 |
| `scaler.update()` | 更新下一步的动态缩放因子，不是更新模型参数 |
| `optimizer.zero_grad()` | 参数更新后清除已有梯度 |

近似有效 batch size：

```text
batch_size * accumulation_steps * DDP 进程数
```

默认单进程配置近似为 `32 * 8 * 1 = 256` 个样本一次参数更新。最后一个不完整 batch 等边界情况需要单独考虑。

当前默认 `dtype=bfloat16` 时，代码中的 `GradScaler` 被禁用，相关调用基本是直通操作；选择 `float16` 时才启用动态梯度缩放。自动混合精度 `autocast` 与梯度缩放 `GradScaler` 是两个相关但不同的机制。

---

## 第五阶段：补齐训练工程能力

入口文件：`trainer/trainer_utils.py`。

按以下顺序阅读：

1. `get_lr`：余弦学习率衰减。
2. `init_model`：创建模型和 tokenizer，按需加载已有权重。
3. `lm_checkpoint`：保存或恢复训练状态。
4. `SkipBatchSampler`：恢复训练时跳过已经完成的 batch。
5. `init_distributed_mode`：初始化 NCCL 和 DDP。
6. `setup_seed`：设置随机种子与确定性选项。

### 两类保存文件

训练代码会保存两类用途不同的文件：

```text
out/pretrain_<hidden_size>.pth
  主要保存模型 state_dict
  用于推理、评估或作为后续 SFT 的初始权重

checkpoints/pretrain_<hidden_size>_resume.pth
  保存模型、优化器、epoch、step、world_size、实验记录 ID 等状态
  用于中断后继续训练
```

只保存 `model.state_dict()` 无法精确恢复优化器动量和训练进度，因此不能替代完整 resume checkpoint。

---

## 建议的首轮实践

第一轮不要直接跑完整数据。准备一小份样本或限制数据规模，完成一次可观察的训练闭环：

1. 打印一条原始 `text`。
2. 打印对应 token 文本、token ID、`input_ids` 和 `labels`。
3. 打印一个 batch 的 Tensor 形状。
4. 打印 `logits`、shift 后 logits/labels 的形状。
5. 观察初始 loss。
6. 运行若干次参数更新，确认 loss 和学习率发生变化。
7. 保存权重并重新加载。
8. 用 `eval_llm.py --weight pretrain` 做一次简单生成测试。

每一轮都沿着以下主线复述：

```text
text
  -> tokens
  -> input_ids / labels
  -> hidden_states
  -> logits
  -> next-token cross entropy
  -> backward
  -> gradients
  -> AdamW parameter update
  -> checkpoint
```

## 第一阶段完成标准

如果能够不看代码解释下面这句话，就可以进入模型结构阶段：

> `PretrainDataset` 把文本编码为包含 BOS、EOS 和 PAD 的 token ID 序列，并把 PAD 对应的 label 标记为 `-100`；模型经过 Embedding、Transformer Blocks 和 LM Head 得到每个位置的词表 logits，再用前一位置的 logits 预测后一位置的 label，通过交叉熵产生 loss，反向传播计算梯度，AdamW 按梯度累积周期更新参数，最后分别保存推理权重和断点续训状态。

