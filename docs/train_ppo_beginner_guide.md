> 本文依据当前仓库的 `trainer/train_ppo.py`、`trainer/rollout_engine.py`、`dataset/lm_dataset.py` 和 `trainer/trainer_utils.py` 整理。重点是弄清楚一批 Prompt 如何经过生成、打分、优势估计，最后变成 Actor 和 Critic 的梯度。
>
> 验证边界：最小 Dataset 实验和张量级调试实验已在本地 CPU 运行；本地检出未提供 `dataset/rlaif.jsonl`、默认 Reward Model，且未检测到 CUDA。这**不限制实验地点**：准备好数据和权重后，可在 L40S 服务器上完成下面的真实 rollout 与训练调试。本文没有声称已在 L40S 上跑通完整 PPO。

# 1. 先抓住主线

PPO 与预训练、SFT 都是在更新语言模型参数，但它们给模型提供“正确方向”的方式不同：

- 预训练：给定文本，让模型预测每一个下一个 token。
- SFT：给定对话，让模型模仿数据中的 assistant 标准回答。
- PPO：只给 Prompt，让 Actor 自己生成回答，再根据奖励判断这次生成值得鼓励还是抑制。

当前代码的一批数据会经历下面这条链：

```text
RLAIFDataset 读取对话
        ↓ 只保留用于提问的上下文
prompt: list[str]
        ↓ tokenizer，左侧 padding
input_ids / attention_mask: [B, P]
        ↓ rollout_engine 调用 Actor 生成
completion_ids: [B, R] + old_logp: [B, R]
        ↓ 规则奖励 + Reward Model
rewards: [B]
        ↓ Critic 估值 + GAE
advantages / returns: [B, R]
        ↓ PPO 多轮小批量更新
Actor 学习“怎样回答” + Critic 学习“回答有多好”
```

其中：

- `B` 是一个 batch 的 Prompt 数量。
- `P` 是 padding 后的 Prompt 长度。
- `R` 是生成部分的长度。
- 一个“时间步”对应回答中的一个 token。

**思考**　为什么 PPO 数据集中不必提供一个固定的标准答案？

<details>
<summary><strong>参考分析</strong></summary>

因为 PPO 的训练信号不是“这个位置必须预测某个标准 token”，而是“当前策略生成的整段回答获得了多少奖励”。程序再用 Critic 和 GAE 把整段回答的标量奖励转换成各 token 的优势值。数据可以有参考答案供其他奖励规则使用，但当前 `train_ppo.py` 没有使用 `RLAIFDataset` 返回的 `answer`。

</details>

# 2. 它和预训练、SFT 到底差在哪里

| 对比项 | 预训练 | SFT | 当前 PPO |
|---|---|---|---|
| 输入数据 | 普通文本 | 完整多轮对话 | 对话 Prompt |
| 回答从哪里来 | 原始文本后续 token | 数据中的 assistant 回答 | Actor 在线生成 |
| 直接训练信号 | token 标签 | assistant 部分的 token 标签 | 奖励、优势值和回报 |
| 主要损失 | 全部有效 token 的交叉熵 | assistant token 的交叉熵 | clipped policy loss + value loss + KL 惩罚 |
| 训练模型 | 一个语言模型 | 一个语言模型 | Actor 和 Critic |
| 只推理不训练的模型 | 无 | 无 | Ref Model 和 Reward Model |
| 初始权重 | 通常从头训练 | 默认加载 `pretrain` | 默认加载 `full_sft` |
| 是否边训练边生成 | 否 | 否 | 是 |
| 同一批数据是否重复更新 | 通常一次 | 通常一次 | 默认进行 2 轮 PPO 更新 |

三者都要经历 token 化、前向传播、反向传播和优化器更新，所以外层训练框架看起来相似。真正的分界点是：

```text
预训练/SFT：标签直接告诉模型“应该生成什么”。
PPO：奖励告诉模型“刚才生成得好不好”，优势值再决定哪些动作应被强化。
```

SFT 的 loss 可以概括为：

$$
\mathcal{L}_{\mathrm{SFT}}
=-\sum_t m_t\log \pi_\theta(y_t\mid x,y_{<t})
$$

`m_t` 在 assistant 回答位置为 1，其余位置被 `-100` 标签忽略。

PPO 则先定义新旧策略概率比：

$$
r_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}
=\exp\left(\log\pi_\theta-\log\pi_{\mathrm{old}}\right)
$$

再限制一次更新不能让这个比值变化得过大：

$$
\mathcal{L}_{\mathrm{policy}}
=\mathbb{E}\left[
\max\left(
-A_t r_t,
-A_t\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)
\right)
\right]
+\beta\mathcal{L}_{\mathrm{KL-ref}}
$$

代码默认 `epsilon=0.2`，所以概率比的裁剪范围是 `[0.8, 1.2]`。

**思考**　既然 Actor 由 `full_sft` 初始化，为什么还需要 Ref Model？

<details>
<summary><strong>参考分析</strong></summary>

Actor 会在奖励驱动下不断变化，可能为了获得高分而偏离原来 SFT 模型的语言分布。Ref Model 是冻结的 SFT 快照，`kl_ref_penalty` 用来约束 Actor 不要偏离它太远。旧策略 `old_logp` 与 Ref Model 不是同一个概念：前者服务于 PPO 的单轮稳定更新，后者服务于长期行为约束。

</details>

# 3. 四个模型分别负责什么

| 组件 | 是否训练 | 输入 | 输出 | 职责 |
|---|---:|---|---|---|
| Actor | 是 | Prompt 或完整序列 | token 概率 | 生成回答，并学习提高高优势动作的概率 |
| Critic | 是 | Prompt + 回答 | 每个位置的价值 `V(s_t)` | 估计从当前状态继续生成的预期回报 |
| Ref Model | 否 | Prompt + 回答 | token 概率 | 提供冻结的 SFT 行为基准 |
| Reward Model | 否 | 对话上下文 + 回答 | 一个分数 | 判断整段回答质量 |

Actor、Ref Model 和 Critic 都从 `full_sft` 权重开始：

```python
actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
ref_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model = ref_model.eval().requires_grad_(False)

state_dict = torch.load(ckp, map_location=args.device)
critic_model = CriticModel(lm_config)
critic_model.load_state_dict(state_dict, strict=False)
```

Critic 额外添加一个 `hidden_size → 1` 的价值头：

```python
self.value_head = nn.Linear(params.hidden_size, 1)
```

如果输入完整序列的形状是 `[B, P+R]`，Critic 就为每个位置输出一个标量，形状仍是 `[B, P+R]`。随后只取回答位置的价值。

这里有两个容易被注释误导的实现细节：

1. 注释说“替换 `lm_head`”，但代码实际是新增 `value_head`；继承得到的 `lm_head` 还存在，只是 `CriticModel.forward()` 不使用它。
2. `MiniMindModel.forward()` 已对隐藏状态做过一次 `norm`，当前 `CriticModel.forward()` 又调用了一次 `self.model.norm(outputs[0])`。这是当前实现，而不是理解 PPO 所必需的标准步骤。

# 4. Prompt 如何变成一次 rollout

## 4.1 `RLAIFDataset` 只构造提问上下文

数据集的关键语句是：

```python
self.tokenizer.apply_chat_template(
    conversations[:-1],
    tokenize=False,
    open_thinking=use_thinking,
    add_generation_prompt=True,
)
```

它做了三件事：

1. `conversations[:-1]` 去掉最后一条消息，通常也就是原数据中的 assistant 回答。
2. `add_generation_prompt=True` 在末尾添加“轮到 assistant 回答”的模板标记。
3. `thinking_ratio` 按概率决定是否打开 thinking 格式。

`__getitem__()` 最终返回：

```python
{"prompt": prompt, "answer": ""}
```

因此，进入 `ppo_train_epoch()` 的 `batch["prompt"]` 是字符串列表，不是 SFT 中已经 padding 好的 `(input_ids, labels)`。

## 4.2 Prompt 在训练循环中才被 token 化

```python
enc = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=args.max_seq_len,
    padding_side="left",
).to(args.device)
```

这里使用左侧 padding，是为了让一个 batch 中所有 Prompt 的最后一个有效 token 对齐，便于从相同的列位置继续生成。

注意，`RLAIFDataset.max_length` 当前只被保存为成员变量，没有在数据集内部参与截断。真正的 Prompt 截断由这里的 `args.max_seq_len` 完成。

## 4.3 Rollout Engine 返回六类数据

`rollout_engine.rollout()` 返回 `RolloutResult`：

| 字段 | 典型形状 | 含义 |
|---|---:|---|
| `output_ids` | `[B, P+R]` | Prompt 和生成回答拼接后的完整 token |
| `completion_ids` | `[B, R]` | 只包含回答 token |
| `per_token_logps` | `[B, R]` | 生成当时旧策略对每个回答 token 的 log probability |
| `completions` | `list[str]` | 解码后的回答文本 |
| `prompt_lens` | `[B]` | 回答开始位置 |
| `completion_mask` | `[B, R]` | 哪些回答位置有效 |

默认 `torch` 引擎直接调用 Actor 的 `generate()`。`sglang` 引擎通过 HTTP 请求外部服务生成，再把文本重新 token 化。

# 5. 回答位置为什么要做索引和 mask

语言模型第 `k` 个 logit 预测的是第 `k+1` 个 token，所以代码先错开一位：

```python
labels = gen_out[:, 1:]
```

回答 token 在这个错位后的 logit 序列里从 `prompt_len - 1` 开始，因此：

```python
resp_idx = torch.arange(R).unsqueeze(0)
logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
```

假设 padding 后 `P=4`，回答有 4 个位置：

```text
完整 token 下标：       0  1  2  3 | 4  5  6  7
                       <---Prompt---> <---回答--->
错位后 logit 下标：     0  1  2  3  4  5  6
预测回答 token 的位置：          3  4  5  6
```

这就是 `P - 1 + [0, 1, 2, 3]`。

接着，代码结合 padding 和第一个 EOS 得到真正的回答长度，并构造：

```python
resp_policy_mask  # 哪些 token 参与策略损失
resp_value_mask   # 哪些 token 参与价值损失
```

没有 mask，EOS 后用于补齐形状的 token 也会进入 loss，模型就会学习本不应存在的动作。

**思考**　为什么 `old_resp_logp` 必须在更新 Actor 前保存，而不能每次都用当前 Actor 重新计算？

<details>
<summary><strong>参考分析</strong></summary>

PPO 要比较“生成数据时的旧策略”和“正在优化的新策略”。如果分子、分母都由更新后的同一个 Actor 计算，`log_ratio` 会一直接近 0，概率比会接近 1，PPO 就失去了衡量策略变化幅度的基准。

</details>

# 6. 一个回答怎样得到奖励

`calculate_rewards()` 把多个信号加到同一个标量上：

```text
最终奖励
= 回答长度奖励
+ thinking 长度奖励
+ thinking 闭合次数奖励
- 重复 3-gram 惩罚
+ Reward Model 分数
```

具体规则是：

- 回答去空白后长度在 `[20, 800]`：`+0.5`，否则 `-0.5`。
- 出现 `</think>` 且思考内容长度在 `[20, 300]`：`+1.0`，否则 `-0.5`。
- `</think>` 恰好出现一次：`+0.25`，否则 `-0.25`。
- 重复 3-gram 最多扣 `0.5`。
- Reward Model 分数由 `LMForRewardModel.get_score()` 限制到 `[-3, 3]`。

规则奖励直接数 Python 字符串长度，不是 token 长度。Reward Model 只评价 `</think>` 后面的答案正文；如果没有该标记，就评价整个回答。

得到 `[B]` 标量奖励后，代码把每条回答的奖励放在最后一个有效 token 上：

```python
token_rewards = torch.zeros_like(old_resp_logp)
last_idx = resp_lengths - 1
token_rewards[batch_index, last_idx] += rewards
```

前面 token 的直接奖励虽然是 0，但 GAE 会把结尾奖励向前传播。

# 7. Critic 和 GAE 如何分配功劳

Critic 先给回答中每个状态估值：

$$
V_t=V(s_t)
$$

程序从后向前计算 TD 误差：

$$
\delta_t=r_t+\gamma V_{t+1}-V_t
$$

再计算广义优势估计：

$$
A_t=\delta_t+\gamma\lambda A_{t+1}
$$

最后构造 Critic 的回归目标：

$$
R_t=A_t+V_t
$$

直观上：

- `A_t > 0`：这个 token 动作比 Critic 原先预期更好，Actor 应提高它的概率。
- `A_t < 0`：这个动作比预期更差，Actor 应降低它的概率。
- `returns`：Critic 下一轮应该逼近的目标。

代码还会在所有有效回答 token 上标准化 `advantages`，使其均值约为 0、方差约为 1，从而减小梯度尺度波动。

**思考**　为什么已经有 Reward Model，还要训练 Critic？

<details>
<summary><strong>参考分析</strong></summary>

Reward Model 给整段回答一个最终分数，却没有直接说明每个 token 的贡献。Critic 学习每个生成状态的预期回报，GAE 再用实际奖励与价值预测之差构造优势，从而为每个回答 token 分配更低方差的学习信号。

</details>

# 8. PPO 更新阶段在优化什么

一批 rollout 生成后，代码默认执行 `ppo_update_iters=2` 轮更新，每轮再打乱并拆成 mini-batch。

## 8.1 Actor：有限度地改变 token 概率

程序重新用当前 Actor 计算 `mb_resp_logp`，再与 rollout 时保存的 `old_resp_logp` 比较：

```python
log_ratio = mb_resp_logp - old_resp_logp
ratio = torch.exp(log_ratio)
```

优势为正时，希望 `ratio` 增大；优势为负时，希望它减小。但 clipped loss 不鼓励一步走得过远。

此外，当前实现还计算 Actor 相对冻结 Ref Model 的非负 KL 近似惩罚：

$$
x=\log\pi_{\mathrm{ref}}-\log\pi_\theta
$$

$$
\mathcal{L}_{\mathrm{KL-ref}}=e^x-x-1
$$

## 8.2 Critic：拟合 return，但限制单次变化

Critic 的新预测既要接近 `returns`，又会被限制在旧预测附近：

$$
\mathcal{L}_{V}
=\frac12\max\left[
(V_\theta-R)^2,
(\operatorname{clip}(V_\theta,V_{old}-c,V_{old}+c)-R)^2
\right]
$$

## 8.3 一次反向传播同时更新两个网络

总损失是：

```python
loss = (
    policy_loss
    + args.vf_coef * value_loss
    + aux_loss
) / args.accumulation_steps
```

- `policy_loss` 的计算图连接 Actor。
- `value_loss` 的计算图连接 Critic。
- 一次 `loss.backward()` 会分别给两个模型产生梯度。
- 随后 `actor_optimizer.step()` 和 `critic_optimizer.step()` 各自更新参数。

当 `approx_kl > early_stop_kl` 时，代码把 loss 乘 0，不再产生有效更新，但仍完成各卡的 forward-backward 通信闭环，避免 DDP 某些进程提前退出造成死锁。

# 9. 动手调试数据流

下面的实验按照“先看字符串，再看 token 位置，最后看真实训练日志”的顺序进行。建议不要一开始就单步跟完整训练，否则四个模型和多层循环会掩盖主线。

## 9.1 实验一：只观察 Dataset 产生的 Prompt

先准备一条最小 JSONL 数据：

```json
{"conversations":[{"role":"user","content":"请用一句话解释梯度下降。"},{"role":"assistant","content":"梯度下降通过沿损失函数下降最快的方向更新参数。"}]}
```

保存为运行实验的那台机器上的 `/tmp/ppo_tiny.jsonl`，在项目根目录、已激活项目 Python 环境后运行：

```bash
HF_HOME=/tmp/ppo_hf_cache python - <<'PY'
import random
from transformers import AutoTokenizer
from dataset.lm_dataset import RLAIFDataset

random.seed(0)
tokenizer = AutoTokenizer.from_pretrained("./model")
dataset = RLAIFDataset(
    "/tmp/ppo_tiny.jsonl",
    tokenizer,
    max_length=64,
    thinking_ratio=0.0,
)
item = dataset[0]
print("keys:", item.keys())
print("prompt:", repr(item["prompt"]))
print("answer:", repr(item["answer"]))
PY
```

重点核对：

1. Prompt 中保留了 user 内容。
2. 原 assistant 标准回答没有进入 Prompt。
3. 末尾存在 assistant 开始标记，等待 Actor 续写。
4. `answer` 是空字符串。

当前环境用上述最小数据实测得到：

```text
keys: dict_keys(['prompt', 'answer'])
prompt: '<|im_start|>user\n请用一句话解释梯度下降。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
answer: ''
```

即使 `thinking_ratio=0.0`，当前聊天模板仍会插入一对空的 `<think>...</think>`；该参数控制的是是否打开 thinking，而不是是否彻底移除 thinking 标记。本实验依赖项目的 tokenizer 文件；本地检出未提供默认 `rlaif.jsonl`，服务器若有该文件也可以换用真实样本观察。

## 9.2 实验二：亲手验证回答位置、EOS mask 和 GAE

下面的例子不加载任何模型，可以直接在 CPU 上运行：

```bash
python - <<'PY'
import torch

prompt_lens = torch.tensor([4, 4])
completion_ids = torch.tensor([
    [21, 22,  2,  0],
    [31, 32, 33, 34],
])
eos_id, pad_id = 2, 0

resp_idx = torch.arange(completion_ids.size(1)).unsqueeze(0)
logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
resp_pad_mask = completion_ids.ne(pad_id)
eos_mask = completion_ids.eq(eos_id) & resp_pad_mask
has_eos = eos_mask.any(dim=1)
eos_pos = eos_mask.int().argmax(dim=1)
resp_lengths = resp_pad_mask.sum(dim=1)
resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths)
policy_mask = (resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask

print("logp_pos =", logp_pos.tolist())
print("resp_lengths =", resp_lengths.tolist())
print("policy_mask =", policy_mask.int().tolist())

old_values = torch.tensor([
    [0.4, 0.3, 0.2, 0.0],
    [0.5, 0.4, 0.3, 0.2],
]) * policy_mask
rewards = torch.tensor([1.0, -0.5])
token_rewards = torch.zeros_like(old_values)
token_rewards[torch.arange(2), resp_lengths - 1] = rewards

gamma, lam = 1.0, 0.95
last = torch.zeros(2)
advs_rev = []
for t in reversed(range(old_values.size(1))):
    nv = old_values[:, t + 1] if t < old_values.size(1) - 1 else 0.0
    delta = token_rewards[:, t] + gamma * nv - old_values[:, t]
    last = delta + gamma * lam * last
    advs_rev.append(last)
advantages = torch.stack(advs_rev[::-1], dim=1) * policy_mask

print("token_rewards =", token_rewards.tolist())
print("advantages =", advantages.round(decimals=4).tolist())
PY
```

当前环境实测输出：

```text
logp_pos = [[3, 4, 5, 6], [3, 4, 5, 6]]
resp_lengths = [3, 4]
policy_mask = [[1, 1, 1, 0], [1, 1, 1, 1]]
token_rewards = [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, -0.5]]
advantages = [[0.527, 0.66, 0.8, 0.0], [-0.8854, -0.8267, -0.765, -0.7]]
```

观察第一条样本：奖励只放在第 3 个回答 token 上，但更早的两个 token 也得到了正优势；这就是 GAE 的“向前分配功劳”。EOS 后的第 4 个位置被 mask 掉。

**思考**　第二条样本的所有优势为什么都是负数？

<details>
<summary><strong>参考分析</strong></summary>

它最终得到 `-0.5` 奖励，而 Critic 在此前还预测了正价值。实际结果比预期更差，所以末尾 TD 误差为负；反向递推后，前面的 token 也得到负优势，Actor 会倾向于降低这些动作的概率。

</details>

## 9.3 实验三：在源码中设置断点跟踪一个 batch

推荐按以下顺序设置断点，每到一处只记录“类型、形状、前几个值”：

| 顺序 | 断点语句 | 重点观察 |
|---:|---|---|
| 1 | `prompts = batch["prompt"]` | `prompts[0]` 是否只有上下文 |
| 2 | `rollout_result = rollout_engine.rollout(...)` 后 | 六个返回字段的形状 |
| 3 | `rewards = calculate_rewards(...)` 后 | 回答文本和最终标量奖励 |
| 4 | `logp_pos = ...` 后 | 第一个回答 token 是否映射到 `P-1` |
| 5 | `resp_policy_mask = ...` 后 | EOS 后是否为 0 |
| 6 | `advantages = ...` 后 | 正负号、均值和有效位置 |
| 7 | `log_ratio = ...` 后 | PPO 第一轮是否接近 0 |
| 8 | `loss.backward()` 后 | Actor 与 Critic 是否都有非空梯度 |

可以临时在调试器的 Watch 区域加入：

```python
enc.input_ids.shape
completion_ids.shape
prompt_lens.tolist()
resp_lengths.tolist()
resp_policy_mask[0].tolist()
rewards.tolist()
advantages[0][resp_policy_mask[0].bool()].tolist()
log_ratio[resp_policy_mask[inds].bool()].abs().max().item()
```

真实数据流调试可以直接放在 L40S 服务器上进行。先在服务器确认当前 Python 环境、GPU 和默认路径：

```bash
# 在项目根目录运行；先激活服务器上的项目 Python 环境
python -c 'import torch; print("CUDA 可用:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")'
ls -lh dataset/rlaif.jsonl out/full_sft_768.pth
ls -ld ../internlm2-1_8b-reward
```

上述检查命令从项目根目录运行；脚本默认从 `trainer/` 目录启动，此时数据是 `../dataset/rlaif.jsonl`，SFT 权重是 `../out/full_sft_768.pth`，Reward Model 是 `../../internlm2-1_8b-reward`。服务器数据或 Reward Model 在别处时可指定 `--data_path`、`--reward_model_path`；`--from_weight` 只指定权重前缀，`init_model()` 仍按 `../out/{from_weight}_768.pth` 查找 Actor 初始权重，不能把任意绝对路径直接传给它。

路径和 CUDA 均就绪后，可先用小 batch、短回答做一次调试运行：

```bash
cd trainer
python train_ppo.py \
  --batch_size 1 \
  --mini_batch_size 1 \
  --ppo_update_iters 1 \
  --max_seq_len 128 \
  --max_gen_len 64 \
  --num_workers 0 \
  --save_interval 999999 \
  --debug_mode \
  --debug_interval 1 \
  --debug_log_ratio
```

`--debug_mode` 会打印 Prompt、回答、长度和奖励；`--debug_log_ratio` 会打印第一轮第一个 mini-batch 的新旧策略差异。
`--save_interval 999999` 只减少中途保存，epoch 最后一个 step 仍会保存 Actor 和续训 checkpoint。L40S 提供了运行实验的硬件条件，但显存是否足够仍取决于 Reward Model、序列长度和实际 batch；如果显存不足，先缩短 `--max_gen_len`、`--max_seq_len`。这组服务器命令尚未在本次环境中实跑。

第一轮 `log_ratio` 理论上应较接近 0，因为 Actor 尚未基于这批 rollout 更新。它不必严格等于 0：Actor 处于 `train()` 模式时的随机层、混合精度路径或外部 rollout 引擎的权重同步时机都可能造成差异。

# 10. 保存、恢复与 Rollout 权重同步

保存时有两类文件：

1. `out/ppo_actor_*.pth`：用于后续推理的 Actor 权重。
2. `checkpoints/...`：续训状态，包括 Actor、Critic、两个优化器和两个 scheduler。

Reward Model 和冻结 Ref Model 不需要保存训练状态。

默认 `torch` rollout 引擎与训练代码持有同一个 Actor 对象，因此优化器更新后生成自然使用新参数。`sglang` 是外部服务，只在下面的时机调用 `update_policy()`：

```python
if step % args.save_interval == 0 or step == iters:
    rollout_engine.update_policy(actor_model)
```

所以在 `sglang` 模式下，两个同步点之间的 rollout 可能来自较旧的策略。这是当前实现的同步边界，调试新旧 log probability 时必须把它考虑进去。

# 11. 当前实现中值得留意的边界

1. **奖励并非纯 Reward Model 分数。** 长度、thinking 格式和重复惩罚都会显著改变最终奖励。
2. **PPO 只在回答 token 上优化。** Prompt、padding 和 EOS 后位置都不应进入策略及价值损失。
3. **`answer` 当前未参与 PPO。** `RLAIFDataset` 返回空字符串，训练循环也只读取 `batch["prompt"]`。
4. **梯度累积会在每个外层 batch 末尾补一次 step。** 因此当一个 batch 内 mini-batch 数少于 `accumulation_steps` 时，当前代码不会把未满的梯度继续累积到下一个 DataLoader batch。
5. **完整训练依赖外部条件。** 至少需要 RLAIF 数据、`full_sft` 权重、Reward Model、足够显存；`sglang` 还需要可访问的服务和共享路径。
6. **一次 reward 上升不等于模型整体更好。** 还要检查回答质量、长度、重复、KL、Critic loss，以及模型是否钻了规则奖励的空子。

# 12. 学完后应能回答的四个问题

**思考**　`old_resp_logp`、`ref_resp_logp` 和 `mb_resp_logp` 分别来自谁，作用是什么？

<details>
<summary><strong>参考分析</strong></summary>

- `old_resp_logp`：rollout 时的策略，作为 PPO 概率比的分母。
- `ref_resp_logp`：冻结的 SFT 模型，限制 Actor 长期偏移。
- `mb_resp_logp`：正在训练的当前 Actor，带梯度，是策略损失真正更新的对象。

</details>

**思考**　标量奖励为什么可以训练回答中的多个 token？

<details>
<summary><strong>参考分析</strong></summary>

标量奖励先放到回答最后一个有效 token；Critic 提供每个状态的价值；GAE 从后向前递推，把最终结果相对预期的好坏传播给前面的动作，于是每个有效回答 token 都能获得优势信号。

</details>

**思考**　PPO 与 SFT 最核心的数据流差异是什么？

<details>
<summary><strong>参考分析</strong></summary>

SFT 从数据集直接取得目标回答和 labels，做一次 teacher-forcing 交叉熵训练；PPO 从数据集取得 Prompt，先由 Actor 在线生成回答，再经过奖励、价值估计和 GAE 构造训练目标。因此 PPO 的目标会随当前策略变化。

</details>

**思考**　调试时发现第一轮 `ratio` 明显偏离 1，应先检查什么？

<details>
<summary><strong>参考分析</strong></summary>

先核对 rollout 与训练 Actor 是否使用同一份最新权重，再核对 `completion_ids`、`prompt_lens`、`logp_pos` 和 mask 是否对齐；随后检查 Actor 的训练/推理模式、dropout、混合精度，以及 SGLang 是否尚未同步最新策略。不要先假设 PPO 公式本身出错。

</details>
