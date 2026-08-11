import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch,           # 当前是第几个 epoch，从 0 开始
                loader,          # DataLoader，每次产出一个 batch: input_ids, labels
                iters,           # 当前 epoch 总共有多少个 step(batch)，用于日志和学习率计算, iters就是iterations per epoch
                start_step=0,    # 断点续训时，从哪个 step 接着训
                wandb=None,      # 日志工具
                ):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)   # 把数据搬到GPU
        labels = labels.to(args.device)
        last_step = step   # 记录当前的step

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate) # 计算当前学习率
        # 更新optimizer里的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播和loss计算
        # - with autocast_ctx表示混合精度上下文，cuda会用bf16/fp16加速部分计算
        with autocast_ctx:
            # res是模型前向传播的返回结果对象，实际是调用MiniMindForCausalLM.forward()
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss            # 主语言模型 loss + MoE辅助loss
            loss = loss / args.accumulation_steps     # 除以梯度累计步数，因为这里是先计算小batch的loss，而代码的计算方法是args.accumulation_steps个小batch的梯度累计起来再更新参数

        # 反向传播
        # - backward就是计算loss对每个参数的偏导数，例如∂loss / ∂W，计算完的梯度存在每个模型参数的.grad属性里
        scaler.scale(loss).backward()

        # 判断是否要进行梯度更新
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 先把梯度缩回正常尺度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪，防止发生梯度爆炸

            # 更新参数
            scaler.step(optimizer)  # 根据梯度更新模型参数，可以理解为启用GradScanler后的optimizer.step()
            scaler.update()         # 更新下一次FP16训练使用的动态缩放因子

            # 清空梯度，否则 PyTorch 默认会累加梯度。set_to_none=True 通常更省显存
            optimizer.zero_grad(set_to_none=True)

        # 每隔log_interval打一次日志，或者当前epoch最后一个step也打日志
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60   # 估算剩余时间
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 保存模型的checkpoint
        # - is_main_process()只允许主进程保存，DDP多卡训练的时候如果多进程同时保存文件，会发生冲突
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 保存模型权重，并生成保存路径
            model.eval()      # 把模型切换为评估/推理模式
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 拿到原始模型，因为模型可能被包过DistributedDataParallel(model)
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # 拿到模型权重字典
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp) # 保存模型权重
            # 保存断点可训的checkpoint，不只是保存模型参数，还保存optimizer状态、scaler状态等
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()    # 把模型切换为训练模式
            del state_dict

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    '''
    Automatic Mixed Precision: 不是所有计算都用float32，让部分适合低精度的计算用bfloat16或者float16，以减少缓存、提高速度
    '''
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    '''
    训练时上传 loss/lr 等指标，保存 checkpoint 时把 run id 一起存进去，方便断点续训继续写到同一个实验记录
    '''
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========

    # 模型、分词器实例
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 创建了一个"会在取样本时把文本变成token"的数据集对象
    # - 注意这个PretrainDataset类的__getitem__函数是怎么把文本转换为input_ids
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 如果当前是DDP分布式训练，就给数据集创建一个DistributedSampler
    # - DDP多卡训练的时候每一张卡都会对应一个进程，如果没有sampler，每个进程可能都会读取完整数据，导致重复训练同一批样本
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 创建AMP混合精度训练用的梯度缩放器
    # - fp16数值范围小，梯度太小可能变成0，造成“下溢”，GradScaler会把loss放大，再反向传播，更新参数之前再缩回来
    # - enabled参数表示只有--dtype float16的时候启用scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 创建优化器，用AdamW算法更新模型参数
    # - 优化器: 根据反向传播得到的梯度，决定模型参数应该“往哪个方向、走多大一步”进行更新的算法
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0  # 默认从第0个epoch，第0个step开始训练
    # 如果有checkpoint的话
    if ckp_data:
        model.load_state_dict(ckp_data['model'])            # 恢复模型已经学到的权重
        optimizer.load_state_dict(ckp_data['optimizer'])    # 恢复优化器状态
        scaler.load_state_dict(ckp_data['scaler'])          # 恢复混合精度训练里的GradScaler状态，主要用于fp16训练时的loss scaling
        # 告诉程序从第几个epoch和step开始跑
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # Python短路写法，等价于 [if] train_sampler [then] train_sampler.set_epoch(epoch)
        # - 主要用于DDP分布式训练，DistributedSampler需要每个epoch设置不同的epoch值，训练数据被打乱（shuffle）后的排列顺序不同
        train_sampler and train_sampler.set_epoch(epoch)
        # 每个epoch设置一个随机数种子，然后生成一个随机打乱的数据索引列表
        # - 种子决定随机序列，42 + epoch的写法表示每轮不同，但可预测，没有用到随机seed
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # DataLoader把数据组织成batch
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            # loader的长度就是batch数，batch数 * batch_size = 训练集样本数
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()