from datetime import datetime
import os
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model import MiniMindLM, LMConfig
from model.dataset import PretrainDataset
from contextlib import nullcontext
import math
import wandb
import time
import psutil  # 用于监控内存
import json  # 用于保存结果到文件
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from train_mapping import EmbeddingMapping
def embed_aligned_embeddings(model, pretrained_path="./model/pretrained/pretrain_512.pth", mapping_path="alignment_mapping_linear.pth", nonlinear=True):
    device = next(model.parameters()).device
    # 加载预训练嵌入 f(c)
    pretrained_state_dict = torch.load(pretrained_path, map_location=device)
    
    f_embedding_weight = pretrained_state_dict["tok_embeddings.weight"]  # [vocab_size, d1]
    vocab_size, d1 = f_embedding_weight.shape
    print(f"aligned embedding:{nonlinear}")
    # 加载映射 T
    T = EmbeddingMapping(d1, model.params.dim, nonlinear=nonlinear).to(device)
    T.load_state_dict(torch.load(mapping_path, map_location=device))
    
    T.eval()

    # 计算 T(f(c))
    with torch.no_grad():
        aligned_embeddings = T(f_embedding_weight.to(device))  # [vocab_size, d2]

    # 将 T(f(c)) 嵌入到模型的 tok_embeddings
    model.tok_embeddings.weight.data = aligned_embeddings
    # 确保 output 层权重与 tok_embeddings 一致（因为它们共享权重）
    model.output.weight.data = aligned_embeddings

    # 冻结嵌入层
    model.tok_embeddings.weight.requires_grad = False
    model.output.weight.requires_grad = False  # 如果共享权重，也冻结 output 层

    return model

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # MB
    return 0

def train_epoch(epoch, wandb, model, train_loader, optimizer, scaler, args, ctx, iter_per_epoch, loss_fct, metrics):
    start_time = time.time()
    total_loss = 0
    num_steps = 0

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * args.accumulation_steps
        num_steps += 1

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            print(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iter_per_epoch}) '
                f'loss:{loss.item() * args.accumulation_steps:.3f} '
                f'lr:{optimizer.param_groups[-1]["lr"]:.12f} '
                f'epoch_Time:{spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60}min'
            )
            if wandb and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr'],
                    "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                })

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if args.use_moe else ''
            suffix = 'before_embedding' if metrics["stage"] == "before" else 'after_embedding'
            ckp = f'{args.save_dir}/pretrain_{args.dim}{moe_path}_{suffix}_epoch{epoch}_step{step}.pth'
            if isinstance(model, DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, ckp)
            model.train()

    avg_loss = total_loss / num_steps
    epoch_time = time.time() - start_time
    cpu_memory = get_memory_usage()
    gpu_memory = get_gpu_memory_usage()

    metrics["loss"].append(avg_loss)
    metrics["time"].append(epoch_time)
    metrics["cpu_memory"].append(cpu_memory)
    metrics["gpu_memory"].append(gpu_memory)

    print(f"Epoch {epoch}: Avg Loss={avg_loss:.3f}, Time={epoch_time:.1f}s, CPU Memory={cpu_memory:.1f}MB, GPU Memory={gpu_memory:.1f}MB")
    return metrics

def init_model(lm_config, args):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config).to(args.device)
    print(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer

def init_distributed_mode(args):
    if not ddp:
        return
    global ddp_local_rank, DEVICE
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)
    args.device = DEVICE

def save_metrics_to_file(metrics, filename, model_name, stage):
    data = {
        "model_name": model_name,
        "stage": stage,
        "avg_loss_per_epoch": metrics["loss"],
        "time_per_epoch_seconds": metrics["time"],
        "cpu_memory_per_epoch_mb": metrics["cpu_memory"],
        "gpu_memory_per_epoch_mb": metrics["gpu_memory"],
        "total_time_seconds": sum(metrics["time"]),
        "avg_cpu_memory_mb": sum(metrics["cpu_memory"]) / len(metrics["cpu_memory"]),
        "avg_gpu_memory_mb": sum(metrics["gpu_memory"]) / len(metrics["gpu_memory"])
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved {stage} metrics to {filename}")
    return data

def plot_metrics(metrics_before, metrics_after, output_dir):
    epochs = range(1, len(metrics_before["avg_loss_per_epoch"]) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, metrics_before["avg_loss_per_epoch"], label="Before Embedding", marker='o')
    plt.plot(epochs, metrics_after["avg_loss_per_epoch"], label="After Embedding", marker='o')
    plt.title("Average Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, metrics_before["time_per_epoch_seconds"], label="Before Embedding", marker='o')
    plt.plot(epochs, metrics_after["time_per_epoch_seconds"], label="After Embedding", marker='o')
    plt.title("Time per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, metrics_before["gpu_memory_per_epoch_mb"], label="Before Embedding", marker='o')
    plt.plot(epochs, metrics_after["gpu_memory_per_epoch_mb"], label="After Embedding", marker='o')
    plt.title("GPU Memory per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Memory (MB)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, "training_comparison.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Saved comparison plot to {plot_file}")

def train_model(model, train_loader, args, stage, metrics_file):
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    loss_fct = nn.CrossEntropyLoss(reduction='none')
    iter_per_epoch = len(train_loader)
    metrics = {"loss": [], "time": [], "cpu_memory": [], "gpu_memory": [], "stage": stage}

    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{stage}: Frozen params={frozen_params / 1e6:.3f}M, Total params={total_params / 1e6:.3f}M")

    for epoch in range(args.epochs):
        metrics = train_epoch(epoch, wandb, model, train_loader, optimizer, scaler, args, ctx, iter_per_epoch, loss_fct, metrics)

    if not ddp or dist.get_rank() == 0:
        model.eval()
        moe_path = '_moe' if args.use_moe else ''
        final_ckp = f'{args.save_dir}/pretrain_{args.dim}{moe_path}_{stage}_trained.pth'
        if isinstance(model, DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, final_ckp)
        print(f"Saved {stage} trained model to {final_ckp}")
        metrics_data = save_metrics_to_file(metrics, metrics_file, f"pretrain_{args.dim}{moe_path}", stage)
        return metrics_data

    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MiniMind Pretraining with Alignment")
    parser.add_argument("--out_dir", type=str, default="exp")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", default=False)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    parser.add_argument("--pretrained_path", type=str, default="./model/pretrained/pretrained_512.pth")
    parser.add_argument("--mapping_path", type=str, default="alignment_mapping_linear.pth")
    parser.add_argument("--nonlinear", default=True, help="Use nonlinear mapping for alignment")
    args = parser.parse_args()

    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"nonlinear_{'true' if args.nonlinear else 'false'}-save_interval_{args.save_interval}-{timestamp}"
    args.save_dir = os.path.join(args.out_dir, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    # 保存 args 到文件
    args_dict = vars(args)
    with open(os.path.join(args.save_dir, "args.json"), 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(f"Saved args to {os.path.join(args.save_dir, 'args.json')}")

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, args.device

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode(args)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        wandb.init(project=args.wandb_project, name=f"MiniMind-Pretrain-{exp_name}")
    else:
        wandb = None

    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config, args)

    # 设置数据加载器
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 1. 训练嵌入前的模型
    print("Training model before embedding...")
    if not ddp or dist.get_rank() == 0:
        torch.save(model.state_dict(), f"{args.save_dir}/pretrain_{args.dim}_before_embedding.pth")
        print("Saved initial model before embedding")
    metrics_before = train_model(model, train_loader, args, "before_embedding", os.path.join(args.save_dir, "metrics_before.json"))

    # 2. 嵌入对齐后的嵌入并冻结
    print("Embedding aligned embeddings and freezing tok_embeddings...")
    model_after = embed_aligned_embeddings(
        model,
        pretrained_path=args.pretrained_path,
        mapping_path=args.mapping_path,
        nonlinear=args.nonlinear
    )

    # 3. 训练嵌入后的模型
    print("Training model after embedding...")
    metrics_after = train_model(model_after, train_loader, args, "after_embedding", os.path.join(args.save_dir, "metrics_after.json"))
    plot_metrics(metrics_before, metrics_after, args.save_dir)
