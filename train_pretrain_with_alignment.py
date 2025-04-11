import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM, LMConfig
from model.dataset import PretrainDataset
from contextlib import nullcontext
import math
import wandb
import time
import psutil
import json
import datetime
from train_mapping import EmbeddingMapping

# 全局变量
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def replace_and_freeze_embeddings(model, g_embedding_path):
    device = next(model.parameters()).device
    g_embedding_state_dict = torch.load(g_embedding_path, map_location=device)
    if isinstance(model.tok_embeddings, nn.Embedding):
        model.tok_embeddings.load_state_dict(g_embedding_state_dict)
        model.tok_embeddings.weight.requires_grad = False
        model.output.weight.data = model.tok_embeddings.weight.data
        model.output.weight.requires_grad = False
        Logger("Replaced and froze tok_embeddings")
    else:
        raise ValueError("tok_embeddings must be nn.Embedding for replacement")
    return model

def convert_torch2transformers(torch_path, transformers_path):
    def export_tokenizer(transformers_path):
        tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
        tokenizer.save_pretrained(transformers_path)

    LMConfig.register_for_auto_class()
    MiniMindLM.register_for_auto_class("AutoModelForCausalLM")
    lm_config = LMConfig(dim=512, n_layers=8, max_seq_len=512)  # 根据实际情况调整参数
    lm_model = MiniMindLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    export_tokenizer(transformers_path)
    print(f"模型已保存为 Transformers 格式: {transformers_path}")

def Logger(content):
    print(content)

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def get_memory_usage():
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024 / 1024
    gpu_mem = torch.cuda.memory_allocated(DEVICE) / 1024**2 if torch.cuda.is_available() else 0
    return cpu_mem, gpu_mem

def train_epoch(epoch, wandb, model, train_loader, optimizer, scaler, args, ctx, iter_per_epoch, loss_fct, metrics):
    start_time = time.time()
    total_loss = 0
    num_steps = 0

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        step_start_time = time.time()
        X, Y, loss_mask = X.to(args.device), Y.to(args.device), loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(Y.size())
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

        step_time = time.time() - step_start_time
        cpu_mem, gpu_mem = get_memory_usage()
        metrics["step_time"].append(step_time)
        metrics["step_cpu_mem"].append(cpu_mem)
        metrics["step_gpu_mem"].append(gpu_mem)

        if step % args.log_interval == 0:
            Logger(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iter_per_epoch}) '
                f'loss:{loss.item() * args.accumulation_steps:.3f} '
                f'lr:{lr:.12f} '
                f'step_time:{step_time:.3f}s '
                f'gpu_mem:{gpu_mem:.2f}MB'
            )
            if wandb:
                wandb.log({"loss": loss.item() * args.accumulation_steps, "lr": lr, "step_time": step_time, "gpu_mem": gpu_mem})

        if (step + 1) % args.save_interval == 0:
            model.eval()
            moe_path = '_moe' if args.use_moe else ''
            suffix = metrics["stage"]
            ckp = f'{args.save_dir}/pretrain_{args.dim}{moe_path}_{suffix}_epoch{epoch}_step{step}.pth'
            torch.save(model.state_dict(), ckp)
            Logger(f"Saved checkpoint to {ckp}")
            transformers_ckp = f'{args.save_dir}/transformers_{suffix}_epoch{epoch}_step{step}'
            convert_torch2transformers(ckp, transformers_ckp)
            model.train()

    avg_loss = total_loss / num_steps
    epoch_time = time.time() - start_time
    cpu_mem, gpu_mem = get_memory_usage()

    metrics["loss"].append(avg_loss)
    metrics["time"].append(epoch_time)
    metrics["memory"].append(cpu_mem)
    metrics["gpu_memory"].append(gpu_mem)
    return metrics

def init_model(lm_config, args, freeze_embedding=True):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config).to(args.device)
    if freeze_embedding:
        model.tok_embeddings.weight.requires_grad = False
        model.output.weight.requires_grad = False
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer

def save_metrics_to_file(metrics, filename, model_name, stage):
    data = {
        "model_name": model_name,
        "stage": stage,
        "avg_loss_per_epoch": metrics["loss"],
        "time_per_epoch_seconds": metrics["time"],
        "memory_per_epoch_mb": metrics["memory"],
        "gpu_memory_per_epoch_mb": metrics["gpu_memory"],
        "total_time_seconds": sum(metrics["time"]),
        "avg_memory_mb": sum(metrics["memory"]) / len(metrics["memory"]),
        "avg_gpu_memory_mb": sum(metrics["gpu_memory"]) / len(metrics["gpu_memory"]),
        "step_time_seconds": metrics["step_time"],
        "step_cpu_mem_mb": metrics["step_cpu_mem"],
        "step_gpu_mem_mb": metrics["step_gpu_mem"]
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    Logger(f"Saved {stage} metrics to {filename}")

def train_model(model, train_loader, args, stage, metrics_file):
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    loss_fct = nn.CrossEntropyLoss(reduction='none')
    iter_per_epoch = len(train_loader)
    metrics = {
        "loss": [], "time": [], "memory": [], "gpu_memory": [],
        "step_time": [], "step_cpu_mem": [], "step_gpu_mem": [],
        "stage": stage
    }

    for epoch in range(args.epochs):
        metrics = train_epoch(epoch, wandb, model, train_loader, optimizer, scaler, args, ctx, iter_per_epoch, loss_fct, metrics)

    model.eval()
    moe_path = '_moe' if args.use_moe else ''
    final_ckp = f'{args.save_dir}/pretrain_{args.dim}{moe_path}_{stage}_trained.pth'
    torch.save(model.state_dict(), final_ckp)
    Logger(f"Saved {stage} trained model to {final_ckp}")
    transformers_final_ckp = f'{args.save_dir}/transformers_{stage}_trained'
    convert_torch2transformers(final_ckp, transformers_final_ckp)
    save_metrics_to_file(metrics, metrics_file, f"pretrain_{args.dim}{moe_path}", stage)

    return model

def save_args(args, filename):
    args_dict = vars(args)
    with open(filename, 'w') as f:
        json.dump(args_dict, f, indent=4)
    Logger(f"Saved args to {filename}")

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
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    parser.add_argument("--pretrained_path", type=str, default="./model/pretrained/pretrain_512.pth")
    parser.add_argument("--mapping_path_linear", type=str, default="alignment_mapping_linear.pth")
    parser.add_argument("--mapping_path_nonlinear", type=str, default="alignment_mapping_nonlinear.pth")
    parser.add_argument("--g_embedding_path_linear", type=str, default="_linear_aligned_g_embedding.pth")
    parser.add_argument("--g_embedding_path_nonlinear", type=str, default="_nonlinear_aligned_g_embedding.pth")
    parser.add_argument("--nonlinear",default=False, help="Run with nonlinear embedding")
    args = parser.parse_args()

    # 生成实验目录
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"nonlinear_{'yes' if args.nonlinear else 'no'}_saveint_{args.save_interval}_{date_str}"
    args.save_dir = os.path.join(args.out_dir, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    # 保存 args
    save_args(args, os.path.join(args.save_dir, "args.json"))

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=f"MiniMind-Pretrain-{exp_name}")
    else:
        wandb = None

    # 数据加载器
    model, tokenizer = init_model(lm_config, args, freeze_embedding=True)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, pin_memory=True, drop_last=False,
        shuffle=False, num_workers=args.num_workers
    )

    # 1. 训练初始模型（冻结 embedding）
    Logger("Training initial model with frozen embedding...")
    torch.save(model.state_dict(), f"{args.save_dir}/pretrain_{args.dim}_initial.pth")
    model_initial = train_model(model, train_loader, args, "initial", 
                                os.path.join(args.save_dir, "metrics_initial.json"))
    g_embedding_path = args.g_embedding_path_nonlinear if args.nonlinear else args.g_embedding_path_linear
    stage_name = "nonlinear_embedding" if args.nonlinear else "linear_embedding"
    Logger(f"Replacing with {'Nonlinear' if args.nonlinear else 'Linear'} g(c)...")
    model_replaced = replace_and_freeze_embeddings(model_initial, g_embedding_path)
    
    # 保存嵌入后未训练的模型
    moe_path = '_moe' if args.use_moe else ''
    untrained_ckp = f'{args.save_dir}/pretrain_{args.dim}{moe_path}_{stage_name}_untrained.pth'
    torch.save(model_replaced.state_dict(), untrained_ckp)
    Logger(f"Saved {stage_name} untrained model to {untrained_ckp}")
    transformers_untrained_ckp = f'{args.save_dir}/transformers_{stage_name}_untrained'
    convert_torch2transformers(untrained_ckp, transformers_untrained_ckp)
# # 3. 训练替换后的模型
#     Logger(f"Training {stage_name} model...")
#     train_model(model_replaced, train_loader, args, stage_name, 
#                 os.path.join(args.save_dir, f"metrics_{stage_name}.json"))