import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.model import MiniMindLM, LMConfig
import math
import time
import psutil
import os
import datetime
import argparse

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义映射类，支持线性或非线性（MLP）
class EmbeddingMapping(nn.Module):
    def __init__(self, input_dim, output_dim, nonlinear=False, hidden_dim=256):
        super().__init__()
        self.nonlinear = nonlinear
        if nonlinear:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
            nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)
        else:
            self.W = nn.Parameter(torch.randn(output_dim, input_dim))
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x):
        if self.nonlinear:
            return self.fc2(self.relu(self.fc1(x)))
        else:
            return x @ self.W.T

def Logger(content, log_file=None):
    print(content)
    if log_file:
        with open(log_file, "a") as f:
            f.write(content + "\n")

def get_memory_usage(device):
    """
    获取 CPU 和 GPU 内存使用量，单位为 MB。
    """
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024 / 1024
    gpu_mem = torch.cuda.memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    return cpu_mem, gpu_mem

def train_mapping(
    llama_model_path="/root/autodl-tmp/DSnoT/llm_weights/",
    nonlinear=False,
    num_epochs=1000,
    batch_size=1024,
    lr=0.001,
    out_dir="exp",
    log_filename="mapping_training_log.txt"
):
    # 生成实验目录
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"nonlinear_{'yes' if nonlinear else 'no'}_epochs_{num_epochs}_batch_{batch_size}_lr_{lr:.6f}_{date_str}"
    save_dir = os.path.join(out_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # 设置日志文件路径
    log_file = os.path.join(save_dir, log_filename)

    # 初始化日志
    Logger(
        f"Training Mapping with {'Nonlinear MLP' if nonlinear else 'Linear Mapping'}\n"
        f"LLaMA model path: {llama_model_path}, Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {lr}\n",
        log_file
    )

    # 1. 加载本地 LLaMA 2 7B 的嵌入层 f(c)
    try:
        llama_model = AutoModelForCausalLM.from_pretrained(
            llama_model_path,
            torch_dtype=torch.float16,
            device_map=device,
            local_files_only=True
        )
        f_embedding_weight = llama_model.model.embed_tokens.weight  # [llama_vocab_size, d1]
        llama_vocab_size, d1 = f_embedding_weight.shape
        Logger(f"LLaMA embedding shape (f(c)): {f_embedding_weight.shape}", log_file)
    except Exception as e:
        Logger(f"Failed to load LLaMA model: {e}", log_file)
        raise

    # 2. 初始化 MiniMindLM 的嵌入层 g(c)
    lm_config = LMConfig(dim=512, n_layers=8, max_seq_len=512, vocab_size=50257)
    model = MiniMindLM(lm_config).to(device)
    g_embedding = model.tok_embeddings  # nn.Embedding 对象，[vocab_size, d2]
    vocab_size, d2 = g_embedding.weight.shape
    Logger(f"MiniMindLM embedding shape (g(c)): {g_embedding.weight.shape}", log_file)

    # 3. 处理词表大小差异
    effective_vocab_size = min(vocab_size, llama_vocab_size)
    Logger(f"Effective vocab size (shared tokens): {effective_vocab_size}", log_file)

    # 4. 定义映射 T: d2 -> d1
    T = EmbeddingMapping(d2, d1, nonlinear=nonlinear).to(device)

    # 5. 定义优化器，同时优化 g(c) 和 T
    optimizer = optim.Adam(
        list(T.parameters()) + list(g_embedding.parameters()),
        lr=lr
    )

    # 训练循环
    total_training_time = 0.0
    num_batches = (effective_vocab_size + batch_size - 1) // batch_size
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        num_batches_processed = 0

        for i in range(0, effective_vocab_size, batch_size):
            batch_end = min(i + batch_size, effective_vocab_size)
            batch_indices = torch.arange(i, batch_end, device=device)

            # 获取 g(c)
            g_batch = g_embedding(batch_indices)  # [batch_size, d2]
            # 计算 T(g(c))
            T_g_batch = T(g_batch)  # [batch_size, d1]
            # 获取 f(c)（LLaMA 嵌入）
            f_batch = f_embedding_weight[i:batch_end].to(device)  # [batch_size, d1]

            # 计算损失：T(g(c)) 与 f(c) 的欧几里得距离平方和均值
            loss = ((T_g_batch - f_batch) ** 2).sum(dim=1).mean()
            total_loss += loss.item()
            num_batches_processed += 1

            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time
        avg_loss = total_loss / num_batches_processed if num_batches_processed > 0 else 0

        # 获取内存使用量
        cpu_mem, gpu_mem = get_memory_usage(device)

        # 日志输出
        log_message = (
            f"Epoch {epoch}, Avg Loss: {avg_loss:.6f}, "
            f"Epoch Time: {epoch_time:.2f}s, "
            f"CPU Memory: {cpu_mem:.2f}MB, "
            f"GPU Memory: {gpu_mem:.2f}MB"
        )
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            Logger(log_message, log_file)

    # 6. 验证结果
    with torch.no_grad():
        g_all = g_embedding(torch.arange(effective_vocab_size, device=device))  # [effective_vocab_size, d2]
        T_g_all = T(g_all)  # [effective_vocab_size, d1]
        f_all = f_embedding_weight[:effective_vocab_size].to(device)  # [effective_vocab_size, d1]
        final_loss = ((T_g_all - f_all) ** 2).sum(dim=1).mean()
        distances = ((T_g_all - f_all) ** 2).sum(dim=1).sqrt()
        avg_distance = distances.mean().item()
        max_distance = distances.max().item()
        pct_below_1 = (distances < 1.0).float().mean().item() * 100

        total_log = (
            f"\nFinal Results:\n"
            f"Final Loss: {final_loss.item():.6f}\n"
            f"Average Distance per Token: {avg_distance:.6f}\n"
            f"Max Distance per Token: {max_distance:.6f}\n"
            f"Percentage of Tokens with Distance < 1.0: {pct_below_1:.2f}%\n"
            f"Total Training Time: {total_training_time:.2f}s\n"
            f"Average Time per Epoch: {total_training_time / num_epochs:.2f}s\n"
            f"Total Batches Processed: {num_batches * num_epochs}\n"
        )
        Logger(total_log, log_file)

    # 7. 保存 T 和 g(c)
    T_save_path = os.path.join(save_dir, f"mapping_T{'_nonlinear' if nonlinear else '_linear'}.pth")
    g_save_path = os.path.join(save_dir, f"{'nonlinear' if nonlinear else 'linear'}_aligned_g_embedding.pth")
    torch.save(T.state_dict(), T_save_path)
    torch.save(g_embedding.state_dict(), g_save_path)
    Logger(f"Saved T to {T_save_path}", log_file)
    Logger(f"Saved g_embedding to {g_save_path}", log_file)

    return T, g_embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Embedding Mapping with LLaMA 2 7B")
    parser.add_argument("--llama_model_path", type=str, default="/root/autodl-tmp/DSnoT/llm_weights/",
                        help="Path to local LLaMA 2 7B model")
    parser.add_argument("--nonlinear", action="store_true", help="Use nonlinear MLP mapping")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--out_dir", type=str, default="mapping_results", help="Output directory for results")
    args = parser.parse_args()

    # 运行线性映射
    print("Training with Linear Mapping:")
    train_mapping(
        llama_model_path=args.llama_model_path,
        nonlinear=False,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out_dir,
        log_filename="linear_mapping_log.txt"
    )

    # 运行非线性映射
    if args.nonlinear:
        print("\nTraining with Nonlinear MLP:")
        train_mapping(
            llama_model_path=args.llama_model_path,
            nonlinear=True,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            out_dir=args.out_dir,
            log_filename="nonlinear_mapping_log.txt"
        )