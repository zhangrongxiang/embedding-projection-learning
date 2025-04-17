import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.model import MiniMindLM, LMConfig
import math
import time
import psutil
import datetime

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
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024 / 1024
    gpu_mem = torch.cuda.memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    return cpu_mem, gpu_mem

def train_mapping(
    qwen_model_name="Qwen/Qwen2.5-3B-Instruct",
    nonlinear=False,
    num_epochs=1000,
    batch_size=1024,
    lr=0.001,
    out_dir="exp",
    log_filename="mapping_training_log.txt"
):
    # 生成实验目录
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"qwen25_nonlinear_{'yes' if nonlinear else 'no'}_epochs_{num_epochs}_batch_{batch_size}_lr_{lr:.6f}_{date_str}"
    save_dir = os.path.join(out_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # 设置日志文件路径
    log_file = os.path.join(save_dir, log_filename)

    # 初始化日志
    Logger(
        f"Training Mapping with {'Nonlinear MLP' if nonlinear else 'Linear Mapping'}\n"
        f"Qwen2.5 Model: {qwen_model_name}, Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {lr}\n",
        log_file
    )

    # 1. 加载 Qwen2.5-3B-Instruct 的嵌入层 f(c)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        qwen_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    f_embedding_weight = qwen_model.model.embed_tokens.weight  # [qwen_vocab_size, d1]
    qwen_vocab_size, d1 = f_embedding_weight.shape
    Logger(f"Qwen2.5 embedding shape (f(c)): {f_embedding_weight.shape}", log_file)

    # 2. 初始化 MiniMind 的嵌入层 g(c)
    lm_config = LMConfig(dim=512, n_layers=8, max_seq_len=512, vocab_size=50257)  # MiniMind 词表大小
    model = MiniMindLM(lm_config).to(device)
    g_embedding = model.tok_embeddings  # nn.Embedding 对象，[minimind_vocab_size, d2]
    minimind_vocab_size, d2 = g_embedding.weight.shape
    Logger(f"MiniMind embedding shape (g(c)): {g_embedding.weight.shape}", log_file)

    # 3. 定义映射 T: d2 -> d1
    T = EmbeddingMapping(d2, d1, nonlinear=nonlinear).to(device)

    # 4. 定义优化器，同时优化 g(c) 和 T
    optimizer = optim.Adam(
        list(T.parameters()) + list(g_embedding.parameters()),
        lr=lr
    )

    # 训练循环（以 MiniMind 的词表为准）
    total_training_time = 0.0
    num_batches = (minimind_vocab_size + batch_size - 1) // batch_size
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        num_batches_processed = 0

        for i in range(0, minimind_vocab_size, batch_size):
            batch_end = min(i + batch_size, minimind_vocab_size)
            batch_indices = torch.arange(i, batch_end, device=device)

            # 获取 g(c)
            g_batch = g_embedding(batch_indices)  # [batch_size, d2]
            # 计算 T(g(c))
            T_g_batch = T(g_batch)  # [batch_size, d1]
            # 获取 f(c)，使用 MiniMind 的 token ID 从 Qwen2.5 嵌入中索引
            # 假设 Qwen2.5 词表包含 MiniMind 的 token（或通过映射处理）
            f_batch = f_embedding_weight[batch_indices % qwen_vocab_size].to(device)  # [batch_size, d1]

            # 计算损失：||T(g(c)) - f(c)||_2^2 的均值
            loss = ((T_g_batch - f_batch) ** 2).sum(dim=1).mean()
            total_loss += loss.item()
            num_batches_processed += 1

            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time
        avg_loss = total_loss / num_batches_processed

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

    # 5. 验证结果
    with torch.no_grad():
        g_all = g_embedding(torch.arange(minimind_vocab_size, device=device))  # [minimind_vocab_size, d2]
        T_g_all = T(g_all)  # [minimind_vocab_size, d1]
        f_all = f_embedding_weight[:minimind_vocab_size].to(device)  # [minimind_vocab_size, d1]
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

    # 保存 T 和 g(c)
    T_save_path = os.path.join(save_dir, f"mapping_T{'_nonlinear' if nonlinear else '_linear'}.pth")
    g_save_path = os.path.join(save_dir, f"{'nonlinear' if nonlinear else 'linear'}_aligned_g_embedding.pth")
    torch.save(T.state_dict(), T_save_path)
    torch.save(g_embedding.state_dict(), g_save_path)
    Logger(f"Saved T to {T_save_path}", log_file)
    Logger(f"Saved g_embedding to {g_save_path}", log_file)

    return T, g_embedding

# 示例调用
if __name__ == "__main__":
    print("Training with Linear Mapping:")
    train_mapping(
        nonlinear=False,
        num_epochs=3000,
        batch_size=1024,
        lr=0.001,
        out_dir="mapping_exp",
        log_filename="linear_mapping_log.txt"
    )

    print("\nTraining with Nonlinear MLP:")
    train_mapping(
        nonlinear=True,
        num_epochs=3000,
        batch_size=1024,
        lr=0.001,
        out_dir="mapping_exp",
        log_filename="nonlinear_mapping_log.txt"
    )