import torch
import torch.nn as nn
import torch.optim as optim
from model.model import MiniMindLM, LMConfig  # 假设 model.py 在 model/ 目录下
import math
import time
import psutil
import os

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

# 主函数：联合训练 g(c) 和 T
def train_mapping(pretrained_path="./model/pretrained/pretrain_512.pth", nonlinear=False, num_epochs=1000, batch_size=1024, lr=0.001, log_file="mapping_training_log.txt"):
    # 初始化日志文件
    with open(log_file, "w") as f:
        f.write(f"Training Mapping with {'Nonlinear MLP' if nonlinear else 'Linear Mapping'}\n")
        f.write(f"Pretrained path: {pretrained_path}, Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {lr}\n\n")

    # 1. 加载预训练模型的嵌入层 f(c)
    pretrained_state_dict = torch.load(pretrained_path, map_location=device)
    f_embedding_weight = pretrained_state_dict["tok_embeddings.weight"]  # [vocab_size, d1]
    vocab_size, d1 = f_embedding_weight.shape
    print(f"Pretrained embedding shape (f(c)): {f_embedding_weight.shape}")

    # 2. 初始化新模型的嵌入层 g(c)
    lm_config = LMConfig(dim=512, n_layers=8, max_seq_len=512, vocab_size=vocab_size)
    model = MiniMindLM(lm_config).to(device)
    g_embedding = model.tok_embeddings  # nn.Embedding 对象，权重形状 [vocab_size, d2]
    d2 = g_embedding.weight.shape[1]
    print(f"New model embedding shape (g(c)): {g_embedding.weight.shape}")

    # 3. 定义映射 T
    T = EmbeddingMapping(d2, d1, nonlinear=nonlinear).to(device)  # T: d2 -> d1

    # 4. 定义优化器，同时优化 g(c) 和 T
    optimizer = optim.Adam(
        list(T.parameters()) + list(g_embedding.parameters()),
        lr=lr
    )

    # 训练循环
    total_training_time = 0.0
    num_batches = (vocab_size + batch_size - 1) // batch_size
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        num_batches_processed = 0

        for i in range(0, vocab_size, batch_size):
            batch_end = min(i + batch_size, vocab_size)
            batch_indices = torch.arange(i, batch_end, device=device)

            # 获取 g(c)
            g_batch = g_embedding(batch_indices)  # [batch_size, d2]
            # 计算 T(g(c))
            T_g_batch = T(g_batch)  # [batch_size, d1]
            # 获取 f(c)
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
        avg_loss = total_loss / num_batches_processed

        # 获取内存使用量
        if device.type == "cuda":
            mem_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**2   # MB
        else:
            mem_allocated = mem_reserved = 0.0
        cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2

        # 日志输出
        log_message = (
            f"Epoch {epoch}, Avg Loss: {avg_loss:.6f}, "
            f"Epoch Time: {epoch_time:.2f}s, "
            f"CPU Memory: {cpu_mem:.2f}MB, "
            f"GPU Memory Allocated: {mem_allocated:.2f}MB, "
            f"GPU Memory Reserved: {mem_reserved:.2f}MB"
        )
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(log_message)
            with open(log_file, "a") as f:
                f.write(log_message + "\n")

    # 5. 验证结果
    with torch.no_grad():
        g_all = g_embedding(torch.arange(vocab_size, device=device))  # [vocab_size, d2]
        T_g_all = T(g_all)  # [vocab_size, d1]
        f_all = f_embedding_weight.to(device)
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
        print(total_log)
        with open(log_file, "a") as f:
            f.write(total_log)

    # 保存 T 和 g(c)
    torch.save(T.state_dict(), f"mapping_T{'_nonlinear' if nonlinear else '_linear'}.pth")
    torch.save(g_embedding.state_dict(), f"{'_nonlinear' if nonlinear else '_linear'}_aligned_g_embedding.pth")
    return T, g_embedding

# 示例调用
if __name__ == "__main__":
    print("Training with Linear Mapping:")
    train_mapping(nonlinear=False, num_epochs=1000, batch_size=1024, lr=0.001, log_file="linear_mapping_log.txt")

    print("\nTraining with Nonlinear MLP:")
    train_mapping(nonlinear=True, num_epochs=1000, batch_size=1024, lr=0.001, log_file="nonlinear_mapping_log.txt")