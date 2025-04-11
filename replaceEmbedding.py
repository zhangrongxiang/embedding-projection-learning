import torch
from model.model import MiniMindLM, LMConfig  # 假设模型定义在 model/model.py 中
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_aligned_embedding(g_embedding_path):
    """加载对齐后的 g(c) 嵌入"""
    if not os.path.exists(g_embedding_path):
        raise ValueError(f"对齐嵌入路径 {g_embedding_path} 不存在")
    g_state_dict = torch.load(g_embedding_path, map_location=device)
    return g_state_dict

def load_pretrained_model(pretrained_path):
    """加载预训练模型"""
    if not os.path.exists(pretrained_path):
        raise ValueError(f"预训练模型路径 {pretrained_path} 不存在")
    pretrained_state_dict = torch.load(pretrained_path, map_location=device)
    return pretrained_state_dict

def replace_embedding_in_pretrained(pretrained_state_dict, g_state_dict):
    """将对齐后的 g(c) 替换到预训练模型的 tok_embeddings"""
    if "tok_embeddings.weight" not in pretrained_state_dict:
        raise KeyError("预训练模型中缺少 'tok_embeddings.weight'")
    if "weight" not in g_state_dict:
        raise KeyError("对齐嵌入中缺少 'weight'")

    pretrained_vocab_size, pretrained_dim = pretrained_state_dict["tok_embeddings.weight"].shape
    g_vocab_size, g_dim = g_state_dict["weight"].shape

    # 检查维度是否匹配
    if pretrained_vocab_size != g_vocab_size or pretrained_dim != g_dim:
        raise ValueError(
            f"维度不匹配：预训练嵌入 {pretrained_vocab_size}x{pretrained_dim}，"
            f"对齐嵌入 {g_vocab_size}x{g_dim}"
        )

    # 替换嵌入
    pretrained_state_dict["tok_embeddings.weight"] = g_state_dict["weight"]
    print("已将对齐后的 g(c) 替换到预训练模型的 tok_embeddings")
    return pretrained_state_dict

def save_updated_model(state_dict, output_path):
    """保存更新后的模型权重"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(state_dict, output_path)
    print(f"更新后的模型已保存到 {output_path}")

def replace_and_save(pretrained_path, g_embedding_path, output_path):
    """主函数：替换嵌入并保存模型"""
    # 加载对齐后的 g(c)
    g_state_dict = load_aligned_embedding(g_embedding_path)
    
    # 加载预训练模型
    pretrained_state_dict = load_pretrained_model(pretrained_path)
    
    # 替换嵌入
    updated_state_dict = replace_embedding_in_pretrained(pretrained_state_dict, g_state_dict)
    
    # 保存更新后的模型
    save_updated_model(updated_state_dict, output_path)

if __name__ == "__main__":
    # 默认路径（根据你的实验调整）
    pretrained_path = "./model/pretrained/pretrain_512.pth"
    g_embedding_path_linear = "./_linear_aligned_g_embedding.pth"
    g_embedding_path_nonlinear = "./_nonlinear_aligned_g_embedding.pth"
    output_path_linear = "./output/pretrain_512_with_linear_g.pth"
    output_path_nonlinear = "./output/pretrain_512_with_nonlinear_g.pth"

    # 替换并保存 Linear 版本
    try:
        print("处理 Linear 对齐嵌入...")
        replace_and_save(pretrained_path, g_embedding_path_linear, output_path_linear)
    except Exception as e:
        print(f"处理 Linear 嵌入时出错: {e}")

    # 替换并保存 Nonlinear 版本
    try:
        print("\n处理 Nonlinear 对齐嵌入...")
        replace_and_save(pretrained_path, g_embedding_path_nonlinear, output_path_nonlinear)
    except Exception as e:
        print(f"处理 Nonlinear 嵌入时出错: {e}")