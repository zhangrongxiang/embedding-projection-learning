import os
import torch
from transformers import AutoTokenizer
from model.model import MiniMindLM, LMConfig

def replace_and_freeze_embeddings(model, g_embedding_path, device, freeze=True):
    """
    替换 MiniMind 模型的嵌入层权重，并可选冻结嵌入层和输出层。
    
    Args:
        model: MiniMindLM 模型实例
        g_embedding_path: 训练好的 g(c) 嵌入权重路径（如 linear_aligned_g_embedding.pth）
        device: 设备（cuda 或 cpu）
        freeze: 是否冻结嵌入层和输出层权重
    
    Returns:
        model: 替换嵌入后的模型
    """
    # 加载训练好的 g(c) 嵌入
    g_embedding_state_dict = torch.load(g_embedding_path, map_location=device)
    
    # 检查嵌入层类型
    if not isinstance(model.tok_embeddings, torch.nn.Embedding):
        raise ValueError("model.tok_embeddings must be nn.Embedding")
    
    # 替换 tok_embeddings 权重
    model.tok_embeddings.load_state_dict(g_embedding_state_dict)
    
    # 同步更新 output 层权重（假设共享权重）
    model.output.weight.data = model.tok_embeddings.weight.data
    
    # 可选：冻结嵌入层和输出层
    if freeze:
        model.tok_embeddings.weight.requires_grad = False
        model.output.weight.requires_grad = False
        print(f"Replaced and froze tok_embeddings and output weights")
    else:
        print(f"Replaced tok_embeddings and output weights (not frozen)")
    
    return model

def save_model(model, save_path, transformers_path=None):
    """
    保存模型权重，并可选转换为 Transformers 格式。
    
    Args:
        model: MiniMindLM 模型实例
        save_path: 保存路径（如 minimind_replaced.pth）
        transformers_path: 可选的 Transformers 格式保存路径
    """
    # 保存模型权重
    torch.save(model.state_dict(), save_path)
    print(f"Saved model weights to {save_path}")
    
    # 可选：转换为 Transformers 格式
    if transformers_path:
        LMConfig.register_for_auto_class()
        MiniMindLM.register_for_auto_class("AutoModelForCausalLM")
        model.save_pretrained(transformers_path, safe_serialization=False)
        
        # 保存 tokenizer
        tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
        tokenizer.save_pretrained(transformers_path)
        print(f"Saved model in Transformers format to {transformers_path}")

def main():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g_embedding_path = "mapping_exp/qwen25_nonlinear_no_epochs_3000_batch_1024_lr_0.001000_20250416_235407/linear_aligned_g_embedding.pth"
    save_dir = "./replaced_model"
    save_path = os.path.join(save_dir, "minimind_replaced.pth")
    transformers_path = os.path.join(save_dir, "transformers_replaced")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化 MiniMind 模型
    lm_config = LMConfig(dim=512, n_layers=8, max_seq_len=512, vocab_size=50257)  # MiniMind 词表大小
    model = MiniMindLM(lm_config).to(device)
    
    # 替换嵌入
    model = replace_and_freeze_embeddings(model, g_embedding_path, device, freeze=True)
    
    # 保存模型
    save_model(model, save_path, transformers_path)

if __name__ == "__main__":
    main()