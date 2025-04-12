import torch
from transformers import AutoTokenizer
from model.model import MiniMindLM, LMConfig  # 假设模型定义在 model/model.py 中

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(pth_path, dim=512, n_layers=8, max_seq_len=512, use_moe=False):
    """加载 .pth 格式的模型和分词器"""
    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    
    # 初始化模型
    lm_config = LMConfig(dim=dim, n_layers=n_layers, max_seq_len=max_seq_len, use_moe=use_moe)
    model = MiniMindLM(lm_config).to(device)
    
    # 加载权重
    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    
    # 统计参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"模型参数量: {params:.2f}M")
    
    return model.eval(), tokenizer

def chat_with_model(pth_path):
    """与模型对话"""
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(pth_path)
    
    print("欢迎与 MiniMindLM 对话！输入 'exit' 退出。")
    while True:
        # 获取用户输入
        user_input = input("你: ")
        if user_input.lower() == "exit":
            print("再见！")
            break
        
        # 构造输入
        prompt = tokenizer.bos_token + user_input if tokenizer.bos_token else user_input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        
        # 生成回复
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=100,  # 控制生成长度
                temperature=0.85,    # 控制随机性
                top_p=0.85,          # 控制多样性
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码并输出回复
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"MiniMindLM: {response}")

if __name__ == "__main__":
    # 示例权重路径（替换为你的实际路径）
    pth_path = "/root/autodl-tmp/zrx/minimind/exp/nonlinear_yes_saveint_10000_20250412_105842/pretrain_512_initial_epoch0_step9999.pth"
    
    # 开始对话
    try:
        chat_with_model(pth_path)
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except Exception as e:
        print(f"发生错误: {e}")