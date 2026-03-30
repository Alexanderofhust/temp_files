import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel

# =========================
# 配置区
# =========================
MODEL_PATH = "/data2/chenxiwu/Drive/recogdrive/tune_output/tune_gptdrive_traj_only_cot"
HIDDEN_FILE = "/data1/chenxiwu/ICoT/data/hidden_states/0a1b4e0aa3824b0a96bafae7105c58cc_21376.npy" # 选一个已生成的文件

def verify_and_decode():
    # 1. 加载 Tokenizer 和模型（仅需加载语言模型部分以节省显存）
    print("加载模型中...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # 只需要语言模型部分的 lm_head 来进行解码
    model = AutoModel.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True, 
        device_map={'': 0}
    ).eval()
    
    # 2. 加载生成的 .npy 文件
    if not os.path.exists(HIDDEN_FILE):
        print(f"文件不存在: {HIDDEN_FILE}")
        return
    
    hidden_state_np = np.load(HIDDEN_FILE)
    print(f"文件加载成功: {HIDDEN_FILE}")
    print(f"特征形状 (Shape): {hidden_state_np.shape}") # 应该是 [2048] 或 [4096]
    
    # 检查数值异常
    if np.isnan(hidden_state_np).any():
        print("警告: 发现 NaN 数值！")
    print(f"数值统计: Max={hidden_state_np.max():.4f}, Min={hidden_state_np.min():.4f}, Mean={hidden_state_np.mean():.4f}")

    # 3. 执行“解码”过程
    # 将 numpy 转回 torch tensor
    hidden_tensor = torch.from_numpy(hidden_state_np).to(torch.bfloat16).cuda().unsqueeze(0) # [1, Dim]

    with torch.no_grad():
        # 在 InternVL/InternLM2 中，解码通常是 hidden_state -> language_model.output -> lm_head
        # 最直接的方法是直接调用 lm_head
        if hasattr(model.language_model, 'output'):
            # 针对部分 InternLM2 结构，需要经过 Final Norm
            hidden_tensor = model.language_model.model.norm(hidden_tensor)
        
        logits = model.language_model.lm_head(hidden_tensor) # [1, Vocab_Size]
        
        # 获取概率最大的前 5 个候选词
        probs = torch.softmax(logits.float(), dim=-1)
        top_k_probs, top_k_ids = torch.topk(probs, 5)
        
        print("\n--- 解码结果 (Top 5 预测) ---")
        for i in range(5):
            token_id = top_k_ids[0][i].item()
            prob = top_k_probs[0][i].item()
            word = tokenizer.decode([token_id])
            print(f"Rank {i+1}: ID={token_id:6d} | Prob={prob:.4f} | Word='{word}'")

if __name__ == "__main__":
    verify_and_decode()