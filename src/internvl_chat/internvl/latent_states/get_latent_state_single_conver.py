import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import argparse

#说明区，不可修改或省略
# 1. 本代码用于从 InternVL 模型中提取教师模型的隐藏状态
# 2. 单卡显存够用，写死在一张卡上推理
# 3. 输入数据为多轮conversation，只有第一轮中标注了<image>标签


def parse_args():
    parser = argparse.ArgumentParser(description="提取教师模型隐藏状态")
    # 核心必选参数（也可设默认值，根据需求调整）
    parser.add_argument("--model_path", type=str, required=True, 
                        help="模型路径")
    parser.add_argument("--data_json", type=str, required=True,
                        help="待处理的JSONL数据文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出根目录")
    # 可选参数（带默认值）
    parser.add_argument("--image_root", type=str, default="",
                        help="图像根目录（无则留空）")
    parser.add_argument("--save_hidden_states", type=bool, default=True,
                        help="是否保存隐藏状态")
    parser.add_argument("--img_size", type=int, default=448,
                        help="图像处理尺寸")
    parser.add_argument("--m_tokens", type=int, default=4,
                        help="对齐的Token数量")
    return parser.parse_args()

# =========================
# 配置区（从命令行参数读取）
# =========================
args = parse_args()

# 核心配置（从命令行参数赋值）
MODEL_PATH = args.model_path
DATA_JSON = args.data_json
IMAGE_ROOT = args.image_root
OUTPUT_DIR = args.output_dir
SAVE_HIDDEN_STATES = args.save_hidden_states
IMG_SIZE = args.img_size
M_TOKENS = args.m_tokens

# 自动推导的路径（无需手动传递）
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "teacher_inference.jsonl")
HIDDEN_SAVE_PATH = os.path.join(OUTPUT_DIR, "m_hidden_states")

# SAVE_HIDDEN_STATES = True 
# IMG_SIZE = 448

# 【新增配置】：隐式推理需要的对齐 Token 数量，必须与学生模型的 M 一致
# M_TOKENS = 4 

# =========================
# 图像工具
# =========================

def build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

def load_image(image_file, input_size=448, transform=None):
    image = Image.open(image_file).convert('RGB')
    pixel_values = transform(image)
    return pixel_values.unsqueeze(0)

def get_file_lines(file_path):
    """获取文件总行数（高效方式）"""
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

# =========================
# 核心类
# =========================

class TeacherInference:
    def __init__(self, model_path):
        print(f"正在加载教师模型到单卡: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.device = torch.device("cuda:0") 
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={'': 0}
        ).eval()
        self.transform = build_transform(IMG_SIZE)
        
        # 自动探测每张图对应的 Context Token 数 (通常为 256)
        self.num_image_token = getattr(self.model.config, 'num_image_token', 256)
        
        # 动态获取特殊 Token ID（比硬编码更安全）
        self.IMG_START_ID = self.tokenizer.convert_tokens_to_ids("<img>") or 151665
        self.IMG_END_ID = self.tokenizer.convert_tokens_to_ids("</img>") or 151666
        self.IMG_CONTEXT_ID = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>") or 151667
        self.IM_START_ID = self.tokenizer.convert_tokens_to_ids("<|im_start|>") or 151644
        self.IM_END_ID = self.tokenizer.convert_tokens_to_ids("<|im_end|>") or 151645

        print(f"Token ID 确认: Start={self.IMG_START_ID}, Context={self.IMG_CONTEXT_ID}, End={self.IMG_END_ID}")

    # 【新增函数】：将不定长的特征和文本ID，均匀下采样对齐到 M 个
    def align_to_M(self, hidden_states: torch.Tensor, text_ids: torch.Tensor, M: int):
        L = hidden_states.shape[0]
        if L < M:
            # 极少数情况下，生成的文本长度小于 M，使用最后一个 token 进行 padding 补齐
            pad_len = M - L
            pad_hs = hidden_states[-1:].repeat(pad_len, 1)
            pad_ids = text_ids[-1:].repeat(pad_len)
            hidden_states = torch.cat([hidden_states, pad_hs], dim=0)
            text_ids = torch.cat([text_ids, pad_ids], dim=0)
            return hidden_states, text_ids

        # 均匀采样 M 个索引
        indices = torch.linspace(0, L - 1, M).long()
        aligned_hs = hidden_states[indices]
        aligned_ids = text_ids[indices]
        
        return aligned_hs, aligned_ids

    @torch.no_grad()
    def extract_single_turn_state(self, image_paths, conversations):
        # 1. 准备图片
        pixel_values_list =[load_image(p, IMG_SIZE, self.transform) for p in image_paths]
        pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).to(self.device)
        num_images = pixel_values.shape[0]
        
        # 2. 先用官方 chat 获取 response
        response, _ = self.model.chat(
            self.tokenizer, 
            pixel_values, 
            conversations[0]["value"], 
            generation_config={'max_new_tokens': 1024, 'do_sample': False},
            history=None,
            return_history=True
        )

        # 3. 构造 100% 精准的 Input IDs 张量
        # A. 构造开头
        prefix_text = "<|im_start|>user\n"
        prefix_ids = self.tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        
        # B. 构造图像块
        single_img_ids = [self.IMG_START_ID] + [self.IMG_CONTEXT_ID] * self.num_image_token + [self.IMG_END_ID]
        all_images_ids = torch.tensor([single_img_ids] * num_images, dtype=torch.long).view(1, -1).to(self.device)
        
        # C. 构造中间问题文本
        clean_q = conversations[0]["value"].replace('<image>', '').replace('<img>', '').strip()
        middle_text = f"\n{clean_q}<|im_end|>\n<|im_start|>assistant\n"
        middle_ids = self.tokenizer(middle_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        
        # D. 构造 Assistant 内容（【修改点】：拆分 response 和 suffix，方便精确定位）
        response_ids = self.tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        suffix_ids = torch.tensor([[self.IM_END_ID]], dtype=torch.long).to(self.device)
        
        # 拼接全量 Tensor
        model_inputs = torch.cat([prefix_ids, all_images_ids, middle_ids, response_ids, suffix_ids], dim=1)

        # 4. 前向传播
        image_flags = torch.ones(num_images, dtype=torch.long, device=self.device)
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=model_inputs,
            image_flags=image_flags,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 5. 【修改点】：精准提取 Response 对应的整段特征
        L = response_ids.shape[1]
        # 我们截取倒数第 L+1 到 倒数第 2 个 token 的特征，这刚好对应生成的 response_ids
        response_hidden_states = outputs.hidden_states[-1][0, -(L + 1):-1, :]
        response_text_ids = response_ids.squeeze(0) # 形状 [L]

        # 6. 【新增】：对齐特征与文本，压缩至 M 个 Token
        aligned_hs, aligned_ids = self.align_to_M(response_hidden_states, response_text_ids, M_TOKENS)
        
        # 转回 CPU，并将 hidden state 转为 numpy，将 text_ids 转为 list
        final_hs_np = aligned_hs.to(torch.float32).cpu().numpy()
        final_ids_list = aligned_ids.cpu().tolist()
        
        return response, final_hs_np, final_ids_list


# =========================
# 主程序
# =========================

def main():
    if not os.path.exists(HIDDEN_SAVE_PATH):
        os.makedirs(HIDDEN_SAVE_PATH, exist_ok=True)

    teacher = TeacherInference(MODEL_PATH)

    with open(DATA_JSON, "r", encoding="utf-8") as f, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        
        total_lines = get_file_lines(DATA_JSON)  # 获取总行数
        for line_idx, line in enumerate(tqdm(f, desc="Single-turn Extracting", total=total_lines)):
            data = json.loads(line)
            scene_token = data.get("scene_token", f"idx_{line_idx}")
            image_paths = data["image"]
            conversations = data["conversations"]

            try:
                # 【修改点】：接收 3 个返回值（包含了采样后的 text_ids）
                response, hidden_feat, teacher_text_ids = teacher.extract_single_turn_state(image_paths, conversations)
                
                # 保存文件名格式：scene_行号.npy
                feat_filename = f"{scene_token}_{line_idx}.npy"
                if SAVE_HIDDEN_STATES:
                    # 保存形状为 [M, hidden_size] 的特征矩阵
                    np.save(os.path.join(HIDDEN_SAVE_PATH, feat_filename), hidden_feat)

                # 记录到 JSONL
                record = {
                    "scene_token": scene_token,
                    "hidden_state_file": feat_filename,
                    "teacher_text_ids": teacher_text_ids, # 【新增字段】用于蒸馏 Verbalizer
                    "response": response
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                # 定期清理显存碎片
                if line_idx % 100 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"处理 {scene_token} 出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n提取完成！结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()