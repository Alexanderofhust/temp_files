import os
import sys
import json
import re
import math
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple
from transformers import AutoTokenizer

# =========================
# 导入自定义模型 (无需重复定义)
# =========================
ROOT_PATH = "/data1/chenxiwu/ICoT/ICoT-Drive/src/internvl_chat"
sys.path.insert(0, ROOT_PATH)

from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatConfig
# 直接导入隐式学生模型
from internvl.model.internvl_chat.latent_internvl_chat import ImplicitCoTDriverStudent

# =========================
# 配置区
# =========================
DATA_JSON = "/data1/chenxiwu/Drive/OpenREAD/cot_data/val/test.jsonl"
#IMAGE_ROOT = "/data1/chenxiwu"
MODEL_PATH = "/data2/chenxiwu/Drive/recogdrive/tune_output/tune_only_gptdrive_traj" 
OUTPUT_JSONL = "/data1/chenxiwu/ICoT/ICoT-Drive/src/internvl_chat/internvl/infer/infer_output/tune_gptdrive_implicit_cot.jsonl"

DEVICE = torch.device("cuda:0")
IMAGE_RESIZE = 448
NUM_POINTS = 6
M_TOKENS = 4  # 必须与训练时设置的隐式Token数量保持一致

# =========================
# 基础工具函数
# =========================
def build_transform(input_size):
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
    return transform(image).unsqueeze(0)

def extract_traj_from_gt(text: str, num_points: int = 6) -> List[Tuple[float, float]]:
    matches = re.findall(r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)", text)
    traj =[(float(x), float(y)) for x, y in matches]
    return traj[:num_points]

def ade_fde(pred: List[Tuple[float, float]], gt: List[Tuple[float, float]]) -> Tuple[float, float]:
    n = min(len(pred), len(gt))
    if n == 0: return float("inf"), float("inf")
    dists =[math.sqrt((pred[i][0] - gt[i][0])**2 + (pred[i][1] - gt[i][1])**2) for i in range(n)]
    return sum(dists) / n, dists[n - 1]

def get_file_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

# =========================
# 推理核心引擎
# =========================
class StudentInferenceEngine:
    def __init__(self, model_path):
        print(f"正在加载隐式推理学生模型到单卡: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        config = InternVLChatConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # 加载并初始化隐式学生模型
        self.model = ImplicitCoTDriverStudent.from_pretrained(
            model_path, 
            config=config, 
            M=M_TOKENS, 
            trust_remote_code=True, 
            ignore_mismatched_sizes=True
        ).to(DEVICE).to(torch.bfloat16).eval()
        
        self.transform = build_transform(IMAGE_RESIZE)
        self.num_image_token = getattr(self.model.config, 'num_image_token', 256)
        
        # 动态获取特殊 Token ID
        self.IMG_START_ID = self.tokenizer.convert_tokens_to_ids("<img>") or 151665
        self.IMG_END_ID = self.tokenizer.convert_tokens_to_ids("</img>") or 151666
        self.IMG_CONTEXT_ID = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>") or 151667
        self.IM_START_ID = self.tokenizer.convert_tokens_to_ids("<|im_start|>") or 151644
        self.IM_END_ID = self.tokenizer.convert_tokens_to_ids("<|im_end|>") or 151645

        self.model.img_context_token_id = self.IMG_CONTEXT_ID

    @torch.no_grad()
    def infer(self, image_paths, question_text):
        """核心推理：绕过自回归，单次 Forward"""
        
        # 1. 准备视觉输入
        if image_paths:
            pixel_values_list =[load_image(p, IMAGE_RESIZE, self.transform) for p in image_paths]
            pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).to(DEVICE)
            num_images = pixel_values.shape[0]
            image_flags = torch.ones(num_images, dtype=torch.long, device=DEVICE)
        else:
            pixel_values, image_flags, num_images = None, None, 0

        # 2. 精准拼接 Token ID 张量
        prefix_text = "<|im_start|>user\n"
        prefix_ids = self.tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
        
        if num_images > 0:
            single_img_ids =[self.IMG_START_ID] + [self.IMG_CONTEXT_ID] * self.num_image_token +[self.IMG_END_ID]
            all_images_ids = torch.tensor([single_img_ids] * num_images, dtype=torch.long).view(1, -1).to(DEVICE)
        else:
            all_images_ids = torch.empty((1, 0), dtype=torch.long, device=DEVICE)
        
        clean_q = question_text.replace('<image>', '').replace('<img>', '').strip()
        # 尾部加上 Assistant 标签，模型最后一层的最后一个 Token 就是触发推理的特征点
        middle_text = f"\n{clean_q}<|im_end|>\n<|im_start|>assistant\n"
        middle_ids = self.tokenizer(middle_text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
        
        model_inputs = torch.cat([prefix_ids, all_images_ids, middle_ids], dim=1)

        # 3. 执行单次前向传播 (One Pass)
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=model_inputs,
                image_flags=image_flags
            )
        
        # 4. 提取输出
        # a. 轨迹预测 [6, 2] -> list of tuples
        pred_traj_tensor = outputs["pred_trajectory"].squeeze(0).cpu().float().tolist()
        pred_traj =[(round(pt[0], 4), round(pt[1], 4)) for pt in pred_traj_tensor]

        # b. 将潜变量解码为离散文本（用于可解释性和日志观察）
        verb_logits = outputs.get("verb_logits")
        if verb_logits is not None:
            pred_text_ids = verb_logits.squeeze(0).argmax(dim=-1)
            implicit_thought_text = self.tokenizer.decode(pred_text_ids, skip_special_tokens=True)
        else:
            implicit_thought_text = ""

        return pred_traj, implicit_thought_text

# =========================
# 主程序
# =========================
def main():
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    engine = StudentInferenceEngine(MODEL_PATH)
    
    total_ade, total_fde, valid_cnt = 0.0, 0.0, 0

    print(">>> 开始执行隐式学生模型推理...")
    with open(DATA_JSON, "r", encoding="utf-8") as f, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        
        total_lines = get_file_lines(DATA_JSON)
        for line_idx, line in enumerate(tqdm(f, desc="Implicit Infer", total=total_lines)):
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
                scene_token = data.get("scene_token", f"idx_{line_idx}")
                question = data["conversations"][0]["value"].strip()
                gt_text = data["conversations"][1]["value"].strip()
                image_paths = data.get("image",[])

                # 核心推理：单次前向传播，直接输出预测轨迹与隐式解码文本
                pred_traj, implicit_thought_text = engine.infer(image_paths, question)

                # 提取 Ground Truth 轨迹并计算误差指标
                gt_traj = extract_traj_from_gt(gt_text, NUM_POINTS)
                ade, fde = ade_fde(pred_traj, gt_traj)

                if math.isfinite(ade):
                    total_ade += ade
                    total_fde += fde
                    valid_cnt += 1

                # 组装结果并写入文件
                record = {
                    "scene_token": scene_token,
                    "question": question,
                    "pred_traj": pred_traj,
                    "gt_traj": gt_traj,
                    "ADE": ade,
                    "FDE": fde,
                    "implicit_thought_text": implicit_thought_text  # 保存隐向量解码出的思维过程
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                # 定期清理显存碎片，保障长序列推理稳定不断电
                if line_idx % 200 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"处理 {scene_token} 出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    # =========================
    # 打印最终评估报告
    # =========================
    print("\n========== Evaluation Result ==========")
    print(f"Valid samples: {valid_cnt}")
    print(f"Mean ADE: {total_ade / max(valid_cnt, 1):.4f}")
    print(f"Mean FDE: {total_fde / max(valid_cnt, 1):.4f}")
    print(f"Results saved to: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()