import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, AutoModel

#说明区，不可修改或省略
# 1. 本代码用于从 InternVL 模型中提取教师模型的隐藏状态
# 2. 单卡显存够用，写死在一张卡上推理
# 3. 输入数据为多轮conversation，只有第一轮中标注了<image>标签

# =========================
# 配置区
# =========================
MODEL_PATH = "/data2/chenxiwu/Drive/recogdrive/tune_output/tune_gptdrive_traj_only_cot"
DATA_JSON = "/data1/chenxiwu/Drive/OpenREAD/cot_data/change_traj/Nuscenes_bev_traj_ema_cot.jsonl"
IMAGE_ROOT = ""
OUTPUT_DIR = "/data1/chenxiwu/ICoT/ICoT-Drive/src/internvl_chat/internvl/latent_states/data"
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "teacher_inference_multiturn.jsonl")
HIDDEN_SAVE_PATH = os.path.join(OUTPUT_DIR, "hidden_states")

SAVE_HIDDEN_STATES = True 
IMG_SIZE = 448

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

# =========================
# 核心类
# =========================

class TeacherInference:
    def __init__(self, model_path):
        print(f"正在加载教师模型到单卡: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.device = torch.device("cuda:0") 
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={'': 0}
        ).eval()
        self.transform = build_transform(IMG_SIZE)
        
        # 【核心修复 1】：强制同步模型内部的所有 ID 属性
        self.vision_token_id = 151667 # <IMG_CONTEXT>
        
        # 必须同时设置这三个地方，缺一不可
        self.model.vision_token_id = self.vision_token_id 
        self.model.config.vision_token_id = self.vision_token_id
        if hasattr(self.model, 'language_model'):
            self.model.language_model.config.vision_token_id = self.vision_token_id

        self.num_image_token = getattr(self.model.config, 'num_image_token', 256)
        
        print(f"强制对齐确认: 模型内部 vision_token_id = {self.model.vision_token_id}")

    @torch.no_grad()
    def extract_multi_turn_latent(self, image_paths, conversations):
        # 1. 准备图片
        pixel_values_list = [load_image(p, IMG_SIZE, self.transform) for p in image_paths]
        pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).to(self.device)
        num_images = pixel_values.shape[0]
        
        # 构造纯净的槽位 Tensor (不带 <img> 标签，只带 Context ID)
        expected_total_slots = num_images * self.num_image_token
        vision_tokens = torch.full((1, expected_total_slots), self.vision_token_id, dtype=torch.long).to(self.device)
        image_flags = torch.ones(num_images, dtype=torch.long, device=self.device)

        accumulated_ids = None
        turn_results = []

        # 2. 遍历多轮对话
        for i in range(0, len(conversations), 2):
            human_text = conversations[i]["value"]
            gpt_text = conversations[i+1]["value"]
            turn_idx = i // 2

            # 【核心修复 2】：绝对干净的文本清洗，防止任何字符串形式的占位符干扰
            # 必须移除所有可能被 tokenizer 识别为特殊标记的字符串
            clean_q = human_text.replace('<image>', '').replace('<img>', '').replace('<IMG_CONTEXT>', '').strip()

            if turn_idx == 0:
                # 第一轮：[Prefix] + [1280个 151667] + [Question]
                prefix_ids = self.tokenizer("<|im_start|>user\n", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                q_ids = self.tokenizer(f"{clean_q}<|im_end|>\n", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                current_user_ids = torch.cat([prefix_ids, vision_tokens, q_ids], dim=1)
            else:
                # 后续轮次
                current_user_ids = self.tokenizer(f"<|im_start|>user\n{clean_q}<|im_end|>\n", 
                                                 return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

            assistant_ids = self.tokenizer(f"<|im_start|>assistant\n{gpt_text}<|im_end|>\n", 
                                          return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

            if accumulated_ids is None:
                accumulated_ids = torch.cat([current_user_ids, assistant_ids], dim=1)
            else:
                accumulated_ids = torch.cat([accumulated_ids, current_user_ids, assistant_ids], dim=1)

            # --- 【核心修复 3】：手动校验 Mask ---
            # 检查 input_ids 里面 ID 为 151667 的数量是否严格等于 1280
            actual_count = (accumulated_ids == self.vision_token_id).sum().item()
            if actual_count != expected_total_slots:
                # 如果因为分词意外产生了额外的 151667，强制将其替换为 pad_token_id (151643)
                # 这种防御性编程能 100% 避免 1528 vs 1280 的报错
                mask = (accumulated_ids == self.vision_token_id)
                found_indices = torch.where(mask[0])[0]
                # 只保留最后 1280 个（或者前 1280 个），将多余的抹掉
                if len(found_indices) > expected_total_slots:
                    accumulated_ids[0, found_indices[expected_total_slots:]] = 151643 

            # 3. 前向传播
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=accumulated_ids,
                image_flags=image_flags,
                output_hidden_states=True,
                return_dict=True
            )

            last_hidden_state = outputs.hidden_states[-1][0, -1, :].to(torch.float32).cpu().numpy()

            turn_results.append({
                "turn_idx": turn_idx,
                "latent_state": last_hidden_state,
                "response": gpt_text
            })

        return turn_results
    
# =========================
# 图像工具与主程序保持不变 (build_transform, load_image, main)
# =========================
def build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(mean, std)
    ])

def load_image(image_file, input_size=448, transform=None):
    image = Image.open(image_file).convert('RGB')
    return transform(image).unsqueeze(0)

def main():
    if not os.path.exists(HIDDEN_SAVE_PATH): os.makedirs(HIDDEN_SAVE_PATH, exist_ok=True)
    teacher = TeacherInference(MODEL_PATH)
    with open(DATA_JSON, "r", encoding="utf-8") as f: lines = f.readlines()
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for line_idx, line in enumerate(tqdm(lines, desc="Multi-turn Latent Extracting")):
            data = json.loads(line)
            try:
                results = teacher.extract_multi_turn_latent(data["image"], data["conversations"])
                files = []
                for res in results:
                    fname = f"{data.get('scene_token','scene')}_{line_idx}_turn{res['turn_idx']}.npy"
                    np.save(os.path.join(HIDDEN_SAVE_PATH, fname), res['latent_state'])
                    files.append(fname)
                fout.write(json.dumps({"scene_token": data.get('scene_token'), "hidden_state_files": files, "responses": [r["response"] for r in results]}, ensure_ascii=False) + "\n")
                if line_idx % 50 == 0: torch.cuda.empty_cache()
            except Exception as e:
                print(f"\n[Fatal Error] {e}")
                import traceback; traceback.print_exc()
                continue
if __name__ == "__main__":
    main()