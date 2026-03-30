import torch
import torch.nn as nn
import torch.cuda
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict

# 保持你之前的路径导入逻辑
import sys
ROOT_PATH = "/data1/chenxiwu/ICoT/ICoT-Drive/src/internvl_chat"
sys.path.insert(0, ROOT_PATH)

from internvl.model.internvl_chat.modeling_internvl_chat import (
    InternVLChatModel,
    InternVLChatConfig,
    CausalLMOutputWithPast
)

class ImplicitCoTDriverStudent(InternVLChatModel):
    def __init__(self, config: InternVLChatConfig, M: int = 4):
        super().__init__(config)
        self.M = M 
        self.hidden_size = self.language_model.config.hidden_size
        self.vocab_size = self.language_model.config.vocab_size
        
        # 如果模型配置里没有指定这个ID，手动设一个默认值（测试用）
        if self.img_context_token_id is None:
            self.img_context_token_id = 111111 # 随便设一个逻辑上的占位符

        self.latent_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            # nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.M * self.hidden_size)
        )

        self.verbalizer = nn.Sequential(
            # nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.vocab_size)
        )
        self.latent_proj = self.latent_proj.to(dtype=torch.bfloat16)
        self.verbalizer = self.verbalizer.to(dtype=torch.bfloat16)
        self._init_new_layers()  # 显式初始化新增层，避免自动初始化异常

        # ========== 强制开启新增层梯度 ==========
        for param in self.latent_proj.parameters():
            param.requires_grad = True
        for param in self.verbalizer.parameters():
            param.requires_grad = True
        # 验证
        print("[GRAD CHECK] latent_proj requires_grad:", all(p.requires_grad for p in self.latent_proj.parameters()))
        print("[GRAD CHECK] verbalizer requires_grad:", all(p.requires_grad for p in self.verbalizer.parameters()))
        # =======================================
        
        # 现stage不用动作头
        # self.trajectory_head = nn.Sequential(
        #     nn.Linear(self.M * self.hidden_size, self.hidden_size),
        #     nn.GELU(),
        #     nn.LayerNorm(self.hidden_size),
        #     nn.Linear(self.hidden_size, 6 * 2) 
        # )

    def _init_new_layers(self):
        """显式初始化新增层（FP32+设备绑定+保守初始化）"""
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def init_linear_layer(layer):
            if isinstance(layer, nn.Linear):
                # Xavier初始化（BF16友好）
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('gelu'))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 1e-4)  # 偏置非0
                layer.weight.data = layer.weight.data.to(device=device, dtype=torch.bfloat16)
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.to(device=device, dtype=torch.bfloat16)
        
        # 初始化LayerNorm层的通用函数（保守版）
        def init_layernorm_layer(layer):
            if isinstance(layer, nn.LayerNorm):
                # 权重初始化为1.0（FP32），偏置为0
                weight = torch.ones(layer.weight.shape, device=device, dtype=dtype)
                bias = torch.zeros(layer.bias.shape, device=device, dtype=dtype)
                layer.weight.data = weight.to(layer.weight.dtype)
                layer.bias.data = bias.to(layer.bias.dtype)
        
        # 1. 初始化latent_proj的所有层
        for layer in self.latent_proj:
            if isinstance(layer, nn.Linear):
                init_linear_layer(layer)
            elif isinstance(layer, nn.LayerNorm):
                init_layernorm_layer(layer)
        
        # 2. 初始化verbalizer的所有层
        for layer in self.verbalizer:
            if isinstance(layer, nn.Linear):
                init_linear_layer(layer)
            elif isinstance(layer, nn.LayerNorm):
                init_layernorm_layer(layer)
        
        # 仅替换NaN/Inf，不裁剪正常数值
        for name, param in self.named_parameters():
            if "latent_proj" in name or "verbalizer" in name:
                param.data = torch.nan_to_num(param.data, nan=1e-4, posinf=1.0, neginf=-1.0)
         
        # 4. 验证初始化结果
        print("[INIT CHECK] 新增层初始化完成，检查latent_proj第一层权重：")
        first_linear = self.latent_proj[0]
        weight = first_linear.weight.data
        print(f"  权重NaN: {torch.isnan(weight).any().item()}")
        print(f"  权重Inf: {torch.isinf(weight).any().item()}")
        print(f"  权重均值: {weight.mean().item():.6f}")
        print(f"  权重范围: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
        # 5. 同样检查verbalizer第一层权重
        first_linear_v = self.verbalizer[1]  # verbalizer的第一层Linear是第二层（第一层是LayerNorm）
        weight_v = first_linear_v.weight.data
        print(f"  verbalizer 权重NaN: {torch.isnan(weight_v).any().item()}")
        print(f"  verbalizer 权重Inf: {torch.isinf(weight_v).any().item()}")
        print(f"  verbalizer 权重均值: {weight_v.mean().item():.6f}")
        print(f"  verbalizer 权重范围: [{weight_v.min().item():.6f}, {weight_v.max().item():.6f}]")

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        teacher_hidden_states: Optional[torch.Tensor] = None,
        teacher_text_ids: Optional[torch.LongTensor] = None,
        gt_trajectory: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 1. image_flags 处理 (参考官方)
        if image_flags is not None:
            image_flags = image_flags.squeeze(-1)
        else:
            image_flags = torch.ones((pixel_values.shape[0],), dtype=torch.long, device=pixel_values.device)

        # 2. 调用父类获取基础输出 (包含 VLM 损失和各层 Hidden States)
        outputs = super().forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        # # 3. 提取特征：必须通过 Mask 找到真正的最后一个 Token，而不是 -1
        # last_hidden_state = outputs.hidden_states[-1] # [B, L, D]
        # if attention_mask is not None:
        #     # 找到每一行最后一个非 0 (非 Mask) 的位置索引
        #     last_token_indices = attention_mask.sum(dim=1) - 1
        #     last_token_feature = last_hidden_state[torch.arange(last_hidden_state.shape[0]), last_token_indices, :]
        # else:
        #     last_token_feature = last_hidden_state[:, -1, :]
        
        # 在训练时，labels 中 -100 的最后一位通常是 Prompt 的最后一个 Token
        # 我们要在这里产生“隐式思维”来引导后面非 -100 部分（轨迹）的生成
        last_hidden_state = outputs.hidden_states[-1] # [B, L, D]
        
        if labels is not None:
            # 找到每一行 labels 中最后一个 -100 的索引
            # 这里的逻辑是：计算 -100 的数量，减 1 得到 Prompt 结尾的索引
            prompt_len = (labels == -100).sum(dim=1)
            last_token_indices = prompt_len - 1
            # 限制索引不小于0
            last_token_indices = torch.clamp(last_token_indices, min=0)
            last_token_feature = last_hidden_state[torch.arange(last_hidden_state.shape[0]), last_token_indices, :]
        elif attention_mask is not None:
            last_token_indices = attention_mask.sum(dim=1) - 1
            last_token_feature = last_hidden_state[torch.arange(last_hidden_state.shape[0]), last_token_indices, :]
        else:
            last_token_feature = last_hidden_state[:, -1, :]

        # 4. 生成隐式思维链与任务预测
        # [B, D] -> [B, M * D] -> [B, M, D]
        # latent_thoughts = self.latent_proj(last_token_feature).view(-1, self.M, self.hidden_size)
        last_token_feature = F.layer_norm(
            last_token_feature,
            (self.hidden_size,)
        )
 
        # ===== Step 2: latent =====
        latent_raw = self.latent_proj(last_token_feature)
        # ==========================================================

        # 防止爆炸（BF16 必备）
        # latent_raw = torch.clamp(latent_raw, -30, 30)
        latent_thoughts = latent_raw.view(-1, self.M, self.hidden_size)

        
        
        # 5. 损失计算初始化
        loss_dict = {}
        # 基础语言损失：防止模型 Backbone 崩塌
        total_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=pixel_values.device, dtype=pixel_values.dtype)
        if outputs.loss is not None:
            loss_dict["loss_vlm"] = outputs.loss

        # 获取权重 (使用 getattr 兼容不同版本的 args 注入)
        w_hs = getattr(self, "distill_hs_weight", 1.0)
        w_verb = getattr(self, "verbalizer_weight", 0.5)
        w_traj = getattr(self, "trajectory_weight", 2.0)

        # (1) Distillation Loss: 隐状态对齐
        if teacher_hidden_states is not None:
            # loss_hs = F.mse_loss(latent_thoughts, teacher_hidden_states)
            teacher_hs_norm = F.layer_norm(
                teacher_hidden_states,
                (self.hidden_size,)
            )

          
            loss_hs = F.mse_loss(latent_thoughts, teacher_hs_norm)
            total_loss = total_loss + w_hs * loss_hs
            loss_dict["loss_hs"] = loss_hs

        # (2) Semantic Alignment: 语义锚点
        if teacher_text_ids is not None:
            # verb_logits = self.verbalizer(latent_thoughts) # [B, M, Vocab]
            # loss_verb = F.cross_entropy(verb_logits.view(-1, self.vocab_size), teacher_text_ids.view(-1))
            verb_input = F.layer_norm(latent_thoughts, (self.hidden_size,))
            verb_logits = self.verbalizer(verb_input)

            loss_verb = F.cross_entropy(
                verb_logits.view(-1, self.vocab_size),
                teacher_text_ids.view(-1),
                ignore_index=0  # 或 tokenizer.pad_token_id
            )
            total_loss = total_loss + w_verb * loss_verb
            loss_dict["loss_verb"] = loss_verb

        # # (3) Driving Task Loss: 轨迹预测
        # if gt_trajectory is not None:
        #     # 展平输入轨迹 head
        #     pred_trajectory = self.trajectory_head(latent_thoughts.view(-1, self.M * self.hidden_size)).view(-1, 6, 2)
        #     loss_traj = F.smooth_l1_loss(pred_trajectory, gt_trajectory)
        #     total_loss = total_loss + w_traj * loss_traj
        #     loss_dict["loss_traj"] = loss_traj
        # else:
        #     pred_trajectory = None


        # # if torch.rand(1).item() < 0.01:  # 避免刷屏
        # print("\n[DEBUG 5-1: loss_hs 检查]")
        # print(f"loss_hs 值: {loss_hs.item():.6f}")
        # print(f"loss_hs 是否为NaN: {torch.isnan(loss_hs).item()}")
        # print(f"loss_hs 是否为Inf: {torch.isinf(loss_hs).item()}")
        # print("\n[DEBUG 5-2: loss_verb 检查]")
        # print(f"loss_verb 值: {loss_verb.item():.6f}")
        # print(f"loss_verb 是否为NaN: {torch.isnan(loss_verb).item()}")
        # print(f"loss_verb 是否为Inf: {torch.isinf(loss_verb).item()}")
        # ========== 调试9：检查关键层梯度 ==========
        if torch.rand(1).item() < 0.25:  # 10%概率打印，避免刷屏
            print("\n[DEBUG 9: 梯度检查]")
            # 检查latent_proj的梯度（对应loss_hs）
            latent_proj_grad = []
            for name, param in self.latent_proj.named_parameters():
                if param.grad is not None:
                    latent_proj_grad.append(f"{name}: grad_mean={param.grad.mean().item():.6f}, grad_norm={param.grad.norm().item():.6f}")
                else:
                    latent_proj_grad.append(f"{name}: 无梯度")
            print(f"latent_proj 梯度: {latent_proj_grad}")
            
            # 检查verbalizer的梯度（对应loss_verb）
            verbalizer_grad = []
            for name, param in self.verbalizer.named_parameters():
                if param.grad is not None:
                    verbalizer_grad.append(f"{name}: grad_mean={param.grad.mean().item():.6f}, grad_norm={param.grad.norm().item():.6f}")
                else:
                    verbalizer_grad.append(f"{name}: 无梯度")
            print(f"verbalizer 梯度: {verbalizer_grad}")
            
            # 检查loss_verb的输入分布
            if teacher_text_ids is not None:
                print(f"teacher_text_ids 唯一值: {torch.unique(teacher_text_ids).cpu().numpy()}")
                print(f"teacher_text_ids 非忽略值数量: {torch.sum(teacher_text_ids != 0).item()}")


        #     torch.cuda.synchronize()

        #     print("\n" + "="*60)
        #     print(f"[GPU 显存占用快照] local_rank: 0")
        #     print(f"  总已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        #     print(f"  总缓存    : {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        #     # 1) 整个模型参数显存
        #     param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        #     print(f"  [模型参数] 总计: {param_bytes / 1024**3:.3f} GB")

        #     # 2) 新增层显存（你自己加的）
        #     latent_proj_bytes = sum(p.numel() * p.element_size() for p in self.latent_proj.parameters())
        #     verbalizer_bytes = sum(p.numel() * p.element_size() for p in self.verbalizer.parameters())
        #     print(f"    ├ latent_proj: {latent_proj_bytes / 1024**3:.4f} GB")
        #     print(f"    └ verbalizer : {verbalizer_bytes / 1024**3:.4f} GB")

        #     # 3) 梯度显存
        #     grad_bytes = sum(p.grad.numel() * p.grad.element_size() for p in self.parameters() if p.grad is not None)
        #     print(f"  [梯度] 总计    : {grad_bytes / 1024**3:.3f} GB")

        #     # 4) 激活显存（近似：总分配 - 参数 - 梯度）
        #     activ_bytes = torch.cuda.memory_allocated() - param_bytes - grad_bytes
        #     print(f"  [激活/中间张量]: {activ_bytes / 1024**3:.3f} GB")
        #     print("="*60)
        # # =========================================================================
        # ==========================================
        if not return_dict:
            # return (total_loss, latent_thoughts, pred_trajectory)
             return (total_loss, latent_thoughts, outputs.logits)

        return {
            "loss": total_loss,
            "loss_items": loss_dict,
            "logits": outputs.logits,
            "latent_thoughts": latent_thoughts,
            # "pred_trajectory": pred_trajectory
            "pred_trajectory": None # 轨迹现在通过 logits 生成
        }
    
def test_student_with_local_weights():
    MODEL_PATH = "/data1/chenxiwu/ReCogDrive/model/internvl3_1B" 
    DEVICE = "cuda:1"
    M = 4 
    
    print(f"正在加载配置...")
    try:
        config = InternVLChatConfig.from_pretrained(MODEL_PATH)
        # 逻辑验证不需要加载几十G权重，直接初始化随机权重的模型
        model = ImplicitCoTDriverStudent(config, M=M).to(DEVICE).to(torch.bfloat16)
        model.train()
        
        # 重要：手动指定一个 ID 并放入 input_ids 中，模拟图像占位符
        IMG_CTX_ID = 111111 
        model.img_context_token_id = IMG_CTX_ID

        # 准备数据
        B = 2
        pixel_values = torch.randn(B, 3, 448, 448).to(DEVICE).to(torch.bfloat16)
        
        # 构造含有图像占位符的 input_ids
        input_ids = torch.full((B, 256), 1, dtype=torch.long).to(DEVICE)
        # 官方代码逻辑：vit_embeds 会填充到 input_ids == img_context_token_id 的位置
        # 假设每个样本有 256 个图像 token
        input_ids[:, :256] = IMG_CTX_ID 
        
        # 模拟 image_flags (B,)
        image_flags = torch.ones((B,), dtype=torch.long).to(DEVICE)
        
        # 模拟标签
        mock_hs = torch.randn(B, M, model.hidden_size).to(DEVICE).to(torch.bfloat16)
        mock_text = torch.randint(0, model.vocab_size, (B, M)).to(DEVICE)
        mock_traj = torch.randn(B, 6, 2).to(DEVICE).to(torch.bfloat16)

        print("开始前向传播...")
        # 修正：使用正确的 autocast API
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                image_flags=image_flags, # 传入这个参数！
                teacher_hidden_states=mock_hs,
                teacher_text_ids=mock_text,
                gt_trajectory=mock_traj
            )

        print(f"测试成功！Total Loss: {outputs['loss'].item():.4f}")

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_student_with_local_weights()