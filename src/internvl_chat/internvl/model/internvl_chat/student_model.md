学生模型架构说明：基于步级知识蒸馏的隐式思维链驾驶模型
1. 模型概述
学生模型旨在通过**隐式思维链（Implicit Chain-of-Thought）**技术，在不显式输出自然语言文本的前提下，利用 M个连续的 Latent Thought Tokens 承载复杂的驾驶推理逻辑。模型继承了教师模型（InternVL3）的感知能力，并通步级（Step-level）特征对齐，将显式的逻辑链压缩至潜空间中，最终实现端到端的轨迹预测。
2. 总体架构设计
学生模型架构由三个核心组件构成：多模态感知编码器、隐式逻辑推理引擎（LLM Backbone） 以及 任务导向型解码头（Task Heads）。
2.1 多模态感知编码器 (Perception Encoder)
视觉模块：沿用教师模型的 InternViT-300M。为了保持特征的一致性，在蒸馏阶段可选择冻结或采用轻量级微调。它负责将环视图像及 BEV 增强特征编码为视觉Token序列。多模态对齐层：通过线性投影层将视觉 Token 与导航指令、自车状态等文本 Token 映射至统一的隐空间(Hidden Dimension).
2.2 隐式逻辑推理引擎 (Implicit Reasoning Backbone)
基座模型：基于 Qwen2.5 架构。与教师模型不同，学生模型在自回归生成过程中不生成单词 Token，而是生成M个特殊的 Latent Thought Tokens.
推理机制：参考 Impromptu 与 Latent CoT 思路，模型在接收到环境输入后，首先自回归地预测M个潜空间向量。这些向量作为“思维载体”，通过多层注意力机制深度融合上下文信息，模拟教师模型中的“场景描述 → 对象分析 → 逻辑推导”过程。
2.3 任务导向型解码头 (Task Heads)
Verbalizer LLM (推理文本解码器)：一个轻量级的线性层或小型 Transformer Head，仅在训练阶段启用。它负责将每个T解码回自然语言，受教师模型显式文本的交叉熵损失 监督。
Trajectory Predictor (轨迹生成器)：聚合 M个 Latent Tokens 的特征，通过多层感知机（MLP）或带有时间嵌入的解码器，输出未来 3 秒内的 6 个 
(x,y) 轨迹点。
3. 核心技术方案：Step-level 知识蒸馏对齐
参考 Sim-CoT (2509.20317) 的步级监督范式，学生模型通过以下路径实现推理能力的内化：
3.1 步级隐状态对齐 (Step-level Hidden State Alignment)
不同于传统的仅对齐输出端的蒸馏，本项目要求学生模型的第 i个 Latent Token T_i 的 Hidden State，必须与教师模型在推理显式 CoT 第 个推理步骤时的 Hidden State 保持一致。逻辑：若教师模型推理“前方有行人”，其第 k步的特征会捕获行人特征。学生模型的T_k通过 L2_hs强制逼近该特征，从而确保隐式推理序列在逻辑节点上与显式推理严密对齐。
3.2 语义锚定与 Verbalizer (Semantic Anchoring)
参考 LatentVLA (2602.01166) 的思路，为了防止隐式 Token 在训练过程中漂移为无意义的噪声，通过 Verbalizer 模块进行语义锚定。
作用：交叉熵损失强制要求 T_i在潜空间中必须包含足以还原出显式推理文本的信息量。这保证了即使在没有文本输出的推理阶段，Latent Tokens 内部依然维持着清晰的语义逻辑结构。
4. 推理流程 (Inference Flow)
在实际部署阶段，Verbalizer 模块被移除，推理流程如下：
感知输入：获取环视图像、导航信息。
思维演化：LLM Backbone 内部生成 M个隐式推理步（Latent Tokens），无需解码为文本，极大提升了推理效率。
轨迹决策：轨迹头基于 M个逻辑步的融合特征，直接输出安全的驾驶轨迹。
5. 参考文献思路借鉴
arXiv:2601.09708 (Impromptu)：借鉴其对多模态 Latent CoT 的建模方式，通过预设数量的 Thought Tokens 替代文本序列。
arXiv:2602.01166 (LatentVLA)：借鉴其将物理约束动作映射为潜空间 Token 的方法，确保隐式推理与物理执行（轨迹预测）的强关联。
arXiv:2509.20317 (Sim-CoT)：借鉴其步级（Step-level）对齐损失设计，解决隐式训练中的逻辑坍塌问题。
架构示意图（逻辑描述）
code
Text
[图像/指令输入] 
      ↓
[InternViT + Qwen2.5 Embedding]
      ↓
[Qwen2.5 Layers] → 生成 [T]1, [T]2, ..., [T]M (Latent Thought Tokens)
      ↓                          ↓
      ↓                  [Distill: L2-hs ↔ Teacher Hidden States]
      ↓                          ↓
      ↓                  [Verbalizer Head] → 解码显式文本 (CE-text Loss)
      ↓
[Trajectory Head] → 输出轨迹 (L2-ans Loss)