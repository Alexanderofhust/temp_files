# ICoT-Drive

## Fast-ThinkAct思路
1. 学生模型生成M个latent token，充当显示CoT中think的作用。引入Verbalizer LLM将latent token decode 为text，交叉熵损失保证reasoning capability
2. 在<answer>对应的token处，记录teacher's hidden states，训练时将student's hidden states与其做minimizing L2 distance，做Trajectory-level supervision,此处改为记录M个step，参考SIM-CoT的监督范式，做step level supervision.
3. 最后生成的answer,将其与gt做L2 loss
