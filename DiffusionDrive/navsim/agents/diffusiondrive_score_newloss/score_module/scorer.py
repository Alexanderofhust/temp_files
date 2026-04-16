import torch.nn as nn


class Scorer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.pred_score = nn.ModuleDict(
            {
                "no_at_fault_collisions": nn.Sequential(
                    nn.Linear(config.tf_d_model, config.tf_d_ffn),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_ffn, 1),
                ),
                "drivable_area_compliance": nn.Sequential(
                    nn.Linear(config.tf_d_model, config.tf_d_ffn),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_ffn, 1),
                ),
                "time_to_collision_within_bound": nn.Sequential(
                    nn.Linear(config.tf_d_model, config.tf_d_ffn),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_ffn, 1),
                ),
                "ego_progress": nn.Sequential(
                    nn.Linear(config.tf_d_model, config.tf_d_ffn),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_ffn, 1),
                ),
                "driving_direction_compliance": nn.Sequential(
                    nn.Linear(config.tf_d_model, config.tf_d_ffn),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_ffn, 1),
                ),
                "comfort": nn.Sequential(
                    nn.Linear(config.tf_d_model, config.tf_d_ffn),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_ffn, 1),
                ),
            }
        )

    def forward(self, proposal_feature):
        pred_logit = {}
        for name, head in self.pred_score.items():
            pred_logit[name] = head(proposal_feature).squeeze(-1)
        return pred_logit

