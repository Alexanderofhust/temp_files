from typing import Callable, Dict, Optional
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

from .transfuser_config import TransfuserConfig
from .transfuser_features import BoundingBox2DIndex


def three_to_two_classes(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    x[x == 0.5] = 0.0
    return x


def score_loss(
    pred_logit: Dict[str, torch.Tensor],
    target_scores: torch.Tensor,
):
    comfort = pred_logit["comfort"]
    dtype = comfort.dtype

    no_at_fault_collisions = pred_logit["no_at_fault_collisions"]
    drivable_area_compliance = pred_logit["drivable_area_compliance"]
    time_to_collision_within_bound = pred_logit["time_to_collision_within_bound"]
    ego_progress = pred_logit["ego_progress"]
    driving_direction_compliance = pred_logit["driving_direction_compliance"]

    gt_no_at_fault_collisions, gt_drivable_area_compliance, gt_ego_progress, gt_time_to_collision_within_bound, gt_comfort, gt_driving_direction_compliance, _ = torch.split(
        target_scores, 1, dim=-1
    )
    gt_no_at_fault_collisions = gt_no_at_fault_collisions.squeeze(-1)
    gt_drivable_area_compliance = gt_drivable_area_compliance.squeeze(-1)
    gt_ego_progress = gt_ego_progress.squeeze(-1)
    gt_time_to_collision_within_bound = gt_time_to_collision_within_bound.squeeze(-1)
    gt_driving_direction_compliance = gt_driving_direction_compliance.squeeze(-1)
    gt_comfort = gt_comfort.squeeze(-1)

    da_loss = F.binary_cross_entropy_with_logits(drivable_area_compliance, gt_drivable_area_compliance.to(dtype))

    mask_valid_ttc = (gt_time_to_collision_within_bound != 2.0).float()
    ttc_loss = (
        F.binary_cross_entropy_with_logits(
            time_to_collision_within_bound,
            gt_time_to_collision_within_bound.to(dtype),
            mask_valid_ttc,
            reduction="sum",
        )
        / mask_valid_ttc.sum().clamp(min=1.0)
    )

    noc_loss = F.binary_cross_entropy_with_logits(
        no_at_fault_collisions,
        three_to_two_classes(gt_no_at_fault_collisions.to(dtype)),
    )
    progress_loss = F.binary_cross_entropy_with_logits(ego_progress, gt_ego_progress.to(dtype))
    ddc_loss = F.binary_cross_entropy_with_logits(
        driving_direction_compliance,
        three_to_two_classes(gt_driving_direction_compliance.to(dtype)),
    )
    comfort_loss = F.binary_cross_entropy_with_logits(comfort, gt_comfort.to(dtype))

    return {
        "da_loss": da_loss,
        "ttc_loss": ttc_loss,
        "noc_loss": noc_loss,
        "progress_loss": progress_loss,
        "ddc_loss": ddc_loss,
        "comfort_loss": comfort_loss,
    }


def proposal_cls_pdm_loss(
    pred_logits: torch.Tensor,
    final_scores: torch.Tensor,
    config: TransfuserConfig,
) -> torch.Tensor:
    if config.trajectory_cls_use_soft_target:
        soft_target = torch.softmax(final_scores.detach() / config.trajectory_cls_tau, dim=1)
        log_prob = F.log_softmax(pred_logits, dim=1)
        return -(soft_target * log_prob).sum(dim=1).mean()

    hard_target = final_scores.detach().argmax(dim=1)
    return F.cross_entropy(pred_logits, hard_target)


def transfuser_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: TransfuserConfig,
    scoring_function: Optional[Callable] = None,
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """
    # import ipdb; ipdb.set_trace()
    if "trajectory_reg_loss" in predictions:
        trajectory_reg_loss = predictions["trajectory_reg_loss"]
    elif "trajectory_loss" in predictions:
        trajectory_reg_loss = predictions["trajectory_loss"]
    else:
        trajectory_reg_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])
    agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)
    bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
    if "diffusion_loss" in predictions:
        diffusion_loss = predictions["diffusion_loss"]
    else:
        diffusion_loss = 0

    trajectory_cls_pdm_loss_0 = trajectory_cls_pdm_loss_1 = None
    trajectory_score = best_score = None
    score_terms = {}
    final_score_loss = trajectory_reg_loss.new_tensor(0.0)
    trajectory_cls_pdm_loss = trajectory_reg_loss.new_tensor(0.0)

    if scoring_function is not None and "pred_logit" in predictions and "proposals" in predictions:
        final_scores, best_scores, target_scores = scoring_function(targets, predictions["proposals"], test=False)
        score_terms = score_loss(predictions["pred_logit"], target_scores)
        final_score_loss = (
            score_terms["da_loss"]
            + score_terms["ttc_loss"]
            + score_terms["noc_loss"]
            + score_terms["progress_loss"]
            + score_terms["ddc_loss"]
            + score_terms["comfort_loss"]
        )
        if (
            config.trajectory_cls_pdm_weight > 0
            and "mode_scores_stage0" in predictions
            and "mode_scores_stage1" in predictions
        ):
            trajectory_cls_pdm_loss_0 = proposal_cls_pdm_loss(
                predictions["mode_scores_stage0"],
                final_scores,
                config,
            )
            trajectory_cls_pdm_loss_1 = proposal_cls_pdm_loss(
                predictions["mode_scores_stage1"],
                final_scores,
                config,
            )
            trajectory_cls_pdm_loss = config.trajectory_cls_pdm_weight * (
                trajectory_cls_pdm_loss_0 + trajectory_cls_pdm_loss_1
            )
        top_proposals = torch.argmax(predictions["pdm_score"].detach(), dim=1)
        trajectory_score = final_scores[torch.arange(len(final_scores), device=final_scores.device), top_proposals].mean()
        best_score = best_scores.mean()

    trajectory_loss = trajectory_reg_loss + trajectory_cls_pdm_loss

    loss = (
        config.trajectory_weight * trajectory_loss
        + config.diff_loss_weight * diffusion_loss
        + config.final_score_weight * final_score_loss
        + config.agent_class_weight * agent_class_loss
        + config.agent_box_weight * agent_box_loss
        + config.bev_semantic_weight * bev_semantic_loss
    )
    loss_dict = {
        "loss": loss,
        "trajectory_loss": config.trajectory_weight * trajectory_loss,
        "trajectory_reg_loss": trajectory_reg_loss,
        "trajectory_cls_pdm_loss": trajectory_cls_pdm_loss,
        "diffusion_loss": config.diff_loss_weight * diffusion_loss,
        "final_score_loss": final_score_loss,
        "agent_class_loss": config.agent_class_weight * agent_class_loss,
        "agent_box_loss": config.agent_box_weight * agent_box_loss,
        "bev_semantic_loss": config.bev_semantic_weight * bev_semantic_loss,
        "score": trajectory_score,
        "best_score": best_score,
    }
    loss_dict.update(score_terms)
    return loss_dict


def _agent_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: detection loss
    """

    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    if config.latent:
        rad_to_ego = torch.arctan2(
            gt_states[..., BoundingBox2DIndex.Y],
            gt_states[..., BoundingBox2DIndex.X],
        )

        in_latent_rad_thresh = torch.logical_and(
            -config.latent_rad_thresh <= rad_to_ego,
            rad_to_ego <= config.latent_rad_thresh,
        )
        gt_valid = torch.logical_and(in_latent_rad_thresh, gt_valid)

    # save constants
    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate cross-entropy cost for cost matrix.
    :param gt_valid: tensor of binary ground-truth labels
    :param pred_logits: tensor of predicted logits of neural net
    :return: bce cost matrix as tensor
    """

    # NOTE: numerically stable BCE with logits
    # https://github.com/pytorch/pytorch/blob/c64e006fc399d528bb812ae589789d0365f3daf4/aten/src/ATen/native/Loss.cpp#L214
    gt_valid_expanded = gt_valid[:, :, None].detach().float()  # (b, n, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, n)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term  # (b, n, n)
    ce_cost = ce_cost.permute(0, 2, 1)

    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """
    Function to calculate L1 cost for cost matrix.
    :param gt_states: tensor of ground-truth bounding boxes
    :param pred_states: tensor of predicted bounding boxes
    :param gt_valid: mask of binary ground-truth labels
    :return: l1 cost matrix as tensor
    """

    gt_states_expanded = gt_states[:, :, None, :2].detach()  # (b, n, 1, 2)
    pred_states_expanded = pred_states[:, None, :, :2].detach()  # (b, 1, n, 2)
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    """
    Helper function to align indices after matching
    :param indices: matched indices
    :return: permuted indices
    """
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
