#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.diffusiondrive_score.transfuser_agent import TransfuserAgent
from navsim.agents.diffusiondrive_score.transfuser_config import TransfuserConfig
from navsim.planning.training.dataset import CacheOnlyDataset


DEFAULT_CHECKPOINT = Path(
    "/data/shengzhenli/DiffusionDrive/diffusiondrive_score_exp/lightning_logs/version_2/checkpoints/epoch99-step16700.ckpt"
)
DEFAULT_CACHE_PATH = Path("/data/shengzhenli/DiffusionDrive/diffusiondrive_score_dataset_cache/training_cache")
DEFAULT_SCORE_CACHE_DIR = Path("/data/shengzhenli/DrivoR-main/exp/train_metric_cache")
DEFAULT_SPLIT_CONFIG = Path(
    "/data/shengzhenli/DiffusionDrive/navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
)
DEFAULT_OUTPUT = Path("/data/shengzhenli/DiffusionDrive/diffusiondrive_score_exp/val_anchor_best_stats_cli.json")


def parse_bool(value: str) -> bool:
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def move_tensor_dict(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def build_result(
    stats: Dict[str, Any],
    examples: list[Dict[str, Any]],
    elapsed_sec: float,
    finished_batches: int,
    total_batches: int,
) -> Dict[str, Any]:
    num_samples = max(int(stats["num_samples"]), 1)
    result = dict(stats)
    result.update(
        {
            "nearest_eq_best_ratio": stats["nearest_eq_best"] / num_samples,
            "pred_eq_best_ratio": stats["pred_eq_best"] / num_samples,
            "pred_eq_nearest_ratio": stats["pred_eq_nearest"] / num_samples,
            "nearest_hits_best_score_ratio": stats["nearest_hits_best_score"] / num_samples,
            "pred_hits_best_score_ratio": stats["pred_hits_best_score"] / num_samples,
            "mean_best_score": stats["best_score_sum"] / num_samples,
            "mean_pred_score": stats["pred_score_sum"] / num_samples,
            "mean_nearest_score": stats["nearest_score_sum"] / num_samples,
            "elapsed_sec": elapsed_sec,
            "finished_batches": finished_batches,
            "total_batches": total_batches,
            "examples": examples,
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统计 diffusiondrive_score 验证集上最近 GT anchor、oracle best PDM proposal、模型选中 proposal 的对应关系。"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--score-cache-dir", type=Path, default=DEFAULT_SCORE_CACHE_DIR)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--score-use-ray", type=parse_bool, default=True)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--examples-limit", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("NAVSIM_DISABLE_RAY_PROGRESS_BAR", "1")
    os.environ.setdefault("NAVSIM_EXP_ROOT", "/data/shengzhenli/DiffusionDrive/diffusiondrive_score_exp")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.cache_path.is_dir():
        raise FileNotFoundError(f"Dataset cache not found: {args.cache_path}")
    if not args.score_cache_dir.is_dir():
        raise FileNotFoundError(f"Score cache not found: {args.score_cache_dir}")
    if not args.split_config.is_file():
        raise FileNotFoundError(f"Split config not found: {args.split_config}")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    split_cfg = OmegaConf.load(args.split_config)
    val_logs = list(split_cfg.val_logs)

    config = TransfuserConfig(score_cache_dir=str(args.score_cache_dir), score_use_ray=args.score_use_ray)
    agent = TransfuserAgent(config=config, lr=args.lr, checkpoint_path=str(args.checkpoint))
    agent = agent.to(device)
    agent.eval()

    dataset = CacheOnlyDataset(
        cache_path=str(args.cache_path),
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=val_logs,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    total_batches = len(loader) if args.max_batches is None else min(len(loader), args.max_batches)
    plan_anchor = agent._transfuser_model._trajectory_head.plan_anchor.detach().to(device)

    stats: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "cache_path": str(args.cache_path),
        "score_cache_dir": str(args.score_cache_dir),
        "split_config": str(args.split_config),
        "device": str(device),
        "score_use_ray": bool(args.score_use_ray),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "num_samples": 0,
        "nearest_eq_best": 0,
        "pred_eq_best": 0,
        "pred_eq_nearest": 0,
        "nearest_hits_best_score": 0,
        "pred_hits_best_score": 0,
        "best_score_sum": 0.0,
        "pred_score_sum": 0.0,
        "nearest_score_sum": 0.0,
    }
    examples: list[Dict[str, Any]] = []
    start = time.time()

    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(tqdm(loader, total=total_batches, desc="val-anchor-best")):
            if batch_idx >= total_batches:
                break

            features = move_tensor_dict(features, device)
            targets_device = move_tensor_dict(targets, device)

            outputs = agent.forward(features)
            proposals = outputs["proposals"]
            pred_idx = outputs["pdm_score"].argmax(dim=1)

            target_traj = targets_device["trajectory"]
            dist = torch.linalg.norm(target_traj.unsqueeze(1)[..., :2] - plan_anchor.unsqueeze(0), dim=-1)
            nearest_anchor_idx = dist.mean(dim=-1).argmin(dim=-1)

            final_scores, best_scores, _ = agent.compute_score(targets, proposals)
            best_pdm_idx = final_scores.argmax(dim=1)

            batch_arange = torch.arange(proposals.shape[0], device=proposals.device)
            pred_scores = final_scores[batch_arange, pred_idx]
            nearest_scores = final_scores[batch_arange, nearest_anchor_idx]

            stats["num_samples"] += int(proposals.shape[0])
            stats["nearest_eq_best"] += int((nearest_anchor_idx == best_pdm_idx).sum().item())
            stats["pred_eq_best"] += int((pred_idx == best_pdm_idx).sum().item())
            stats["pred_eq_nearest"] += int((pred_idx == nearest_anchor_idx).sum().item())
            stats["nearest_hits_best_score"] += int(
                torch.isclose(nearest_scores, best_scores, atol=1e-6, rtol=1e-5).sum().item()
            )
            stats["pred_hits_best_score"] += int(
                torch.isclose(pred_scores, best_scores, atol=1e-6, rtol=1e-5).sum().item()
            )
            stats["best_score_sum"] += float(best_scores.sum().item())
            stats["pred_score_sum"] += float(pred_scores.sum().item())
            stats["nearest_score_sum"] += float(nearest_scores.sum().item())

            if len(examples) < args.examples_limit:
                tokens = targets["token"]
                keep = min(len(tokens), args.examples_limit - len(examples))
                for i in range(keep):
                    examples.append(
                        {
                            "token": tokens[i],
                            "nearest_anchor_idx": int(nearest_anchor_idx[i].item()),
                            "best_pdm_idx": int(best_pdm_idx[i].item()),
                            "pred_idx": int(pred_idx[i].item()),
                            "nearest_pdm_score": float(nearest_scores[i].item()),
                            "best_pdm_score": float(best_scores[i].item()),
                            "pred_pdm_score": float(pred_scores[i].item()),
                        }
                    )

            if args.save_every > 0 and (batch_idx + 1) % args.save_every == 0:
                partial = build_result(
                    stats=stats,
                    examples=examples,
                    elapsed_sec=time.time() - start,
                    finished_batches=batch_idx + 1,
                    total_batches=total_batches,
                )
                output_path.write_text(json.dumps(partial, indent=2, ensure_ascii=False))

    result = build_result(
        stats=stats,
        examples=examples,
        elapsed_sec=time.time() - start,
        finished_batches=total_batches,
        total_batches=total_batches,
    )
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
