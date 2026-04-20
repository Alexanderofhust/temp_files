from typing import Any, Dict, List, Optional, Union

import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.theia_score.modules.scheduler import WarmupCosLR
from navsim.agents.theia_score.transfuser_callback import TransfuserCallback
from navsim.agents.theia_score.transfuser_config import TransfuserConfig
from navsim.agents.theia_score.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.agents.theia_score.transfuser_loss import transfuser_loss
from navsim.agents.theia_score.transfuser_model_v2 import V2TransfuserModel as TransfuserModel
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.planning.training.callbacks.console_epoch_summary_callback import ConsoleEpochSummaryCallback


def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type_name = cfg.pop("type")
    return getattr(obj, type_name)(**cfg, **kwargs)


class TransfuserAgent(AbstractAgent):
    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()

        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._transfuser_model = TransfuserModel(config)
        self.train_metric_cache_paths: Dict[str, Path] = {}
        self.test_metric_cache_paths: Dict[str, Path] = {}
        self.get_scores = None
        self.score_use_ray = bool(config.use_score_head and config.score_use_ray)
        self.worker = None
        self.worker_map = None
        if config.use_score_head:
            self._initialize_score_supervision()
        self.init_from_pretrained()

    def _initialize_score_supervision(self) -> None:
        from navsim.agents.theia_score.score_module.compute_navsim_score import get_scores

        self.get_scores = get_scores
        score_cache_dir = Path(self._config.score_cache_dir)
        cache_path = score_cache_dir
        if not cache_path.is_dir():
            navsim_exp_root = os.getenv("NAVSIM_EXP_ROOT")
            if navsim_exp_root:
                cache_path = Path(navsim_exp_root) / self._config.score_cache_dir

        if cache_path.is_dir():
            metric_cache = MetricCacheLoader(cache_path)
            self.train_metric_cache_paths = metric_cache.metric_cache_paths
            self.test_metric_cache_paths = metric_cache.metric_cache_paths

        if self.score_use_ray:
            import ray

            if ray.is_initialized():
                print(
                    "Ray is already initialized in this process; "
                    "disabling internal score Ray worker to avoid nested Ray initialization."
                )
                self.score_use_ray = False
                self._config.score_use_ray = False

        if self.score_use_ray:
            from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
            from nuplan.planning.utils.multithreading.worker_utils import worker_map

            try:
                self.worker = RayDistributedNoTorch(threads_per_node=8)
                self.worker_map = worker_map
            except Exception as exc:
                print(
                    "Failed to initialize internal score Ray worker; "
                    f"falling back to sequential score computation. Error: {exc}"
                )
                self.score_use_ray = False
                self._config.score_use_ray = False
                self.worker = None
                self.worker_map = None

    def compute_score(self, targets: Dict[str, torch.Tensor], proposals: torch.Tensor, test: bool = False):
        metric_cache_paths = self.train_metric_cache_paths if self.training else self.test_metric_cache_paths
        if not metric_cache_paths:
            raise FileNotFoundError(
                "Could not find train metric cache paths. "
                "Set NAVSIM_EXP_ROOT or agent.config.score_cache_dir and generate train_metric_cache before score training."
            )
        if "token" not in targets:
            raise KeyError("Score supervision requires `token` in targets.")

        data_points = []
        for token, poses in zip(targets["token"], proposals.detach().cpu().numpy()):
            if token not in metric_cache_paths:
                raise KeyError(f"Missing train metric cache for token `{token}`.")
            data_points.append({"token": metric_cache_paths[token], "poses": poses, "test": test})

        if self.score_use_ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.as_tensor(
            np.stack([result[0] for result in all_res]),
            dtype=proposals.dtype,
            device=proposals.device,
        )
        final_scores = target_scores[:, :, -1]
        best_scores = final_scores.amax(dim=-1)
        return final_scores, best_scores, target_scores

    def init_from_pretrained(self):
        if self._checkpoint_path:
            if torch.cuda.is_available():
                checkpoint = torch.load(self._checkpoint_path)
            else:
                checkpoint = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))

            state_dict = checkpoint["state_dict"]
            state_dict = {k.replace("agent.", ""): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        else:
            print("No checkpoint path provided. Initializing from scratch.")

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        return self._transfuser_model(features, targets=targets)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return transfuser_loss(
            targets,
            predictions,
            self._config,
            self.compute_score if self._config.use_score_head else None,
        )

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        return self.get_coslr_optimizers()

    def get_step_lr_optimizers(self):
        optimizer = torch.optim.Adam(self._transfuser_model.parameters(), lr=self._lr, weight_decay=self._config.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._config.lr_steps, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_coslr_optimizers(self):
        optimizer_cfg = dict(
            type=self._config.optimizer_type,
            lr=self._lr,
            weight_decay=self._config.weight_decay,
            paramwise_cfg=self._config.opt_paramwise_cfg,
        )
        scheduler_cfg = dict(
            type=self._config.scheduler_type,
            milestones=self._config.lr_steps,
            gamma=0.1,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)
        scheduler_cfg = DictConfig(scheduler_cfg)

        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop("paramwise_cfg", None)

        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg["name"]]

            for k, v in self._transfuser_model.named_parameters():
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg["name"].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
        else:
            params = self._transfuser_model.parameters()

        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        if paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg["name"].items()):
                cfg = {}
                if "lr_mult" in pg_cfg:
                    cfg["lr"] = optimizer_cfg["lr"] * pg_cfg["lr_mult"]
                optimizer.add_param_group({"params": pg, **cfg})

        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=100,
            warmup_epochs=3,
        )

        if "interval" in scheduler_cfg:
            scheduler = {"scheduler": scheduler, "interval": scheduler_cfg["interval"]}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [
            ModelCheckpoint(
                save_top_k=1,
                monitor="val/score_epoch",
                filename="best-score-{epoch}-{step}",
                mode="max",
            ),
            ModelCheckpoint(save_last=True),
            ConsoleEpochSummaryCallback(log_every_n_steps=100),
            TransfuserCallback(self._config),
        ]
