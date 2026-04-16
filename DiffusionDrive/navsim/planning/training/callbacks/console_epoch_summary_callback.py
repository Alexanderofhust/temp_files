from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch


class ConsoleEpochSummaryCallback(pl.Callback):
    """Prints step updates and epoch summaries in a compact console format."""

    def __init__(self, log_every_n_steps: int = 100) -> None:
        self.log_every_n_steps = log_every_n_steps
        self._last_printed_epoch: Optional[int] = None

    @staticmethod
    def _is_scalar_metric(value: Any) -> bool:
        if isinstance(value, (int, float)):
            return True
        return isinstance(value, torch.Tensor) and value.numel() == 1

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    @staticmethod
    def _normalize_metric_name(name: str, epoch_only: bool) -> Optional[str]:
        if name in {"epoch", "step", "v_num"}:
            return None
        if epoch_only:
            if name.endswith("_step"):
                return None
            if name.endswith("_epoch"):
                return name[: -len("_epoch")]
        else:
            if name.endswith("_epoch"):
                return None
            if name.endswith("_step"):
                return name[: -len("_step")]
        return name

    def _collect_metrics(self, metrics: Dict[str, Any], epoch_only: bool) -> Dict[str, float]:
        collected: Dict[str, float] = {}
        for key, value in metrics.items():
            key = str(key)
            normalized_key = self._normalize_metric_name(key, epoch_only=epoch_only)
            if normalized_key is None or not self._is_scalar_metric(value):
                continue
            collected[normalized_key] = self._to_float(value)
        return collected

    @staticmethod
    def _group_metrics(metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        grouped = {"train": {}, "val": {}, "other": {}}
        for key, value in sorted(metrics.items()):
            if key.startswith("train/"):
                grouped["train"][key] = value
            elif key.startswith("val/"):
                grouped["val"][key] = value
            else:
                grouped["other"][key] = value
        return grouped

    @staticmethod
    def _format_metrics(metrics: Dict[str, float]) -> str:
        return ", ".join(f"{key}={value:.3f}" for key, value in sorted(metrics.items()))

    @staticmethod
    def _num_batches(num_batches: Any) -> int:
        if isinstance(num_batches, (list, tuple)):
            return int(num_batches[0]) if num_batches else 0
        return int(num_batches)

    def _should_print(self, trainer: pl.Trainer) -> bool:
        return not trainer.sanity_checking and getattr(trainer, "global_rank", 0) == 0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self._should_print(trainer):
            return
        if self.log_every_n_steps <= 0 or batch_idx % self.log_every_n_steps != 0:
            return
        metrics = self._collect_metrics(trainer.progress_bar_metrics, epoch_only=False)
        print(
            f"Epoch {trainer.current_epoch} - train {batch_idx} / {self._num_batches(trainer.num_training_batches)}"
            f" - {self._format_metrics(metrics)}"
        )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self._should_print(trainer) or dataloader_idx != 0:
            return
        if self.log_every_n_steps <= 0 or batch_idx % self.log_every_n_steps != 0:
            return
        metrics = self._collect_metrics(trainer.progress_bar_metrics, epoch_only=False)
        print(
            f"Epoch {trainer.current_epoch} - val {batch_idx} / {self._num_batches(trainer.num_val_batches)}"
            f" - {self._format_metrics(metrics)}"
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._should_print(trainer):
            return
        current_epoch = int(trainer.current_epoch)
        if self._last_printed_epoch == current_epoch:
            return

        metrics = self._collect_metrics(trainer.callback_metrics, epoch_only=True)
        grouped_metrics = self._group_metrics(metrics)

        print(f"\n###########  Epoch {current_epoch} ##########")
        for group_name in ("train", "val", "other"):
            for key, value in grouped_metrics[group_name].items():
                print(f"{key},{value:.3f}")
        print("###########\n")
        self._last_printed_epoch = current_epoch
