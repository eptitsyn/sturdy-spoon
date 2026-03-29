"""Training pipeline для AI text detector — PyTorch Lightning."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    category=DeprecationWarning,
    module=r"pytorch_lightning\.utilities\._pytree",
)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from model import AITextDetector
from data import WordTokenizer, TextClassificationDataset, collate_fn


@dataclass
class TrainConfig:
    # Model
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    max_len: int = 512
    dropout: float = 0.1

    # Training
    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_vocab: int = 30_000
    val_split: float = 0.1
    seed: int = 42
    use_amp: bool = True
    token_dropout: float = 0.05
    label_smoothing: float = 0.02
    patience: int = 3
    grad_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    num_workers: int = -1

    # Paths
    save_dir: str = "checkpoints"


class SmoothedBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets)


def _compute_optimal_threshold(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    probs = 1 / (1 + np.exp(-logits))
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    preds = (probs >= threshold).astype(int)
    return threshold, float(f1_score(labels, preds, average="macro"))


def _build_stratify_labels(labels: list[int]) -> list[int] | None:
    if len(set(labels)) < 2:
        return None
    class_counts = {label: labels.count(label) for label in set(labels)}
    if min(class_counts.values()) < 2:
        return None
    return labels


def _split_train_val(
    texts: list[str],
    labels: list[int],
    val_split: float,
    seed: int,
) -> tuple[list[str], list[str], list[int], list[int]]:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")

    if not 0 < val_split < 1 or len(texts) < 2:
        return texts, [], labels, []

    stratify = _build_stratify_labels(labels)
    try:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=val_split,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=val_split,
            random_state=seed,
            stratify=None,
        )
    return train_texts, val_texts, train_labels, val_labels


def _calibrate_threshold(
    model: AITextDetector,
    loader: DataLoader,
    use_amp: bool,
) -> float:
    if len(loader.dataset) == 0:
        return 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    logits_list: list[float] = []
    labels_list: list[int] = []

    with torch.no_grad():
        for input_ids, targets in loader:
            input_ids = input_ids.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp and device.type != "cpu"):
                logits = model(input_ids)
            logits_list.extend(logits.cpu().tolist())
            labels_list.extend(targets.int().cpu().tolist())

    logits_arr = np.array(logits_list, dtype=np.float32)
    labels_arr = np.array(labels_list, dtype=np.int64)
    threshold, _ = _compute_optimal_threshold(logits_arr, labels_arr)
    return threshold


def _configure_torch_runtime() -> None:
    if not torch.cuda.is_available():
        return
    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True


def _resolve_num_workers(num_workers: int) -> int:
    if num_workers >= 0:
        return num_workers
    cpu_count = os.cpu_count() or 2
    return max(1, min(11, cpu_count - 1))


class AIDetectorModule(pl.LightningModule):
    """LightningModule wrapping AITextDetector."""

    def __init__(self, model: AITextDetector, cfg: TrainConfig, total_steps: int):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.total_steps = total_steps
        self.criterion = SmoothedBCEWithLogitsLoss(cfg.label_smoothing)
        self.validation_logits: list[torch.Tensor] = []
        self.validation_targets: list[torch.Tensor] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        input_ids, targets = batch
        logits = self(input_ids)
        loss = self.criterion(logits, targets)
        acc = ((logits > 0).long() == targets.long()).float().mean()
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.validation_logits.clear()
        self.validation_targets.clear()

    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch
        logits = self(input_ids)
        loss = self.criterion(logits, targets)
        self.validation_logits.append(logits.detach().cpu())
        self.validation_targets.append(targets.detach().cpu())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        if not self.validation_logits:
            return

        logits = torch.cat(self.validation_logits).numpy()
        labels = torch.cat(self.validation_targets).numpy().astype(int)
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        threshold, best_f1 = _compute_optimal_threshold(logits, labels)

        self.log("val_acc", float((preds == labels).mean()), prog_bar=True)
        self.log("val_f1_macro", f1_score(labels, preds, average="macro"), prog_bar=True)
        self.log("val_best_f1", best_f1, prog_bar=False)
        self.log("val_best_threshold", threshold, prog_bar=False)
        if len(np.unique(labels)) > 1:
            self.log("val_roc_auc", roc_auc_score(labels, probs), prog_bar=False)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        warmup_steps = min(self.cfg.warmup_steps, max(self.total_steps - 1, 0))

        def lr_lambda(current_step: int) -> float:
            if self.total_steps <= 1:
                return 1.0
            if warmup_steps > 0 and current_step < warmup_steps:
                return 0.1 + 0.9 * (current_step / max(1, warmup_steps))

            progress = (current_step - warmup_steps) / max(1, self.total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def train(
    texts: list[str],
    labels: list[int],
    config: TrainConfig | None = None,
) -> tuple[AITextDetector, WordTokenizer, float]:
    """
    Полный цикл тренировки с PyTorch Lightning.

    Args:
        texts: список текстов
        labels: 1 = AI, 0 = Human
        config: гиперпараметры

    Returns:
        (model, tokenizer, calibrated_threshold)
    """
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    _configure_torch_runtime()
    num_workers = _resolve_num_workers(cfg.num_workers)

    # ── Tokenizer & Data ─────────────────────────────────────────────────
    train_texts, val_texts, train_labels, val_labels = _split_train_val(
        texts,
        labels,
        cfg.val_split,
        cfg.seed,
    )
    tokenizer = WordTokenizer.from_texts(
        train_texts,
        max_vocab=cfg.max_vocab,
        max_len=cfg.max_len,
    )
    train_ds = TextClassificationDataset(train_texts, train_labels, tokenizer)
    val_ds = TextClassificationDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Vocab: {tokenizer.vocab_size} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ────────────────────────────────────────────────────────────
    model = AITextDetector(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        max_len=cfg.max_len,
        dropout=cfg.dropout,
        token_dropout=cfg.token_dropout,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(cfg.accumulate_grad_batches, 1)))
    total_steps = cfg.epochs * steps_per_epoch
    lit_model = AIDetectorModule(model, cfg, total_steps)

    # ── Callbacks & Trainer ──────────────────────────────────────────────
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=save_dir,
        filename="best_checkpoint",
        monitor="val_f1_macro",
        mode="max",
        save_top_k=1,
    )
    early_stopping_cb = EarlyStopping(
        monitor="val_f1_macro",
        mode="max",
        patience=cfg.patience,
    )

    use_amp = cfg.use_amp and torch.cuda.is_available()
    logger = CSVLogger(save_dir=str(save_dir), name="logs")
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        callbacks=[checkpoint_cb, early_stopping_cb],
        precision="16-mixed" if use_amp else "32-true",
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=cfg.grad_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        logger=logger,
        enable_model_summary=False,
    )

    trainer.fit(lit_model, train_loader, val_loader)

    # ── Save in inference-compatible format ──────────────────────────────
    best_ckpt = torch.load(checkpoint_cb.best_model_path, weights_only=True)
    # Lightning prefixes keys with "model." — strip it for inference.py
    state_dict = {
        k.removeprefix("model."): v
        for k, v in best_ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    torch.save(state_dict, save_dir / "best_model.pt")
    tokenizer.save(save_dir / "tokenizer.json")
    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": cfg.d_model,
        "nhead": cfg.nhead,
        "num_layers": cfg.num_layers,
        "dim_feedforward": cfg.dim_feedforward,
        "max_len": cfg.max_len,
        "dropout": cfg.dropout,
        "pad_idx": 0,
        "unk_idx": 1,
        "token_dropout": 0.0,
    }
    (save_dir / "model_config.json").write_text(
        json.dumps(model_config, indent=2),
        encoding="utf-8",
    )

    best_val_acc = checkpoint_cb.best_model_score
    print(f"\nBest val macro F1: {best_val_acc:.3f}")

    model.load_state_dict(state_dict)
    calibrated_threshold = _calibrate_threshold(model, val_loader, use_amp)
    (save_dir / "inference_config.json").write_text(
        json.dumps({"threshold": calibrated_threshold}, indent=2),
        encoding="utf-8",
    )
    print(f"Calibrated threshold: {calibrated_threshold:.3f}")
    return model, tokenizer, calibrated_threshold
