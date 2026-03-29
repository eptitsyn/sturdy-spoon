"""Training pipeline для AI text detector — PyTorch Lightning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

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

    # Paths
    save_dir: str = "checkpoints"


class AIDetectorModule(pl.LightningModule):
    """LightningModule wrapping AITextDetector."""

    def __init__(self, model: AITextDetector, cfg: TrainConfig, total_steps: int):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.total_steps = total_steps
        self.criterion = nn.BCEWithLogitsLoss()

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

    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch
        logits = self(input_ids)
        loss = self.criterion(logits, targets)
        acc = ((logits > 0).long() == targets.long()).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def train(
    texts: list[str],
    labels: list[int],
    config: TrainConfig | None = None,
) -> tuple[AITextDetector, WordTokenizer]:
    """
    Полный цикл тренировки с PyTorch Lightning.

    Args:
        texts: список текстов
        labels: 1 = AI, 0 = Human
        config: гиперпараметры

    Returns:
        (model, tokenizer)
    """
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)

    # ── Tokenizer & Data ─────────────────────────────────────────────────
    tokenizer = WordTokenizer.from_texts(texts, max_vocab=cfg.max_vocab, max_len=cfg.max_len)
    dataset = TextClassificationDataset(texts, labels, tokenizer)

    val_size = int(len(dataset) * cfg.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=9
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, collate_fn=collate_fn, num_workers=9
    )

    print(f"Vocab: {tokenizer.vocab_size} | Train: {train_size} | Val: {val_size}")

    # ── Model ────────────────────────────────────────────────────────────
    model = AITextDetector(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        max_len=cfg.max_len,
        dropout=cfg.dropout,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    total_steps = cfg.epochs * len(train_loader)
    lit_model = AIDetectorModule(model, cfg, total_steps)

    # ── Callbacks & Trainer ──────────────────────────────────────────────
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=save_dir,
        filename="best_checkpoint",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    use_amp = cfg.use_amp and torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        callbacks=[checkpoint_cb],
        precision="16-mixed" if use_amp else "32-true",
        enable_progress_bar=True,
        log_every_n_steps=10,
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

    best_val_acc = checkpoint_cb.best_model_score
    print(f"\nBest val accuracy: {best_val_acc:.3f}")

    model.load_state_dict(state_dict)
    return model, tokenizer
