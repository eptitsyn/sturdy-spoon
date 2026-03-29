"""Training pipeline для AI text detector."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

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


def train(
    texts: list[str],
    labels: list[int],
    config: TrainConfig | None = None,
) -> tuple[AITextDetector, WordTokenizer]:
    """
    Полный цикл тренировки.

    Args:
        texts: список текстов
        labels: 1 = AI, 0 = Human
        config: гиперпараметры

    Returns:
        (model, tokenizer)
    """
    cfg = config or TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    print(f"Device: {device}")

    # ── Tokenizer & Data ─────────────────────────────────────────────────
    tokenizer = WordTokenizer.from_texts(texts, max_vocab=cfg.max_vocab, max_len=cfg.max_len)
    dataset = TextClassificationDataset(texts, labels, tokenizer)

    val_size = int(len(dataset) * cfg.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, collate_fn=collate_fn, num_workers=0
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
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(train_loader))
    scaler = GradScaler(enabled=cfg.use_amp and device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()

    # ── Training Loop ────────────────────────────────────────────────────
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.perf_counter()

        for input_ids, targets in train_loader:
            input_ids, targets = input_ids.to(device), targets.to(device)

            with autocast(device.type, enabled=cfg.use_amp):
                logits = model(input_ids)
                loss = criterion(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * targets.size(0)
            preds = (logits > 0).long()
            correct += (preds == targets.long()).sum().item()
            total += targets.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, cfg.use_amp)
        elapsed = time.perf_counter() - t0

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            tokenizer.save(save_dir / "tokenizer.json")
            print(f"  ✓ Saved best model (val_acc={val_acc:.3f})")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    model.load_state_dict(torch.load(save_dir / "best_model.pt", weights_only=True))
    return model, tokenizer


@torch.no_grad()
def evaluate(
    model: AITextDetector,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for input_ids, targets in loader:
        input_ids, targets = input_ids.to(device), targets.to(device)
        with autocast(device.type, enabled=use_amp):
            logits = model(input_ids)
            loss = criterion(logits, targets)
        total_loss += loss.item() * targets.size(0)
        preds = (logits > 0).long()
        correct += (preds == targets.long()).sum().item()
        total += targets.size(0)
    return total_loss / total, correct / total
