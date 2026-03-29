"""Инференс для AI text detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.amp import autocast

from model import AITextDetector
from data import WordTokenizer


@dataclass
class DetectionResult:
    label: str          # "AI" or "Human"
    confidence: float   # 0..1
    logit: float        # raw logit


class Detector:
    """Обёртка для инференса."""

    def __init__(self, model: AITextDetector, tokenizer: WordTokenizer, device: str = "auto"):
        self.device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path, device: str = "auto", **model_kwargs) -> Detector:
        """Загрузка из сохранённого чекпоинта."""
        ckpt = Path(checkpoint_dir)
        tokenizer = WordTokenizer.load(ckpt / "tokenizer.json")

        model = AITextDetector(vocab_size=tokenizer.vocab_size, **model_kwargs)
        model.load_state_dict(torch.load(ckpt / "best_model.pt", weights_only=True, map_location="cpu"))
        return cls(model, tokenizer, device)

    @torch.no_grad()
    def predict(self, text: str) -> DetectionResult:
        ids = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long, device=self.device)
        with autocast(self.device.type):
            logit = self.model(ids).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()
        return DetectionResult(
            label="AI" if logit > 0 else "Human",
            confidence=prob if logit > 0 else 1 - prob,
            logit=logit,
        )

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[DetectionResult]:
        from data import collate_fn
        encoded = [torch.tensor(self.tokenizer.encode(t), dtype=torch.long) for t in texts]
        from torch.nn.utils.rnn import pad_sequence
        padded = pad_sequence(encoded, batch_first=True, padding_value=0).to(self.device)

        with autocast(self.device.type):
            logits = self.model(padded)

        results = []
        for logit in logits.cpu().tolist():
            prob = torch.sigmoid(torch.tensor(logit)).item()
            results.append(DetectionResult(
                label="AI" if logit > 0 else "Human",
                confidence=prob if logit > 0 else 1 - prob,
                logit=logit,
            ))
        return results
