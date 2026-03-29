"""Инференс для AI text detector."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch.amp import autocast

from model import AITextDetector
from data import BPETokenizer


@dataclass
class DetectionResult:
    label: str          # "AI" or "Human"
    confidence: float   # 0..1
    logit: float        # raw logit


class Detector:
    """Обёртка для инференса."""

    def __init__(
        self,
        model: AITextDetector,
        tokenizer: BPETokenizer,
        device: str = "auto",
        threshold: float = 0.5,
    ):
        self.device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.threshold = threshold

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path, device: str = "auto", **model_kwargs) -> Detector:
        """Загрузка из сохранённого чекпоинта."""
        ckpt = Path(checkpoint_dir)
        tokenizer_path = ckpt / "tokenizer"
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                "Tokenizer directory not found in checkpoint. "
                "Retrain the model with the new BPE tokenizer."
            )
        tokenizer = BPETokenizer.load(tokenizer_path)
        saved_model_config = {}
        model_config_path = ckpt / "model_config.json"
        if model_config_path.exists():
            saved_model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        saved_model_config.update(model_kwargs)
        saved_model_config.setdefault("vocab_size", tokenizer.vocab_size)

        threshold = 0.5
        inference_config_path = ckpt / "inference_config.json"
        if inference_config_path.exists():
            inference_config = json.loads(inference_config_path.read_text(encoding="utf-8"))
            threshold = float(inference_config.get("threshold", 0.5))

        model = AITextDetector(**saved_model_config)
        model.load_state_dict(torch.load(ckpt / "best_model.pt", weights_only=True, map_location="cpu"))
        return cls(model, tokenizer, device, threshold=threshold)

    @torch.no_grad()
    def predict(self, text: str) -> DetectionResult:
        ids = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long, device=self.device)
        with autocast(self.device.type):
            logit = self.model(ids).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()
        return DetectionResult(
            label="AI" if prob >= self.threshold else "Human",
            confidence=prob if prob >= self.threshold else 1 - prob,
            logit=logit,
        )

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[DetectionResult]:
        encoded = [torch.tensor(self.tokenizer.encode(t), dtype=torch.long) for t in texts]
        from torch.nn.utils.rnn import pad_sequence
        padded = pad_sequence(encoded, batch_first=True, padding_value=0).to(self.device)

        with autocast(self.device.type):
            logits = self.model(padded)

        results = []
        for logit in logits.cpu().tolist():
            prob = torch.sigmoid(torch.tensor(logit)).item()
            is_ai = prob >= self.threshold
            results.append(DetectionResult(
                label="AI" if is_ai else "Human",
                confidence=prob if is_ai else 1 - prob,
                logit=logit,
            ))
        return results
