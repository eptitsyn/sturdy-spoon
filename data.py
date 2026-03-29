"""Токенизатор и Dataset для AI text detector."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# ─── Tokenizer ────────────────────────────────────────────────────────────────

SPECIAL_TOKENS = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}


@dataclass
class WordTokenizer:
    """
    Простой word-level токенизатор с фиксированным словарём.
    Для продакшена замени на BPE / SentencePiece / HF tokenizer.
    """

    word2idx: dict[str, int] = field(default_factory=lambda: dict(SPECIAL_TOKENS))
    idx2word: dict[int, str] = field(default_factory=dict)
    max_len: int = 512

    def __post_init__(self):
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    # ── Build vocab ──────────────────────────────────────────────────────
    @classmethod
    def from_texts(cls, texts: Sequence[str], max_vocab: int = 30_000, max_len: int = 512) -> WordTokenizer:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(cls._tokenize(text))

        word2idx = dict(SPECIAL_TOKENS)
        for word, _ in counter.most_common(max_vocab - len(SPECIAL_TOKENS)):
            word2idx[word] = len(word2idx)

        tok = cls(word2idx=word2idx, max_len=max_len)
        tok.idx2word = {v: k for k, v in word2idx.items()}
        return tok

    # ── Encode / Decode ──────────────────────────────────────────────────
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def encode(self, text: str) -> list[int]:
        tokens = self._tokenize(text)[: self.max_len]
        unk = self.word2idx["[UNK]"]
        return [self.word2idx.get(t, unk) for t in tokens]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.idx2word.get(i, "[UNK]") for i in ids if i != 0)

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    # ── Persistence ──────────────────────────────────────────────────────
    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.word2idx, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, max_len: int = 512) -> WordTokenizer:
        word2idx = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(word2idx=word2idx, max_len=max_len)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TextClassificationDataset(Dataset):
    """
    Dataset для пар (текст, label).
    label: 1 = AI-generated, 0 = human-written
    """

    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer: WordTokenizer):
        assert len(texts) == len(labels)
        self.encodings = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encodings[idx], self.labels[idx]


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Паддинг батча до максимальной длины в батче."""
    seqs, labels = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, torch.stack(labels)
