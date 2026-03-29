"""Subword tokenizer and dataset for AI text detector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["[PAD]", "[UNK]"]


def _require_tokenizers():
    try:
        from tokenizers import Tokenizer
        from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        raise ModuleNotFoundError(
            "Missing dependency 'tokenizers'. Install it with `pip install tokenizers`."
        ) from exc
    return Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


class BPETokenizer:
    """Byte-level BPE tokenizer trained on project texts."""

    def __init__(self, backend_tokenizer, max_len: int = 512):
        self.backend = backend_tokenizer
        self.max_len = max_len
        self.backend.enable_truncation(max_length=max_len)
        self._pad_id = self.backend.token_to_id("[PAD]")
        self._unk_id = self.backend.token_to_id("[UNK]")

        if self._pad_id is None or self._unk_id is None:
            raise ValueError("Tokenizer is missing required special tokens [PAD]/[UNK].")

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        max_vocab: int = 30_000,
        max_len: int = 512,
        min_frequency: int = 2,
    ) -> BPETokenizer:
        Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers = _require_tokenizers()

        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=max_vocab,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer, max_len=max_len)

    def encode(self, text: str) -> list[int]:
        ids = self.backend.encode(text).ids[: self.max_len]
        if not ids:
            ids = [self.unk_idx]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.backend.decode([i for i in ids if i != self.pad_idx])

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()

    @property
    def pad_idx(self) -> int:
        return int(self._pad_id)

    @property
    def unk_idx(self) -> int:
        return int(self._unk_id)

    def save(self, path: str | Path) -> None:
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.backend.save(str(save_dir / "tokenizer.json"))
        (save_dir / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "type": "bytelevel_bpe",
                    "max_len": self.max_len,
                    "special_tokens": SPECIAL_TOKENS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, max_len: int | None = None) -> BPETokenizer:
        Tokenizer, *_ = _require_tokenizers()

        load_path = Path(path)
        if load_path.is_file():
            tokenizer_path = load_path
            config_path = load_path.with_name("tokenizer_config.json")
        else:
            tokenizer_path = load_path / "tokenizer.json"
            config_path = load_path / "tokenizer_config.json"

        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))

        effective_max_len = max_len if max_len is not None else int(config.get("max_len", 512))
        backend = Tokenizer.from_file(str(tokenizer_path))
        return cls(backend, max_len=effective_max_len)


class TextClassificationDataset(Dataset):
    """
    Dataset for (text, label) pairs.
    label: 1 = AI-generated, 0 = human-written
    """

    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer: BPETokenizer):
        assert len(texts) == len(labels)
        self.encodings = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encodings[idx], self.labels[idx]


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad batch to the maximum length in the batch."""
    seqs, labels = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, torch.stack(labels)
