"""Subword tokenizer, stylometric features, and dataset for AI text detector."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["[PAD]", "[UNK]"]
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
PUNCT_CHARS = set(".,!?;:…—-\"'`«»()[]{}")
QUOTE_CHARS = set("\"'`«»")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+|\n+")


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


class StylometricVectorizer:
    """Deterministic stylometric features for text-level AI detection."""

    BOS_CHAR = "\u0002"
    EOS_CHAR = "\u0003"

    def __init__(
        self,
        hash_dim: int = 128,
        char_ngram_range: tuple[int, int] = (3, 5),
        char_lm_order: int = 3,
    ):
        self.hash_dim = hash_dim
        self.char_ngram_range = char_ngram_range
        self.char_lm_order = char_lm_order
        self.scalar_mean = torch.zeros(len(self.scalar_feature_names()), dtype=torch.float32)
        self.scalar_std = torch.ones(len(self.scalar_feature_names()), dtype=torch.float32)
        self.lm_context_counts: Counter[str] = Counter()
        self.lm_transition_counts: Counter[tuple[str, str]] = Counter()
        self.lm_vocab: set[str] = {self.EOS_CHAR}

    @staticmethod
    def scalar_feature_names() -> list[str]:
        return [
            "log_char_len",
            "log_token_count",
            "avg_token_len",
            "std_token_len",
            "type_token_ratio",
            "repeated_token_ratio",
            "repeated_bigram_ratio",
            "uppercase_ratio",
            "digit_ratio",
            "whitespace_ratio",
            "punct_ratio",
            "exclamation_ratio",
            "question_ratio",
            "comma_ratio",
            "semicolon_ratio",
            "quote_ratio",
            "ellipsis_ratio",
            "punct_gap_mean",
            "punct_gap_std",
            "repeated_punct_ratio",
            "log_line_count",
            "nonempty_line_ratio",
            "avg_line_len",
            "std_line_len",
            "log_sentence_count",
            "avg_sentence_len",
            "std_sentence_len",
            "sentence_burstiness",
            "max_char_run_ratio",
            "repeated_line_ratio",
            "log_char_perplexity",
        ]

    @property
    def scalar_dim(self) -> int:
        return len(self.scalar_feature_names())

    @property
    def feature_dim(self) -> int:
        return self.scalar_dim + self.hash_dim

    @classmethod
    def fit(
        cls,
        texts: Sequence[str],
        hash_dim: int = 128,
        char_ngram_range: tuple[int, int] = (3, 5),
        char_lm_order: int = 3,
    ) -> StylometricVectorizer:
        vectorizer = cls(
            hash_dim=hash_dim,
            char_ngram_range=char_ngram_range,
            char_lm_order=char_lm_order,
        )
        for text in texts:
            vectorizer._update_char_language_model(text)

        if texts:
            scalar_matrix = torch.tensor(
                [vectorizer._extract_scalar_features(text) for text in texts],
                dtype=torch.float32,
            )
            vectorizer.scalar_mean = scalar_matrix.mean(dim=0)
            vectorizer.scalar_std = scalar_matrix.std(dim=0, unbiased=False).clamp_min(1e-4)
        return vectorizer

    def save(self, path: str | Path) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hash_dim": self.hash_dim,
                "char_ngram_range": self.char_ngram_range,
                "char_lm_order": self.char_lm_order,
                "scalar_mean": self.scalar_mean,
                "scalar_std": self.scalar_std,
                "lm_context_counts": dict(self.lm_context_counts),
                "lm_transition_counts": dict(self.lm_transition_counts),
                "lm_vocab": sorted(self.lm_vocab),
            },
            save_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> StylometricVectorizer:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        vectorizer = cls(
            hash_dim=int(state["hash_dim"]),
            char_ngram_range=tuple(state["char_ngram_range"]),
            char_lm_order=int(state["char_lm_order"]),
        )
        vectorizer.scalar_mean = state["scalar_mean"].float()
        vectorizer.scalar_std = state["scalar_std"].float().clamp_min(1e-4)
        vectorizer.lm_context_counts = Counter(state["lm_context_counts"])
        vectorizer.lm_transition_counts = Counter(state["lm_transition_counts"])
        vectorizer.lm_vocab = set(state["lm_vocab"])
        if not vectorizer.lm_vocab:
            vectorizer.lm_vocab = {vectorizer.EOS_CHAR}
        return vectorizer

    def transform_text(self, text: str) -> torch.Tensor:
        scalar_features = torch.tensor(self._extract_scalar_features(text), dtype=torch.float32)
        scalar_features = (scalar_features - self.scalar_mean) / self.scalar_std
        hashed_ngrams = self._extract_hashed_char_ngrams(text)
        return torch.cat([scalar_features, hashed_ngrams], dim=0)

    def transform_batch(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.feature_dim), dtype=torch.float32)
        return torch.stack([self.transform_text(text) for text in texts], dim=0)

    def _update_char_language_model(self, text: str) -> None:
        normalized = self._normalize_for_lm(text)
        if not normalized:
            return

        context_size = max(1, self.char_lm_order - 1)
        padded = (self.BOS_CHAR * context_size) + normalized + self.EOS_CHAR
        self.lm_vocab.update(padded)

        for idx in range(context_size, len(padded)):
            context = padded[idx - context_size: idx]
            next_char = padded[idx]
            self.lm_context_counts[context] += 1
            self.lm_transition_counts[(context, next_char)] += 1

    def _char_perplexity(self, text: str) -> float:
        normalized = self._normalize_for_lm(text)
        if not normalized or not self.lm_vocab:
            return 1.0

        context_size = max(1, self.char_lm_order - 1)
        padded = (self.BOS_CHAR * context_size) + normalized + self.EOS_CHAR
        vocab_size = max(1, len(self.lm_vocab))
        total_nll = 0.0
        n_steps = 0

        for idx in range(context_size, len(padded)):
            context = padded[idx - context_size: idx]
            next_char = padded[idx]
            context_count = self.lm_context_counts.get(context, 0)
            transition_count = self.lm_transition_counts.get((context, next_char), 0)
            prob = (transition_count + 1.0) / (context_count + vocab_size)
            total_nll -= math.log(prob)
            n_steps += 1

        return math.exp(total_nll / max(1, n_steps))

    def _extract_scalar_features(self, text: str) -> list[float]:
        text = text or ""
        char_len = len(text)
        tokens = WORD_RE.findall(text.lower())
        token_count = len(tokens)
        token_lengths = [len(token) for token in tokens]
        unique_tokens = len(set(tokens))

        token_counter = Counter(tokens)
        repeated_token_ratio = (
            sum(count for count in token_counter.values() if count > 1) / max(1, token_count)
        )
        bigrams = list(zip(tokens, tokens[1:]))
        bigram_counter = Counter(bigrams)
        repeated_bigram_ratio = (
            sum(count for count in bigram_counter.values() if count > 1) / max(1, len(bigrams))
        )

        line_lengths = [len(line.strip()) for line in text.splitlines()]
        nonempty_lines = [line for line in line_lengths if line > 0]
        sentence_lengths = [
            len(WORD_RE.findall(chunk))
            for chunk in SENTENCE_SPLIT_RE.split(text)
            if chunk.strip()
        ]

        punct_positions = [idx for idx, ch in enumerate(text) if ch in PUNCT_CHARS]
        punct_gaps = [
            punct_positions[idx] - punct_positions[idx - 1]
            for idx in range(1, len(punct_positions))
        ]
        punct_runs = [
            len(match.group(0))
            for match in re.finditer(r"[.,!?;:…\-—]{2,}", text)
        ]
        repeated_lines = Counter(line.strip() for line in text.splitlines() if line.strip())
        repeated_line_ratio = (
            sum(count for count in repeated_lines.values() if count > 1)
            / max(1, len(nonempty_lines))
        )

        punctuation_count = sum(1 for ch in text if ch in PUNCT_CHARS)
        uppercase_count = sum(1 for ch in text if ch.isupper())
        digit_count = sum(1 for ch in text if ch.isdigit())
        whitespace_count = sum(1 for ch in text if ch.isspace())
        max_char_run = self._max_char_run(text)
        char_perplexity = self._char_perplexity(text)

        line_count = max(1, len(line_lengths))
        sentence_count = max(1, len(sentence_lengths))

        return [
            math.log1p(char_len),
            math.log1p(token_count),
            self._safe_mean(token_lengths),
            self._safe_std(token_lengths),
            unique_tokens / max(1, token_count),
            repeated_token_ratio,
            repeated_bigram_ratio,
            uppercase_count / max(1, char_len),
            digit_count / max(1, char_len),
            whitespace_count / max(1, char_len),
            punctuation_count / max(1, char_len),
            text.count("!") / max(1, char_len),
            text.count("?") / max(1, char_len),
            text.count(",") / max(1, char_len),
            (text.count(";") + text.count(":")) / max(1, char_len),
            sum(1 for ch in text if ch in QUOTE_CHARS) / max(1, char_len),
            text.count("...") / max(1, char_len),
            self._safe_mean(punct_gaps) / max(1, char_len),
            self._safe_std(punct_gaps) / max(1, char_len),
            sum(punct_runs) / max(1, char_len),
            math.log1p(line_count),
            len(nonempty_lines) / line_count,
            self._safe_mean(nonempty_lines) / max(1, char_len),
            self._safe_std(nonempty_lines) / max(1, char_len),
            math.log1p(sentence_count),
            self._safe_mean(sentence_lengths),
            self._safe_std(sentence_lengths),
            self._burstiness(sentence_lengths),
            max_char_run / max(1, char_len),
            repeated_line_ratio,
            math.log1p(char_perplexity),
        ]

    def _extract_hashed_char_ngrams(self, text: str) -> torch.Tensor:
        features = torch.zeros(self.hash_dim, dtype=torch.float32)
        normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
        if not normalized:
            return features

        total = 0
        min_n, max_n = self.char_ngram_range
        for n in range(min_n, max_n + 1):
            if len(normalized) < n:
                continue
            for idx in range(len(normalized) - n + 1):
                ngram = normalized[idx: idx + n]
                bucket, sign = self._signed_hash(ngram)
                features[bucket] += sign
                total += 1

        if total > 0:
            features /= float(total)
            norm = torch.linalg.vector_norm(features)
            if norm > 0:
                features /= norm
        return features

    def _signed_hash(self, text: str) -> tuple[int, float]:
        hash_value = 1469598103934665603
        for byte in text.encode("utf-8", errors="ignore"):
            hash_value ^= byte
            hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        bucket = int(hash_value % self.hash_dim)
        sign = 1.0 if ((hash_value >> 63) & 1) == 0 else -1.0
        return bucket, sign

    @staticmethod
    def _normalize_for_lm(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").lower()).strip()

    @staticmethod
    def _safe_mean(values: Sequence[int | float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _safe_std(values: Sequence[int | float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return float(math.sqrt(max(variance, 0.0)))

    @staticmethod
    def _burstiness(values: Sequence[int | float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = StylometricVectorizer._safe_mean(values)
        std = StylometricVectorizer._safe_std(values)
        return float(std / max(mean, 1e-6))

    @staticmethod
    def _max_char_run(text: str) -> int:
        if not text:
            return 0
        max_run = 1
        current_run = 1
        prev_char = text[0]
        for char in text[1:]:
            if char == prev_char:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                prev_char = char
                current_run = 1
        return max_run


class TextClassificationDataset(Dataset):
    """
    Dataset for (text, features, label) tuples.
    label: 1 = AI-generated, 0 = human-written
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: BPETokenizer,
        vectorizer: StylometricVectorizer | None = None,
    ):
        assert len(texts) == len(labels)
        self.encodings = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]
        if vectorizer is not None:
            self.features = vectorizer.transform_batch(texts)
        else:
            self.features = torch.zeros((len(texts), 0), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encodings[idx], self.features[idx], self.labels[idx]


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad token sequences and stack stylometric features."""
    seqs, features, labels = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    feature_batch = torch.stack(features)
    return padded, feature_batch, torch.stack(labels)
