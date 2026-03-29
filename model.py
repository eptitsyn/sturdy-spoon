"""Transformer encoder для бинарной классификации текста (AI vs Human)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Синусоидальное позиционное кодирование."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class AITextDetector(nn.Module):
    """
    Transformer Encoder → [CLS]-pooling → binary classifier.

    Архитектура:
        Embedding + PosEnc → N × TransformerEncoderLayer → CLS token → MLP head → logit
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len + 1, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN — стабильнее тренируется
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """True там, где padding — формат для PyTorch TransformerEncoder."""
        # prepend False for [CLS]
        cls_mask = torch.zeros(input_ids.size(0), 1, dtype=torch.bool, device=input_ids.device)
        pad_mask = input_ids == self.pad_idx  # (B, S)
        return torch.cat([cls_mask, pad_mask], dim=1)  # (B, 1+S)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) — token ids

        Returns:
            logits: (batch,) — >0 → AI, <0 → Human
        """
        B = input_ids.size(0)

        # Token embeddings + scale
        x = self.embedding(input_ids) * math.sqrt(self.d_model)  # (B, S, D)

        # Prepend [CLS]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+S, D)

        x = self.pos_encoder(x)

        # Padding mask
        padding_mask = self._make_padding_mask(input_ids)

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # (B, 1+S, D)
        x = self.layer_norm(x)

        # [CLS] token representation → classification
        cls_repr = x[:, 0]  # (B, D)
        logits = self.head(cls_repr).squeeze(-1)  # (B,)
        return logits
