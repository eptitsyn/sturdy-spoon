"""Transformer encoder для бинарной классификации текста (AI vs Human)."""

import math
import torch
import torch.nn as nn


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
        unk_idx: int = 1,
        token_dropout: float = 0.05,
        stylometric_dim: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.token_dropout = token_dropout
        self.stylometric_dim = stylometric_dim

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
        transformer_kwargs = {"num_layers": num_layers}
        # Nested tensors are incompatible with norm_first=True in current torch,
        # so disable that path explicitly to avoid the startup warning.
        try:
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                enable_nested_tensor=False,
                **transformer_kwargs,
            )
        except TypeError:
            self.transformer = nn.TransformerEncoder(encoder_layer, **transformer_kwargs)
        self.layer_norm = nn.LayerNorm(d_model)

        # CLS-only pooling is usually weak when training from scratch, so we
        # fuse CLS, masked mean, and masked max pooled features.
        pooled_dim = d_model * 3
        if stylometric_dim > 0:
            self.stylometric_projector = nn.Sequential(
                nn.Linear(stylometric_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            pooled_dim += d_model
        else:
            self.stylometric_projector = None
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, d_model),
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

    def _apply_token_dropout(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.training or self.token_dropout <= 0:
            return input_ids

        drop_mask = torch.rand_like(input_ids, dtype=torch.float32) < self.token_dropout
        drop_mask &= input_ids != self.pad_idx
        if not torch.any(drop_mask):
            return input_ids

        dropped = input_ids.clone()
        dropped[drop_mask] = self.unk_idx
        return dropped

    def forward(
        self,
        input_ids: torch.Tensor,
        stylometric_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) — token ids

        Returns:
            logits: (batch,) — >0 → AI, <0 → Human
        """
        B = input_ids.size(0)
        input_ids = self._apply_token_dropout(input_ids)

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

        # Multi-pooling improves signal capture for style classification tasks.
        cls_repr = x[:, 0]  # (B, D)
        token_repr = x[:, 1:]  # (B, S, D)
        token_mask = ~padding_mask[:, 1:]  # (B, S)
        token_mask_f = token_mask.unsqueeze(-1).to(token_repr.dtype)

        denom = token_mask_f.sum(dim=1).clamp_min(1.0)
        mean_repr = (token_repr * token_mask_f).sum(dim=1) / denom

        masked_tokens = token_repr.masked_fill(~token_mask.unsqueeze(-1), float("-inf"))
        max_repr = masked_tokens.max(dim=1).values
        max_repr = torch.where(torch.isfinite(max_repr), max_repr, torch.zeros_like(max_repr))

        pooled = torch.cat([cls_repr, mean_repr, max_repr], dim=-1)
        if self.stylometric_projector is not None:
            if stylometric_features is None:
                stylometric_features = torch.zeros(
                    B,
                    self.stylometric_dim,
                    device=pooled.device,
                    dtype=pooled.dtype,
                )
            if stylometric_features.ndim == 1:
                stylometric_features = stylometric_features.unsqueeze(0)
            if stylometric_features.size(-1) != self.stylometric_dim:
                raise ValueError(
                    f"Expected stylometric features with dim={self.stylometric_dim}, "
                    f"got {stylometric_features.size(-1)}"
                )
            style_repr = self.stylometric_projector(
                stylometric_features.to(device=pooled.device, dtype=pooled.dtype)
            )
            pooled = torch.cat([pooled, style_repr], dim=-1)
        logits = self.head(pooled).squeeze(-1)  # (B,)
        return logits
