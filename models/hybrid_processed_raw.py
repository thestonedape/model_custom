"""
Hybrid processed+raw EEG word classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridProcessedRawWordClassifier(nn.Module):
    def __init__(
        self,
        processed_dim: int = 840,
        eeg_dim: int = 105,
        et_dim: int = 4,
        metrics_dim: int = 5,
        model_dim: int = 192,
        processed_hidden_dim: int = 256,
        aux_hidden_dim: int = 64,
        num_heads: int = 6,
        num_layers: int = 3,
        ff_dim: int = 384,
        dropout: float = 0.1,
        num_classes: int = 500,
        norm_first: bool = False,
        patch_kernel_size: int = 7,
        patch_stride: int = 4,
    ):
        super().__init__()

        self.processed_encoder = nn.Sequential(
            nn.LayerNorm(processed_dim),
            nn.Linear(processed_dim, processed_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(processed_hidden_dim, model_dim),
            nn.GELU(),
        )

        self.eeg_projection = nn.Sequential(
            nn.LayerNorm(eeg_dim),
            nn.Linear(eeg_dim, model_dim),
            nn.GELU(),
        )
        self.et_projection = nn.Sequential(
            nn.LayerNorm(et_dim),
            nn.Linear(et_dim, model_dim),
            nn.GELU(),
        )
        self.eeg_temporal_stem = nn.Sequential(
            nn.Conv1d(
                in_channels=eeg_dim,
                out_channels=eeg_dim,
                kernel_size=patch_kernel_size,
                stride=patch_stride,
                padding=patch_kernel_size // 2,
            ),
            nn.GELU(),
        )
        self.et_temporal_stem = nn.Sequential(
            nn.Conv1d(
                in_channels=et_dim,
                out_channels=et_dim,
                kernel_size=patch_kernel_size,
                stride=patch_stride,
                padding=patch_kernel_size // 2,
            ),
            nn.GELU(),
        )
        self.raw_gate_projection = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.Sigmoid(),
        )
        self.temporal_refine = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim),
        )

        self.aux_projection = nn.Sequential(
            nn.LayerNorm(metrics_dim + 2),
            nn.Linear(metrics_dim + 2, aux_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(aux_hidden_dim, aux_hidden_dim),
            nn.GELU(),
        )
        self.raw_post = nn.Sequential(
            nn.LayerNorm(model_dim + aux_hidden_dim),
            nn.Linear(model_dim + aux_hidden_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.hybrid_gate = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        )

    @staticmethod
    def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(sequence.dtype)
        summed = (sequence * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return summed / denom

    @staticmethod
    def downsample_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
        mask_float = mask.to(torch.float32).unsqueeze(1)
        resized = F.interpolate(mask_float, size=target_len, mode="nearest")
        return resized.squeeze(1) > 0.5

    def encode_raw(
        self,
        eeg: torch.Tensor,
        eeg_mask: torch.Tensor,
        et: torch.Tensor,
        et_mask: torch.Tensor,
        metrics: torch.Tensor,
        mean_pupil_size: torch.Tensor,
        n_fixations: torch.Tensor,
    ) -> torch.Tensor:
        eeg_stem = self.eeg_temporal_stem(eeg.transpose(1, 2)).transpose(1, 2)
        et_stem = self.et_temporal_stem(et.transpose(1, 2)).transpose(1, 2)
        target_len = eeg_stem.shape[1]
        eeg_features = self.eeg_projection(eeg_stem)
        et_features = self.et_projection(et_stem)
        eeg_mask_ds = self.downsample_mask(eeg_mask, target_len)
        et_mask_ds = self.downsample_mask(et_mask, target_len)
        fused_mask = eeg_mask_ds & et_mask_ds

        gate = self.raw_gate_projection(torch.cat([eeg_features, et_features], dim=-1))
        fused = gate * eeg_features + (1.0 - gate) * et_features
        fused = self.temporal_refine(fused)
        encoded = self.temporal_encoder(fused, src_key_padding_mask=~fused_mask)
        pooled = self.masked_mean(encoded, fused_mask)

        aux = torch.cat(
            [
                metrics,
                mean_pupil_size.unsqueeze(-1),
                n_fixations.to(metrics.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        aux_features = self.aux_projection(aux)
        return self.raw_post(torch.cat([pooled, aux_features], dim=-1))

    def forward(
        self,
        processed_eeg: torch.Tensor,
        eeg: torch.Tensor,
        eeg_mask: torch.Tensor,
        et: torch.Tensor,
        et_mask: torch.Tensor,
        metrics: torch.Tensor,
        mean_pupil_size: torch.Tensor,
        n_fixations: torch.Tensor,
    ) -> torch.Tensor:
        processed_embed = self.processed_encoder(processed_eeg)
        raw_embed = self.encode_raw(
            eeg=eeg,
            eeg_mask=eeg_mask,
            et=et,
            et_mask=et_mask,
            metrics=metrics,
            mean_pupil_size=mean_pupil_size,
            n_fixations=n_fixations,
        )
        hybrid_gate = self.hybrid_gate(torch.cat([processed_embed, raw_embed], dim=-1))
        fused = hybrid_gate * raw_embed + (1.0 - hybrid_gate) * processed_embed
        return self.classifier(fused)
