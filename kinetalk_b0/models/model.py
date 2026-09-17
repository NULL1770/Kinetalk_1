from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .encoders import AudioContentEncoder, NeutralArticulationDecoder


def _dimensions(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return cfg["data"], cfg["model"]


class NativeFeatureAggregator(nn.Module):
    """Reduce optional native HuBERT windows for the restore DLP path."""

    def __init__(self, dim: int, context: int = 9):
        super().__init__()
        if context < 1:
            raise ValueError("native context must be positive")
        self.context = int(context)

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            if not return_aux:
                return x
            if x.ndim < 3:
                raise ValueError("native content must have batch and time axes")
            lag = x.new_zeros(x.shape[0])
            position = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).view(1, -1).expand(x.shape[0], -1)
            weights = x.new_ones(x.shape[0], x.shape[1], 1)
            return x, lag, position, weights
        if x.shape[2] != self.context:
            raise ValueError(f"Expected native context {self.context}, got {x.shape[2]}")
        output = x[:, :, self.context // 2]
        lag = x.new_zeros(x.shape[0])
        native_position = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).view(1, -1).expand(x.shape[0], -1) * 2.0
        weights = x.new_zeros(x.shape[0], x.shape[1], self.context)
        weights[:, :, self.context // 2] = 1.0
        if return_aux:
            return output, lag, native_position, weights
        return output


class Stage1Model(nn.Module):
    """Emotional or neutral content audio -> the same neutral articulation B0."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        data, model = _dimensions(cfg)
        # Stage1 tensor shapes and computation are unchanged; keep v7
        # compatibility so the existing frozen Stage1 checkpoint remains
        # valid while Stage2-4 are retrained under the new residual protocol.
        self.register_buffer("architecture_version", torch.tensor(7))
        self.motion_dim = int(data["motion_dim"])
        self.neutral_indices = list(dict.fromkeys(int(i) for i in data["neutral_output_indices"]))
        self.art_indices = list(dict.fromkeys(int(i) for i in model.get("articulatory_indices", self.neutral_indices)))
        if not set(self.art_indices).issubset(self.neutral_indices):
            raise ValueError("articulatory_indices must be a subset of neutral_output_indices")
        self.decoder_indices = list(self.neutral_indices)
        self.native_aggregator = NativeFeatureAggregator(
            int(data["content_dim"]), int(data.get("stage1_native_context", 9))
        )
        self.content = AudioContentEncoder(
            int(data["content_dim"]), int(model["content_dim"]), int(model["hidden_dim"]), int(model["heads"]), float(model["dropout"])
        )
        self.neutral = NeutralArticulationDecoder(
            int(model["content_dim"]),
            int(data["motion_dim"]),
            int(model["hidden_dim"]),
            int(model["heads"]),
            self.decoder_indices,
            float(model["dropout"]),
            raw_content_dim=int(data["content_dim"]),
            output_activation=str(model.get("stage1_output_activation", "linear")),
        )

    def forward(
        self,
        content: torch.Tensor,
        mask: torch.Tensor | None = None,
        reference_motion: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        content, lag_frames, native_position, attention_weights = self.native_aggregator(content, return_aux=True)
        h0 = self.content(content, mask)
        canonical = self.neutral(h0, mask, content)
        output = {
            "h0": h0, "b0": canonical, "b0_canonical": canonical,
            "lag_frames": lag_frames,
            "native_position": native_position, "attention_weights": attention_weights,
            "alignment_offset": lag_frames.unsqueeze(1).expand(-1, content.shape[1]),
        }
        return output
