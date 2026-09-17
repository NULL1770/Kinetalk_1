from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, device=time.device) / max(half - 1, 1))
        embedding = torch.cat([torch.sin(time[:, None] * frequencies), torch.cos(time[:, None] * frequencies)], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return self.mlp(embedding)


class DiTBlock(nn.Module):
    """AdaLN self-attention plus temporal condition cross-attention."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.style_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.style_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 3))
        # shift/scale/gates for self attention, cross attention and MLP.
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 9))

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        global_condition: torch.Tensor,
        style_condition: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        values = self.modulation(global_condition).chunk(9, dim=-1)
        shift_self, scale_self, gate_self, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = values
        self_input = self._modulate(self.norm_self(tokens), shift_self, scale_self)
        self_output, _ = self.self_attention(
            self_input, self_input, self_input, key_padding_mask=None if mask is None else ~mask, need_weights=False
        )
        tokens = tokens + torch.tanh(gate_self).unsqueeze(1) * self_output
        style_shift, style_scale, style_gate = self.style_modulation(style_condition).chunk(3, dim=-1)
        style_tokens = self.style_norm(tokens) * (1.0 + style_scale.unsqueeze(1)) + style_shift.unsqueeze(1)
        tokens = tokens + 0.25 * torch.tanh(style_gate).unsqueeze(1) * style_tokens
        cross_input = self._modulate(self.norm_cross(tokens), shift_cross, scale_cross)
        cross_output, _ = self.cross_attention(
            cross_input, context, context, key_padding_mask=None if mask is None else ~mask, need_weights=False
        )
        tokens = tokens + torch.tanh(gate_cross).unsqueeze(1) * cross_output
        mlp_input = self._modulate(self.norm_mlp(tokens), shift_mlp, scale_mlp)
        tokens = tokens + torch.tanh(gate_mlp).unsqueeze(1) * self.mlp(mlp_input)
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        return tokens


class ResidualDiT(nn.Module):
    """Flow-matching DiT that generates normalized residuals and returns velocity."""

    def __init__(
        self,
        motion_dim: int,
        content_dim: int,
        emotion_dim: int,
        style_dim: int,
        dim: int,
        depth: int,
        heads: int,
        dropout: float,
        intensity_dim: int = 1,
        global_dropout: float = 0.1,
        style_dropout: float = 0.1,
    ):
        super().__init__()
        self.global_dropout = global_dropout
        self.style_dropout = style_dropout
        self.motion_input = nn.Linear(motion_dim, dim)
        self.context_input = nn.Sequential(nn.Linear(content_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time = TimeEmbedding(dim)
        self.emotion = nn.Linear(emotion_dim + intensity_dim, dim)
        self.local_emotion = nn.Linear(emotion_dim, dim)
        self.style = nn.Linear(style_dim, dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, dropout) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
        self.output = nn.Linear(dim, motion_dim)

    @staticmethod
    def _drop_condition(value: torch.Tensor, probability: float, training: bool) -> torch.Tensor:
        if not training or probability <= 0.0:
            return value
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        keep = (torch.rand(shape, device=value.device) >= probability).to(value.dtype)
        return value * keep

    def forward(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor,
        content: torch.Tensor,
        global_emotion: torch.Tensor,
        intensity: torch.Tensor,
        style: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        local_emotion: torch.Tensor | None = None,
        condition_dropout: bool = True,
    ) -> torch.Tensor:
        if condition_dropout:
            global_emotion = self._drop_condition(global_emotion, self.global_dropout, self.training)
            style = self._drop_condition(style, self.style_dropout, self.training)
        context = self.context_input(content)
        if local_emotion is not None:
            context = context + self.local_emotion(local_emotion)
        tokens = self.motion_input(x_t) + context
        global_condition = self.time(time) + self.emotion(torch.cat([global_emotion, intensity], dim=-1)) + self.style(style)
        for block in self.blocks:
            tokens = block(tokens, context, global_condition, self.style(style), mask)
        shift, scale = self.output_modulation(global_condition).chunk(2, dim=-1)
        tokens = self.output_norm(tokens) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        velocity = self.output(tokens)
        if mask is not None:
            velocity = velocity * mask.unsqueeze(-1).to(velocity.dtype)
        return velocity

    def flow_inputs(self, target_residual: torch.Tensor, residual_scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        target = target_residual / residual_scale
        noise = torch.randn_like(target)
        time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        interpolation = time[:, None, None]
        x_t = (1.0 - interpolation) * noise + interpolation * target
        return x_t, time, target - noise

    def decode(
        self,
        content: torch.Tensor,
        global_emotion: torch.Tensor,
        intensity: torch.Tensor,
        style: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        residual_scale: float,
        steps: int = 4,
        stochastic: bool = False,
        local_emotion: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if steps < 1:
            raise ValueError("DiT decode steps must be at least one")
        if initial_noise is not None:
            expected = (content.shape[0], content.shape[1], self.output.out_features)
            if tuple(initial_noise.shape) != expected or not torch.isfinite(initial_noise).all():
                raise ValueError(f"initial_noise must be finite with shape {expected}")
            state = initial_noise.to(device=content.device, dtype=content.dtype).clone()
        else:
            state = torch.randn(
            content.shape[0], content.shape[1], self.output.out_features, device=content.device, dtype=content.dtype
            ) if stochastic else torch.zeros(
            content.shape[0], content.shape[1], self.output.out_features, device=content.device, dtype=content.dtype
            )
        times = torch.linspace(0.0, 1.0, steps + 1, device=content.device, dtype=content.dtype)
        for index in range(steps):
            current = torch.full((content.shape[0],), times[index], device=content.device, dtype=content.dtype)
            velocity = self(state, current, content, global_emotion, intensity, style, mask, local_emotion=local_emotion, condition_dropout=False)
            state = state + (times[index + 1] - times[index]) * velocity
            if mask is not None:
                state = state * mask.unsqueeze(-1).to(state.dtype)
        return state * residual_scale
