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
        temporal_adapter: bool = False,
    ):
        super().__init__()
        self.global_dropout = global_dropout
        self.style_dropout = style_dropout
        self.motion_input = nn.Linear(motion_dim, dim)
        self.context_input = nn.Sequential(nn.Linear(content_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time = TimeEmbedding(dim)
        self.emotion = nn.Linear(emotion_dim + intensity_dim, dim)
        # Temporal audio context. The parameter keeps the old local_emotion key
        # for strict checkpoint compatibility; callers should pass it as u_a.
        self.local_emotion = nn.Linear(emotion_dim, dim)
        self.style = nn.Linear(style_dim, dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, dropout) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
        self.output = nn.Linear(dim, motion_dim)
        self.register_buffer("coordinate_mean", torch.zeros(motion_dim), persistent=False)
        self.register_buffer("coordinate_std", torch.ones(motion_dim), persistent=False)
        self.channel_coordinates = False
        self.motion_temporal = None
        self.set_temporal_adapter(temporal_adapter)
        # Explicit output-side affect route. It reuses the existing emotion
        # and output projections so historical checkpoints remain loadable;
        # no new trainable state is introduced. The deployable default now
        # covers the complete mouth region as well as upper-face expression.
        # Legacy protected-mouth recipes still mask these channels at the
        # NeutralAffectSystem residual-support boundary.
        affect_indices = tuple(range(14, 41)) + (5, 6, 12, 13, 41, 42, 43, 44, 45)
        affect_mask = torch.zeros(motion_dim, dtype=torch.bool)
        for index in affect_indices:
            if index < motion_dim:
                affect_mask[index] = True
        self.register_buffer("affect_output_mask", affect_mask, persistent=False)

    def set_channel_coordinates(self, mean=None, std=None) -> None:
        """Use TRAIN-centered coordinates while returning physical velocities.

        x_t is in the system's existing residual_scale units. Subtracting
        t*mean preserves a zero-centered source and a centered target. The
        derivative of that moving center is added back to every velocity.
        """
        if mean is None and std is None:
            self.coordinate_mean.zero_()
            self.coordinate_std.fill_(1.)
            self.channel_coordinates = False
            return
        if mean is None or std is None:
            raise ValueError("Channel coordinates require both TRAIN mean and std")
        mean = torch.as_tensor(mean, device=self.coordinate_mean.device, dtype=self.coordinate_mean.dtype)
        std = torch.as_tensor(std, device=self.coordinate_std.device, dtype=self.coordinate_std.dtype)
        if (mean.shape != self.coordinate_mean.shape or std.shape != self.coordinate_std.shape or
                not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any()):
            raise ValueError("Channel coordinates require finite mean and positive std [motion_dim]")
        self.coordinate_mean.copy_(mean)
        self.coordinate_std.copy_(std)
        self.channel_coordinates = True

    def set_temporal_adapter(self, enabled: bool) -> None:
        """Opt-in local motion-token interaction; zero-start, no output filtering."""
        if type(enabled) is not bool:
            raise ValueError("temporal_adapter must be Boolean")
        if enabled and self.motion_temporal is None:
            # Preserve RNG state so enabling a zero-start branch cannot shift
            # matched training noise, data order or dropout draws.
            with torch.random.fork_rng(devices=[]):
                adapter = nn.Conv1d(self.motion_input.out_features,
                                    self.motion_input.out_features, 3, padding=1, bias=False)
                nn.init.zeros_(adapter.weight)
            self.motion_temporal = adapter.to(self.motion_input.weight)
        elif not enabled:
            self.motion_temporal = None

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
        temporal_condition: torch.Tensor | None = None,
        condition_dropout: bool = True,
    ) -> torch.Tensor:
        if condition_dropout:
            global_emotion = self._drop_condition(global_emotion, self.global_dropout, self.training)
            style = self._drop_condition(style, self.style_dropout, self.training)
        context = self.context_input(content)
        if temporal_condition is not None and local_emotion is not None:
            raise ValueError("Pass only one temporal condition alias")
        temporal = temporal_condition if temporal_condition is not None else local_emotion
        if temporal is not None:
            context = context + self.local_emotion(temporal)
        coordinate_input = ((x_t - time[:, None, None] * self.coordinate_mean) / self.coordinate_std
                            if self.channel_coordinates else x_t)
        tokens = self.motion_input(coordinate_input) + context
        if self.motion_temporal is not None:
            local_input = tokens if mask is None else torch.where(mask[..., None], tokens, 0.)
            local = self.motion_temporal(local_input.transpose(1, 2)).transpose(1, 2)
            if mask is not None:
                local = torch.where(mask[..., None], local, 0.)
            tokens = tokens + local
        global_condition = self.time(time) + self.emotion(torch.cat([global_emotion, intensity], dim=-1)) + self.style(style)
        for block in self.blocks:
            tokens = block(tokens, context, global_condition, self.style(style), mask)
        shift, scale = self.output_modulation(global_condition).chunk(2, dim=-1)
        tokens = self.output_norm(tokens) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        velocity = self.output(tokens)
        # The ordinary AdaLN path can learn to ignore a clip-level emotion
        # code while fitting the flow objective.  This output-side route makes
        # the requested global affect directly observable at the coefficients
        # that carry brow/eye and affective mouth expression.
        affect_tokens = self.emotion(torch.cat([global_emotion, intensity], dim=-1))
        affect_drive = self.output(affect_tokens).unsqueeze(1)
        velocity = velocity + 0.1 * affect_drive * self.affect_output_mask.view(1, 1, -1).to(velocity.dtype)
        if self.channel_coordinates:
            velocity = self.coordinate_mean + self.coordinate_std * velocity
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
        temporal_condition: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        motion_support: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if steps < 1:
            raise ValueError("DiT decode steps must be at least one")
        expected = (content.shape[0], content.shape[1], self.output.out_features)
        support = torch.ones(expected, dtype=torch.bool, device=content.device)
        if motion_support is not None:
            if (motion_support.dtype != torch.bool or motion_support.shape != (self.output.out_features,)
                    or motion_support.device != content.device or not motion_support.any()):
                raise ValueError("motion_support must be fixed nonempty Boolean [motion_dim] on the content device")
            support = support & motion_support[None, None]
        if mask is not None:
            support = support & mask[..., None]
        if initial_noise is not None:
            if tuple(initial_noise.shape) != expected:
                raise ValueError(f"initial_noise must have shape {expected}")
            state = initial_noise.to(device=content.device, dtype=content.dtype).clone()
        else:
            state = torch.randn(
            content.shape[0], content.shape[1], self.output.out_features, device=content.device, dtype=content.dtype
            ) if stochastic else torch.zeros(
            content.shape[0], content.shape[1], self.output.out_features, device=content.device, dtype=content.dtype
            )
        if not torch.isfinite(state[support]).all():
            raise ValueError(f"initial_noise must be finite on fixed support with shape {expected}")
        state = torch.where(support, state, 0.)
        times = torch.linspace(0.0, 1.0, steps + 1, device=content.device, dtype=content.dtype)
        for index in range(steps):
            current = torch.full((content.shape[0],), times[index], device=content.device, dtype=content.dtype)
            velocity = self(state, current, content, global_emotion, intensity, style, mask, local_emotion=local_emotion, temporal_condition=temporal_condition, condition_dropout=False)
            velocity = torch.where(support, velocity, 0.)
            state = torch.where(support, state + (times[index + 1] - times[index]) * velocity, 0.)
        return state * residual_scale
