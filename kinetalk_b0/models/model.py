from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..utils import freeze_module
from .dit import ResidualDiT
from .encoders import (
    AudioContentEncoder,
    AudioEmotionDistributionEncoder,
    IdentityCalibrator,
    NeutralArticulationDecoder,
    ResidualEmotionEncoder,
    ResidualStyleEncoder,
    GradientReversal,
)


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


class Stage2Model(nn.Module):
    """Residual factorization into global affect and motion-only style."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        data, model = _dimensions(cfg)
        self.residual_scale = float(model.get("residual_scale", 0.25))
        num_emotions = len(data["emotion_classes"])
        num_intensities = int(data["num_intensity_levels"])
        common = {
            "motion_dim": int(data["motion_dim"]),
            "hidden_dim": int(model["hidden_dim"]),
            "heads": int(model["heads"]),
            "dropout": float(model["dropout"]),
        }
        self.emotion = ResidualEmotionEncoder(
            **common,
            emotion_dim=int(model["emotion_dim"]),
            num_emotions=num_emotions,
            num_intensities=num_intensities,
        )
        self.style_encoder = ResidualStyleEncoder(
            **common,
            style_dim=int(model["style_dim"]),
            reference_audio_dim=int(data.get("audio_emotion_dim", data.get("audio_dim", 0))),
            use_reference_audio=bool(model.get("style_use_audio_hint", False)),
        )
        self.style_emotion_probe = nn.Linear(int(model["style_dim"]), num_emotions)
        self.style_content_probe = nn.Linear(int(model["style_dim"]), int(data["content_dim"]))
        self.style_grl = GradientReversal(float(model.get("style_grl_lambda", 0.1)))
        self.register_buffer("architecture_version", torch.tensor(8))
        self.renderer = ResidualDiT(
            motion_dim=int(data["motion_dim"]),
            content_dim=int(model["content_dim"]),
            emotion_dim=int(model["emotion_dim"]),
            style_dim=int(model["style_dim"]),
            intensity_dim=int(model.get("intensity_condition_dim", 1)),
            dim=int(model["dit_dim"]),
            depth=int(model["dit_depth"]),
            heads=int(model["heads"]),
            dropout=float(model["dropout"]),
            global_dropout=float(model.get("global_condition_dropout", 0.10)),
            style_dropout=float(model.get("style_condition_dropout", 0.10)),
        )

    @property
    def style(self) -> ResidualStyleEncoder:
        """Motion-only execution-style encoder used by the renderer."""
        return self.style_encoder

    def encode_factors(self, residual: torch.Tensor, mask: torch.Tensor | None, reference_audio: torch.Tensor | None = None, *, style_residual: torch.Tensor | None = None, emotion_residual: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # Affect is estimated from the emotion-minus-neutral residual when a
        # paired neutral view is available. Style still sees the raw residual
        # so the deployment reference contract remains unchanged.
        emotion = self.emotion(residual if emotion_residual is None else emotion_residual, mask)
        emotion["style"] = self.style_encoder(residual if style_residual is None else style_residual, mask, reference_audio)
        emotion["style_emotion_probe_logits"] = self.style_emotion_probe(self.style_grl(emotion["style"]))
        emotion["style_content_probe"] = self.style_content_probe(self.style_grl(emotion["style"]))
        emotion["style_emotion_probe_detached"] = self.style_emotion_probe(emotion["style"].detach())
        emotion["style_content_probe_detached"] = self.style_content_probe(emotion["style"].detach())
        return emotion

    @staticmethod
    def intensity_condition(factors: dict[str, torch.Tensor]) -> torch.Tensor:
        return factors["intensity_value"]

    def flow_prediction(
        self,
        target_residual: torch.Tensor,
        h0: torch.Tensor,
        factors: dict[str, torch.Tensor],
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_t, time, velocity_target = self.renderer.flow_inputs(target_residual, self.residual_scale)
        prediction = self.renderer(
            x_t,
            time,
            h0,
            factors["global"],
            self.intensity_condition(factors),
            factors["style"],
            mask,
            local_emotion=factors.get("local"),
        )
        return prediction, velocity_target

    def render(
        self,
        h0: torch.Tensor,
        factors: dict[str, torch.Tensor],
        mask: torch.Tensor | None,
        *,
        steps: int,
        stochastic: bool = False,
    ) -> torch.Tensor:
        return self.renderer.decode(
            h0,
            factors["global"],
            self.intensity_condition(factors),
            factors["style"],
            mask,
            residual_scale=self.residual_scale,
            steps=steps,
            stochastic=stochastic,
            local_emotion=factors.get("local"),
        )


class Stage3Model(nn.Module):
    """Audio affect field distilled from frozen Stage-2 factors."""

    def __init__(self, cfg: dict[str, Any], stage2: Stage2Model):
        super().__init__()
        data, model = _dimensions(cfg)
        # Keep Stage-3 checkpoints self-describing, matching all other stages.
        # The loader validates this marker before Stage-4 can consume the
        # audio prior, which prevents mixing checkpoints from incompatible
        # architecture revisions.
        self.register_buffer("architecture_version", torch.tensor(8))
        self.audio = AudioEmotionDistributionEncoder(
            int(data["audio_emotion_dim"]), int(model["emotion_dim"]),
            int(model["hidden_dim"]), int(model["heads"]),
            len(data["emotion_classes"]), int(data["num_intensity_levels"]),
            float(model["dropout"]),
        )
        # These are registered so a Stage-3 checkpoint is self-describing. All
        # their parameters remain frozen.
        self.teacher_emotion = stage2.emotion
        self.renderer = stage2.renderer
        self.residual_scale = stage2.residual_scale
        # Compatibility shim for the first v2 Stage3 run, which briefly
        # serialized the Stage4 gate before its constructor was corrected.
        # It is frozen and unused; keeping the key lets that valid Stage3
        # checkpoint remain loadable without retraining Stage3.
        self.residual_gate_logit = nn.Parameter(torch.zeros(()), requires_grad=False)
        freeze_module(self.teacher_emotion)
        freeze_module(self.renderer)

    def forward(self, audio_emotion: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.audio(audio_emotion, mask)

    def teacher(
        self,
        residual_gt: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        mc_samples: int = 1,
        reference_audio: torch.Tensor | None = None,
        reference_style_audio: torch.Tensor | None = None,
        style_residual: torch.Tensor | None = None,
        neutral_residual: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Frozen Stage-2 factors as the audio distillation target.

        The teacher provides only global emotion and intensity targets.
        """
        with torch.no_grad():
            affect = residual_gt if neutral_residual is None else residual_gt - neutral_residual
            factors = self.teacher_emotion(affect, mask)
        return factors

    def render_closure(
        self,
        h0: torch.Tensor,
        audio_factors: dict[str, torch.Tensor],
        style: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        steps: int,
    ) -> torch.Tensor:
        return self.renderer.decode(
            h0,
            audio_factors["global"],
            audio_factors["intensity_value"],
            style,
            mask,
            residual_scale=self.residual_scale,
            steps=steps,
            stochastic=False,
        )


class FullFaceStyleAdapter(nn.Module):
    """Zero-initialized full-face residual adapter conditioned only on Style.

    The frozen DiT remains the authoritative content/affect generator. This
    branch can alter all motion channels continuously, but it receives no
    audio-affect code and therefore cannot rewrite or suppress that pathway.
    """

    def __init__(self, content_dim: int, style_dim: int, hidden_dim: int, motion_dim: int):
        super().__init__()
        self.content_norm = nn.LayerNorm(content_dim)
        self.content = nn.Linear(content_dim, hidden_dim)
        self.style = nn.Sequential(nn.SiLU(), nn.Linear(style_dim, hidden_dim * 2))
        self.temporal = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, motion_dim, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.temporal[-1].weight)
        nn.init.zeros_(self.temporal[-1].bias)

    def forward(
        self, content: torch.Tensor, style: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        tokens = self.content(self.content_norm(content))
        shift, scale = self.style(style).chunk(2, dim=-1)
        tokens = tokens * (1.0 + 0.5 * torch.tanh(scale).unsqueeze(1)) + shift.unsqueeze(1)
        output = self.temporal(tokens.transpose(1, 2)).transpose(1, 2)
        if mask is not None:
            output = output * mask.unsqueeze(-1).to(output.dtype)
        return output


class Stage4Model(nn.Module):
    """Deployment path: audio B0 + audio emotion + BS style -> raw BS."""

    def __init__(self, cfg: dict[str, Any], stage1: Stage1Model, stage2: Stage2Model, stage3: Stage3Model):
        super().__init__()
        model = cfg["model"]
        self.register_buffer("architecture_version", torch.tensor(12))
        self.stage1 = stage1
        self.factor_emotion = stage2.emotion
        self.factor_style = stage2.style
        self.audio_prior = stage3.audio
        self.renderer = stage3.renderer
        self.residual_scale = stage2.residual_scale
        self.logit_residual_bound = float(model.get("stage4_logit_residual_bound", 0.35))
        self.max_residual_gate = float(model.get("stage4_max_residual_gate", 0.6))
        init_gate = float(model.get("stage4_initial_residual_gate", 0.10))
        init_gate = min(max(init_gate, 1e-4), self.max_residual_gate - 1e-4)
        gate_ratio = torch.tensor(init_gate / self.max_residual_gate, dtype=torch.float32)
        self.residual_gate_logit = nn.Parameter(torch.logit(gate_ratio))
        adapter_max_gate = float(model.get("stage4_style_adapter_max_gate", 1.0))
        adapter_initial_gate = float(model.get("stage4_style_adapter_initial_gate", 0.10))
        adapter_initial_gate = min(max(adapter_initial_gate, 1e-4), adapter_max_gate - 1e-4)
        self.style_adapter_max_gate = adapter_max_gate
        self.style_adapter_gate_logit = nn.Parameter(
            torch.logit(torch.tensor(adapter_initial_gate / adapter_max_gate))
        )
        self.style_adapter = FullFaceStyleAdapter(
            int(model["content_dim"]),
            int(model["style_dim"]),
            int(model.get("stage4_style_adapter_hidden", model["hidden_dim"])),
            int(cfg["data"]["motion_dim"]),
        )
        # Stage 4 is a deployment adapter, not a second factor-learning
        # stage.  Global emotion and Stage-2 style coordinates stay frozen.
        self.global_calibrator = IdentityCalibrator(int(model["emotion_dim"]))
        self.style_calibrator = IdentityCalibrator(int(model["style_dim"]))
        freeze_module(self.stage1)
        freeze_module(self.factor_emotion)
        freeze_module(self.factor_style)
        freeze_module(self.audio_prior)
        freeze_module(self.renderer)
        # The calibrators are identity bridges kept for checkpoint/API
        # compatibility. Stage4 must optimize only DiT condition modulation
        # layers; otherwise these unrestricted adapters can absorb factor
        # corrections and hide a renderer failure.
        freeze_module(self.global_calibrator)
        freeze_module(self.style_calibrator)
        # The complete pre-trained generator stays frozen. Stage4 learns only
        # a zero-initialized parallel Style correction, preserving the base
        # audio-affect response by construction.
        self.residual_gate_logit.requires_grad_(False)
        self.style_adapter.requires_grad_(True)
        self.style_adapter_gate_logit.requires_grad_(True)

    def load_frozen_base(self, path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        """Strictly import an architecture-v9 Stage4 deployment as the base."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        state = payload.get("model", payload)
        version = state.get("architecture_version")
        if version is None or int(version) != 9:
            raise RuntimeError(f"Stage4 adapter base must be architecture v9, got {version!r}")
        target = self.state_dict()
        new_keys = {
            key for key in target
            if key.startswith("style_adapter.") or key == "style_adapter_gate_logit"
        }
        required_base = set(target) - new_keys - {"architecture_version"}
        missing = sorted(required_base - set(state))
        shape_mismatch = sorted(
            key for key in required_base & set(state) if target[key].shape != state[key].shape
        )
        unexpected = sorted(set(state) - required_base - {"architecture_version"})
        if missing or shape_mismatch or unexpected:
            raise RuntimeError(
                "Stage4 v9 base is not structurally compatible: "
                f"missing={missing[:3]}, shape_mismatch={shape_mismatch[:3]}, "
                f"unexpected={unexpected[:3]}"
            )
        incompatible = self.load_state_dict(
            {key: state[key] for key in required_base}, strict=False
        )
        expected_missing = new_keys | {"architecture_version"}
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Audited Stage4 base load failed: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        self.architecture_version.fill_(12)
        return payload

    def bounded_motion(self, b0: torch.Tensor, raw_residual: torch.Tensor) -> torch.Tensor:
        """Map residual around B0 with asymmetric bounds, guaranteeing [0,1].

        Unlike a plain ``B0 + residual``, the positive/negative room is
        data-dependent: a zero upper-face B0 channel can only move upward,
        while an already-active channel cannot cross either endpoint.
        """
        gate = self.max_residual_gate * torch.sigmoid(self.residual_gate_logit)
        normalized = torch.tanh(raw_residual / max(self.residual_scale, 1e-6))
        room = torch.where(normalized >= 0.0, 1.0 - b0, b0)
        delta = gate * self.logit_residual_bound * normalized * room
        return (b0 + delta).clamp(0.0, 1.0)

    def conditions(
        self,
        batch: dict[str, Any],
        *,
        training_target: bool = False,
        style_mode: str = "deployment",
    ) -> dict[str, torch.Tensor]:
        query = batch["query"]
        reference = batch["style_reference"]
        # A framewise flow target is valid only when the style condition comes
        # from the same query clip.  Deployment uses the external reference;
        # Stage-4 training uses query self-style and therefore does not ask a
        # cross-sentence/cross-speaker style render to match query frames.
        b0_reference_motion = query["b0_gt"] if training_target else reference["motion"]
        b0_reference_mask = query["mask"] if training_target else reference["mask"]
        # Cross-style must use the external style reference.  The emotion_pair
        # branch is same-speaker and is useful for factor invariance in Stage2,
        # but using it here cannot teach a cross-speaker Style intervention.
        if style_mode not in {"deployment", "self", "cross"}:
            raise ValueError(f"Unknown Stage4 style mode: {style_mode}")
        if training_target and style_mode == "self":
            style_source = query
        else:
            style_source = reference
        # Keep Stage-4 training on the same raw-reference residual contract as
        # deployment. Stage-2's paired neutral/emotional view constraint is
        # what removes the emotion shortcut from this input, rather than a
        # train/deploy distribution change here.
        style_motion = style_source["motion"]
        style_mask = style_source["mask"]
        style_audio = style_source["audio_emotion"]
        with torch.no_grad():
            query_stage1 = self.stage1(query["content"], query["mask"])
            reference_stage1 = self.stage1(
                reference["content"], reference["mask"],
            )
            # Stage 4 deliberately uses predicted B0 for its deployment-facing
            # target/reference. This is distinct from the Stage-2 R_gt teacher.
            style_stage1 = query_stage1 if style_source is query else reference_stage1
            style_residual = style_motion - style_stage1["b0"]
            raw_style = self.factor_style(style_residual, style_mask, style_audio)
            audio = self.audio_prior(query["audio_emotion"], query["mask"])
        global_code = self.global_calibrator(audio["global"])
        style_code = self.style_calibrator(raw_style)
        return {
            "b0_pred": query_stage1["b0"],
            "b0_canonical": query_stage1["b0_canonical"],
            "h0": query_stage1["h0"],
            "global": global_code,
            "intensity_value": audio["intensity_value"],
            "local": audio["local"],
            "style": style_code,
            "style_gain": style_code.new_ones(()),
        }

    def generate(self, query_audio: torch.Tensor, reference_audio: torch.Tensor, reference_motion: torch.Tensor, query_content: torch.Tensor, reference_content: torch.Tensor, query_mask: torch.Tensor, reference_mask: torch.Tensor, *, steps: int = 4, stochastic: bool = True) -> torch.Tensor:
        batch = {"query": {"audio_emotion": query_audio, "content": query_content, "mask": query_mask, "motion": torch.zeros(query_audio.shape[0], query_audio.shape[1], reference_motion.shape[-1], device=query_audio.device)}, "style_reference": {"audio_emotion": reference_audio, "motion": reference_motion, "content": reference_content, "mask": reference_mask}}
        conditions = self.conditions(batch)
        return self.render(batch, conditions, steps=steps, stochastic=stochastic)[0]

    def flow_prediction(self, batch: dict[str, Any], conditions: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = batch["query"]
        deployment_target = query["motion"] - conditions["b0_pred"]
        x_t, time, velocity_target = self.renderer.flow_inputs(deployment_target, self.residual_scale)
        prediction = self.renderer(
            x_t, time, conditions["h0"], conditions["global"], conditions["intensity_value"], conditions["style"], query["mask"], local_emotion=conditions.get("local")
        )
        return prediction, velocity_target, deployment_target

    def render(self, batch: dict[str, Any], conditions: dict[str, torch.Tensor], *, steps: int, stochastic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        query = batch["query"]
        residual = self.renderer.decode(
            conditions["h0"],
            conditions["global"],
            conditions["intensity_value"],
            conditions["style"],
            query["mask"],
            residual_scale=self.residual_scale,
            steps=steps,
            stochastic=stochastic,
            local_emotion=conditions.get("local"),
        )
        adapter_gate = self.style_adapter_max_gate * torch.sigmoid(self.style_adapter_gate_logit)
        style_delta = adapter_gate * self.residual_scale * torch.tanh(
            self.style_adapter(conditions["h0"], conditions["style"], query["mask"])
        )
        combined_residual = residual + style_delta
        final = self.bounded_motion(conditions["b0_pred"], combined_residual)
        return final, combined_residual

