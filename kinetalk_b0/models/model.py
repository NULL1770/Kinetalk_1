from __future__ import annotations

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

    def encode_factors(self, residual: torch.Tensor, mask: torch.Tensor | None, reference_audio: torch.Tensor | None = None, *, style_residual: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        emotion = self.emotion(residual, mask)
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
    ) -> dict[str, torch.Tensor]:
        """Frozen Stage-2 factors as the audio distillation target.

        The teacher provides only global emotion and intensity targets.
        """
        with torch.no_grad():
            factors = self.teacher_emotion(residual_gt, mask)
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


class Stage4Model(nn.Module):
    """Deployment path: audio B0 + audio emotion + BS style -> raw BS."""

    def __init__(self, cfg: dict[str, Any], stage1: Stage1Model, stage2: Stage2Model, stage3: Stage3Model):
        super().__init__()
        model = cfg["model"]
        self.register_buffer("architecture_version", torch.tensor(8))
        self.stage1 = stage1
        self.factor_emotion = stage2.emotion
        self.factor_style = stage2.style
        self.audio_prior = stage3.audio
        self.renderer = stage3.renderer
        self.residual_scale = stage2.residual_scale
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
        # Unfreeze only condition modulation layers. Content, motion backbone,
        # Stage1, and factor encoders remain frozen.
        for name, parameter in self.renderer.named_parameters():
            if any(key in name for key in ("modulation", "style_modulation", "local_emotion")):
                parameter.requires_grad_(True)

    def conditions(self, batch: dict[str, Any], *, training_target: bool = False) -> dict[str, torch.Tensor]:
        query = batch["query"]
        reference = batch["style_reference"]
        # A framewise flow target is valid only when the style condition comes
        # from the same query clip.  Deployment uses the external reference;
        # Stage-4 training uses query self-style and therefore does not ask a
        # cross-sentence/cross-speaker style render to match query frames.
        b0_reference_motion = query["b0_gt"] if training_target else reference["motion"]
        b0_reference_mask = query["mask"] if training_target else reference["mask"]
        # During Stage4 training use the same-sentence cross-emotion donor when
        # available. This makes audio emotion the causal source of emotion,
        # while the donor style is counterfactual.
        donor = batch.get("emotion_pair") if training_target else reference
        use_donor = training_target and donor is not None and bool(batch.get("relations", {}).get("emotion", torch.zeros(1, dtype=torch.bool)).any())
        style_source = donor if use_donor else (query if training_target else reference)
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
            donor_stage1 = self.stage1(style_source["content"], style_source["mask"])
            style_stage1 = donor_stage1 if use_donor else (reference_stage1 if not training_target else query_stage1)
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
        return conditions["b0_pred"] + residual, residual

