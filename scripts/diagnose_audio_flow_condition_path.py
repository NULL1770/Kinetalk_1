"""Read-only fit-only diagnosis of the frozen audio/DiT condition path.

Restores the original uniform epoch8 source via the unchanged source loader.
That loader validates its existing internal405 reference; diagnostic scores use
only 96 metadata-selected fit clips. No development curves or external/test
artifacts are used. Source development curves are byte-hashed by the loader,
not deserialized. No weight, control gain, or alignment is fitted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_teacher_schedule_probe import GROUPS
from scripts.audit_multiseed_stochasticity import center
from scripts.train_audio_conditioned_flow_probe import load_source, source_paths
from scripts.train_formal_predictable_projection import canonical_hash, save_json
from scripts.train_predictable_renderer import (
    audio_activity_gate, audio_features, batch_to_device, observed,
    projected_affect, reverse_controls, sha, state_hash,
)
from scripts.train_projection_schedule_ablation import read_allowlist

SEEDS = (42, 123, 2026)
TIMES = (0., .1, .5, .9)
MODES = ("full", "zero", "reverse", "oracle")
SELECTION_SEED = 20260924
RECONSTRUCTION_RTOL = 2e-4
RECONSTRUCTION_ATOL = 2e-5


def select_fit_metadata(query, fit_ids, *, count=96):
    """Select lowest SHA256 from metadata only, never values/outcomes."""
    if list(query["clip_id"]) != list(fit_ids) or len(set(fit_ids)) != len(fit_ids):
        raise ValueError("Fit metadata/allowlist order differs")
    if len(fit_ids) < count or count < 1:
        raise ValueError("Insufficient fit metadata")
    rows = []
    for index, clip in enumerate(fit_ids):
        metadata = {"clip_id": str(clip), "sentence_id": str(query["sentence_id"][index]),
                    "speaker_id": int(query["speaker_id"][index]), "emotion_id": int(query["emotion_id"][index])}
        digest = canonical_hash({"seed": SELECTION_SEED, "metadata": metadata})
        rows.append({"index": index, **metadata, "selection_sha256": digest})
    rows.sort(key=lambda row: (row["selection_sha256"], row["clip_id"]))
    return rows[:count]


def reconstruct_attention(module, query, key, value, *, key_padding_mask=None):
    """Explicit batch-first MHA math; diagnostic copies only, no parameter edits."""
    if (not module.batch_first or not module._qkv_same_embed_dim or module.bias_k is not None
            or module.bias_v is not None or module.add_zero_attn or module.training):
        raise ValueError("Require eval batch-first equal-dimension plain MHA")
    if query.ndim != 3 or key.shape != value.shape:
        raise ValueError("Unexpected MHA input shapes")
    dim, heads = module.embed_dim, module.num_heads
    bias = module.in_proj_bias
    q, k, v = [F.linear(x, module.in_proj_weight[i * dim:(i + 1) * dim],
                       None if bias is None else bias[i * dim:(i + 1) * dim])
               for i, x in enumerate((query, key, value))]
    b, target_t, _ = q.shape
    source_t = k.shape[1]
    q = q.reshape(b, target_t, heads, dim // heads).transpose(1, 2)
    k = k.reshape(b, source_t, heads, dim // heads).transpose(1, 2)
    v = v.reshape(b, source_t, heads, dim // heads).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / (dim // heads) ** .5
    if key_padding_mask is not None:
        if key_padding_mask.dtype != torch.bool or key_padding_mask.shape != (b, source_t):
            raise ValueError("Expected Boolean MHA key padding mask")
        if key_padding_mask.all(1).any():
            raise ValueError("No valid attention keys")
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], -torch.inf)
    weights = scores.softmax(-1)
    combined = (weights @ v).transpose(1, 2).reshape(b, target_t, dim)
    return F.linear(combined, module.out_proj.weight, module.out_proj.bias), weights


class ConditionCapture:
    """Temporary observational hooks removed on all exits."""
    def __init__(self, renderer):
        self.renderer = renderer
        self.handles = []
        self.gates = {}
        self.attention = None
        self.output = None
        self.reconstruction_max_abs_error = None

    def __enter__(self):
        for index, block in enumerate(self.renderer.blocks):
            def capture_gate(module, inputs, output, index=index):
                self.gates[index] = output.chunk(9, -1)[5].tanh().detach()
            self.handles.append(block.modulation.register_forward_hook(capture_gate))
        self.handles.append(self.renderer.blocks[-1].cross_attention.register_forward_hook(self._attention_hook, with_kwargs=True))
        return self

    def _attention_hook(self, module, args, kwargs, output):
        if kwargs.get("attn_mask") is not None or kwargs.get("is_causal", False):
            raise ValueError("Unexpected attention mask/causal path")
        rebuilt, weights = reconstruct_attention(module, *args[:3], key_padding_mask=kwargs.get("key_padding_mask"))
        actual = output[0]
        padding = kwargs.get("key_padding_mask")
        valid = torch.ones(actual.shape[:2], dtype=torch.bool, device=actual.device) if padding is None else ~padding
        # Query and key clocks are identical in ResidualDiT; only observed query
        # rows are relevant to the generated output and numerical comparison.
        torch.testing.assert_close(rebuilt[valid], actual[valid], rtol=RECONSTRUCTION_RTOL, atol=RECONSTRUCTION_ATOL)
        self.reconstruction_max_abs_error = (rebuilt[valid] - actual[valid]).abs().max().detach()
        self.attention, self.output = weights.detach(), actual.detach()

    def take(self):
        if self.attention is None or len(self.gates) != len(self.renderer.blocks):
            raise RuntimeError("Not all condition hooks ran")
        value = {"attention": self.attention, "output": self.output, "gates": dict(self.gates),
                 "reconstruction_max_abs_error": self.reconstruction_max_abs_error}
        self.gates.clear(); self.attention = None; self.output = None
        return value

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def per_clip_rms(values, mask):
    if values.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("RMS mask must match values")
    if not torch.isfinite(values[mask]).all():
        raise ValueError("Nonfinite observed diagnostic")
    count = mask.flatten(1).sum(1)
    if (count == 0).any():
        raise ValueError("No observed diagnostic values")
    clean = torch.where(mask, values.double(), 0.)
    return (clean.flatten(1).square().sum(1) / count).sqrt()


def attention_entropy(weights, valid):
    """Mean across heads/observed queries, normalized by log(valid key count)."""
    if weights.ndim != 4 or weights.shape[0] != len(valid) or weights.shape[2:] != (valid.shape[1], valid.shape[1]):
        raise ValueError("Attention and observed clock shapes differ")
    entropy = -(weights.double() * weights.double().clamp_min(1e-300).log()).sum(-1)
    query_mask = valid[:, None, :].expand_as(entropy)
    raw = torch.where(query_mask, entropy, 0.).sum((1, 2)) / query_mask.sum((1, 2))
    denom = valid.sum(1).double().log()
    normalized = torch.where(denom > 0, raw / denom.clamp_min(1e-12), torch.zeros_like(raw))
    return raw, normalized


def condition_responses(left, right, valid, channel_mask):
    """Same x_t/time: isolate condition changes, without a lag/gain search."""
    amask = (valid[:, None, :, None] & valid[:, None, None, :]).expand_as(left["attention"])
    omask = valid[..., None].expand_as(left["output"])
    last = max(left["gates"])
    lg, rg = left["gates"][last][:, None], right["gates"][last][:, None]
    values = {"attention_weight_rms_change": per_clip_rms(left["attention"] - right["attention"], amask),
              "attention_output_rms_change": per_clip_rms(left["output"] - right["output"], omask),
              "gated_output_rms_change": per_clip_rms(left["output"] * lg - right["output"] * rg, omask)}
    for group, cc in GROUPS.items():
        mask = valid[..., None] & channel_mask[:, None, cc]
        values["velocity_rms_change/" + group] = per_clip_rms(left["velocity"][..., cc] - right["velocity"][..., cc], mask)
    values["attention_output_centered_rms_change"] = per_clip_rms(center(left["output"] - right["output"], omask), omask)
    values["gated_output_centered_rms_change"] = per_clip_rms(center(left["output"] * lg - right["output"] * rg, omask), omask)
    for group, cc in GROUPS.items():
        mask = valid[..., None] & channel_mask[:, None, cc]
        values["velocity_centered_rms_change/" + group] = per_clip_rms(center(left["velocity"][..., cc] - right["velocity"][..., cc], mask), mask)
    return values


def diagnostic_flow_state(system, batch, noise, time):
    """Match actual flow inputs exactly; channel masks restrict scores only."""
    q = batch["q"]
    valid = q["valid"][..., None]
    target = torch.where(valid, q["motion"] - batch["base"]["b0"] - batch["identity"]["baseline"][:, None], 0.) / system.residual_scale
    return (1 - time[:, None, None]) * noise + time[:, None, None] * target, target - noise


@torch.no_grad()
def run_conditions(system, batch, conditions, x_t, time, capture, *, velocity_target=None):
    q, base, identity = batch["q"], batch["base"], batch["identity"]
    valid, channel_mask = q["valid"], q["channel_mask"]
    context = system.renderer.context_input(base["h0"])
    feature_mask = valid[..., None].expand_as(context)
    all_values, records = {}, {mode: {} for mode in MODES}
    for mode in MODES:
        affect = conditions[mode]
        velocity = system.renderer(x_t, time, base["h0"], affect["global"], affect["intensity_value"],
            identity["code"], valid, local_emotion=affect["local"], condition_dropout=False)
        value = capture.take(); value["velocity"] = velocity
        all_values[mode] = value
        local_embedding = system.renderer.local_emotion(affect["local"])
        local_dynamic = F.linear(affect["local"], system.renderer.local_emotion.weight, None)
        rms = {"content_embedding_rms": per_clip_rms(context, feature_mask),
               "local_embedding_with_bias_rms": per_clip_rms(local_embedding, feature_mask),
               "local_bias_free_contribution_rms": per_clip_rms(local_dynamic, feature_mask),
               "attention_output_rms": per_clip_rms(value["output"], feature_mask),
               "content_embedding_centered_rms": per_clip_rms(center(context, feature_mask), feature_mask),
               "local_embedding_centered_rms": per_clip_rms(center(local_embedding, feature_mask), feature_mask),
               "local_temporally_centered_rms": per_clip_rms(center(local_dynamic, feature_mask), feature_mask),
               "attention_output_centered_rms": per_clip_rms(center(value["output"], feature_mask), feature_mask),
               "audio_activity_gate": audio_activity_gate(affect)}
        for key in ("controls_pre_gate_rms", "controls_post_gate_rms"):
            if key in affect:
                rms[key] = affect[key]
        entropy, normalized = attention_entropy(value["attention"], valid)
        rms.update(attention_entropy=entropy, normalized_attention_entropy=normalized)
        last = max(value["gates"])
        rms["gated_attention_output_rms"] = per_clip_rms(value["output"] * value["gates"][last][:, None], feature_mask)
        rms["gated_attention_output_centered_rms"] = per_clip_rms(center(value["output"] * value["gates"][last][:, None], feature_mask), feature_mask)
        rms["local_bias_free_to_content_rms"] = rms["local_bias_free_contribution_rms"] / rms["content_embedding_rms"].clamp_min(1e-12)
        rms["local_temporally_centered_to_content_centered_rms"] = rms["local_temporally_centered_rms"] / rms["content_embedding_centered_rms"].clamp_min(1e-12)
        for block, gate in value["gates"].items():
            rms[f"block_{block}/gate_cross_rms"] = gate.double().square().mean(-1).sqrt()
            rms[f"block_{block}/gate_cross_abs_mean"] = gate.double().abs().mean(-1)
            rms[f"block_{block}/gate_abs_below_001_fraction"] = (gate.abs() < .01).double().mean(-1)
        for group, cc in GROUPS.items():
            mask = valid[..., None] & channel_mask[:, None, cc]
            rms["velocity_rms/" + group] = per_clip_rms(velocity[..., cc], mask)
            rms["velocity_centered_rms/" + group] = per_clip_rms(center(velocity[..., cc], mask), mask)
            if velocity_target is not None:
                rms["flow_velocity_error_rms/" + group] = per_clip_rms(velocity[..., cc] - velocity_target[..., cc], mask)
        records[mode] = {key: val.cpu().tolist() for key, val in rms.items()}
        records[mode]["mha_reconstruction_max_abs_error"] = [float(value["reconstruction_max_abs_error"])] * len(valid)
    differences = {left + "_minus_" + right: {key: value.cpu().tolist() for key, value in
        condition_responses(all_values[left], all_values[right], valid, channel_mask).items()}
        for left, right in (("full", "zero"), ("oracle", "zero"), ("full", "reverse"), ("oracle", "full"))}
    return records, differences, all_values["full"]["velocity"]


def collect(rows, metadata, records, differences, *, phase, seed, time, step=None):
    for i, item in enumerate(metadata):
        rows.append({"clip_id": item["clip_id"], "fit_index": item["index"], "emotion_id": item["emotion_id"],
            "speaker_id": item["speaker_id"], "phase": phase, "seed": seed, "time": time, "step": step,
            "modes": {mode: {key: val[i] for key, val in values.items()} for mode, values in records.items()},
            "contrasts": {name: {key: val[i] for key, val in values.items()} for name, values in differences.items()}})


def summarize(rows):
    groups = {}
    for row in rows:
        populations = ["all", "neutral" if row["emotion_id"] == 0 else "nonneutral"]
        for population in populations:
            key = (row["phase"], row["time"], population)
            output = groups.setdefault(key, {"observations": 0, "sums": {}, "max_mha_reconstruction_error": 0.})
            output["observations"] += 1
            for section in ("modes", "contrasts"):
                for name, metrics in row[section].items():
                    for metric, value in metrics.items():
                        output["sums"].setdefault(section + "/" + name + "/" + metric, 0.)
                        output["sums"][section + "/" + name + "/" + metric] += value
                        if metric == "mha_reconstruction_max_abs_error":
                            output["max_mha_reconstruction_error"] = max(output["max_mha_reconstruction_error"], value)
    return [{"phase": key[0], "time": key[1], "population": key[2], "observations_clip_times_seed": row["observations"],
             "mean_of_per_clip_values": {name: value / row["observations"] for name, value in row["sums"].items()},
             "max_mha_reconstruction_error": row["max_mha_reconstruction_error"]}
            for key, row in groups.items()]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--rollout", action="store_true", help="Also probe all conditions at the same state of a frozen audio-full 12-step rollout")
    args = p.parse_args()
    if args.output.exists() or args.batch_size < 1:
        raise ValueError("Fresh output and positive batch size required")
    torch.set_num_threads(4)
    loaded = load_source(args.source_run, args.device)
    system, head = loaded["system"], loaded["head"]
    if any(module.training for module in system.modules()) or any(parameter.requires_grad for parameter in system.parameters()):
        raise ValueError("Read-only diagnostic requires frozen eval model")
    split, train = loaded["cache"]["splits"]["train"], loaded["bundle"]["bundles"]["internal"]
    source_recipe = loaded["source_recipe"]
    fit_ids = read_allowlist(source_recipe["args"]["fit_ids"])
    if len(fit_ids) != 2315 or list(train["clip_id"]) != fit_ids:
        raise ValueError("Require exact original2315fit set")
    metadata = select_fit_metadata(split["q"], fit_ids)
    selected = torch.tensor([row["index"] for row in metadata], dtype=torch.long)
    model_before, head_before = state_hash(system.state_dict()), state_hash(head.state_dict())
    source_files = source_paths() + [Path(__file__), Path(__file__).with_name("audit_teacher_schedule_probe.py"),
        Path(__file__).with_name("audit_multiseed_stochasticity.py"),
        Path(__file__).parents[1] / "tests/test_diagnose_audio_flow_condition_path.py"]
    source_hashes = {str(path.resolve()): sha(path) for path in source_files}
    features, weight = audio_features(train)[selected], train["weight"][selected].float()
    noise_shape = (len(selected), *split["q"]["motion"].shape[1:])
    noises = {seed: torch.randn(noise_shape, generator=torch.Generator().manual_seed(seed)) for seed in SEEDS}
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    try:
        with ConditionCapture(system.renderer) as capture:
            for start in range(0, len(selected), args.batch_size):
                local_ids = torch.arange(start, min(start + args.batch_size, len(selected)))
                ids = selected[local_ids]
                batch = batch_to_device(split, ids, args.device)
                w = weight[local_ids].to(args.device)
                controls = head(features[local_ids].to(args.device), w)
                oracle = head.teacher(train["motion_bins"][ids].float().to(args.device), w)
                mode_controls = {mode: (oracle if mode == "oracle" else reverse_controls(controls, w) if mode == "reverse"
                    else torch.zeros_like(controls) if mode == "zero" else controls) for mode in MODES}
                gate = audio_activity_gate(batch["affect"])
                conditions = {}
                for mode, z in mode_controls.items():
                    affect = projected_affect(system, batch, z, w, zero=mode == "zero", audio_gate=True)
                    count = (w.sum(1) * z.shape[-1]).clamp_min(1)
                    clean_z = torch.where((w > 0)[..., None], z, 0.).double()
                    affect["controls_pre_gate_rms"] = ((clean_z.square() * w[..., None]).sum((1, 2)) / count).sqrt()
                    affect["controls_post_gate_rms"] = affect["controls_pre_gate_rms"] * gate
                    conditions[mode] = affect
                q = batch["q"]
                batch_metadata = [metadata[int(i)] for i in local_ids]
                for seed in SEEDS:
                    noise = noises[seed][local_ids].to(args.device)
                    for flow_time in TIMES:
                        time = torch.full((len(ids),), flow_time, device=args.device)
                        x_t, velocity_target = diagnostic_flow_state(system, batch, noise, time)
                        rec, delta, _ = run_conditions(system, batch, conditions, x_t, time, capture, velocity_target=velocity_target)
                        collect(rows, batch_metadata, rec, delta, phase="fixed_noised_target", seed=seed, time=flow_time)
                    if args.rollout:
                        state = noise.clone()
                        rollout_times = torch.linspace(0., 1., 13, device=args.device, dtype=noise.dtype)
                        for step in range(12):
                            value = float(rollout_times[step])
                            time = torch.full((len(ids),), rollout_times[step], device=args.device, dtype=noise.dtype)
                            rec, delta, full_velocity = run_conditions(system, batch, conditions, state, time, capture)
                            collect(rows, batch_metadata, rec, delta, phase="shared_state_audio_rollout", seed=seed, time=value, step=step)
                            # Advance only full condition. Other modes are probes at
                            # this same state, not independent counterfactual paths.
                            state = (state + (rollout_times[step + 1] - rollout_times[step]) * full_velocity) * q["valid"][..., None]
                print(json.dumps({"fit_clips_done": start + len(ids), "total": len(selected)}), flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
    model_after, head_after = state_hash(system.state_dict()), state_hash(head.state_dict())
    if model_before != model_after or head_before != head_after:
        raise RuntimeError("Read-only diagnostic modified model/head state")
    report = {"schema": "audio_flow_condition_path_diagnostic_v1", "args": {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_sha256": source_hashes, "input_sha256": loaded["input_sha256"],
        "source_recipe_sha256": loaded["source_recipe_sha256"], "source_adapter_sha256": loaded["source_adapter_sha256"],
        "source_provenance_sha256": loaded["source_provenance_sha256"], "model_before_sha256": model_before,
        "model_after_sha256": model_after, "head_before_sha256": head_before, "head_after_sha256": head_after,
        "model_unchanged": True, "head_unchanged": True, "selection_seed": SELECTION_SEED,
        "selection_rule": "Lowest canonical SHA256(seed, clip/sentence/speaker/emotion metadata) of original2315 fit; no outcome stratification",
        "selected_metadata": metadata, "selection_sha256": canonical_hash(metadata), "noise_seeds": SEEDS,
        "noise_sha256": {str(seed): state_hash({"noise": noise}) for seed, noise in noises.items()},
        "flow_times": TIMES, "modes": MODES, "last_block": len(system.renderer.blocks) - 1,
        "local_rms_definition": "local_bias_free excludes Linear bias; centered metrics additionally subtract each observed per-clip time mean. With-bias local embedding is reported separately.",
        "flow_state_definition": "Exact NeutralAffectSystem.flow valid-frame mask for x_t; observed(channel) mask only for metric scoring.",
        "position_path": "Stage1 h0 uses TemporalBackbone TCN then SinusoidalPosition then Transformer, norm and output linear. DiT has no additional frame PE; Q/K receive mixed h0/local context. native_position metadata is not directly consumed.",
        "mha_reconstruction_rtol": RECONSTRUCTION_RTOL, "mha_reconstruction_atol": RECONSTRUCTION_ATOL,
        "tf32_during_diagnostic": False, "summary": summarize(rows), "per_clip": rows,
        "training_performed": False, "gain_or_alignment_fitted": False, "default_replaced": False,
        "source_loader_validates_existing_internal405": True, "diagnostic_dev_targets_used": False,
        "dev_curves_byte_hashed_by_loader": True, "dev_curves_deserialized": False, "outer280_loaded": False, "new_identity439_loaded": False, "test_loaded": False,
        "scope": "Same-state condition-path sensitivity on selected fit examples, not heldout quality/calibration. Native velocity response magnitude is not beneficial alignment. Gate/allocation correlation is diagnostic, not a causal proof of a bottleneck."}
    save_json(args.output, report)
    print(json.dumps({"complete": True, "fit_clips": len(selected), "model_unchanged": True, "rows": len(rows), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
