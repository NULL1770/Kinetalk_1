"""Independent tensor metrics for neutral-identity / affect experiments.

The target is the observed expressive motion.  B0 is a frozen articulation
baseline and is evaluated against that target, never treated as expressive GT.
Call ``motion_metrics`` separately for training and held-out clips, and for
full/constant/reversed affect conditions generated with the same initial noise.
Nonzero condition response alone is not evidence of better motion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch


_UPPER = tuple(range(14))
_BROWS = tuple(range(41, 46))
_MOUTH = tuple(range(14, 41))


def _inputs(pred, target, b0, valid, channel_mask, times):
    tensors = [x.detach().to(device="cpu", dtype=torch.float64) for x in (pred, target, b0)]
    if tensors[0].ndim == 2:
        tensors = [x.unsqueeze(0) for x in tensors]
    pred, target, b0 = tensors
    if pred.ndim != 3 or target.shape != pred.shape or b0.shape != pred.shape:
        raise ValueError("pred, target and b0 must share [B,T,D] or [T,D] shape")
    batch, frames, channels = pred.shape
    valid = valid.detach().to(device="cpu", dtype=torch.bool)
    if valid.ndim == 1:
        valid = valid.unsqueeze(0)
    if valid.shape != (batch, frames):
        raise ValueError("valid must match motion batch/time axes")
    cm = channel_mask.detach().to(device="cpu", dtype=torch.bool)
    if cm.ndim == 1:
        cm = cm.unsqueeze(0).expand(batch, -1)
    if cm.shape != (batch, channels):
        raise ValueError("channel_mask must be [D] or [B,D]")
    times = times.detach().to(device="cpu", dtype=torch.float64)
    if times.ndim == 1:
        times = times.unsqueeze(0).expand(batch, -1)
    if times.shape != (batch, frames):
        raise ValueError("times must be [T] or [B,T]")
    observed = valid.unsqueeze(-1) & cm.unsqueeze(1)
    if not torch.isfinite(times[valid]).all():
        raise ValueError("Valid timestamps must be finite")
    for name, tensor in (("pred", pred), ("target", target), ("b0", b0)):
        if not torch.isfinite(tensor[observed]).all():
            raise ValueError(f"Nonfinite observed {name}")
    # Invalid placeholders must not leak NaNs into differencing or plotting.
    pred, target, b0 = [torch.where(observed, x, torch.zeros_like(x)) for x in (pred, target, b0)]
    return pred, target, b0, valid, cm, times, observed


def _mean(value, mask):
    return float(value[mask].mean()) if bool(mask.any()) else None


def _corr(x, y, mask, eps=1e-10):
    """Correlation after independent per-clip/channel temporal centering.

    Only channels with at least three paired samples and nonconstant target
    and prediction contribute.  Static baselines return None, not fake zero.
    """
    count = mask.sum(1, keepdim=True)
    clean_x = torch.where(mask, x, torch.zeros_like(x))
    clean_y = torch.where(mask, y, torch.zeros_like(y))
    dx = torch.where(mask, x - clean_x.sum(1, keepdim=True) / count.clamp_min(1), 0.0)
    dy = torch.where(mask, y - clean_y.sum(1, keepdim=True) / count.clamp_min(1), 0.0)
    xx, yy = dx.square().sum(1), dy.square().sum(1)
    usable = (count.squeeze(1) >= 3) & (xx > eps) & (yy > eps)
    if not usable.any():
        return None, 0
    coefficients = (dx * dy).sum(1) / (xx * yy).sqrt().clamp_min(eps)
    return float(coefficients[usable].mean()), int(usable.sum())


def _std_ratio(x, y, mask, eps=1e-10):
    """Ratio of summed within-clip/channel standard deviations.

    The same target-active channels are included for both numerator and
    denominator, so a static prediction produces zero rather than exclusion.
    """
    count = mask.sum(1)
    mean_x = torch.where(mask, x, 0.0).sum(1) / count.clamp_min(1)
    mean_y = torch.where(mask, y, 0.0).sum(1) / count.clamp_min(1)
    var_x = torch.where(mask, (x - mean_x[:, None]).square(), 0.0).sum(1) / count.clamp_min(1)
    var_y = torch.where(mask, (y - mean_y[:, None]).square(), 0.0).sum(1) / count.clamp_min(1)
    active = (count >= 3) & (var_y > eps)
    if not active.any():
        return None, 0
    return float(var_x[active].sqrt().sum() / var_y[active].sqrt().sum()), int(active.sum())


def _velocity(x, valid, cm, times, max_gap_s=0.12):
    dt = times[:, 1:] - times[:, :-1]
    pairs = valid[:, 1:] & valid[:, :-1] & torch.isfinite(dt) & (dt > 0) & (dt <= max_gap_s)
    mask = pairs.unsqueeze(-1) & cm.unsqueeze(1)
    safe_dt = torch.where(pairs, dt, torch.ones_like(dt))
    velocity = (x[:, 1:] - x[:, :-1]) / safe_dt.unsqueeze(-1)
    return torch.where(mask, velocity, 0.0), mask


def _lag(x, y, mask, times, max_lag_s=0.32):
    """Mean per-clip best lag; positive means prediction follows target.

    Searches sampled lags within +/- max_lag_s.  Each lag is scored with
    per-channel centered correlation.  Ties favor the smallest absolute lag.
    This descriptive best-lag score must accompany zero-lag correlation.
    """
    result_lag, result_corr = [], []
    for i in range(x.shape[0]):
        frame_valid = mask[i].any(-1)
        dt = times[i, 1:] - times[i, :-1]
        dt = dt[frame_valid[1:] & frame_valid[:-1] & (dt > 0) & (dt <= 0.12)]
        if not dt.numel():
            continue
        max_shift = min(int(max_lag_s / float(dt.median())), x.shape[1] - 3)
        best = None
        for shift in sorted(range(-max_shift, max_shift + 1), key=lambda s: (abs(s), s)):
            if shift > 0:
                a, b = slice(shift, None), slice(None, -shift)
            elif shift < 0:
                a, b = slice(None, shift), slice(-shift, None)
            else:
                a, b = slice(None), slice(None)
            common = mask[i:i+1, a] & mask[i:i+1, b]
            offset = times[i:i+1, a] - times[i:i+1, b]
            common &= torch.isfinite(offset).unsqueeze(-1) & (offset.abs() <= max_lag_s + 1e-8).unsqueeze(-1)
            corr, _ = _corr(x[i:i+1, a], y[i:i+1, b], common)
            if corr is None:
                continue
            used_frames = common.any(-1)
            lag = float(offset[used_frames].mean())
            if best is None or corr > best[0] + 1e-10:
                best = corr, lag
        if best is not None:
            result_corr.append(best[0])
            result_lag.append(best[1])
    if not result_lag:
        return None, None, 0
    return sum(result_lag) / len(result_lag), sum(result_corr) / len(result_corr), len(result_lag)


def motion_metrics(pred, target, b0, valid, channel_mask, times) -> dict:
    """Return JSON-serializable, mask-aware motion metrics in raw BS units.

    ``upper`` means ARKit indices 0:14 (eyes), as specified by this experiment;
    it does not include eyebrow channels 41:46, reported separately as ``brows``.
    Velocity uses seconds and skips
    gaps >120 ms. Pearson correlations are macro averaged over valid clip /
    channel pairs. Undefined metrics are None. No clipping is performed.
    """
    pred, target, b0, valid, cm, times, observed = _inputs(pred, target, b0, valid, channel_mask, times)
    pv, pair_mask = _velocity(pred, valid, cm, times)
    tv, _ = _velocity(target, valid, cm, times)
    bv, _ = _velocity(b0, valid, cm, times)
    metrics = {
        "clips": int(pred.shape[0]),
        "valid_frames": int(valid.sum()),
        "observed_values": int(observed.sum()),
        "masked_mse": _mean((pred - target).square(), observed),
        "b0_masked_mse": _mean((b0 - target).square(), observed),
    }
    for label, indices in (("upper", _UPPER), ("brows", _BROWS), ("mouth", _MOUTH), ("jaw17", (17,))):
        ids = [j for j in indices if j < pred.shape[-1]]
        if not ids:
            continue
        mask, vm = observed[..., ids], pair_mask[..., ids]
        metrics[f"{label}_masked_mse"] = _mean((pred[..., ids] - target[..., ids]).square(), mask)
        for name, motion, velocity in (("pred", pred, pv), ("b0", b0, bv)):
            prefix = f"{label}_{name}"
            corr, count = _corr(motion[..., ids], target[..., ids], mask)
            vcorr, vcount = _corr(velocity[..., ids], tv[..., ids], vm)
            ratio, active = _std_ratio(motion[..., ids], target[..., ids], mask)
            squared_velocity_error = _mean((velocity[..., ids] - tv[..., ids]).square(), vm)
            metrics.update({
                f"{prefix}_corr": corr,
                f"{prefix}_corr_pairs": count,
                f"{prefix}_std_ratio": ratio,
                f"{prefix}_active_pairs": active,
                f"{prefix}_velocity_rmse": None if squared_velocity_error is None else squared_velocity_error ** 0.5,
                f"{prefix}_velocity_corr": vcorr,
                f"{prefix}_velocity_corr_pairs": vcount,
            })
            if label in ("mouth", "jaw17"):
                lag, peak_corr, lag_count = _lag(motion[..., ids], target[..., ids], mask, times)
                metrics.update({f"{prefix}_lag_s": lag, f"{prefix}_best_lag_corr": peak_corr, f"{prefix}_lag_clips": lag_count})
        if label == "mouth":
            # Magnitude timing combines available mouth channels without
            # treating a change in overall expressive gain as a phase change.
            speed_mask = vm.any(-1, keepdim=True)
            gt_speed = tv[..., ids].square().sum(-1, keepdim=True).sqrt()
            for name, velocity in (("pred", pv), ("b0", bv)):
                speed = velocity[..., ids].square().sum(-1, keepdim=True).sqrt()
                corr, count = _corr(speed, gt_speed, speed_mask)
                metrics[f"mouth_{name}_speed_corr"] = corr
                metrics[f"mouth_{name}_speed_corr_clips"] = count
    return metrics


def plot_motion_curves(
    pred, target, b0, valid, times, path: str | Path, *,
    sample_index: int = 0, channels: Sequence[int] | None = None,
    title: str | None = None, channel_mask: torch.Tensor | None = None,
) -> str:
    """Save target/prediction/B0 curves for one clip; return absolute PNG path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if channel_mask is None:
        channel_mask = torch.ones(pred.shape[-1], dtype=torch.bool)
    pred, target, b0, valid, cm, times, _ = _inputs(pred, target, b0, valid, channel_mask, times)
    if not 0 <= sample_index < pred.shape[0]:
        raise ValueError("sample_index is outside the batch")
    channels = list(channels if channels is not None else (0, 6, 7, 13, 17, 41, 42, 43))
    channels = [j for j in channels if 0 <= j < pred.shape[-1] and bool(cm[sample_index, j])]
    if not channels or not valid[sample_index].any():
        raise ValueError("No observed channels/frames available to plot")
    labels = {0: "eyeBlinkLeft", 6: "eyeWideLeft", 7: "eyeBlinkRight", 13: "eyeWideRight", 17: "jawOpen", 41: "browDownLeft", 42: "browDownRight", 43: "browInnerUp"}
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, max(3, 1.8 * len(channels))), sharex=True, squeeze=False)
    x = times[sample_index].clone()
    frame_valid = valid[sample_index]
    x = x - x[frame_valid][0]
    for ax, j in zip(axes[:, 0], channels):
        for name, motion, color, style in (("Observed target", target, "black", "-"), ("Prediction", pred, "tab:red", "-"), ("B0 articulation baseline", b0, "tab:blue", "--")):
            y = motion[sample_index, :, j].clone()
            y[~frame_valid] = float("nan")
            ax.plot(x.numpy(), y.numpy(), color=color, linestyle=style, label=name, linewidth=1.2)
        ax.set_ylabel(labels.get(j, f"BS {j}"), fontsize=8)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("Time (s)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() != ".png":
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return str(output)
