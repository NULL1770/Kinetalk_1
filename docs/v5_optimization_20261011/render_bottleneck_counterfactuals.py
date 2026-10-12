"""Render frozen decoder-bottleneck counterfactuals for visual diagnosis only."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--project", type=Path, default=Path("D:/实验室项目/新实验/kinetalk_b0_residual_train"))
    p.add_argument("--blender", type=Path, default=Path("D:/3d_engine/blender-4.3.0-windows-x64/blender.exe"))
    p.add_argument("--blend", type=Path, default=Path("D:/实验室项目/新实验/arkit2.blend"))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.input, allow_pickle=False) as z:
        names = [str(x) for x in z["mode_names"]]
        motions = z["motions"].copy()
        keep = [0, 1, 2, names.index("teacher_raw_oracle_fixed_schedule"),
                names.index("anchor_mean_oracle_fixed_schedule"),
                names.index("geometry_oracle_fixed_schedule")]
        selected = motions[keep]
        selected_names = np.asarray([names[i] for i in keep])
        out_input = args.output.parent / (args.output.name + "_input.npz")
        out_input.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_input, motions=selected, mode_names=selected_names,
            times=z["times"], valid=z["valid"], channel_mask=z["channel_mask"],
            channels=z["channels"], clip_id=z["clip_id"], noise_seed=z["noise_seed"])
    renderer = args.project / "scripts/render_dynamic_rig_comparison.py"
    cmd = ["python", str(renderer), "--input", str(out_input), "--output", str(args.output),
           "--audio", str(args.audio), "--tile-size", "320", "--columns", "3", "--samples", "16"]
    subprocess.run(cmd, check=True)
    report = json.loads((args.output / "display_report.json").read_text(encoding="utf8"))
    if report.get("status") != "complete":
        raise RuntimeError(report)
    print(json.dumps({"status":"complete", "output":str(args.output), "modes":selected_names.tolist()}, indent=2))


if __name__ == "__main__":
    main()
