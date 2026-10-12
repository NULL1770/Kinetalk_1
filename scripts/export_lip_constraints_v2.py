"""Export provisional lip observations from the immutable fixed ARKit rig.

Normal Python launches Blender; Blender workers never save the source .blend.
Pairs are selected from independently named inner/middle skin-weight regions
before seeing any fitted motion, model output or test split. Only core lip
vertices are included; this does not imply a physiological collision surface.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

SCHEMA = "kinetalk_lip_intervals_v2.1"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rig", type=Path, default=Path("final_experiment/rig/fixed_arkit_rig_20260923.npz"))
    p.add_argument("--output", type=Path, default=Path("final_experiment/rig/lip_constraints_v2_20261010.npz"))
    p.add_argument("--blend", type=Path)
    p.add_argument("--blender", type=Path, default=Path("D:/3d_engine/blender-4.3.0-windows-x64/blender.exe"))
    p.add_argument("--object", default="body")
    p.add_argument("--max-pairs", type=int, default=12)
    p.add_argument("--min-group-weight", type=float, default=.25)
    p.add_argument("--max-xy-distance-mm", type=float, default=2.5)
    p.add_argument("--central-width-fraction", type=float, default=.6)
    p.add_argument("--skip-audit", action="store_true")
    return p.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None)


def export_worker(args, bpy):
    rig_path, output = args.rig.resolve(), args.output.resolve()
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    meta = json.loads(rig_path.with_suffix(".json").read_text(encoding="utf8"))
    blend = args.blend.resolve() if args.blend else Path(meta["source_blend"])
    if Path(bpy.data.filepath).resolve() != blend.resolve():
        bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False, use_scripts=False)
    source_sha = hashlib.sha256(blend.read_bytes()).hexdigest()
    if source_sha != meta["source_blend_sha256"]:
        raise ValueError("Source .blend changed since fixed rig export")
    obj = bpy.data.objects[args.object]
    if bpy.context.scene.unit_settings.system != "METRIC":
        raise ValueError("Explicit metric scene required")
    with np.load(rig_path, allow_pickle=False) as data:
        ids = data["vertex_indices"].copy()
        channels = data["channels"].copy()
        neutral = data["neutral_vertices"].astype(np.float64)*1000.
        deltas = data["blendshape_deltas"].astype(np.float64)*1000.
        lip_mask = data["lip_mask"].copy()
    linear = np.asarray(obj.matrix_world.to_3x3(), dtype=np.float64)*bpy.context.scene.unit_settings.scale_length*1000.
    def coords(key):
        values = np.empty(len(key.data)*3, dtype=np.float64)
        key.data.foreach_get("co", values)
        return values.reshape(-1,3)[ids] @ linear.T
    keys = obj.data.shape_keys.key_blocks
    if not np.allclose(coords(keys[0]), neutral, rtol=1e-5, atol=.0002):
        raise ValueError("Selected rest geometry disagrees with frozen rig")
    source_deltas = np.stack([coords(keys[str(name)])-coords(keys[str(name)].relative_key) for name in channels])
    if not np.allclose(source_deltas, deltas, rtol=1e-5, atol=.0002):
        raise ValueError("Shape-key basis disagrees with frozen rig")
    group_names = sorted(meta["lip_groups"])
    group_ids = {obj.vertex_groups[name].index: k for k, name in enumerate(group_names)}
    weights = np.zeros((len(ids), len(group_names)), np.float32)
    for local, source in enumerate(ids):
        for entry in obj.data.vertices[int(source)].groups:
            if entry.group in group_ids:
                weights[local, group_ids[entry.group]] = entry.weight
    def region(names):
        return weights[:,[group_names.index(name) for name in names]].sum(1) >= args.min_group_weight
    upper = region(["LipUpperMiddle", "lLipUpperInner", "rLipUpperInner"])
    lower = region(["LipLowerMiddle", "lLipLowerInner", "rLipLowerInner"])
    # Exclusive membership avoids picking the same mixed-boundary vertex twice.
    ambiguous = upper & lower
    upper &= ~ambiguous; lower &= ~ambiguous
    center = np.median(neutral[lip_mask,0])
    halfwidth = (neutral[lip_mask,0].max()-neutral[lip_mask,0].min())/2
    central = np.abs(neutral[:,0]-center) <= halfwidth*args.central_width_fraction
    upper &= central & lip_mask; lower &= central & lip_mask
    up, low = np.flatnonzero(upper), np.flatnonzero(lower)
    if min(len(up),len(low)) < 3:
        raise ValueError("Insufficient independently named inner-lip candidates")
    distances = np.linalg.norm(neutral[up,None,:2]-neutral[None,low,:2],axis=-1)
    order = np.argsort(distances,axis=None,kind="stable")
    chosen = []; used_up = set(); used_low = set()
    for flat in order:
        a,b = np.unravel_index(flat,distances.shape)
        u,l = int(up[a]),int(low[b])
        if distances[a,b] > args.max_xy_distance_mm:
            break
        if u in used_up or l in used_low:
            continue
        # Reject inverted or distant surfaces; record the remaining rest offset.
        gap = neutral[u,2]-neutral[l,2]
        if gap < -.5 or gap > 8.:
            continue
        midpoint = (neutral[u]+neutral[l])/2
        if any(np.linalg.norm(midpoint[:2]-(neutral[x]+neutral[y])[:2]/2) < .75 for x,y in chosen):
            continue
        chosen.append((u,l));used_up.add(u);used_low.add(l)
        if len(chosen) >= args.max_pairs:
            break
    if len(chosen) < 3:
        raise ValueError("Insufficient geometrically admissible point pairs")
    pairs = np.asarray(chosen,dtype=np.int64)
    pairs = pairs[np.argsort((neutral[pairs[:,0],0]+neutral[pairs[:,1],0])/2)]
    xy = np.linalg.norm(neutral[pairs[:,0],:2]-neutral[pairs[:,1],:2],axis=-1)
    j = (deltas[:,pairs[:,0],2]-deltas[:,pairs[:,1],2]).T
    rest = neutral[pairs[:,0],2]-neutral[pairs[:,1],2]
    quality = np.exp(-xy/args.max_xy_distance_mm).astype(np.float32)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output, schema=np.array(SCHEMA), jacobian_mm=j.astype(np.float32),
        rest_gap_mm=rest.astype(np.float32), pair_quality=quality,
        pair_local_indices=pairs, pair_source_indices=ids[pairs], pairing_xy_distance_mm=xy,
        channels=channels, vertex_indices=ids, neutral_mm=neutral.astype(np.float32),
        blendshape_deltas_mm=deltas.astype(np.float32), lip_mask=lip_mask,
        lip_group_weights=weights, group_names=np.asarray(group_names),
        upper_candidates=upper, lower_candidates=lower, provisional=np.array(True))
    report = dict(schema=SCHEMA,source_blend=str(blend),source_blend_sha256=source_sha,
        rig=str(rig_path),rig_sha256=hashlib.sha256(rig_path.read_bytes()).hexdigest(),
        object=obj.name,blender=bpy.app.version_string,coordinate_unit="millimetres",
        selected_vertices=len(ids),pair_count=len(pairs),jacobian_rank=int(np.linalg.matrix_rank(j)),
        candidates_upper=len(up),candidates_lower=len(low),ambiguous_candidates=int(ambiguous.sum()),
        pair_selection="unique greedy nearest XY in inner/middle named skin groups; central mouth only; fixed before motion data",
        args={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        observation="signed world-Z upper minus lower vertex separation; rest gap + J @ ARKit52",
        status="provisional_geometry_observation_not_physiological_contact",
        phoneme_alignment_available=False,train_tolerance_calibration_available=False,
        limitations=["Named skin weights do not define an independently validated collision surface.",
            "No p/b/m phase timings or contact labels have been extracted.",
            "Intervals require independent closure/release evidence and TRAIN-only fitted tolerances.",
            "Shape-key rest geometry excludes armature, subdivision and other modifiers."],
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix(".json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(report,ensure_ascii=False))


def main():
    args = arguments()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    try:
        import bpy
    except ImportError:
        command = [str(args.blender.resolve()), "--factory-startup", "--disable-autoexec",
                   "-b", "--python-exit-code", "1", "--python", str(Path(__file__).resolve()), "--",
                   "--rig",str(args.rig.resolve()),"--output",str(args.output.resolve()),
                   "--object",args.object,"--max-pairs",str(args.max_pairs),
                   "--min-group-weight",str(args.min_group_weight),
                   "--max-xy-distance-mm",str(args.max_xy_distance_mm),
                   "--central-width-fraction",str(args.central_width_fraction)]
        if args.blend:
            command += ["--blend",str(args.blend.resolve())]
        subprocess.run(command,check=True)
        if not args.output.exists() or not args.output.with_suffix(".json").exists():
            raise RuntimeError("Blender finished without a complete lip asset")
        if not args.skip_audit:
            subprocess.run([sys.executable,str(Path(__file__).with_name("audit_lip_constraints_v2.py")),
                            "--asset",str(args.output.resolve())],check=True)
    else:
        export_worker(args,bpy)


if __name__ == "__main__":
    main()
