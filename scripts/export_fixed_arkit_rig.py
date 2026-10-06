"""Blender worker: export fixed rest-pose shape-key geometry and rig regions.

The named skin-weight regions are frozen before any model/test scoring. The
source .blend is opened read-only in practice (never saved). Armature and
subdivision modifiers are deliberately excluded from the linear fixed rig.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import bpy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--object',default='body')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    obj=bpy.data.objects[args.object];keys=obj.data.shape_keys.key_blocks
    if set(ARKIT_NAMES)-set(keys.keys()):raise ValueError('Incomplete ARKit52 shape keys')
    if bpy.context.scene.unit_settings.system!='METRIC':raise ValueError('Explicit metric scene required')
    linear=np.array(obj.matrix_world.to_3x3(),dtype=np.float64)*bpy.context.scene.unit_settings.scale_length
    def coords(key):
        value=np.empty(len(key.data)*3,dtype=np.float64);key.data.foreach_get('co',value)
        return value.reshape(-1,3)@linear.T
    neutral=coords(keys[0]);deltas=np.stack([coords(keys[name])-coords(keys[name].relative_key) for name in ARKIT_NAMES])
    lip_names={'lLipCorner','rLipCorner','lLipLowerOuter','lLipLowerInner','LipLowerMiddle','rLipLowerInner','rLipLowerOuter',
               'LipUpperMiddle','lLipUpperOuter','lLipUpperInner','rLipUpperInner','rLipUpperOuter'}
    expression_names={g.name for g in obj.vertex_groups if ('Eyelid' in g.name or 'Brow' in g.name) and not g.name.startswith('DEF-')}
    def region(names):
        ids={obj.vertex_groups[name].index for name in names}
        weights=np.array([sum(g.weight for g in v.groups if g.group in ids) for v in obj.data.vertices])
        return weights >= .5
    lip=region(lip_names);expression=region(expression_names)
    if min(lip.sum(),expression.sum())<10:raise ValueError('Empty/insufficient fixed anatomical regions')
    # Keep only vertices used by paper regions. This exact restriction does
    # not affect max-region LVE/EVE or regional energy FDD and lowers memory.
    selected=lip|expression;indices=np.flatnonzero(selected)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError(args.output)
    np.savez_compressed(args.output,neutral_vertices=neutral[selected].astype(np.float32),
        blendshape_deltas=deltas[:,selected].astype(np.float32),lip_mask=lip[selected],expression_mask=expression[selected],
        fdd_mask=expression[selected],coordinate_scale_to_mm=np.array(1000.),coordinate_unit=np.array('metres'),
        vertex_indices=indices,channels=np.asarray(ARKIT_NAMES))
    audit=dict(source_blend=bpy.data.filepath,source_blend_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
        object=obj.name,blender=bpy.app.version_string,source_vertices=len(neutral),selected_vertices=int(selected.sum()),
        lip_vertices=int(lip.sum()),expression_vertices=int(expression.sum()),lip_groups=sorted(lip_names),expression_groups=sorted(expression_names),
        region_rule='sum of named original skin weights >= 0.5, fixed independently of data/predictions',
        geometry='raw relative shape-key rest geometry; world linear transform and metric scene scale applied; modifiers excluded',
        fdd_region='same fixed eye/forehead region as EVE',output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        official_dataset_vertex_topology=False)
    args.output.with_suffix('.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(audit,ensure_ascii=False))


main()
