"""Blender-only worker. Operates in memory; never saves or changes the source blend."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np


BROWS = ['browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight']


def point_at(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def material(name, color, roughness=.7):
    value = bpy.data.materials.new(name)
    value.diffuse_color = (*color, 1)
    value.use_nodes = True
    shader = value.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Roughness'].default_value = roughness
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--object', required=True)
    parser.add_argument('--fps', type=int, required=True)
    parser.add_argument('--tile-size', type=int, required=True)
    parser.add_argument('--columns', type=int, required=True)
    parser.add_argument('--samples', type=int, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    with np.load(args.input, allow_pickle=False) as data:
        channels, modes = data['channels'].tolist(), data['mode_names'].tolist()
        motions, times, valid = data['motions'], data['times'], data['valid']
    if motions.shape != (len(modes), len(times), len(channels)) or not np.isfinite(motions).all():
        raise ValueError('Invalid prepared display arrays')
    prototype = bpy.data.objects.get(args.object)
    if prototype is None or prototype.type != 'MESH' or prototype.data.shape_keys is None:
        raise ValueError('Source object must be a mesh with shape keys')
    keys = prototype.data.shape_keys.key_blocks
    missing = sorted(set(channels) - set(keys.keys()))
    if missing:
        raise ValueError('Rig lacks named shape keys: ' + ', '.join(missing))
    geometry = {}
    for name in BROWS:
        distances = [(a.co - b.co).length for a, b in zip(keys[name].data, keys[0].data)]
        geometry[name] = {'max_local_displacement': max(distances),
                          'nonzero_vertices': sum(x > 1e-7 for x in distances)}
        if geometry[name]['max_local_displacement'] <= 1e-7:
            raise ValueError('Brow shape key has no geometry: ' + name)
    # Fresh scene avoids stale visibility, animation, compositor, camera and strip state.
    scene = bpy.data.scenes.new('Matched_Dynamic_Display')
    bpy.context.window.scene = scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.eevee.taa_render_samples = args.samples
    scene.render.fps = args.fps
    scene.render.fps_base = 1
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.render.resolution_percentage = 100
    cols = min(args.columns, len(modes))
    rows = math.ceil(len(modes) / cols)
    scene.render.resolution_x = cols * args.tile_size
    scene.render.resolution_y = rows * args.tile_size
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'Medium High Contrast'
    scene.view_settings.exposure = -2
    scene.view_settings.gamma = 1
    scene.world = bpy.data.worlds.new('Display_World')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.92, .94, .96, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .55
    clay = material('Shared_Neutral_Clay', (.32, .35, .39))
    ink = material('Label_Ink', (.025, .035, .045))
    spacing, center_z = .49, .02
    clones = []
    for i, mode in enumerate(modes):
        x = (i % cols - (cols - 1) / 2) * spacing
        z = ((rows - 1) / 2 - i // cols) * spacing
        obj = prototype.copy()
        obj.data = prototype.data.copy()
        obj.name = f'Display_{i:02d}'
        scene.collection.objects.link(obj)
        obj.animation_data_clear()
        obj.data.shape_keys.animation_data_clear()
        obj.location = (x, 0, z - center_z)
        obj.hide_viewport = False
        obj.hide_render = False
        obj.hide_set(False)
        obj.data.materials.clear()
        obj.data.materials.append(clay)
        for k in obj.data.shape_keys.key_blocks:
            k.value = 0
        # Key by actual names, preserving integer native frames. Linear at subframes.
        for c, name in enumerate(channels):
            key = obj.data.shape_keys.key_blocks[name]
            key.slider_min, key.slider_max = 0, 1
            for frame in range(len(times)):
                key.value = float(motions[i, frame, c])
                key.keyframe_insert('value', frame=frame + 1)
        action = obj.data.shape_keys.animation_data.action
        for curve in action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = 'LINEAR'
        clones.append(obj)
        font = bpy.data.curves.new(f'Label_{i}', 'FONT')
        font.body, font.align_x, font.size = mode, 'CENTER', .019
        font.extrude = 0
        text = bpy.data.objects.new(f'Label_{i}', font)
        scene.collection.objects.link(text)
        text.location = (x, -.27, z + .217)
        text.rotation_euler = (math.pi / 2, 0, 0)
        text.data.materials.append(ink)
    if len({o.data.shape_keys.as_pointer() for o in clones}) != len(clones):
        raise ValueError('Comparison objects share shape-key animation data')
    camera_data = bpy.data.cameras.new('Comparison_Camera')
    camera = bpy.data.objects.new('Comparison_Camera', camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0, -3, 0)
    point_at(camera, (0, 0, 0))
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = cols * spacing
    scene.camera = camera
    for i, (location, energy, size) in enumerate([((-1.2, -1.7, 2), 120, 2), ((1.5, -1, .3), 65, 2), ((0, .8, 1.8), 100, 1.5)]):
        light_data = bpy.data.lights.new(f'Display_Light_{i}', 'AREA')
        light_data.energy, light_data.shape, light_data.size = energy, 'DISK', size
        light = bpy.data.objects.new(f'Display_Light_{i}', light_data)
        scene.collection.objects.link(light)
        light.location = location
        point_at(light, (0, 0, 0))
    frames_dir = args.output / 'frames'
    frames_dir.mkdir(exist_ok=False)
    audit = {'schema': 'blender_dynamic_rig_v1', 'blender': bpy.app.version_string,
             'source_object': args.object, 'source_vertices': len(prototype.data.vertices),
             'brow_geometry': geometry, 'mapping': {name: name for name in channels},
             'modes': modes, 'independent_shape_keys': True, 'old_actions_cleared': True,
             'camera': {'type': 'ORTHO', 'scale': camera_data.ortho_scale, 'tile_spacing': spacing},
             'display_material': 'same neutral clay, geometry unchanged',
             'resolution': [scene.render.resolution_x, scene.render.resolution_y],
             'frame_start': 1, 'frame_end': len(times), 'fps': args.fps,
             'original_blend_saved': False, 'frame_value_max_error': 0.0,
             'display_filled_frames': int((~valid).sum())}
    scene.frame_start, scene.frame_end = 1, len(times)
    for frame in range(len(times)):
        scene.frame_set(frame + 1)
        for i, obj in enumerate(clones):
            actual = np.asarray([obj.data.shape_keys.key_blocks[name].value for name in channels])
            error = float(np.abs(actual - motions[i, frame]).max())
            audit['frame_value_max_error'] = max(audit['frame_value_max_error'], error)
            if error > 1e-6:
                raise ValueError('Rendered keys differ from prepared coefficients')
        scene.render.filepath = str(frames_dir / f'{frame + 1:06d}.png')
        bpy.ops.render.render(write_still=True)
    (args.output / 'rig_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf8')
    print('MATCHED_RIG_RENDER_COMPLETE', len(times), 'frames', len(modes), 'modes')


if __name__ == '__main__':
    main()
