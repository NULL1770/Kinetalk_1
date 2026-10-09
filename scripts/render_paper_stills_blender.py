"""Blender-only fixed-camera transparent scientific stills. No source save."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def point_at(obj,p):obj.rotation_euler=(Vector(p)-obj.location).to_track_quat('-Z','Y').to_euler()


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    spec=json.loads(Path(a.manifest).read_text(encoding='utf8'));out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    before=digest(bpy.data.filepath)
    prototype=bpy.data.objects['face.001'];assert prototype.data.shape_keys
    scene=bpy.data.scenes.new('PaperStill');bpy.context.window.scene=scene
    scene.render.engine='BLENDER_EEVEE_NEXT';scene.eevee.taa_render_samples=64
    scene.render.resolution_x=1024;scene.render.resolution_y=1024;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA'
    scene.render.film_transparent=True
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
    scene.view_settings.exposure=-1.2;scene.view_settings.gamma=1
    scene.world=bpy.data.worlds.new('PaperWorld');scene.world.use_nodes=True
    bg=scene.world.node_tree.nodes['Background'];bg.inputs['Color'].default_value=(.92,.94,.96,1);bg.inputs['Strength'].default_value=.55
    obj=prototype.copy();obj.data=prototype.data.copy();scene.collection.objects.link(obj)
    obj.animation_data_clear();obj.data.shape_keys.animation_data_clear();obj.location=(0,0,-.02)
    obj.hide_viewport=False;obj.hide_render=False;obj.hide_set(False)
    material=bpy.data.materials.new('SharedClay');material.use_nodes=True
    shader=material.node_tree.nodes.get('Principled BSDF');shader.inputs['Base Color'].default_value=(.46,.49,.53,1);shader.inputs['Roughness'].default_value=.7
    obj.data.materials.clear();obj.data.materials.append(material)
    keys=obj.data.shape_keys.key_blocks
    for key in keys:key.value=0
    camera_data=bpy.data.cameras.new('PaperCamera');camera=bpy.data.objects.new('PaperCamera',camera_data);scene.collection.objects.link(camera)
    camera.location=(0,-3,.05);point_at(camera,(0,0,.05));camera_data.type='ORTHO';camera_data.ortho_scale=.38;scene.camera=camera
    for i,(location,energy,size) in enumerate([((-1.2,-1.7,2),120,2),((1.5,-1,.3),65,2),((0,.8,1.8),100,1.5)]):
        ld=bpy.data.lights.new('PaperLight'+str(i),'AREA');ld.energy=energy;ld.shape='DISK';ld.size=size
        lo=bpy.data.objects.new('PaperLight'+str(i),ld);scene.collection.objects.link(lo);lo.location=location;point_at(lo,(0,0,0))
    receipts=[];cache={}
    rows=spec['stills'][:a.limit] if a.limit else spec['stills']
    for row in rows:
        source=row['source']
        if source not in cache:
            assert digest(source)==row['source_sha256']
            with np.load(source,allow_pickle=False) as z:cache[source]={k:z[k].copy() for k in z.files}
        z=cache[source];frame=row['frame'];mode=row['mode']
        assert bool(z['valid'][frame]) and abs(float(z['times'][frame])-row['time_seconds'])<1e-7
        mask=z['channel_mask'];mask=mask[frame] if mask.ndim==2 else mask
        raw=z['motions'][mode,frame]
        assert np.isfinite(raw[mask]).all()
        values=np.clip(np.where(mask,raw,0),0,1)
        for key in keys:key.value=0
        for name,value in zip(z['channels'].tolist(),values):
            keys[name].slider_min=0;keys[name].slider_max=1;keys[name].value=float(value)
        bpy.context.view_layer.update()
        actual=np.array([keys[name].value for name in z['channels'].tolist()])
        assert np.max(np.abs(actual-values))<=1e-7
        path=out/(row['name']+'.png')
        if not path.exists():
            scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
        receipts.append(dict(**row,image_sha256=digest(path),key_max_error=float(np.max(np.abs(actual-values))),
            display_clipped_values=int(((raw<0)|(raw>1))[mask].sum())))
    assert digest(bpy.data.filepath)==before
    (out/'render_receipt.json').write_text(json.dumps(dict(blend_sha256=before,original_blend_unchanged=True,
        blender=bpy.app.version_string,resolution=[1024,1024],transparent=True,rig='face.001',
        camera=dict(location=list(camera.location),ortho_scale=.38),same_settings_all_methods=True,
        manifest_sha256=digest(a.manifest),stills=receipts),indent=2),encoding='utf8')


if __name__=='__main__':main()
