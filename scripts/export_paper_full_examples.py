"""Export one metadata-selected full sequence per validation identity.

The lexicographically first clip per identity is fixed without viewing outputs.
Only saved validation curves and hash-bound prepared validation shards are read.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_paper_full_run import sha, read, require
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES, inspect_input


def export(run, artifacts, data, output):
    require(not output.exists(), 'Fresh visual export directory required')
    manifest = read(data / 'manifest.json')
    rows = manifest['roles']['val']['query']
    picks = [min((r for r in rows if r['speaker'] == s), key=lambda r:r['clip_id'])
             for s in sorted({r['speaker'] for r in rows})]
    curves = {}
    for arm in ('audio', 'static'):
        p = artifacts / arm / 'dynamics/curves.pt'
        require(sha(p) == read(run / arm / 'dynamics/complete.json')['curves_sha256'], 'Curve SHA mismatch')
        curves[arm] = torch.load(p, map_location='cpu', weights_only=False, mmap=True)
    a, s = curves['audio'], curves['static']
    require(a['clip_id'] == s['clip_id'], 'Paired curve membership differs')
    for key in ('target', 'valid', 'times', 'channel_mask'):
        require(torch.equal(a[key], s[key]), 'Paired ' + key + ' differs')
    index = {r['clip_id']:r for r in read(data / 'index.json')['records'] if r['role'] == 'val' and r['kind'] == 'query'}
    output.mkdir(parents=True)
    jobs = []
    modes = ['GT', 'Stage4 audio', 'Audio dynamics', 'Trained static', 'Reversed local audio', 'Oracle target state']
    for row in picks:
        cid = row['clip_id'];i = a['clip_id'].index(cid);n = row['frames']
        shard_path = data / index[cid]['path']
        require(sha(shard_path) == index[cid]['sha256'], 'Prepared shard hash differs')
        shard = torch.load(shard_path, map_location='cpu', weights_only=False)
        require(shard['row'] == row, 'Prepared row differs')
        require(torch.equal(shard['valid'], a['valid'][i, :n]), 'Saved native mask differs')
        require(torch.equal(shard['times'], a['times'][i, :n]), 'Saved native clock differs')
        require(torch.equal(shard['motion'], a['target'][i, :n]), 'Saved target differs')
        native = shard['native_provenance'];wave = Path(native['audio_path'])
        require(sha(wave) == native['audio_sha256'], 'Waveform hash differs')
        values = torch.stack([a['target'][i], a['predictions']['42/base'][i], a['predictions']['42/full'][i],
                              s['predictions']['42/full'][i], a['predictions']['42/reverse_audio'][i],
                              a['predictions']['42/oracle_state'][i]])[:, :n].numpy()
        target = output / (cid + '.npz')
        np.savez_compressed(target, channels=np.asarray(ARKIT_NAMES), mode_names=np.asarray(modes),
            clip_id=np.asarray(cid), noise_seed=np.asarray(42), times=a['times'][i, :n].numpy(),
            valid=a['valid'][i, :n].numpy(), channel_mask=a['channel_mask'][i].numpy(), motions=values)
        copied = output / (cid + wave.suffix);shutil.copy2(wave, copied)
        jobs.append({'clip_id':cid, 'speaker':row['speaker'], 'frames':n, 'input':target.name, 'audio':copied.name,
            'audio_sha256':native['audio_sha256'], 'audio_offset_seconds':native['audio_offset_s'],
            'input_sha256':sha(target), 'display':inspect_input(target,25)[-1]})
    result = {'selection':'lexicographically first validation query per speaker; metadata only; no output selection',
        'noise_seed':42, 'mode_names':modes, 'manifest_sha256':manifest['manifest_sha256'], 'jobs':jobs,
        'full_native_frames':True, 'amplitude_or_lag_fitting':False, 'test_loaded':False,
        'display_only_clipping':True, 'oracle_target_conditioned':True, 'actual_rendering_performed':False,
        'script_sha256':sha(Path(__file__))}
    (output/'provenance.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','artifacts','data','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=export(a.run,a.artifacts,a.data,a.output)
    print(json.dumps({'output':str(a.output),'clips':[v['clip_id'] for v in r['jobs']]}))
