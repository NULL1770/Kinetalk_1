"""Reconstruct MEAD reference h0 on each original full native clock.

Old arrays are copied byte-for-byte into a new data directory. Full packed
TRAIN/validation provide independent enrollment clips; no test is loaded and
no weights are trained. Unknown HDTF metadata is explicitly unavailable.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_expression_response import load_runtime,configure,write
from scripts.train_full_staged import base_forward
from scripts.train_v7_pilot import sha
from kinetalk_v2.hdtf_dataset import HDTFSplit


@torch.no_grad()
def main(a):
    configure(47); device=torch.device(a.device)
    old=Path(a.data);out=Path(a.output)
    if out.exists():raise ValueError('New reference data directory required')
    out.mkdir(parents=True)
    provenance=json.loads((old/'provenance.json').read_text(encoding='utf8'))
    runtime,base=load_runtime(json.loads(Path(a.binding).read_text()),device)
    lookup={}
    for split in ('train','validation'):
        q=runtime['splits'][split]
        for i,cid in enumerate(q['clip_id']):lookup[cid]=(split,i)
    hdtf = None
    if a.hdtf_manifest:
        hdtf = HDTFSplit(a.hdtf_manifest, a.hdtf_features, split='train', motion_policy='raw')
        for i,cid in enumerate(hdtf['clip_id']): lookup[cid]=('hdtf',i)
    cache={};stats={}
    for role in ('fit','held_sentence','unseen_speaker'):
        with np.load(old/(role+'.npz'),allow_pickle=False) as z:
            values={k:z[k].copy() for k in z.files}
        n,r,t,_=values['ref_motion'].shape;dim=values['h0'].shape[-1]
        anchors=np.zeros((n,r,t,dim),dtype=np.float32)
        available=np.zeros((n,r),dtype=bool)
        meta=sorted([x for x in provenance['records'] if x['role']==role],key=lambda x:x['row'])
        for item in meta:
            row=item['row']
            for j,ref in enumerate(item['references']):
                cid=ref['clip_id']
                if cid not in lookup:continue
                split,i=lookup[cid]
                if role in ('fit','held_sentence') and split not in ('train','hdtf'):
                    raise ValueError('Fit/held enrollment must remain TRAIN-only')
                if cid not in cache:
                    dataset = hdtf if split == 'hdtf' else runtime['splits'][split]
                    b=dataset.batch(torch.tensor([i]),device)
                    h0=base_forward(base,b['content'],b['valid'])['h0'][0].cpu().numpy()
                    cache[cid]=(h0,b['valid'][0].cpu().numpy())
                h0,valid=cache[cid];start,end=ref['start_frame'],ref['end_frame']
                cropped=h0[start:end];vm=valid[start:end]
                if not np.array_equal(values['ref_valid'][row,j,:len(vm)],vm):
                    raise ValueError(f'Reference validity mismatch: {cid}')
                anchors[row,j,:len(cropped)]=cropped
                available[row,j]=bool(values['ref_valid'][row,j].any())
        values['ref_h0']=anchors;values['ref_h0_available']=available
        np.savez(out/(role+'.npz'),**values)
        # Old query, motion, enrollment and clocks must remain exactly equal.
        with np.load(out/(role+'.npz'),allow_pickle=False) as written, np.load(old/(role+'.npz'),allow_pickle=False) as original:
            for k in original.files:np.testing.assert_array_equal(written[k],original[k])
        stats[role]={'reference_slots':int(n*r),'anchors_available':int(available.sum()),
                     'unavailable_policy':'use original audio-only reference route, never zero h0 anchor',
                     'old_sha256':sha(old/(role+'.npz')),'new_sha256':sha(out/(role+'.npz'))}
    provenance['reference_anchor_restoration']={'roles':stats,'query_arrays_unchanged':True,
                                               'full_native_clock_before_crop':True,
                                               'sealed_or_test_loaded':False,
                                               'neutral_checkpoint_sha256':sha(json.loads(Path(a.binding).read_text())['neutral_checkpoint']['path'])}
    write(out/'provenance.json',provenance)
    print(json.dumps(provenance['reference_anchor_restoration']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--binding',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--hdtf-manifest');p.add_argument('--hdtf-features');main(p.parse_args())
