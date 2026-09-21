"""Losslessly remove batch padding using locked native clip lengths."""
from pathlib import Path
import torch
from scripts.train_formal_predictable_projection import save_checkpoint
from scripts.extract_emotion2vec_pilot import sha


def compact_curves(curves,length_by_id):
    ids=curves['clip_id']
    if len(ids)!=len(set(ids)):raise ValueError('Repeated clip ID')
    lengths=[int(length_by_id[c]) for c in ids];limit=curves['valid'].shape[1]
    if any(n<1 or n>limit for n in lengths):raise ValueError('Native length outside stored padding')
    if any(curves['valid'][i,n:].any() for i,n in enumerate(lengths)):raise ValueError('Length would discard observed frames')
    def pack(x):return torch.cat([x[i,:n] for i,n in enumerate(lengths)])
    packed={k:v for k,v in curves.items() if k not in ('target','valid','times','b0','predictions')}
    packed.update(compaction_schema='native_curve_pack_v1',native_lengths=lengths,padded_length=limit)
    for k in ('target','valid','times','b0'):
        if k in curves:packed[k]=pack(curves[k])
    packed['predictions']={k:pack(v) for k,v in curves['predictions'].items()}
    return packed


def expand_curves(packed):
    if packed.get('compaction_schema')!='native_curve_pack_v1':raise ValueError('Wrong curve pack schema')
    ns=packed['native_lengths'];limit=packed['padded_length']
    def unpack(x):
        output=x.new_zeros(len(ns),limit,*x.shape[1:]);start=0
        for i,n in enumerate(ns):output[i,:n]=x[start:start+n];start+=n
        if start!=len(x):raise ValueError('Packed length differs')
        return output
    out={k:v for k,v in packed.items() if k not in ('compaction_schema','native_lengths','padded_length','target','valid','times','b0','predictions')}
    for k in ('target','valid','times','b0'):
        if k in packed:out[k]=unpack(packed[k])
    out['predictions']={k:unpack(v) for k,v in packed['predictions'].items()}
    return out


def persist_native_curves(source,destination,length_by_id):
    curves=torch.load(source,map_location='cpu',weights_only=False)
    packed=compact_curves(curves,length_by_id)
    save_checkpoint(Path(destination),packed)
    restored=expand_curves(torch.load(destination,map_location='cpu',weights_only=False))
    # Verify every native value after the durable write, including invalid
    # native edge frames. Only batch padding beyond native lengths is omitted.
    for i,n in enumerate(packed['native_lengths']):
        for key in ('target','valid','times','b0'):
            if key in curves:
                torch.testing.assert_close(restored[key][i,:n],curves[key][i,:n],rtol=0,atol=0,equal_nan=True)
        for key,value in curves['predictions'].items():
            torch.testing.assert_close(restored['predictions'][key][i,:n],value[i,:n],rtol=0,atol=0,equal_nan=True)
    return {'source':str(source),'compact':str(destination),'clips':len(curves['clip_id']),
            'native_frames':sum(packed['native_lengths']),'source_bytes':Path(source).stat().st_size,
            'compact_bytes':Path(destination).stat().st_size,'dtype':'unchanged','quantization':False,
            'native_values_verified':True,'source_sha256':sha(source),'compact_sha256':sha(destination),
            'omitted':'padding beyond manifest native lengths only; reconstruct with expand_curves'}
