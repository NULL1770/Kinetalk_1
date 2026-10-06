"""Bind immutable historical inputs and preselect metadata-only D0 cells."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

root = Path('/root/kinetalk_phase31_neutral_role_20261006')
target = root/'factor_binding_v3.json'
if target.exists(): raise FileExistsError('Fresh binding required')
diagnostic = root/'tools/frozen_factor_diagnostic_v2.py'
protocol = root/'tools/phase30_ua_protocol.py'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
spec=importlib.util.spec_from_file_location('factor_tools',diagnostic)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
data=Path('/root/autodl-tmp/kinetalk_data/packed_trainval_20260923')
marker=json.loads((data/'packed_cache.json').read_text())
assert marker['test_loaded'] is False
meta=torch.load(data/marker['metadata'],map_location='cpu',weights_only=False)
assert meta['test_loaded'] is False
info=meta['split_meta']['validation']
fields={key:np.load(data/'validation'/f'{key}.npy',allow_pickle=False)
        for key in ('speaker_id','emotion_id','intensity_id')}
selected=module.stratified_indices(info['clip_id'],fields['speaker_id'],fields['emotion_id'],fields['intensity_id'])
models={}
paths={
 'neutral':('/root/autodl-tmp/kinetalk_final_20260922/checkpoints/phase2_fullmouth_timing000_20261004',
            '/root/autodl-tmp/kinetalk_final_20260922/code/phase2_support_complete_20261004'),
 'native':('/root/kinetalk_channel_coordinates_stable_budget_20261006/seed47/checkpoints/standardized',
           '/root/kinetalk_channel_coordinates_stable_budget_20261006/code')}
for name,(cp,code) in paths.items():
    cp,code=Path(cp),Path(code)
    recipe=json.loads((cp/'provenance.json').read_text())['recipe']
    assert recipe['test_loaded'] is False
    for k,v in recipe['source_sha256'].items():assert sha(code/k)==v,(name,k)
    complete=json.loads((cp/'audio/complete.json').read_text())
    files={key:sha(cp/key) for key in ('audio/final.pt','audio/curves.pt')}
    assert files['audio/final.pt']==complete['final_sha256']
    assert files['audio/curves.pt']==complete['curves_sha256']
    models[name]=dict(checkpoint_root=str(cp),code_root=str(code),source_sha256=recipe['source_sha256'],files=files,
        stride=recipe['args']['stride'],batch_size=recipe['args']['batch_size'],stable_runtime=name=='native')
binding=dict(schema='phase31_frozen_factor_binding_v1',data=str(data),models=models,
    diagnostic_sha256=sha(diagnostic),protocol_sha256=sha(protocol),preparer_sha256=sha(__file__),
    packed_metadata_sha256=sha(data/marker['metadata']),data_marker_sha256=sha(data/'packed_cache.json'),
    validation_array_sha256={str(p.relative_to(data)):sha(p) for p in (data/'validation').glob('*.npy')},
    selected_indices=selected,selected_clip_ids=[info['clip_id'][i] for i in selected],
    selection='Lexicographically first ID per observed validation speaker/emotion/intensity cell; no generation consulted',
    replay_draws=[42,123,2026],intervention_draw=42,replay_atol=0,replay_rtol=0,
    test_loaded=False,training_performed=False)
target.write_text(json.dumps(binding,indent=2))
print(json.dumps(dict(binding_sha256=sha(target),selected_cells=len(selected),models=list(models))),flush=True)
