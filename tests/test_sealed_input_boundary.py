import json
from pathlib import Path
import numpy as np
import torch
from scripts.prepare_sealed_inputs import read_input
from scripts.evaluate_sealed_models import SealedInputs


def test_query_reader_never_indexes_motion(monkeypatch,tmp_path):
    n=5
    class Archive:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __getitem__(self,key):
            if key=='motion':raise AssertionError('Sealed query target accessed before inference')
            return dict(content=np.zeros((n,768),np.float32),times=np.arange(n)*.04,
                mask=np.ones(n,bool),channel_mask=np.ones(52,bool),
                provenance=np.array(json.dumps(dict(clock_evidence='embedded_video',fps=25))))[key]
    monkeypatch.setattr('scripts.prepare_sealed_inputs.sha',lambda p:'hash')
    monkeypatch.setattr('scripts.prepare_sealed_inputs.np.load',lambda *a,**k:Archive())
    row=dict(source_split='test',artifact='a.npz',artifact_sha256='hash',frames=n,valid_frames=n,clip_id='test',sentence='s')
    value=read_input(row,tmp_path)
    assert 'motion' not in value and value['content'].shape==(n,768)


def test_sealed_batch_only_uses_audio_and_enrollment():
    data=object.__new__(SealedInputs)
    data.rows=[{'speaker':'a','emotion':99},{'speaker':'a','emotion':99}]
    data.offsets=np.array([0,3,8]);data.audio=np.ones((8,1540),np.float16)
    data.valid=np.ones(8,bool);data.anchors={'a':torch.zeros(52)}
    value,lengths=data.batch([0,1],'cpu')
    assert set(value)=={'audio_features','content','valid','anchors'}
    assert lengths==[3,5] and not value['valid'][0,3:].any()
