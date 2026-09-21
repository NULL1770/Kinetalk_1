import torch
import pytest
from scripts.compact_native_curves import compact_curves,expand_curves,persist_native_curves


def test_native_pack_keeps_every_native_value_and_mask_including_invalid_edges():
    q={'clip_id':['a','b'],'valid':torch.tensor([[False,True,True,False,False],[True,False,True,True,False]]),
       'target':torch.randn(2,5,52),'times':torch.arange(5).expand(2,5)/25,
       'channel_mask':torch.ones(2,52,dtype=torch.bool),'predictions':{'42/full':torch.randn(2,5,52)}}
    lengths={'a':4,'b':5};packed=compact_curves(q,lengths);out=expand_curves(packed)
    for i,c in enumerate(q['clip_id']):
        n=lengths[c]
        for k in ('target','times','valid'):torch.testing.assert_close(q[k][i,:n],out[k][i,:n],rtol=0,atol=0)
        torch.testing.assert_close(q['predictions']['42/full'][i,:n],out['predictions']['42/full'][i,:n],rtol=0,atol=0)
    assert packed['target'].shape[0]==9
    with pytest.raises(ValueError,match='discard'):compact_curves(q,{'a':2,'b':5})


def test_persistence_checks_saved_native_values_and_distinct_hashes(tmp_path):
    from scripts.run_calibrated_temporal_queue import persist_stage_artifacts
    import json
    source=tmp_path/'volatile';source.mkdir()
    q={'clip_id':['a'],'valid':torch.tensor([[True,False,False]]),'target':torch.randn(1,3,52),
       'times':torch.arange(3).expand(1,3)/25,'b0':torch.randn(1,3,52),
       'predictions':{'42/full':torch.randn(1,3,52)}}
    torch.save(q,source/'curves.pt')
    (source/'temporal_full.json').write_text('[{"clip_id":"a"}]')
    destination=tmp_path/'durable'
    record=persist_stage_artifacts(source,destination,{'a':2})
    assert record['native_values_verified'] and record['native_frames']==2
    assert record['source_sha256']!=record['compact_sha256']
    assert (destination/'temporal_full.json').read_bytes()==(source/'temporal_full.json').read_bytes()
    assert json.loads((destination/'persistence.json').read_text())==record
