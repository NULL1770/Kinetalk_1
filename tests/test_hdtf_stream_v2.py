import json
import pytest
from scripts.stream_hdtf_features_v2 import available


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')


def test_feature_stream_excludes_zero_window_and_sealed_sources(tmp_path):
    inventory=[{'source_id':str(i),'speaker_group':str(i),'split':split} for i,split in enumerate(('train','train','val','test','quarantine'))]
    write_rows(tmp_path/'inventory.jsonl',inventory)
    records=[dict(r,processing_status='ok',n_training_eligible_windows=0 if r['source_id']=='1' else 3) for r in inventory]
    write_rows(tmp_path/'processed_sources.jsonl',records)
    assert [r['source_id'] for r in available(tmp_path)]==['0','2']


def test_feature_stream_rejects_identity_split_leakage_before_target_io(tmp_path):
    write_rows(tmp_path/'inventory.jsonl',[{'speaker_group':'same','split':'train'},{'speaker_group':'same','split':'test'}])
    # No source motion/audio/receipt exists: metadata must reject before IO.
    with pytest.raises(ValueError,match='crosses splits'):
        available(tmp_path)
