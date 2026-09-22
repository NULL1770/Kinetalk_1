import copy
import json
from pathlib import Path

import pytest

from scripts import expand_mead_full_emotion_manifest as runner
from scripts.prepare_paper_full_data import validate_manifest
from scripts.train_formal_predictable_projection import canonical_hash


def fixture():
    rows={};roles={}
    for role in ('train','val','test'):
        person=role+'_person'
        def row(cid,sentence,emotion,intensity):
            return {'clip_id':role+'_'+cid,'dataset':'mead','speaker':person,'sentence':sentence,
                'emotion':emotion,'intensity':intensity,'split':role,'artifact':f'clips/{role}_{cid}.npz',
                'artifact_sha256':'a'*64,'frames':40,'valid_frames':38,'stage1':False,
                'native_extra':{'untouched':'metadata'}}
        part=[row('r0','ref0',0,0),row('r1','ref1',0,0)]
        part.append(row('q0','query',0,0))
        part.extend(row(f'q{e}_{i}','query',e,i) for e in range(1,8) for i in (1,2,3))
        refs=[{**r,'source_split':role,'manifest_role':'enrollment'} for r in part[:2]]
        query=[{**r,'source_split':role,'manifest_role':'query'} for r in part[2:]
               if r['emotion'] in (0,1,5,6) and r['intensity'] in (0,1,3)]
        roles[role]={'query':query,'enrollment':refs,'legacy_note':'unchanged'}
        if role!='test':rows[role]=part
    base={'status':'approved_train_val_only','roles':roles,'sealed_test_targets_loaded':False,
          'reserved_sentences':['sealed'],'exclusions':[]}
    base['manifest_sha256']=canonical_hash(base)
    return base,rows


def test_expands_all_emotions_levels_preserves_refs_test_and_native_metadata():
    base,rows=fixture();before=copy.deepcopy(base)
    out,summary=runner.build_manifest(base,rows,{'train':'x','val':'y'},'z')
    validate_manifest(out)
    assert base==before and out['roles']['test']==base['roles']['test']
    for role in ('train','val'):
        assert out['roles'][role]['enrollment']==base['roles'][role]['enrollment']
        assert {r['emotion'] for r in out['roles'][role]['query']}==set(range(8))
        native={r['clip_id']:r for r in rows[role]}
        for row in out['roles'][role]['query']:
            assert all(row[k]==v for k,v in native[row['clip_id']].items())
    assert summary['human_review_claimed'] is False
    assert summary['test_is_eight_emotion_benchmark'] is False


def test_exclusions_are_real_rows_with_auditable_reasons_and_partition_counts():
    base,rows=fixture()
    template=rows['train'][3]
    changes=[('reserved',{'sentence':'sealed'}),('same_ref_sentence',{'sentence':'ref0'}),
             ('short',{'valid_frames':31}),('other_dataset',{'dataset':'crema_d','intensity':-1}),
             ('other_person',{'speaker':'foreign'})]
    for name,change in changes:
        rows['train'].append({**copy.deepcopy(template),'clip_id':name,**change})
    out,summary=runner.build_manifest(base,rows,{'train':'x','val':'y'},'z')
    assert len(out['exclusions'])==len(changes)
    assert {r['clip_id'] for r in out['exclusions']}=={x[0] for x in changes}
    assert sum(summary['source_partition_counts']['train'].values())==len(rows['train'])
    assert not any(r['sentence']=='sealed' for r in out['roles']['train']['query'])
    assert all('native_metadata' in r for r in out['exclusions'])


@pytest.mark.parametrize('kind',['enrollment_hash','duplicate','source_split','unknown_emotion','unsafe_path','missing_level'])
def test_rejects_metadata_drift_or_false_full_coverage(kind):
    base,rows=fixture()
    if kind=='enrollment_hash':rows['train'][0]['artifact_sha256']='b'*64
    if kind=='duplicate':rows['val'].append(copy.deepcopy(rows['val'][0]))
    if kind=='source_split':rows['train'][4]['split']='test'
    if kind=='unknown_emotion':rows['train'][4]['emotion']=10
    if kind=='unsafe_path':rows['train'][4]['artifact']='../test.npz'
    if kind=='missing_level':rows['train']=[r for r in rows['train'] if (r['emotion'],r['intensity'])!=(2,2)]
    with pytest.raises(ValueError):runner.build_manifest(base,rows,{'train':'x','val':'y'},'z')


def test_cli_reads_no_test_metadata_or_native_artifact_and_refuses_overwrite(tmp_path,monkeypatch):
    base,rows=fixture();native=tmp_path/'native';native.mkdir()
    for role in ('train','val'):
        (native/f'{role}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows[role]),encoding='utf8')
    (native/'test.jsonl').write_text('THIS MUST NEVER BE OPENED',encoding='utf8')
    path=tmp_path/'base.json';path.write_text(json.dumps(base),encoding='utf8')
    original_open=Path.open
    def guarded(self,*args,**kwargs):
        assert self.name!='test.jsonl' and self.suffix!='.npz'
        return original_open(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    out=tmp_path/'expanded';runner.expand(native,path,out)
    result=json.loads((out/'manifest.json').read_text(encoding='utf8'))
    validate_manifest(result)
    assert result['roles']['test']==base['roles']['test']
    with pytest.raises(FileExistsError):runner.expand(native,path,out)
