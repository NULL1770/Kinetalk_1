import copy
import json
import pytest
import torch
from scripts.run_isolated_state_queue import (assess_state,assess_residual,checked_rows,
    validate_stage_complete,check_support,RESIDUAL_CHECKS,STATE_CHECKS)
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.extract_emotion2vec_pilot import sha
from scripts.train_isolated_residual import SEEDS,MODES


def setup(tmp_path):
    ids=['a','b','c']
    names=('mean_no_worse','mouth_preserved','mbe_no_worse','internal_emotion_no_large_drop',
           'train_state_fit','val_state_better_than_static','val_state_better_than_reverse')
    for arm,value in [('audio',.1),('static',.2)]:
        dest=tmp_path/('state_'+arm);dest.mkdir()
        r={'mode':arm,'clips':len(ids),'test_loaded':False,
            'checks':{k:True for k in names},'per_clip_state_mse':{'full':[
            {'clip_id':c,'sentence':c,'mse':value,'valid_frames':10,'valid_runs':1} for c in ids]}}
        if arm=='static':r['checks']['train_state_fit']=False
        (dest/'evaluation.json').write_text(json.dumps(r))
    return ids


def test_state_acceptance_requires_audio_fit_and_static_pair_not_static_fit(tmp_path):
    ids=setup(tmp_path);result=assess_state(tmp_path,ids)
    assert result['passed'] and result['independent_static']['delta']==pytest.approx(-.1)
    p=tmp_path/'state_audio/evaluation.json';r=json.loads(p.read_text());r['checks']['train_state_fit']=False
    p.write_text(json.dumps(r));assert not assess_state(tmp_path,ids)['passed']


@pytest.mark.parametrize('kind',['duplicate','missing','nonfinite','badverdict'])
def test_gate_rejects_incomplete_or_invalid_evidence(tmp_path,kind):
    ids=setup(tmp_path);p=tmp_path/'state_audio/evaluation.json';r=json.loads(p.read_text())
    rows=r['per_clip_state_mse']['full']
    if kind=='duplicate':rows[1]=copy.deepcopy(rows[0])
    elif kind=='missing':rows.pop()
    elif kind=='nonfinite':rows[0]['mse']=float('nan')
    else:r['checks']['mean_no_worse']=1
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):assess_state(tmp_path,ids)


def setup_residual(tmp_path):
    ids=['a','b','c'];keys=[f'{s}/{m}' for s in SEEDS for m in MODES]
    for arm in ('audio','static'):
        dest=tmp_path/('residual_'+arm);(dest/'scores/residual').mkdir(parents=True)
        report={'mode':arm,'clips':len(ids),'test_loaded':False,'protection_passed':True,
            'checks':{key:True for key in RESIDUAL_CHECKS},
            'mouth_protection':{key:{'passed':True} for key in keys},
            'nonupper43_exact':{key:True for key in keys},
            'generated_teacher_accuracy_nonindependent':{key:.7 for key in keys},
            'max_absolute_upper_mean_drift':{key:1e-8 for key in ('full','static','reverse')}}
        benchmark={m:{'prediction_keys':[f'{s}/{m}' for s in SEEDS],
            'coefficient':{'arkit_mbe':{'value':.5}}} for m in MODES}
        (dest/'evaluation.json').write_text(json.dumps(report))
        (dest/'benchmark_summary.json').write_text(json.dumps(benchmark))
        for mode in ('full','static','reverse'):
            value=.1 if arm=='audio' and mode=='full' else .2
            rows=[{'clip_id':cid,'sentence':cid,'sample_count':3,'valid_frames':10,
                'valid_runs':1,'scales':[1.]*9,'joint_fair_es':{'centered':value},
                'variogram':{'aggregate':value}} for cid in ids]
            (dest/f'scores/residual/temporal_{mode}.json').write_text(json.dumps(rows))
    return ids


def test_residual_full_comparison_passes(tmp_path):
    ids=setup_residual(tmp_path)
    assert assess_residual(tmp_path,ids)['quantitative_passed']


@pytest.mark.parametrize('kind',['checks_missing','check_unknown','frames','runs','draws','sentence','seeds','false_verdict'])
def test_residual_rejects_incomplete_or_mixed_support(tmp_path,kind):
    ids=setup_residual(tmp_path)
    if kind in ('checks_missing','check_unknown','false_verdict'):
        path=tmp_path/'residual_audio/evaluation.json';record=json.loads(path.read_text())
        if kind=='checks_missing':record['checks'].pop('frozen_models_unchanged')
        elif kind=='check_unknown':record['checks']={'other':True}
        else:record['checks']['mouth_preserved']=False
    elif kind=='seeds':
        path=tmp_path/'residual_audio/benchmark_summary.json';record=json.loads(path.read_text())
        record['full']['prediction_keys']=['42/full']
    else:
        path=tmp_path/'residual_static/scores/residual/temporal_full.json';record=json.loads(path.read_text())
        key={'frames':'valid_frames','runs':'valid_runs','draws':'sample_count','sentence':'sentence'}[kind]
        record[0][key]='other' if kind=='sentence' else record[0][key]+1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):assess_residual(tmp_path,ids)


def complete_stage_fixture(tmp_path):
    code=tmp_path/'code';(code/'scripts').mkdir(parents=True)
    source='scripts/example.py';(code/source).write_text('x=1\n');digest=sha(code/source)
    out=tmp_path/'runs/state_audio';(out/'source/scripts').mkdir(parents=True)
    (out/'source'/source).write_bytes((code/source).read_bytes())
    ids=['a','b'];lengths=[4,3];frames=7
    plan={'source_sha256':'source','manifest_sha256':'manifest','data_index_sha256':'index',
        'state_epochs':2,'residual_epochs':2,'train_clips':5,'validation_ids':ids,
        'validation_metadata':{'a':{'frames':4,'valid_frames':3,'sentence':'one'},
            'b':{'frames':3,'valid_frames':3,'sentence':'two'}},'sources':{source:digest}}
    protocol={'schema':'isolated_mean_state_v1','source_sha256':'source','mode':'audio','epochs':2,
        'smoke':False,'test_loaded':False,'default_replaced':False,'train_clips':5,'validation_clip_ids':ids,
        'noise_seeds':list(SEEDS),'seed':47,'batch_size':16,'selection':'fixed final epoch',
        'data':{'manifest_sha256':'manifest','index_sha256':'index'},'sources':{source:digest}}
    (out/'protocol.json').write_text(json.dumps(protocol));ph=canonical_hash(protocol)
    torch.save({'protocol':protocol,'protocol_sha256':ph,'test_loaded':False},out/'final.pt')
    torch.save({'epoch':2,'protocol_sha256':ph},out/'last.pt')
    curves={'compaction_schema':'native_curve_pack_v1','clip_id':ids,'noise_seeds':list(SEEDS),
        'native_lengths':lengths,'valid':torch.tensor([True,False,True,True,True,True,True]),
        'target':torch.zeros(frames,52),'channel_mask':torch.ones(2,52,dtype=torch.bool),
        'predictions':{f'{s}/{m}':torch.zeros(frames,52) for s in SEEDS for m in ('full','base','static','reverse')}}
    torch.save(curves,out/'native_curves.pt')
    status={'status':'complete','test_loaded':False,'final_sha256':sha(out/'final.pt'),
        'native_curves_sha256':sha(out/'native_curves.pt')}
    (out/'status.json').write_text(json.dumps(status))
    (out/'evaluation.json').write_text(json.dumps({'mode':'audio','clips':2,'test_loaded':False}))
    return out,plan,code


def test_completed_stage_verification_can_be_safely_repeated(tmp_path):
    out,plan,code=complete_stage_fixture(tmp_path)
    support=validate_stage_complete(out,'state_audio',plan,code)
    assert support['a']['valid_runs']==2 and support['a']['valid_frames']==3
    assert validate_stage_complete(out,'state_audio',plan,code)==support


@pytest.mark.parametrize('kind',['source','hash','epoch','scope','native','seed','smoke','report_after_verification'])
def test_completed_skip_rejects_changed_or_unbound_artifacts(tmp_path,kind):
    out,plan,code=complete_stage_fixture(tmp_path)
    if kind=='source':(code/'scripts/example.py').write_text('x=2\n')
    elif kind=='hash':(out/'final.pt').write_bytes(b'corrupted')
    elif kind=='epoch':torch.save({'epoch':1,'protocol_sha256':canonical_hash(json.loads((out/'protocol.json').read_text()))},out/'last.pt')
    elif kind in ('scope','smoke'):
        p=out/'protocol.json';r=json.loads(p.read_text());r['train_clips' if kind=='scope' else 'smoke']=1;p.write_text(json.dumps(r))
    elif kind in ('native','seed'):
        p=out/'native_curves.pt';r=torch.load(p,weights_only=False)
        if kind=='native':r['valid'][1]=True
        else:r['predictions'].pop('42/full')
        torch.save(r,p);s=json.loads((out/'status.json').read_text());s['native_curves_sha256']=sha(p)
        (out/'status.json').write_text(json.dumps(s))
    else:
        validate_stage_complete(out,'state_audio',plan,code)
        p=out/'evaluation.json';r=json.loads(p.read_text());r['modified']=True;p.write_text(json.dumps(r))
    with pytest.raises(ValueError):validate_stage_complete(out,'state_audio',plan,code)
