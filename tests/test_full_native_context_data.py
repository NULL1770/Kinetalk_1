"""Synthetic native context contracts; no saved experiment/test split is read."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import full_native_context_data as d


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fixture(tmp_path, *, poison_invalid=False):
    native = tmp_path/'native'; native.mkdir()
    delta = tmp_path/'delta'; (delta/'clips').mkdir(parents=True)
    rows, qs, payloads = [], [], []
    generator = torch.Generator().manual_seed(31)
    for i, (length, start) in enumerate(((120, 13), (83, 0))):
        valid = torch.ones(length, dtype=torch.bool); valid[5] = False
        if i == 0: valid[44] = False
        content = torch.randn(length, 768, generator=generator)
        motion = torch.rand(length, 52, generator=generator)
        audio = torch.randn(length, 83, generator=generator)
        middle = torch.randn(length, 768, generator=generator).half()
        prosody = torch.randn(length, 4, generator=generator)
        times = torch.arange(length, dtype=torch.float64)/25
        features = torch.cat((content, middle.float(), prosody), -1)
        cid = 'clip'+str(i)
        provenance = {'schema':'native_affect_style_v4.1','clock_evidence':'embedded_video','fps':25.,
                      'audio_path':'bound-unused.wav','audio_sha256':'wave'+str(i),'audio_offset_s':0.}
        arrays = {'motion':motion.numpy(),'content':content.numpy(),'audio':audio.numpy(),
                  'times':times.numpy(),'mask':valid.numpy(),'channel_mask':np.ones(52,dtype=bool),
                  'provenance':np.asarray(json.dumps(provenance))}
        if poison_invalid:
            for key in ('motion','content','audio'): arrays[key][~valid.numpy()] = np.nan
        path = native/(cid+'.npz'); np.savez(path,**arrays)
        rows.append({'clip_id':cid,'split':'train','artifact':path.name,'artifact_sha256':d.sha(path),
                     'sentence':'sentence'+str(i),'speaker':'person'+str(i),'speaker_id':i,'emotion':i})
        count=min(96,length-start)
        q={'clip_id':cid,'sentence_id':'sentence'+str(i),'speaker':'person'+str(i),
           'speaker_id':torch.tensor(i),'emotion_id':torch.tensor(i),'intensity_id':torch.tensor(0),
           'channel_mask':torch.ones(52,dtype=torch.bool),'anchors':torch.zeros(52),'anchor_valid':torch.ones(52,dtype=torch.bool),
           'valid':torch.zeros(96,dtype=torch.bool),'times':times[start]+torch.arange(96,dtype=torch.float64)/25,
           'h0':torch.randn(96,7,generator=generator),'b0':torch.randn(96,52,generator=generator),
           'target_intensity':torch.randn(96,3,generator=generator),'target_intensity_valid':torch.ones(96,3,dtype=torch.bool),
           'audio_global':torch.randn(4,generator=generator)}
        q['valid'][:count]=valid[start:start+count]
        for key,value in (('motion',motion.half()),('content',content),('audio',audio.half()),('audio_features',features)):
            q[key]=value.new_zeros(96,value.shape[-1]);q[key][:count]=value[start:start+count]
            if key=='audio_features':q[key][~q['valid']]=0
        qs.append(q)
        extra=valid.clone();extra[start:start+count]=False;ix=extra.nonzero(as_tuple=True)[0]
        payloads.append({'schema':d.SCHEMA,'clip_id':cid,'role':'train','sentence_id':q['sentence_id'],
            'native_frames':length,'center_start':start,'center_length':96,'native_indices':ix,
            'middle':middle[ix],'prosody':prosody[ix],'binding':{'native_artifact_sha256':rows[-1]['artifact_sha256'],
            'wave_sha256':provenance['audio_sha256'],'center_features_sha256':d.tensor_sha(q['audio_features']),
            'center_valid_sha256':d.tensor_sha(q['valid']),'center_times_sha256':d.tensor_sha(q['times'])},
            'overlap_check':{key:{'max_abs':0.,'rms':0.,'changed_fraction':0.,'exact':True,'count':int(q['valid'].sum())}
                             for key in ('content','middle','prosody')}})
    manifest=tmp_path/'native.jsonl';manifest.write_text('\n'.join(json.dumps(row) for row in rows))
    source={'audio_sha256':'old-audio','manifest_sha256':d.sha(manifest),'model':{'model_dir':'old-location','code_sha':'pinned'},
            'geometry':{'hop':320},'layers':[2,4,6]}
    index={'schema':d.SCHEMA,'source':source,'config':{'middle_max_atol':.002,'middle_rms_atol':.0001,
        'prosody_max_atol':.000001,'rtol':0,'normalization_recomputed':False,'storage':{'middle':'float16','prosody':'float32'}},
        'selected_roles':{'train':[x['clip_id'] for x in rows],'validation':[]},'clips':{},'status':'complete'}
    for payload in payloads:
        payload['binding'].update(old_audio_sha256=source['audio_sha256'],native_manifest_sha256=source['manifest_sha256'],
            model_sha256=d.canonical_hash({'code_sha':'pinned'}),geometry_sha256=d.canonical_hash(source['geometry']),layers=[2,4,6])
        file='clips/'+payload['clip_id']+'.pt';torch.save(payload,delta/file)
        index['clips'][payload['clip_id']]={key:payload[key] for key in ('role','sentence_id','native_frames','center_start','center_length')}
        index['clips'][payload['clip_id']].update(file=file,sha256=d.sha(delta/file),status='complete',extra_valid_frames=len(payload['native_indices']),
            native_artifact_sha256=payload['binding']['native_artifact_sha256'],wave_sha256=payload['binding']['wave_sha256'])
    write_index(delta,index)
    return d._stack(qs),native,manifest,delta,index,payloads


def write_index(delta,index):
    (delta/'manifest.json').write_text(json.dumps(index))
    (delta/'complete.json').write_text(json.dumps({'schema':d.SCHEMA,'status':'complete',
        'manifest_sha256':d.sha(delta/'manifest.json'),'selected_count':len(index['clips']),'completed_count':len(index['clips'])}))


def bind_payload(data,i=0):
    _,_,_,root,index,payloads=data
    entry=index['clips'][payloads[i]['clip_id']]
    torch.save(payloads[i],root/entry['file']);entry['sha256']=d.sha(root/entry['file']);write_index(root,index)


def store(data,**kw):return d.NativeContextStore(*data[:4],**kw)


def test_full_preserves_all_native_frames_clock_features_and_center_exact(tmp_path):
    data=fixture(tmp_path);s=store(data);q=data[0]
    b=s.batch([0,1],mode='full')
    assert b['valid'].shape==(2,128)
    assert b['native_lengths'].tolist()==[120,83] and b['center_starts'].tolist()==[13,0]
    assert torch.equal(s.native_lengths,b['native_lengths'])
    for i in range(2):
        start=int(s.center_starts[i]);count=min(96,int(s.native_lengths[i])-start)
        good=q['valid'][i,:count]
        assert torch.equal(b['audio_features'][i,start:start+count][good],q['audio_features'][i,:count][good])
        assert torch.equal(b['motion'][i,start:start+count].half()[good],q['motion'][i,:count][good])
    assert torch.count_nonzero(b['audio_features'][~b['valid']])==0
    assert torch.count_nonzero(b['motion'][~b['valid']])==0
    assert b['motion'].dtype==torch.float32 and b['times'].dtype==torch.float64
    assert torch.allclose(b['times'][:,1:]-b['times'][:,:-1],torch.full_like(b['times'][:,1:],.04),atol=1e-7,rtol=0)
    c=s.batch([1,0],mode='center')
    for key,v in q.items():
        if torch.is_tensor(v):assert torch.equal(c[key],v[[1,0]])
        else:assert c[key]==[v[1],v[0]]


def test_full_does_not_reuse_96frame_encodings_supervision_or_global(tmp_path):
    data=fixture(tmp_path);b=store(data).batch([0])
    for key in ('b0','h0','target_intensity','target_intensity_valid','audio_global'):
        assert key not in b
    assert torch.equal(b['anchors'],data[0]['anchors'][:1])
    assert torch.equal(b['speaker_id'],data[0]['speaker_id'][:1])


def test_invalid_poison_is_zeroed_not_interpolated_or_compressed(tmp_path):
    data=fixture(tmp_path,poison_invalid=True);b=store(data).batch([0,1])
    assert not b['valid'][0,5] and not b['valid'][0,44]
    for key in ('motion','content','audio','audio_features'):
        assert torch.isfinite(b[key]).all()
        assert not torch.count_nonzero(b[key][~b['valid']])
    assert b['times'][0,44]==44/25


def test_cached_loading_reads_only_authorized_clips_and_returns_safe_copies(tmp_path,monkeypatch):
    data=fixture(tmp_path);s=store(data);opened=[];old=np.load
    def track(path,*args,**kw):opened.append(str(path));return old(path,*args,**kw)
    monkeypatch.setattr(np,'load',track)
    first=s.clip(0);first['motion'].fill_(100)
    again=s.clip(0);s.batch([0]);s.batch([0],mode='center')
    assert len(opened)==1 and not torch.all(again['motion']==100)
    with pytest.raises(ValueError,match='allowlist'):s.clip('unapproved')


@pytest.mark.parametrize('case',['incomplete','hash','role','escape','extra_role','missing_entry'])
def test_rejects_incomplete_or_unauthorized_index_before_native_reads(tmp_path,monkeypatch,case):
    data=fixture(tmp_path);index=data[4]
    if case=='incomplete':index['status']='incomplete'
    elif case=='hash':index['source']['manifest_sha256']='wrong'
    elif case=='role':index['clips']['clip0']['role']='test'
    elif case=='escape':index['clips']['clip0']['file']='../outside.pt'
    elif case=='extra_role':index['selected_roles']['test']=[]
    else:index['clips'].pop('clip0')
    write_index(data[3],index)
    def forbidden(*a,**kw):raise AssertionError('native read before scope validation')
    monkeypatch.setattr(np,'load',forbidden)
    with pytest.raises(ValueError):store(data)


@pytest.mark.parametrize('case',['missing_extra','extra_invalid','center_replace','feature_bind','nan_overlap','large_overlap','wave_bind','wrong_storage'])
def test_rejects_delta_corruption_even_with_updated_file_hash(tmp_path,case):
    data=fixture(tmp_path);p=data[5][0]
    if case=='missing_extra':p['native_indices']=p['native_indices'][:-1];p['middle']=p['middle'][:-1];p['prosody']=p['prosody'][:-1]
    elif case=='extra_invalid':p['native_indices'][0]=5
    elif case=='center_replace':p['native_indices'][0]=20
    elif case=='feature_bind':p['binding']['center_features_sha256']='wrong'
    elif case=='nan_overlap':p['overlap_check']['middle']['rms']=float('nan')
    elif case=='large_overlap':p['overlap_check']['middle']['rms']=.0002
    elif case=='wave_bind':p['binding']['wave_sha256']='wrong'
    else:p['middle']=p['middle'].float()
    bind_payload(data)
    with pytest.raises(ValueError):store(data).clip(0)


def test_query_target_change_cannot_change_audio_features(tmp_path):
    data=fixture(tmp_path);a=store(data).clip(0)
    # Targets may not change silently: they are bound against native source.
    data[0]['motion'][0,10]+=1
    with pytest.raises(ValueError,match='storage differs'):store(data).clip(0)
    assert torch.isfinite(a['audio_features']).all()


def test_clock_and_native_bytes_cannot_change_silently(tmp_path):
    data=fixture(tmp_path)
    data[0]['times'][0]+=.04
    with pytest.raises(ValueError,match='clock'):store(data).clip(0)


def test_modified_native_bytes_rejected_before_decoding(tmp_path):
    data=fixture(tmp_path)
    with (data[1]/'clip0.npz').open('ab') as stream:stream.write(b'tampered')
    with pytest.raises(ValueError,match='SHA256'):store(data).clip(0)


def test_smoke_subset_still_keeps_exact_authorized_role(tmp_path):
    data=fixture(tmp_path);q=data[0]
    data=( {k:v[1:] if torch.is_tensor(v) else v[1:] for k,v in q.items()},*data[1:] )
    s=store(data,role='train');assert s.clip_ids==['clip1']
    with pytest.raises(ValueError,match='role'):store(data,role='validation')


def precompute_fixture(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts import prepare_full_native_audio_delta as p
    data=fixture(tmp_path);q,native,manifest,_,_,_=data
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    full={};records=[]
    for i,row in enumerate(rows):
        path=native/row['artifact']
        with np.load(path,allow_pickle=False) as z:arrays={k:np.asarray(z[k]).copy() for k in z.files}
        wave=tmp_path/(row['clip_id']+'.wav');wave.write_bytes(b'synthetic-pinned-wave')
        prov=json.loads(str(arrays['provenance'].item()));prov.update(audio_path=str(wave.resolve()),audio_sha256=d.sha(wave))
        arrays['provenance']=np.asarray(json.dumps(prov));np.savez(path,**arrays)
        row['artifact_sha256']=d.sha(path)
        length=len(arrays['times']);start=13 if i==0 else 0;count=min(96,length-start)
        f=torch.zeros(length,1540);f[:,:768]=torch.from_numpy(arrays['content'])
        f[start:start+count]=q['audio_features'][i,:count]
        full[row['clip_id']]={'middle':f[:,768:1536].half(),'prosody':f[:,1536:].float()}
        records.append({'clip_id':row['clip_id'],'split':'train' if i==0 else 'validation',
            'artifact_sha256':row['artifact_sha256'],'crop_start':start,'native_frames':length,
            'crop_in_range_frames':count,'wave_path':str(wave.resolve()),'wave_sha256':d.sha(wave),'audio_offset_s':0.})
    manifest.write_text('\n'.join(json.dumps(row) for row in rows))
    model={key:'bound-'+key for key in p.MODEL_KEYS};model.update(model_dir='different-directory',label='中文')
    geometry={'hop_samples':320}
    audio={'schema':'label_guided_audio_cache_v1','splits':{},'records':records,
           'feature_stats':{'untouched':True},'provenance':{'model':model,'geometry':geometry,'layers':[2,4,6],
               'source_sha256':{name:d.sha(Path(p.__file__).with_name(name)) for name in
                                ('extract_predictable_audio.py','extract_emotion2vec_pilot.py')},
               'native_train_manifest_sha256':d.sha(manifest)}}
    for i,role in enumerate(('train','validation')):
        audio['splits'][role]={'features':q['audio_features'][i:i+1],'valid':q['valid'][i:i+1],
            'times':q['times'][i:i+1],'clip_id':[q['clip_id'][i]],'sentence_id':[q['sentence_id'][i]]}
    audio_path=tmp_path/'audio.pt';torch.save(audio,audio_path)
    args=SimpleNamespace(audio=audio_path,native_manifest=manifest,native_root=native,model_dir=tmp_path/'model',
                         output=tmp_path/'new_delta',max_clips=2,clip_ids=None,device='cpu')
    monkeypatch.setattr(p,'load_extractor',lambda *a:(SimpleNamespace(blocks=list(range(12))),geometry,{**model,'model_dir':'new-location'}))
    calls=[]
    def extractor(clip,*unused):
        assert set(clip)=={'clip_id','sentence_id','valid','times','metadata'}
        assert 'motion' not in clip and 'audio' not in clip
        calls.append(clip['clip_id'])
        return {'clip_id':clip['clip_id'],'sentence_id':clip['sentence_id'],'valid':clip['valid'],'times':clip['times'],
                **full[clip['clip_id']],'record':{'wave_sha256':clip['metadata']['provenance']['audio_sha256'],'audio_offset_s':0.}}
    monkeypatch.setattr(p,'extract',extractor)
    return p,args,q,calls


def test_precompute_end_to_end_resume_and_store_are_compatible(tmp_path,monkeypatch):
    p,args,q,calls=precompute_fixture(tmp_path,monkeypatch)
    result=p.prepare(args)
    assert result['status']=='complete' and calls==['clip0','clip1']
    assert result['config']['normalization_recomputed'] is False
    p.prepare(args);assert calls==['clip0','clip1']
    for i,role in enumerate(('train','validation')):
        part={k:v[i:i+1] for k,v in q.items()}
        s=d.NativeContextStore(part,args.native_root,args.native_manifest,args.output,role=role)
        b=s.batch([0]);assert b['native_lengths'].item()==(120 if i==0 else 83)
        assert s.batch([0],mode='center')['clip_id']==[q['clip_id'][i]]


@pytest.mark.parametrize('dtype', [np.float16, np.float32])
def test_native_content_historical_storage_promotes_exactly(dtype):
    from scripts.prepare_full_native_audio_delta import validate_native_arrays
    rng = np.random.default_rng(4)
    content = rng.normal(size=(125, 768)).astype(dtype)
    times = np.arange(125, dtype=np.float64) * .04
    mask = np.ones(125, dtype=bool); mask[27] = False
    start = 13
    features = torch.zeros(96, 1540)
    features[:, :768] = torch.from_numpy(content[start:start + 96].astype(np.float32))
    valid = torch.from_numpy(mask[start:start + 96])
    provenance = {'schema': 'native_affect_style_v4', 'clock_evidence': 'embedded_video', 'fps': 25}
    assert validate_native_arrays(content, times, mask, provenance, features,
                                  torch.from_numpy(times[start:start + 96]), valid) == start
    features[0, 0] += .001
    with pytest.raises(ValueError, match='content differs'):
        validate_native_arrays(content, times, mask, provenance, features,
                               torch.from_numpy(times[start:start + 96]), valid)


def test_precompute_never_reads_native_motion_or_audio(tmp_path,monkeypatch):
    p,args,_,_=precompute_fixture(tmp_path,monkeypatch)
    original=np.load
    class ReadGuard:
        def __init__(self,z):self.z=z
        def __enter__(self):return self
        def __exit__(self,*a):self.z.close()
        def __getitem__(self,key):
            assert key not in ('motion','audio','channel_mask'),'target/native audio read by extractor'
            return self.z[key]
    monkeypatch.setattr(np,'load',lambda *a,**kw:ReadGuard(original(*a,**kw)))
    p.prepare(args)


@pytest.mark.parametrize('problem',['middle_max','middle_rms','prosody','model','wrapper_source','scope','orphan'])
def test_precompute_fail_closed_for_source_overlap_scope_and_partial_files(tmp_path,monkeypatch,problem):
    p,args,q,calls=precompute_fixture(tmp_path,monkeypatch)
    if problem=='model':
        old=p.load_extractor
        def altered(*a):
            model,geometry,info=old(*a);info['frontend_source_sha256']='wrong';return model,geometry,info
        monkeypatch.setattr(p,'load_extractor',altered)
    elif problem=='wrapper_source':
        audio=torch.load(args.audio,weights_only=False)
        audio['provenance']['source_sha256']['extract_predictable_audio.py']='wrong'
        torch.save(audio,args.audio)
    elif problem=='scope':
        args.max_clips=None;args.clip_ids=tmp_path/'ids.json';args.clip_ids.write_text('["unapproved"]')
    elif problem=='orphan':
        p.prepare(args);(args.output/'orphan.pt').write_bytes(b'orphan')
    else:
        old=p.extract
        def altered(*a):
            out=old(*a)
            if problem=='prosody':out['prosody']=out['prosody']+.00001
            elif problem=='middle_max':out['middle'][13,0]=out['middle'][13,0]+.1
            else:out['middle']=out['middle']+.0005
            return out
        monkeypatch.setattr(p,'extract',altered)
    with pytest.raises(ValueError):p.prepare(args)
    if problem!='orphan':assert not (args.output/'complete.json').exists()


def test_precompute_smoke_allowlist_rejects_duplicates_and_formal_wrong_count(tmp_path,monkeypatch):
    p,args,_,_=precompute_fixture(tmp_path,monkeypatch)
    args.max_clips=None
    with pytest.raises(ValueError,match='2315'):p.prepare(args)
    args.clip_ids=tmp_path/'ids.json';args.clip_ids.write_text('["clip0","clip0"]')
    with pytest.raises(ValueError,match='unique'):p.prepare(args)
    args.clip_ids.write_text('["clip1"]')
    out=p.prepare(args)
    assert out['selected_roles']=={'train':[],'validation':['clip1']}


def test_precompute_incomplete_committed_records_can_resume(tmp_path,monkeypatch):
    p,args,q,calls=precompute_fixture(tmp_path,monkeypatch);original=p.extract
    def interrupted(clip,*rest):
        if clip['clip_id']=='clip1':raise RuntimeError('synthetic interruption')
        return original(clip,*rest)
    monkeypatch.setattr(p,'extract',interrupted)
    with pytest.raises(RuntimeError,match='interruption'):p.prepare(args)
    assert not (args.output/'complete.json').exists()
    with pytest.raises(ValueError,match='complete'):
        d.NativeContextStore({k:v[:1] for k,v in q.items()},args.native_root,args.native_manifest,args.output)
    monkeypatch.setattr(p,'extract',original)
    out=p.prepare(args);assert out['status']=='complete' and calls==['clip0','clip1']


def test_future_native_targets_do_not_enter_or_change_audio_features(tmp_path):
    data=fixture(tmp_path);original=store(data).clip(0)['audio_features']
    q,native,manifest,delta,index,payloads=data
    path=native/'clip0.npz'
    with np.load(path,allow_pickle=False) as z:arrays={k:np.asarray(z[k]).copy() for k in z.files}
    # Change only target frames after the center ends; transparently rebind
    # native/delta lineage. No target statistic is fed into feature assembly.
    arrays['motion'][115:]+=100
    np.savez(path,**arrays)
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]['artifact_sha256']=d.sha(path)
    manifest.write_text('\n'.join(json.dumps(row) for row in rows))
    index['source']['manifest_sha256']=d.sha(manifest)
    index['clips']['clip0']['native_artifact_sha256']=d.sha(path)
    for i,payload in enumerate(payloads):
        payload['binding']['native_manifest_sha256']=d.sha(manifest)
        if i==0:payload['binding']['native_artifact_sha256']=d.sha(path)
        bind_payload(data,i)
    changed=store(data).clip(0)
    assert torch.equal(original,changed['audio_features'])
    assert changed['motion'][115:].min()>99
