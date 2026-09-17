"""Training storage, lineage, paired exposure and actual optimizer resume."""
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scripts import train_native_context as trainer
from scripts import train_audio_prefix_adaptation as adaptation
from scripts.native_context_runtime import paired_noise


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_query_subset_retains_quantized_storage_then_casts_model_inputs():
    original = torch.tensor([.123456, .654321], dtype=torch.float32)
    q = {'valid':torch.ones(2, 3, dtype=torch.bool),
         'motion':original[:,None,None].expand(2,3,52).half(),
         'audio':torch.randn(2,3,83).half(), 'content':torch.randn(2,3,768),
         'times':torch.arange(3, dtype=torch.float64)[None].expand(2,3)/25,
         'clip_id':['a','b'], 'nested':{'independent':'metadata'}}
    selected = trainer.subset_query(q, [1,0])
    assert selected['motion'].dtype == torch.float16
    assert selected['audio'].dtype == torch.float16
    assert selected['times'].dtype == torch.float64
    assert selected['clip_id'] == ['b','a']
    assert torch.equal(original[[1,0]].half(), selected['motion'][:,0,0])
    assert not torch.equal(original[[1,0]], selected['motion'][:,0,0].float())
    model_batch = trainer.cast_batch(selected, 'cpu')
    assert model_batch['motion'].dtype == model_batch['audio'].dtype == torch.float32
    assert model_batch['times'].dtype == torch.float64
    selected['motion'][0].zero_()
    assert q['motion'][1].count_nonzero()


@pytest.mark.parametrize('ids', [[], [-1], [2], [0,0], [[0]]])
def test_query_subset_rejects_invalid_membership(ids):
    with pytest.raises(ValueError, match='indices'):
        trainer.subset_query({'valid':torch.ones(2,3,dtype=torch.bool)}, ids)


@pytest.mark.parametrize('flag', [True, None, 'false'])
def test_formal_delta_policy_rejects_smoke_or_missing_nested_flag(flag):
    store = SimpleNamespace(index={'smoke':False, 'config':{'smoke':flag}})
    with pytest.raises(ValueError, match='Formal'):
        trainer.validate_delta_policy({'fit':store}, False)
    trainer.validate_delta_policy({'fit':store}, True)


def test_formal_delta_policy_uses_nested_false():
    trainer.validate_delta_policy({'fit':SimpleNamespace(index={'config':{'smoke':False}})}, False)


def test_only_allowlisted_full_encoding_fields_are_attached_in_query_order():
    rows = [{'h0':torch.full((length,3), float(i)),
             'audio_global':torch.ones(2)*i, 'audio_intensity':torch.tensor([i]),
             'static_upper':torch.ones(9)*i, 'motion':torch.ones(length,52)*999,
             'emotion_id':i+1000} for i,length in enumerate((11,5))]
    batch = {'valid':torch.ones(2,16,dtype=torch.bool), 'native_lengths':torch.tensor([5,11]),
             'motion':'original training target', 'emotion_id':'original labels'}
    out = trainer.attach_encoded(batch, [1,0], rows)
    assert out['motion'] == batch['motion'] and out['emotion_id'] == batch['emotion_id']
    assert out['h0'][0,:5].eq(1).all() and out['h0'][0,5:].eq(0).all()
    assert out['h0'][1].eq(0).all()
    assert set(out)-set(batch) == set(trainer.ENCODED)
    rows[1]['h0'] = torch.zeros(6,3)
    with pytest.raises(ValueError, match='native clip length'):
        trainer.attach_encoded(batch, [1,0], rows)


class Local(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(3,4)
        self.blocks = nn.Sequential(nn.Linear(4,4), nn.SiLU())
        self.local_head = nn.Linear(4,2)
        self.global_head = nn.Linear(4,2)
        self.state_head = nn.Linear(4,2)
        self.register_buffer('feature_mean', torch.zeros(3))

    def forward(self, values):
        return self.local_head(self.blocks(self.input(values)))


def payload_fixture():
    local = Local()
    draw = 'a'*64
    row = {'arm':'center96', 'epoch':1, 'updates':2, 'total_steps':2, 'draw_sha256':draw}
    return {'schema':trainer.SCHEMA,'arm':'center96','recipe_sha256':'recipe',
            'completed_epochs':1,'draws':[draw],'epochs':[row],'total_steps':2,
            'upper':{},'local':local.state_dict(),'optimizer':{},'rng':{},
            'frozen':{'system':'s','audio':'a','local':'l'},'scales':torch.ones(9)}, local


@pytest.mark.parametrize('damage', ['recipe','epoch','updates','draw','steps','optimizer','frozen','scales','protected'])
def test_resume_rejects_changed_lineage_or_incomplete_epoch(damage):
    payload, local = payload_fixture()
    before = trainer.p.state_hash(adaptation.protected_local_state(local))
    if damage == 'recipe': payload['recipe_sha256']='wrong'
    elif damage == 'epoch': payload['epochs'][0]['epoch']=2
    elif damage == 'updates': payload['epochs'][0]['updates']=1
    elif damage == 'draw': payload['draws'][0]='b'*64
    elif damage == 'steps': payload['total_steps']=3
    elif damage == 'optimizer': del payload['optimizer']
    elif damage == 'frozen': payload['frozen']['audio']='changed'
    elif damage == 'scales': payload['scales'][0]=2
    else: payload['local']['state_head.bias']=payload['local']['state_head.bias']+1
    with pytest.raises(ValueError):
        trainer.verify_resume(payload,'recipe','center96',2,
            frozen={'system':'s','audio':'a','local':'l'},scales=torch.ones(9),
            protected_local_sha256=before,expected_updates=2)


def test_actual_optimizer_and_all_rng_resume_matches_uninterrupted(tmp_path):
    torch.manual_seed(101);random.seed(101);np.random.seed(101)
    local = adaptation.configure_local(Local(), True)
    upper = nn.Linear(2,1)
    initial_local = copy.deepcopy(local.state_dict())
    initial_upper = copy.deepcopy(upper.state_dict())
    protected = trainer.p.state_hash(adaptation.protected_local_state(local))

    def modules():
        loc = adaptation.configure_local(Local(), True);loc.load_state_dict(initial_local)
        up = nn.Linear(2,1);up.load_state_dict(initial_upper)
        params = list(up.parameters())+[p for p in loc.parameters() if p.requires_grad]
        return loc,up,params,torch.optim.AdamW(params,lr=.001,weight_decay=1e-5)

    def run_epoch(loc,up,params,opt,gen):
        order = torch.randperm(7,generator=gen)
        values=[]
        for ids in order.split(4):
            noise,times,draw=paired_noise(torch.tensor([37,120,155,96,45,101,121])[ids],
                torch.tensor([0,12,29,0,0,3,14])[ids],gen,mode='center')
            values.append((ids.clone(),draw['full_noise'].clone(),times.clone()))
            # Include every captured RNG and real AdamW moments in the update.
            x=noise[:,0,:3]+torch.randn(len(ids),3)*.01
            x=x+random.random()*.01+float(np.random.random())*.01
            loss=(up(loc(x)).squeeze(-1)-times).square().mean()
            trainer.p.r.optimize(loss,opt,params)
        return values

    loc,up,params,opt = modules()
    gen=torch.Generator().manual_seed(97)
    run_epoch(loc,up,params,opt,gen)
    checkpoint={'local':loc.state_dict(),'upper':up.state_dict(),'optimizer':opt.state_dict(),
                'rng':trainer.p.capture_rng(gen)}
    trainer.p.save_checkpoint(tmp_path/'last.pt',checkpoint)
    expected=run_epoch(loc,up,params,opt,gen)
    expected_state={key:value.clone() for key,value in loc.state_dict().items()}
    expected_upper={key:value.clone() for key,value in up.state_dict().items()}
    resumed_loc,resumed_up,resumed_params,resumed_opt=modules()
    saved=trainer.load_pt(tmp_path/'last.pt')
    resumed_loc.load_state_dict(saved['local']);resumed_up.load_state_dict(saved['upper'])
    resumed_opt.load_state_dict(saved['optimizer'])
    resumed_gen=torch.Generator()
    trainer.restore_rng(saved['rng'],resumed_gen)
    actual=run_epoch(resumed_loc,resumed_up,resumed_params,resumed_opt,resumed_gen)
    for a,b in zip(expected,actual):
        for left,right in zip(a,b):assert torch.equal(left,right)
    for key,value in resumed_loc.state_dict().items():assert torch.equal(expected_state[key],value)
    for key,value in resumed_up.state_dict().items():assert torch.equal(expected_upper[key],value)
    assert trainer.p.state_hash(adaptation.protected_local_state(resumed_loc))==protected
    assert all(value.grad is None for name,value in resumed_loc.named_parameters()
               if name.startswith(('global_head.','state_head.')))


def complete_fixture(path):
    payload,local=payload_fixture()
    initial={'upper':'initial-upper','local':'initial-local'}
    matched={'initial':initial,'draws':payload['draws'],'updates':2}
    path.mkdir()
    for name in trainer.completed_files(1):(path/name).write_bytes(b'bound artifact')
    trainer.p.save_checkpoint(path/'last.pt',payload)
    trainer.p.save_json(path/'matched.json',matched)
    trainer.p.save_json(path/'complete.json',{'status':'complete','schema':trainer.SCHEMA,
        'recipe_sha256':'recipe','completed_epochs':1,'total_steps':2,'frozen':payload['frozen'],
        'files':{name:trainer.sha(path/name) for name in trainer.completed_files(1)}})
    args=(path,'recipe','center96',1,payload['frozen'],initial,
          trainer.p.state_hash(adaptation.protected_local_state(local)),2)
    return args,matched


def test_complete_resume_checks_all_bound_artifacts(tmp_path):
    args,matched=complete_fixture(tmp_path/'arm')
    assert trainer.verify_completed(*args)==matched


@pytest.mark.parametrize('name',['matched.json','step0_holdout.json','step0_holdout_upper9.pt',
    'hold_evaluation.json','hold_upper9.pt','dev_evaluation.json','dev_upper9.pt','last.pt','epoch001.json'])
def test_complete_resume_rejects_any_changed_output_not_only_final(tmp_path,name):
    args,_=complete_fixture(tmp_path/'arm')
    (args[0]/name).write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):
        trainer.verify_completed(*args)


def test_complete_resume_rejects_truncated_artifact_manifest(tmp_path):
    args,_=complete_fixture(tmp_path/'arm')
    done=trainer.read(args[0]/'complete.json');del done['files']['hold_upper9.pt']
    trainer.p.save_json(args[0]/'complete.json',done)
    with pytest.raises(ValueError,match='manifest'):
        trainer.verify_completed(*args)


def test_seed_specific_base_subset_preserves_bits_and_prediction_order():
    tensor=torch.randn(3,96,52,dtype=torch.float16)
    tensor[2,0,0]=-0.
    bases={'valid':torch.ones(3,96,dtype=torch.bool), 'target':tensor,
        'clip_id':['a','b','c'], 'schema':'old-schema', 'predictions':{'42':tensor.clone()}}
    actual=trainer.subset_bases(bases,torch.tensor([2,0]))
    assert actual['clip_id']==['c','a'] and actual['schema']=='old-schema'
    assert actual['target'].dtype==torch.float16 and torch.signbit(actual['target'][0,0,0])
    assert torch.equal(actual['predictions']['42'],tensor[[2,0]])
