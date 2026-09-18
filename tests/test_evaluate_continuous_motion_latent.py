import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from scripts import evaluate_continuous_motion_latent as e
from scripts.render_dynamic_rig_comparison import inspect_input


class StubAE(nn.Module):
    def __init__(self, block_size=5, latent_dim=16):
        super().__init__(); self.block_size=block_size; self.latent_dim=latent_dim
        self.encode_lengths=[]; self.decode_lengths=[]

    def encode(self, residual, valid):
        self.encode_lengths.append(residual.shape[1])
        assert valid.all()
        length=(residual.shape[1]+self.block_size-1)//self.block_size
        latent=residual.new_zeros(1,length,self.latent_dim)
        for index in range(length):
            latent[:,index,:9]=residual[:,index*self.block_size:(index+1)*self.block_size].mean(1)
        return latent,torch.ones(1,length,dtype=torch.bool,device=residual.device)

    def decode(self, latent, valid):
        self.decode_lengths.append(valid.shape[1])
        assert valid.all()
        return latent[:,:,:9].repeat_interleave(self.block_size,dim=1)[:,:valid.shape[1]]


class StubFlow(nn.Module):
    latent_dim=16
    block_size=5
    def __init__(self):super().__init__();self.calls=[]
    def sample(self,valid,context,audio_blocks,noise,steps=24,use_audio=True,*,audio_frame_valid=None):
        self.calls.append({'valid':valid.clone(),'audio':audio_blocks.clone(),'frame_valid':audio_frame_valid.clone(),
                           'noise':noise.clone(),'context':context.clone(),'use_audio':use_audio})
        assert torch.equal(valid,audio_frame_valid.any(-1))
        assert torch.count_nonzero(audio_blocks[~audio_frame_valid])==0
        contribution=audio_blocks[...,0].sum(-1)/audio_frame_valid.sum(-1)
        return noise*.02+(contribution[...,None]*.01 if use_audio else 0.)


def fixture():
    torch.manual_seed(219)
    frames=17
    valid=torch.ones(frames,dtype=torch.bool);valid[7:9]=False;valid[-1]=False
    motion_mask=valid[:,None].expand(-1,9).clone();motion_mask[3,0]=False
    target=torch.rand(frames,52)*.3+.2
    baseline=torch.rand(frames,52)*.2+.3
    features=torch.randn(frames,1540);features[~valid]=float('nan')
    clip={'clip_id':'clip_a','sentence':'s1','split':'holdout','speaker':0,'emotion':1,
          'valid':valid,'motion_mask':motion_mask,'motion9':target[:,e.UPPER].clone(),
          'target52':target,'baseline52':baseline,'b9':torch.full((9,),.3),
          'features':features,'context':torch.tensor([1.,2.,3.]),'times':torch.arange(frames,dtype=torch.float64)/25}
    stats={'residual_scale':torch.full((9,),.1),'metric_scale':torch.full((9,),.2),
           'latent_mean':torch.linspace(-.1,.1,16),'latent_scale':torch.full((16,),.3),
           'audio_mean':torch.zeros(1540),'audio_scale':torch.ones(1540),
           'context_mean':torch.tensor([.5,1.,1.5]),'context_scale':torch.full((3,),.5)}
    return clip,stats


def test_generation_does_not_read_motion_mask_or_targets_and_keeps_native_run_clock():
    clip,stats=fixture(); flow=StubFlow();ae=StubAE()
    only_audio={k:v for k,v in clip.items() if k not in ('motion_mask','motion9','target52')}
    output,records=e.generate_clip(flow,ae,only_audio,stats,seeds=(42,77),steps=2)
    assert output.shape==(2,17,9)
    assert ae.decode_lengths==[7,7,7,7]
    assert [r['start'] for r in records]==[0,0,9,9]
    assert [r['latent_frames'] for r in records]==[2,2,2,2]
    for call in flow.calls:
        assert call['frame_valid'].sum()==7
        assert call['frame_valid'][0,-1].tolist()==[True,True,False,False,False]
        torch.testing.assert_close(call['context'],torch.tensor([[1.,2.,3.]]))
    invalid=~clip['valid'].numpy()
    np.testing.assert_array_equal(output[:,invalid],np.broadcast_to(clip['baseline52'][~clip['valid']][:,e.UPPER].numpy(),output[:,invalid].shape))


def test_seed_pairing_is_reproducible_and_interventions_do_not_change_noise():
    clip,stats=fixture(); ae=StubAE();flow=StubFlow()
    a,ra=e.generate_clip(flow,ae,clip,stats,seeds=(42,77),steps=2)
    b,rb=e.generate_clip(StubFlow(),StubAE(),clip,stats,seeds=(42,77),steps=2)
    np.testing.assert_array_equal(a,b);assert ra==rb
    reverse=StubFlow();e.generate_clip(reverse,StubAE(),clip,stats,seeds=(42,77),steps=2,intervention='reverse')
    static=StubFlow();e.generate_clip(static,StubAE(),clip,stats,seeds=(42,77),steps=2,intervention='static')
    for original,rev,st in zip(flow.calls,reverse.calls,static.calls):
        assert torch.equal(original['noise'],rev['noise']) and torch.equal(original['noise'],st['noise'])
        source=original['audio'][original['frame_valid']]
        torch.testing.assert_close(rev['audio'][rev['frame_valid']],source.flip(0))
        flat=st['audio'][st['frame_valid']]
        torch.testing.assert_close(flat,source.mean(0).expand_as(flat),atol=1e-6,rtol=1e-6)


def test_ae_reconstruction_splits_joint_motion_runs_and_raw_composition_is_correct():
    clip,stats=fixture();ae=StubAE(block_size=1,latent_dim=16)
    samples,records,joint=e.reconstruct_clip(ae,clip,stats)
    assert ae.encode_lengths==[3,3,7]
    np.testing.assert_allclose(samples[0,joint],clip['motion9'][joint].numpy(),atol=5e-8)
    assert [r['start'] for r in records]==[0,4,9]


def test_saved_full52_npz_preserves_native_masks_and_protected_channels(tmp_path):
    clip,stats=fixture()
    result=e.evaluate_generation(StubFlow(),StubAE(),[clip],stats,tmp_path/'eval',seeds=(42,77),steps=2)
    assert result['status']=='needs_visual_review' and not result['naturalness_certified']
    assert result['numerical_gate']['passed'] and result['scored_clips']==1
    packed=torch.load(tmp_path/'eval/curves.pt',weights_only=False)
    raw=packed['clips']['clip_a']['samples']
    with np.load(tmp_path/'eval/npz/clip_a.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['valid'],clip['valid'].numpy())
        assert z['motions'].shape==(4,17,52)
        np.testing.assert_array_equal(z['motions'][2:,:,e.UPPER],raw)
        np.testing.assert_array_equal(z['motions'][2:,:,e.OTHER],np.broadcast_to(clip['baseline52'][:,e.OTHER].numpy(),(2,17,43)))
        assert not z['reference_valid'][3] and z['reference_display_baseline_filled'][3].all()
    inspect_input(tmp_path/'eval/npz/clip_a.npz',25)
    assert (tmp_path/'eval/index.html').is_file() and (tmp_path/'eval/svg/clip_a.svg').is_file()
    assert json.loads((tmp_path/'eval/render_jobs.json').read_text())['rendered'] is False


def test_motion_scoring_mask_change_never_changes_free_generation(tmp_path):
    clip,stats=fixture(); changed=copy.deepcopy(clip)
    changed['motion_mask'][1:5]=False;changed['motion9'][1:5]=float('nan')
    a,_=e.generate_clip(StubFlow(),StubAE(),clip,stats,seeds=(42,77),steps=2)
    b,_=e.generate_clip(StubFlow(),StubAE(),changed,stats,seeds=(42,77),steps=2)
    np.testing.assert_array_equal(a,b)
    ra=e.evaluate_generation(StubFlow(),StubAE(),[clip],stats,tmp_path/'a',seeds=(42,77),steps=2)
    rb=e.evaluate_generation(StubFlow(),StubAE(),[changed],stats,tmp_path/'b',seeds=(42,77),steps=2)
    assert ra['per_clip_scores'][0]['valid_frames']>rb['per_clip_scores'][0]['valid_frames']


def test_raw_out_of_bounds_are_reported_not_silently_clipped(tmp_path):
    clip,stats=fixture();clip['b9']=torch.full((9,),1.3)
    result=e.evaluate_generation(StubFlow(),StubAE(),[clip],stats,tmp_path/'out',seeds=(42,77),steps=2)
    assert result['numerics'][0]['raw_upper_out_of_range_fraction']>0
    assert result['numerics'][0]['prediction_clipped'] is False
    packed=torch.load(tmp_path/'out/curves.pt',weights_only=False)
    assert packed['clips']['clip_a']['samples'][:,clip['valid'].numpy()].max()>1
    # Out-of-range is reported separately; numerical success is not quality.
    assert result['naturalness_certified'] is False


def test_deterministic_reconstruction_artifact_is_explicitly_oracle(tmp_path):
    clip,stats=fixture()
    result=e.evaluate_ae(StubAE(block_size=1),[clip],stats,tmp_path/'ae')
    assert result['mode']=='ae_reconstruction' and 'not stochastic' in result['deterministic_score_note']
    assert result['summary']['joint_fair_es']['raw']<1e-6
    assert 'Oracle motion reconstruction only' in (tmp_path/'ae/index.html').read_text()


def test_real_model_tiny_smoke_respects_latent_tail_contract(tmp_path):
    from kinetalk_b0.models.continuous_upper_motion import ContinuousUpperAE,ContinuousLatentFlow
    clip,stats=fixture();torch.manual_seed(201)
    ae=ContinuousUpperAE(hidden=16,depth=1)
    flow=ContinuousLatentFlow(context_dim=3,hidden=16,depth=1)
    result=e.evaluate_generation(flow,ae,[clip],stats,tmp_path/'real',seeds=(42,77),steps=2)
    assert result['numerical_gate']['finite']


def test_fresh_output_and_fair_score_draw_count_required(tmp_path):
    clip,stats=fixture();out=tmp_path/'exists';out.mkdir();(out/'marker').write_text('x')
    with pytest.raises(FileExistsError):e.evaluate_generation(StubFlow(),StubAE(),[clip],stats,out)
    with pytest.raises(ValueError,match='two'):e.evaluate_generation(StubFlow(),StubAE(),[clip],stats,tmp_path/'single',seeds=(42,))


def test_mismatch_uses_only_within_split_audio_and_preserves_query_conditions(tmp_path):
    clip,stats=fixture(); donor=copy.deepcopy(clip)
    donor.update(clip_id='donor',sentence='different')
    donor['features'][donor['valid']]=2.
    donor['motion9'][:]=float('nan'); donor['target52'][:]=float('nan')
    changed,records,excluded=e._mismatched_audio([clip,donor])
    assert not excluded and records[0]['donor_clip_id']=='donor'
    torch.testing.assert_close(changed['clip_a']['features'][clip['valid']],torch.full((14,1540),2.))
    assert changed['clip_a']['context'] is clip['context'] and changed['clip_a']['b9'] is clip['b9']
    assert changed['clip_a']['motion_mask'] is clip['motion_mask']
    other_split=copy.deepcopy(donor);other_split['split']='train'
    assert not e._mismatched_audio([clip,other_split])[0]
    # Donor motion values never enter the feature replacement.
    donor['motion9'][:]=100.; donor['target52'][:]=100.
    again,_,_=e._mismatched_audio([clip,donor])
    torch.testing.assert_close(changed['clip_a']['features'],again['clip_a']['features'],equal_nan=True)
    donor['motion9']=clip['motion9'].clone();donor['target52']=clip['target52'].clone()
    result=e.evaluate_generation(StubFlow(),StubAE(),[clip,donor],stats,tmp_path/'mismatch',intervention='mismatch',seeds=(42,77),steps=2)
    assert result['clips']==2 and len(result['mismatch_donors'])==2
    assert result['numerical_gate']['passed']


def test_mismatch_without_donor_is_reported_as_unsupported(tmp_path):
    clip,stats=fixture()
    result=e.evaluate_generation(StubFlow(),StubAE(),[clip],stats,tmp_path/'none',intervention='mismatch',seeds=(42,77),steps=2)
    assert result['clips']==0 and result['mismatch_excluded_no_donor']==['clip_a']
    assert not result['numerical_gate']['passed'] and result['summary'] is None
