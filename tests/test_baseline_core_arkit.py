import torch
import pytest
from torch.nn import functional as F
from scripts.baseline_core_arkit import (VocaCoreARKit, EmoTalkCoreARKit,
                                       PairedUtterances, align_native, same_stride2)


def test_same_stride_matches_tensorflow_even_and_odd_padding():
    conv=torch.nn.Conv1d(1,1,3,stride=2,bias=False)
    conv.weight.data.fill_(1.)
    torch.testing.assert_close(same_stride2(torch.arange(4.).reshape(1,1,4),conv),torch.tensor([[[3.,5.]]]))
    torch.testing.assert_close(same_stride2(torch.arange(3.).reshape(1,1,3),conv),torch.tensor([[[1.,3.]]]))


@pytest.mark.parametrize('kind',['voca','emotalk'])
def test_padding_and_other_batch_items_do_not_change_native_prediction(kind):
    torch.set_num_threads(1);torch.manual_seed(18)
    model=VocaCoreARKit() if kind=='voca' else EmoTalkCoreARKit(max_seq_len=32,dropout=0.)
    head=model.expression if kind=='voca' else model.bs_map_r
    torch.nn.init.normal_(head.weight,std=.03)
    model.eval();audio=torch.randn(1,7,1540);anchor=torch.randn(1,52)
    valid=torch.ones(1,7,dtype=torch.bool)
    def pred(a,b,v):
        value=model(a,b,v);return value[0] if isinstance(value,tuple) else value
    with torch.no_grad():
        expected=pred(audio,anchor,valid)
        batch=torch.randn(2,11,1540);batch[0,:7]=audio[0]
        anchors=torch.cat((anchor,torch.randn_like(anchor)))
        mask=torch.ones(2,11,dtype=torch.bool);mask[0,7:]=False
        actual=pred(batch,anchors,mask)
    torch.testing.assert_close(actual[0,:7],expected[0],atol=1e-5,rtol=1e-5)


def test_paired_alignment_ignores_padding_and_gradients_are_finite():
    torch.set_num_threads(1);torch.manual_seed(2)
    model=EmoTalkCoreARKit(max_seq_len=32,dropout=0.)
    a=torch.randn(2,8,1540);v=torch.ones(2,8,dtype=torch.bool);v[0,5:]=False
    target_valid=torch.ones(2,9,dtype=torch.bool);target_valid[0,6:]=False
    changed=a.clone();changed[0,5:]=1e4
    torch.testing.assert_close(align_native(a,v,target_valid),align_native(changed,v,target_valid))
    y,e,l=model.paired(a,v,a,v,torch.randn(2,52),target_valid)
    loss=y.square().mean()+F.cross_entropy(e,torch.tensor([1,4]))+F.cross_entropy(l,torch.tensor([0,3]))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_pairing_uses_only_same_speaker_sentence_and_intensity_for_content():
    split=dict(speaker=['a','a','a','b'],sentence_id=['s','s','t','s'],
               emotion_id=torch.tensor([1,2,1,2]),intensity_id=torch.tensor([1,1,1,1]))
    pairs=PairedUtterances(split)
    c,e=pairs.sample(torch.arange(4),torch.Generator().manual_seed(42))
    assert c.tolist()==[1,0,2,3]
    assert e.tolist()==[2,1,0,3]
    assert pairs.audit['cross_emotion_content_available']==2


def test_core_matches_upstream_decoder_topology():
    v=VocaCoreARKit();e=EmoTalkCoreARKit(max_seq_len=32)
    assert [m.out_channels for m in v.convs]==[32,32,64,64]
    assert v.fc1.out_features==128 and v.fc2.out_features==50
    layer=e.transformer_decoder.layers[0]
    assert len(e.transformer_decoder.layers)==1
    assert layer.self_attn.embed_dim==832 and layer.self_attn.num_heads==4
    assert layer.linear1.out_features==832
    assert e.audio_feature_map_cont.out_features==512
    assert e.audio_feature_map_emo2.out_features==256


def test_final_table_rejects_legacy_simplified_test_result(tmp_path):
    from scripts.build_final_paper_tables import _require_scope
    with pytest.raises(ValueError,match='Legacy simplified'):
        _require_scope(tmp_path/'report.json',dict(method='emotalk',test_loaded=True),True)


def test_paired_trainer_completes_optimizer_step_and_checkpoint(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from scripts import baseline_training as trainer
    torch.set_num_threads(1)
    def split():
        return dict(audio_features=torch.randn(2,5,1540),motion=torch.randn(2,5,52),
            valid=torch.ones(2,5,dtype=torch.bool),anchors=torch.zeros(2,52),
            channel_mask=torch.ones(2,52,dtype=torch.bool),emotion_id=torch.tensor([0,1]),
            intensity_id=torch.tensor([1,1]),speaker=['a','a'],sentence_id=['s','s'])
    data=dict(splits={'train':split(),'validation':split()},provenance={})
    monkeypatch.setattr(trainer,'load_prepared',lambda *a,**k:data)
    monkeypatch.setattr(trainer,'evaluate_baseline',lambda *a,**k:dict(validation_mse=1.,validation_emotion_accuracy=.5))
    args=SimpleNamespace(threads=1,seed=42,data='unused',smoke=True,adapter='core_arkit_v1',
        device='cpu',lr=2e-4,batch_size=2,epochs=1,steps=None,output=tmp_path/'run',resume=False)
    model,report=trainer.train_baseline(args,'emotalk')
    assert report['steps']==1 and (args.output/'training_complete.json').is_file()
    assert model.bs_map_r.weight.abs().sum()>0
