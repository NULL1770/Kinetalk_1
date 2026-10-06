"""Source-audited VOCA/EmoTalk cores with explicit shared-audio ARKit adapters.

These are MEAD adaptations, not original encoder/dataset reproductions. See
docs/baseline_adaptation_20260923.md for the complete deviation contract.
Legacy simplified checkpoints keep their own classes and are never upgraded.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from scripts.faceformer_arkit_model import init_biased_mask

VERSION = 'core_arkit_v1'


def adaptation_record(method):
    shared = dict(version=VERSION, official_end_to_end_reproduction=False,
        audio='frozen shared 1540-D cache (content768 + emotion768 + prosody4)',
        output='ARKit52; score train-observed 51 channels',
        identity='independent neutral enrollment coefficients; no test speaker fitting',
        selection='fixed final epoch; no test selection')
    if method == 'voca':
        return dict(shared, label='VOCA-core (ARKit, shared audio)',
            upstream='TimoBolkart/voca@50bf785a880acd3e55b3e8f28eba2fc1e3c7fbfb',
            retained=['16-frame window', 'input normalization', 'identity at input and bottleneck',
                      'four SAME stride-2 ReLU convolutions 32/32/64/64',
                      'tanh FC128 -> expression50 -> linear residual + neutral template'],
            deviations=['shared audio replaces DeepSpeech29; first convolution input resized',
                        'neutral52 replaces categorical training-speaker condition',
                        'linear52 replaces FLAME mesh basis; no FLAME PCA initialization',
                        '25Hz native clock; masked coefficient MSE + 10*velocity MSE'])
    if method == 'emotalk':
        return dict(shared, label='EmoTalk-core (ARKit, shared audio)',
            upstream='psyai-net/EmoTalk_release/model.py and utils.py (source hashes in protocol)',
            retained=['separate content512 and emotion832->256 branches',
                      'content/emotion/level32/person32 concatenation to832',
                      'one 832-D four-head TransformerDecoder, FF832, causal periodic bias',
                      'frame-diagonal cross attention to emotion832',
                      'paired cross-emotion/content reconstruction and emotion classifier'],
            deviations=['frozen content768/emotion768 cache replaces two trainable wav2vec encoders',
                        'audio-predicted four-level condition replaces supplied two-level control',
                        'neutral52->32 replaces 24-person category embedding',
                        'native25Hz; no random blinking or smoothing postprocess',
                        'release lacks training runner: declared MSE/velocity/CE objective',
                        'unavailable paired combinations use self reconstruction, counted in protocol'])
    raise ValueError(method)


def same_stride2(x, conv):
    # TensorFlow SAME: for even lengths/kernel3, left=0 right=1.
    needed = max(((x.shape[-1] + 1) // 2 - 1) * 2 + 3 - x.shape[-1], 0)
    return conv(F.pad(x, (needed // 2, needed - needed // 2)))


class VocaCoreARKit(nn.Module):
    def __init__(self, audio_dim=1540, motion_dim=52):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(1, eps=1e-5, momentum=.1)
        self.convs = nn.ModuleList([nn.Conv1d(audio_dim + motion_dim,32,3,stride=2),
            nn.Conv1d(32,32,3,stride=2),nn.Conv1d(32,64,3,stride=2),nn.Conv1d(64,64,3,stride=2)])
        self.fc1 = nn.Linear(64 + motion_dim,128)
        self.fc2 = nn.Linear(128,50)
        self.expression = nn.Linear(50,motion_dim)
        nn.init.zeros_(self.expression.weight); nn.init.zeros_(self.expression.bias)

    def forward(self, audio, anchor, valid=None):
        if valid is None: valid=torch.ones(audio.shape[:2],device=audio.device,dtype=torch.bool)
        b,t,d=audio.shape
        # No padding may affect normalization or another clip's local window.
        clean=torch.where(valid[...,None],audio,0.)
        windows=F.pad(clean.transpose(1,2),(8,7)).unfold(-1,16,1).permute(0,2,1,3)
        selected=windows[valid]  # [valid frames, audio channels, 16]
        selected=self.input_norm(selected.reshape(-1,1,d*16)).reshape(-1,d,16)
        identity=anchor[:,None].expand(b,t,-1)[valid]
        x=torch.cat((selected,identity[:,:,None].expand(-1,-1,16)),1)
        for conv in self.convs:x=F.relu(same_stride2(x,conv))
        latent=self.fc2(torch.tanh(self.fc1(torch.cat((x.flatten(1),identity),-1))))
        delta=torch.zeros(b,t,anchor.shape[-1],device=audio.device,dtype=audio.dtype)
        delta[valid]=self.expression(latent)
        return anchor[:,None]+delta


def align_native(features, source_valid, target_valid):
    """Resample each paired utterance on its own clock, never padded extent."""
    out=features.new_zeros(target_valid.shape[0],target_valid.shape[1],features.shape[-1])
    for i in range(len(features)):
        src=features[i,source_valid[i]]
        count=int(target_valid[i].sum())
        if count and len(src):
            aligned=F.interpolate(src.T[None],size=count,mode='linear',align_corners=False)[0].T
            out[i,target_valid[i]]=aligned
    return out


class EmoTalkCoreARKit(nn.Module):
    def __init__(self, max_seq_len=2048, period=25, dropout=.1):
        super().__init__()
        self.audio_feature_map_cont=nn.Linear(768,512)
        self.audio_feature_map_emo=nn.Linear(768,832)
        self.audio_feature_map_emo2=nn.Linear(832,256)
        self.obj_vector_level=nn.Linear(4,32)
        self.obj_vector_person=nn.Linear(52,32)
        self.classifier=nn.Sequential(nn.Dropout(dropout),nn.Linear(768,768),nn.Tanh(),
                                      nn.Dropout(dropout),nn.Linear(768,8))
        self.level_head=nn.Linear(768,4)
        layer=nn.TransformerDecoderLayer(832,4,832,dropout=dropout,batch_first=True)
        self.transformer_decoder=nn.TransformerDecoder(layer,1)
        self.bs_map_r=nn.Linear(832,52)
        nn.init.zeros_(self.bs_map_r.weight);nn.init.zeros_(self.bs_map_r.bias)
        self.register_buffer('biased_mask',init_biased_mask(4,max_seq_len,period),persistent=False)

    def _decode(self, content, emotion, anchor, valid):
        pooled=torch.where(valid[...,None],emotion,0.).sum(1)/valid.sum(1).clamp_min(1)[:,None]
        logits=self.classifier(pooled);level_logits=self.level_head(pooled)
        cont=self.audio_feature_map_cont(content)
        memory=self.audio_feature_map_emo(emotion)
        emo=F.relu(self.audio_feature_map_emo2(memory))
        level=self.obj_vector_level(level_logits.softmax(-1))[:,None].expand(-1,content.shape[1],-1)
        person=self.obj_vector_person(anchor)[:,None].expand_as(level)
        hidden=torch.cat((cont,emo,level,person),-1)
        b,t=valid.shape
        if t>self.biased_mask.shape[-1]:raise ValueError('EmoTalk sequence exceeds fixed mask')
        temporal=self.biased_mask[:,:t,:t].repeat(b,1,1).to(hidden)
        # Mask invalid keys in causal self attention. Diagonal memory has no
        # cross-frame padding leakage; masking its invalid rows would be all-inf.
        temporal=temporal.masked_fill((~valid)[:,None,None,:].expand(b,4,t,t).reshape(b*4,t,t),float('-inf'))
        # Invalid query rows are unused. Keep one finite key to avoid NaN grads.
        invalid=(~valid)[:,None,:,None].expand(b,4,t,t).reshape(b*4,t,t)
        temporal=temporal.masked_fill(invalid,0.)
        memory_mask=~torch.eye(t,dtype=torch.bool,device=hidden.device)
        out=self.transformer_decoder(hidden,memory,tgt_mask=temporal,memory_mask=memory_mask)
        prediction=self.bs_map_r(out)
        return torch.where(valid[...,None],prediction,0.),logits,level_logits

    def forward(self,audio,anchor,valid):
        clean=torch.where(valid[...,None],audio,0.)
        return self._decode(clean[...,:768],clean[...,768:1536],anchor,valid)[:2]

    def paired(self,content_audio,content_valid,emotion_audio,emotion_valid,anchor,target_valid):
        content=align_native(content_audio[...,:768],content_valid,target_valid)
        emotion=align_native(emotion_audio[...,768:1536],emotion_valid,target_valid)
        return self._decode(content,emotion,anchor,target_valid)


class PairedUtterances:
    """Metadata-only train pairing; no test list or motion similarity involved."""
    def __init__(self, split):
        from collections import defaultdict
        self.content=defaultdict(list);self.emotion=defaultdict(list)
        self.keys=[];self.labels=[]
        for i,(speaker,sentence) in enumerate(zip(split['speaker'],split['sentence_id'])):
            emo=int(split['emotion_id'][i]);level=int(split['intensity_id'][i])
            self.keys.append((speaker,sentence,level));self.labels.append(emo)
            self.content[(speaker,sentence,level)].append(i)
            self.emotion[(speaker,emo,level)].append(i)
        self.content_choices=[];self.emotion_choices=[]
        for i,(speaker,sentence,level) in enumerate(self.keys):
            self.content_choices.append([j for j in self.content[self.keys[i]] if self.labels[j]!=self.labels[i]])
            self.emotion_choices.append([j for j in self.emotion[(speaker,self.labels[i],level)] if self.keys[j][1]!=sentence])
        self.audit=dict(clips=len(self.keys),cross_emotion_content_available=sum(bool(v) for v in self.content_choices),
                        same_emotion_other_sentence_available=sum(bool(v) for v in self.emotion_choices),
                        source='train metadata only; same speaker and intensity; no motion matching')

    def sample(self,ids,generator):
        def choose(values,i):
            return values[int(torch.randint(len(values),(),generator=generator))] if values else i
        return (torch.tensor([choose(self.content_choices[i],i) for i in ids.tolist()]),
                torch.tensor([choose(self.emotion_choices[i],i) for i in ids.tolist()]))
