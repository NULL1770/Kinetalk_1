"""Compare audio semantic students through the same frozen 300-epoch receiver.

This is a condition-distribution transfer diagnostic, not matched retraining.
All draws and native masks are paired. No teacher value enters generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from kinetalk_b0.models.semantic_upper_flow import SemanticUpperFlow
from scripts.train_visual_semantic_pilot import UPPER, SEEDS, sha, save_json, validate_inputs
from scripts.joint_motion_metrics import score_clip, summarize

SCHEMA = 'semantic_student_transfer_v1'
MODES = ('old', 'new', 'static', 'reverse', 'constant')


def conditions(clip, prediction, kind, mode, scales, device):
    """Only stored acoustic predictions, native mask, audio global and identity."""
    mapping = {'old': 'actual', 'new': 'actual', 'static': 'static',
               'reverse': 'reverse', 'constant': 'constant_audio'}
    if kind not in ('va', 'posterior') or mode not in mapping:
        raise ValueError('Unknown condition')
    valid = clip['valid'][None].to(device)
    x = prediction[kind][mapping[mode]]
    if x.shape != (valid.shape[1], 2 if kind == 'va' else 8) or not torch.isfinite(x[clip['valid']]).all():
        raise ValueError('Invalid acoustic semantic sequence')
    x = (x-scales[kind+'_mean'])/scales[kind+'_std']
    semantic = torch.nn.functional.pad(x, (0, 8-x.shape[-1]))[None].to(device)
    semantic = torch.where(valid[..., None], semantic, 0.)
    global_value = ((clip['affect_global']-scales['affect_global_mean'])/scales['affect_global_std'])[None].to(device)
    identity = ((clip['identity_code']-scales['identity_code_mean'])/scales['identity_code_std'])[None].to(device)
    return valid, semantic, global_value, identity


@torch.no_grad()
def sample(model, clip, prediction, kind, mode, scales, device):
    valid, sem, global_value, identity = conditions(clip, prediction, kind, mode, scales, device)
    output = []
    for seed in SEEDS:
        key = int.from_bytes(hashlib.sha256(f'visual_semantic:{seed}:{clip["clip_id"]}'.encode()).digest()[:8], 'little') % (2**63-1)
        z = model.decode(valid, sem, global_value, identity, steps=16, seed=key)
        values = torch.sigmoid(z*scales['motion_std'].to(z)+scales['motion_mean'].to(z))[0].cpu().numpy()
        values[~clip['valid'].numpy()] = clip['baseline52'][~clip['valid']][:, UPPER].numpy()
        output.append(values)
    return np.stack(output)


def bound_student(path, dataset_hash):
    provenance = json.loads(path.with_name('provenance.json').read_text(encoding='utf8'))
    if provenance['dataset_sha256'] != dataset_hash:
        raise ValueError('Student dataset hash differs')
    for name in ('predictions.pt', 'models.pt'):
        if provenance['outputs'][name] != sha(path.with_name(name)):
            raise ValueError('Student output hash differs: '+name)
    return torch.load(path, map_location='cpu', weights_only=False), provenance


def run(args):
    if args.output.exists(): raise FileExistsError('Fresh diagnostic output required')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    dataset_hash = sha(args.dataset)
    data = torch.load(args.dataset, map_location='cpu', weights_only=False)
    old, old_source = bound_student(args.old_student, dataset_hash)
    new, new_source = bound_student(args.new_student, dataset_hash)
    if new_source['baseline_student_sha256'] != sha(args.old_student):
        raise ValueError('New student was fitted against different static baseline')
    clips, old_rows = validate_inputs(data, old)
    _, new_rows = validate_inputs(data, new)
    for c, a, b in zip(clips, old_rows, new_rows):
        for kind in ('va', 'posterior'):
            if not torch.equal(a[kind]['static'], b[kind]['static']):
                raise ValueError('Frozen static outputs changed')
            if not torch.isfinite(b[kind]['constant_audio'][c['valid']]).all():
                raise ValueError('Missing constant-audio diagnostic')
    models = torch.load(args.new_student.with_name('models.pt'), map_location='cpu', weights_only=False)
    fit = {c['clip_id'] for c in clips if c['split'] == 'train'}
    if set(models['all_train']['fit_clip_ids']) != fit: raise ValueError('New full student fit members differ')
    for c, row in zip(clips, new_rows):
        if c['split'] != 'train': continue
        subset = {d['clip_id'] for d in clips if d['split']=='train' and new['fold_by_sentence'][d['sentence']] != row['fold']}
        model = models['oof'][row['fold']]
        if set(model['fit_clip_ids']) != subset or c['sentence'] in model['fit_sentences']:
            raise ValueError('New OOF student fit members differ')
    source_manifest = json.loads((args.receiver/'manifest.json').read_text())
    source_hashes = {name: sha(args.receiver/name) for name in ('va_audio_last.pt','posterior_audio_last.pt','predictions.pt')}
    if any(source_manifest[name]['sha256'] != value for name,value in source_hashes.items()):
        raise ValueError('Frozen receiver manifest mismatch')
    original = torch.load(args.receiver/'predictions.pt', map_location='cpu', weights_only=False)
    checkpoints = {kind: torch.load(args.receiver/(kind+'_audio_last.pt'), map_location='cpu', weights_only=False) for kind in ('va','posterior')}
    for checkpoint in checkpoints.values():
        if (checkpoint['epoch'] != 300 or checkpoint['protocol']['dataset_sha256'] != dataset_hash
                or checkpoint['protocol']['student_sha256'] != sha(args.old_student)):
            raise ValueError('Expected frozen 300-epoch receiver with original student/data')
    holdout = [i for i,c in enumerate(clips) if c['split']=='holdout']
    fit_ids = original['protocol']['fit_example_ids']
    fit_examples = [i for i,c in enumerate(clips) if c['clip_id'] in fit_ids]
    if len(fit_examples) != len(fit_ids): raise ValueError('Fixed fit examples differ')
    protocol = {'schema':SCHEMA, 'dataset_sha256':dataset_hash,
        'old_student_sha256':sha(args.old_student), 'new_student_sha256':sha(args.new_student),
        'receiver_hashes':source_hashes, 'script_sha256':sha(__file__),
        'flow_model_sha256':sha(Path(__file__).parents[1]/'kinetalk_b0/models/semantic_upper_flow.py'),
        'sample_seeds':list(SEEDS), 'steps':16, 'generation_mask':'native acoustic valid only',
        'scoring_mask':'native valid AND visual semantic availability; all arms same',
        'receiver':'frozen original va_audio/posterior_audio 300 epoch; no receiver training',
        'modes':list(MODES), 'static':'same receiver with exactly frozen old ridge static prediction',
        'reverse':'new TCN applied to window-reversed acoustic input; static level held fixed',
        'constant':'same new TCN fed repeated acoustic clip mean; distinct from static prediction',
        'scope':'previously used inner development holdout; not sealed test or unbiased final generalization',
        'distribution_shift_caveat':True, 'default_replaced':False, 'dev405_or_sealed_test_loaded':False}
    args.output.mkdir(parents=True);save_json(args.output/'protocol.json',protocol)
    packed = {}; reports = {}; replay_error = {}; start = time.time()
    for kind, checkpoint in checkpoints.items():
        receiver = SemanticUpperFlow(**checkpoint['config']).to(args.device)
        receiver.load_state_dict(checkpoint['state']);receiver.eval()
        scales = checkpoint['scales']; torch.save(scales,args.output/(kind+'_scales.pt'))
        for mode in MODES:
            arm = kind+'_'+mode; reports[arm]={}; maximum = 0.
            for role, ids in (('holdout',holdout),('fit_examples',fit_examples)):
                rows=[]
                for index in ids:
                    clip=clips[index];cid=clip['clip_id'];prediction=(old_rows if mode=='old' else new_rows)[index]
                    values=sample(receiver,clip,prediction,kind,mode,scales,args.device)
                    if mode=='old':
                        maximum=max(maximum,float(np.abs(values-original['clips'][cid]['samples'][kind+'_audio']).max()))
                    common=(clip['valid']&clip['semantic_valid']).numpy()
                    scored=score_clip(values,clip['motion'][:,UPPER].numpy(),common,scales['metric_scale'].numpy())
                    rows.append({'clip_id':cid,'sentence':clip['sentence'],'speaker':clip['speaker'],'emotion':clip['emotion'],**scored})
                    if cid not in packed:
                        packed[cid]={'metadata':{k:clip[k] for k in ('clip_id','sentence','split','speaker','emotion')},
                            'valid':common,'native_valid':clip['valid'].numpy(),'times':clip['times'].numpy(),
                            'target':clip['motion'][:,UPPER].numpy(),'target52':clip['motion'].numpy(),
                            'baseline52':clip['baseline52'].numpy(),'samples':{},'seeds':list(SEEDS)}
                    packed[cid]['samples'][arm]=values
                reports[arm][role]={'summary':summarize(rows),'rows':rows}
            if mode=='old':
                replay_error[kind]=maximum
                if maximum>2e-6: raise ValueError('Original frozen receiver replay differs: '+str(maximum))
            save_json(args.output/'reports.json',reports)
            save_json(args.output/'status.json',{'status':'evaluating','arm':arm,'elapsed_seconds':time.time()-start})
            print('TRANSFER_ARM_COMPLETE',arm,json.dumps(reports[arm]['holdout']['summary']['joint_fair_es']),flush=True)
    payload={'schema':SCHEMA,'protocol':protocol,'clips':packed}
    torch.save(payload,args.output/'predictions.pt')
    # Same metadata-first cases, not chosen by score or trajectory.
    preview=[]
    for emotion in sorted({clips[i]['emotion'] for i in holdout}):
        preview.append(min(clips[i]['clip_id'] for i in holdout if clips[i]['emotion']==emotion))
    torch.save({**payload,'clips':{cid:packed[cid] for cid in preview}},args.output/'preview4.pt')
    if any(sha(args.receiver/name)!=value for name,value in source_hashes.items()):
        raise ValueError('Frozen receiver source changed during evaluation')
    save_json(args.output/'status.json',{'status':'complete','elapsed_seconds':time.time()-start,
        'original_replay_maxabs':replay_error,'default_replaced':False,'naturalness_certified':False})
    save_json(args.output/'manifest.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size}
                                         for p in sorted(args.output.iterdir()) if p.is_file() and p.name!='manifest.json'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','old-student','new-student','receiver','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    run(parser.parse_args())
