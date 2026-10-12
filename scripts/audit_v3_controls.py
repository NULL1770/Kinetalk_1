"""Read-only v3 control sensitivity on independent validation references.

This is not a trained ablation or a style accuracy benchmark. Generation never
loads query motion. Same audio/base/seed is used for each condition. A second
same-person reference tests stability and a donor tests conditional response;
neither gives paired counterfactual ground truth.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_v3.unified_model_v3 import UnifiedV3Config, UnifiedExpressionV3Model
from scripts.train_expression_response import configure, load_runtime, sha, state_digest
from scripts.train_unified_v3 import build_base_cache, attach_base
from scripts.evaluate_v2_factor_interventions import (
    balanced_validation_ids, validate_reference, donor_speaker, emotion_indices,
    per_query_seed, motion_descriptors, descriptor_delta, content_change,
    schedule_change, directional_control, _mean_nested, write_json,
)


@torch.no_grad()
def evaluate(args):
    configure(47)
    device = torch.device(args.device)
    binding = json.loads(Path(args.binding).read_text(encoding='utf8'))
    data, base = load_runtime(binding, device)
    q = data['splits']['validation']
    pool = sorted(set(int(q['speaker_id'][i]) for i in range(len(q['_lengths']))))
    ids, excluded = [], []
    for i in balanced_validation_ids(q, 0, 47):
        cid, sentence = q['clip_id'][i], q['sentence_id'][i]
        sid = int(q['speaker_id'][i])
        donor = donor_speaker(sid, pool, 47, cid)
        support = (data['refs'][sid], data['refs'][donor])
        if any(cid in r['clip_id'] or sentence in r.get('sentence_id', []) for r in support):
            excluded.append(cid)
            continue
        ids.append(i)
        if len(ids) >= args.limit:
            break
    if len(ids) < args.limit:
        raise ValueError('Insufficient independent validation controls')
    cache = build_base_cache(q, base, ids, device)
    names = data['config']['data']['emotion_classes']
    controls = emotion_indices(names)
    if sha(binding['rig']) != binding['rig_sha256']:
        raise ValueError('Rig binding differs')
    import numpy as np
    with np.load(binding['rig'], allow_pickle=False) as z:
        channels = z['channels'].tolist()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for checkpoint in args.checkpoint:
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        digest = sha(checkpoint)
        model = UnifiedExpressionV3Model(UnifiedV3Config(**ck['config']),
            ck['model']['feature_mean'], ck['model']['feature_std'], ck['model']['scales']).to(device)
        model.load_state_dict(ck['model'], strict=True)
        model.eval().requires_grad_(False)
        initial = (state_digest(model), state_digest(base.stage1))
        rows = []
        for ordinal, i in enumerate(ids):
            cid, sentence = q['clip_id'][i], q['sentence_id'][i]
            sid = int(q['speaker_id'][i])
            donor = donor_speaker(sid, pool, 47, cid)
            ra, rd = data['refs'][sid], data['refs'][donor]
            if len(ra['clip_id']) < 2:
                raise ValueError('Need two independent same-person references for stability')
            refs = {}
            refids = {}
            for name, ref, slot in [('A', ra, 0), ('same_person_B', ra, 1), ('donor', rd, 0)]:
                validate_reference(cid, sentence, ref, slot)
                refs[name] = {k: ref[k][slot:slot+1].to(device)
                              for k in ('motion', 'valid', 'channel_mask')}
                refids[name] = ref['clip_id'][slot]
            batch = attach_base(q.batch(torch.tensor([i]), device,
                keys=('audio_features', 'content', 'valid', 'channel_mask', 'times')), [i], cache)
            assert 'motion' not in batch
            positional = (batch['audio_features'], batch['h0'], batch['b0'], batch['valid'])
            kwargs = dict(times=batch['times'], seed=per_query_seed(cid, 47))
            outputs = {name: model.predict(*positional, ref, **kwargs) for name, ref in refs.items()}
            for name, index in controls.items():
                probs = F.one_hot(torch.tensor([index], device=device), model.cfg.classes).float()
                outputs[name] = model.predict(*positional, refs['A'], emotion=probs, **kwargs)
            for value in (.5, 1.5):
                outputs[f'intensity_{value}'] = model.predict(*positional, refs['A'],
                    intensity=torch.tensor([value], device=device), **kwargs)
            normal = outputs['A']
            outputs['no_events'] = {**model.decode(batch['h0'], batch['b0'], batch['valid'],
                normal['student']['state'], normal['student']['emotion_logits'].softmax(-1),
                normal['style'], torch.zeros_like(normal['schedule']['events'])),
                'student': normal['student'], 'schedule': {'events': torch.zeros_like(normal['schedule']['events'])}}
            valid = batch['valid'][0].cpu().numpy()
            times = batch['times'][0].cpu().numpy()
            mask = batch['channel_mask'][0].cpu().numpy()
            b0 = batch['b0'][0].cpu().numpy()
            desc = {name: motion_descriptors(out['motion'][0].cpu().numpy(), b0, valid, times, mask, channels)
                    for name, out in outputs.items()}
            differences = {}
            for name, out in outputs.items():
                if name == 'A':
                    continue
                delta = descriptor_delta(desc['A'], desc[name])
                differences[name] = {
                    'content_proxies': content_change(normal['motion'][0].cpu().numpy(),
                        out['motion'][0].cpu().numpy(), valid, times, mask, channels),
                    'descriptor_delta': delta,
                    'state_max_abs_change': float((out['student']['state']-normal['student']['state']).abs().max()),
                    'event_schedule': schedule_change(normal['schedule']['events'][0].cpu().numpy(),
                        out['schedule']['events'][0].cpu().numpy(), valid),
                }
                if name in controls:
                    differences[name]['directional_response'] = directional_control(delta, name)
            rows.append(dict(clip_id=cid, reference_ids=refids, query_motion_loaded_for_generation=False,
                             differences=differences, descriptors=desc))
            print(json.dumps(dict(checkpoint=digest, clips=ordinal+1, total=len(ids))), flush=True)
        if initial != (state_digest(model), state_digest(base.stage1)):
            raise ValueError('Control diagnostic mutated model')
        report = dict(schema='v3-control-sensitivity-v1', clips=len(rows), test_loaded=False,
            checkpoint_sha256=digest, binding_sha256=sha(args.binding), source_sha256=sha(__file__),
            scope='validation sensitivity only, not an independent style/affect accuracy benchmark',
            query_reference_sentence_disjoint=True, excluded_reference_overlap_clips=excluded,
            summary={name:_mean_nested([row['differences'][name] for row in rows])
                     for name in rows[0]['differences']}, records=rows,
            limitations=['ARKit signs are proxies, not psychological or semantic emotion truth.',
                'Reference response is not proof that the target personal style was transferred.',
                'Same audio and seed may produce different schedules under changed conditioning.',
                'No retraining, query-motion teacher, best-of-N selection, or test data used.'])
        write_json(root/(digest[:12]+'.json'), report)
    write_json(root/'complete.json', dict(status='complete', checkpoints=len(args.checkpoint), clips=len(ids)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding', required=True)
    p.add_argument('--checkpoint', required=True, action='append')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, default=32)
    evaluate(p.parse_args())
