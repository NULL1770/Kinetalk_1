"""Fixed-input output-path intervention; neither fit nor checkpoint mutation."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.load_candidate_model import load_candidate_model
from scripts.train_expression_response import configure, load_runtime, sha, state_digest, write
from scripts.train_unified_v4 import build_base_cache, attach_base, select_balanced


def rms(value, valid):
    return float(value[valid].square().mean().sqrt())


@torch.no_grad()
def run(a):
    configure(47)
    root = Path(a.output)
    if root.exists():
        raise FileExistsError('Fresh diagnostic output required')
    root.mkdir(parents=True)
    device = torch.device(a.device)
    data, base = load_runtime(json.loads(Path(a.binding).read_text()), device)
    q = data['splits']['validation']
    ids = select_balanced(q, list(range(len(q['_lengths']))), a.limit, 47)
    cache = build_base_cache(q, base, ids, device)
    rows = []
    for checkpoint in a.checkpoints:
        model = load_candidate_model(torch.load(checkpoint, map_location='cpu', weights_only=False), device).eval()
        if not hasattr(model, 'envelope_head'):
            raise ValueError('Dual-timescale checkpoint required')
        before = state_digest(model)
        for start in range(0, len(ids), 8):
            use = ids[start:start+8]
            b = attach_base(q.batch(torch.tensor(use), device), use, cache)
            enrolled = [data['refs'][int(q['speaker_id'][i])] for i in use]
            if any(q['clip_id'][i] in r['clip_id'] or q['sentence_id'][i] in r['sentence_id']
                   for i, r in zip(use, enrolled)):
                raise ValueError('Reference overlap')
            refs = {k: torch.stack([r[k][0] for r in enrolled]).to(device)
                    for k in ('motion','valid','channel_mask')}
            full = model.predict(b['audio_features'], b['h0'], b['b0'], b['valid'], refs,
                                 times=b['times'], seed=47+start)
            def compose(proposal):
                y = model.decoder._project(proposal, full['geometry_target'], b['valid'])
                return model.composer(y)
            no_adapters = compose(full['proposal'] - full['affect_residual'] - full['phonetic_residual'])
            no_joint = compose(full['proposal'] - full['residual_base'])
            # A decode-only envelope intervention keeps the generated schedule fixed.
            raw_student = {**full['student'], 'state': full['student']['raw_state']}
            raw_cond = model.condition(raw_student, b['valid'])
            no_envelope = model.decode(b['h0'], b['b0'], b['valid'], raw_cond, full['style'],
                full['events'], geometry_audio=full['geometry_audio_context'], times=b['times'])
            for j, i in enumerate(use):
                valid = b['valid'][j]
                row = dict(clip_id=q['clip_id'][i], checkpoint_sha256=sha(checkpoint),
                    joint_rms=rms(full['residual_base'][j], valid),
                    low_adapter_rms=rms(full['affect_residual'][j], valid),
                    high_adapter_rms=rms(full['phonetic_residual'][j], valid),
                    no_adapters_motion_change=rms(no_adapters[j]-full['motion'][j],valid),
                    no_joint_motion_change=rms(no_joint[j]-full['motion'][j],valid),
                    no_envelope_motion_change=rms(no_envelope['motion'][j]-full['motion'][j],valid),
                    no_envelope_proposal_change=rms(no_envelope['proposal'][j]-full['proposal'][j],valid),
                    geometry_projection_rms=rms(full['motion'][j]-full['proposal'][j],valid))
                rows.append(row)
        if state_digest(model) != before:
            raise RuntimeError('Frozen diagnostic changed weights')
    summaries = {}
    for checksum in sorted(set(r['checkpoint_sha256'] for r in rows)):
        group = [r for r in rows if r['checkpoint_sha256']==checksum]
        summaries[checksum] = {k: sum(r[k] for r in group)/len(group)
            for k in group[0] if k not in ('clip_id','checkpoint_sha256')}
    write(root/'report.json', dict(schema='v5_fixed_output_intervention_v1',indices=ids,
        rows=rows,means=summaries,test_loaded=False,weights_changed=False,
        binding_sha256=sha(a.binding),source_sha256=sha(__file__),
        limits='Frozen output interventions establish local path sensitivity, not trained ablation gains or the sole cause of optimization failure. Events and geometry targets held fixed for adapter/joint interventions.'))
    write(root/'complete.json', dict(status='complete',clips=len(ids)))
    print(json.dumps(summaries), flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True)
    p.add_argument('--checkpoints',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=32);p.add_argument('--device',default='cuda')
    run(p.parse_args())
