"""Build a truthful post-run report without loading extra target tensors."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def load(path):
    p=Path(path)
    return json.loads(p.read_text(encoding='utf8')) if p.is_file() else None

def metric(report,name):
    x=(report or {}).get('summary',{}).get(name,{})
    return x.get('value') if x.get('status') in ('computed','partial') else None

def run(root, output):
    root=Path(root); output=Path(output)
    if output.exists(): raise FileExistsError(output)
    e=root/'event'; m=root/'mouth'
    predictor=load(e/'predictor_results.json'); decision=load(e/'decision.json')
    gate=load(e/'receiver_gate.json'); quant=load(e/'duration_quantization.json')
    mouth=load(m/'results.json'); status=load(root.parent/'status.json') or load(root/'status.json')
    gen={k:load(e/f'generation/{k}/arkit_benchmark.json') for k in ('audio','matched_static','reverse')}
    rows=[]
    for seed, value in (predictor or {}).items():
        a=value.get('summary',{}).get('audio',{}); s=value.get('summary',{}).get('matched_static',{})
        rows.append({'seed':int(seed),'audio_brier':a.get('brier'),'static_brier':s.get('brier'),
          'delta_brier':value.get('audio_vs_static',{}).get('audio_minus_control'),
          'delta_brier_ci95':value.get('audio_vs_static',{}).get('ci95'),
          'audio_joint_nll':a.get('joint_nll'),'static_joint_nll':s.get('joint_nll'),
          'delta_joint_nll':value.get('audio_vs_static_joint_nll',{}).get('audio_minus_control'),
          'delta_joint_nll_ci95':value.get('audio_vs_static_joint_nll',{}).get('ci95'),
          'audio_duration_nll':a.get('duration_nll'),'static_duration_nll':s.get('duration_nll'),
          'delta_duration_nll':value.get('audio_vs_static_duration_nll',{}).get('audio_minus_control'),
          'audio_vs_reverse_joint_nll':value.get('audio_vs_reverse',{}).get('audio_minus_control')})
    mouth_rows=[]
    if mouth:
        for seed, arms in mouth.get('seeds',{}).items():
            for arm in ('audio','matched_static','audio_reverse'):
                if arm in arms:
                    x=arms[arm]; d=x.get('diagnostics',{}); a=x.get('arkit',{})
                    mouth_rows.append({'seed':int(seed),'arm':arm,'mbe':metric({'summary':a},'arkit_mbe'),
                      'lbe':metric({'summary':a},'arkit_lbe'),'mouth27_mae':d.get('mouth27_mae'),
                      'mouth27_velocity_mae_per_second':d.get('mouth27_velocity_mae_per_second')})
    summary={'schema':'event_mouth_summary_v1','scope':'inner development only; not sealed test',
      'execution_status':status,'decision':decision,'receiver_gate':gate,'duration_quantization':quant,
      'predictor_by_seed':rows,'generation_arkit':{k:{n:metric(v,n) for n in ('arkit_mbe','arkit_lbe','arkit_fdd_absolute','supp_upper9_fdd_absolute')} for k,v in gen.items()},
      'mouth_by_seed':mouth_rows,'baseline_mouth':mouth.get('baseline') if mouth else None,
      'interpretation':{'receiver':'The oracle schedule changed motion and passed the engineering control gate; this does not prove audio predicts timing.',
        'audio_predictor':'Audio is worse than independently trained static on all three seeds for onset Brier and joint/duration NLL; audio beats reverse in some seeds but fails the predeclared static gate.',
        'mouth':'Mouth values are candidates only; no default promotion, AV certification or sealed-test claim.'}}
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    lines=['# Event schedule and mouth candidate results','',
      'Scope: inner development split with inherited upstream exposure. No default model was replaced.','',
      '## Decision', '','- Receiver control path: '+str((decision or {}).get('receiver_control_path_passed')),
      '- Audio predictor gate: '+str((decision or {}).get('audio_predictability_passed')),'',
      '## Audio predictor (audio minus matched static)', '']
    lines += [f"- seed {r['seed']}: Brier {r['delta_brier']:.6f}; joint NLL {r['delta_joint_nll']:.6f}; duration NLL {r['delta_duration_nll']:.6f}; reverse joint NLL delta {r['audio_vs_reverse_joint_nll']:.6f}" for r in rows]
    lines += ['', 'Positive deltas mean the audio model is worse. The static gate therefore fails on all three seeds.', '', '## Mouth candidate', '']
    lines += [f"- seed {r['seed']} {r['arm']}: MBE {r['mbe']:.6f}, LBE {r['lbe']:.6f}, mouth27 MAE {r['mouth27_mae']:.6f}" for r in mouth_rows]
    lines += ['', 'AV offset/confidence, FD, WInD and rendered perceptual review remain pending.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); run(a.run_root,a.output)
