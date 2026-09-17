"""Read-only same-clock diagnosis of the interrupted centered dynamic adapter."""
import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_projection_delivery import delivery_diagnostic
from scripts.audit_teacher_schedule_probe import load_arm, validate_training_inputs
from scripts.train_predictable_renderer import audio_features, center, observed, sha, state_hash
from scripts.evaluate_cross_identity_projection import RestoredFixedAudioHead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); p = args.root
    torch.set_num_threads(4)
    output = p / 'centered_delivery_diagnosis.json'
    if output.exists(): raise ValueError('Fresh output required')
    run = p / 'centered_epoch010_audit'
    report = json.loads((run / 'report.json').read_text())
    checkpoint_path = run / 'epoch010_interrupted.pt'
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    recipe = ck['recipe']
    curves_path = run / 'epoch010_curves.pt'
    sidecar = json.loads((run / 'epoch010_curves.provenance.json').read_text())
    if sidecar['curve_sha256'] != sha(curves_path) or sidecar['checkpoint_sha256'] != sha(checkpoint_path):
        raise ValueError('Curve/checkpoint binding differs')
    if ck['completed_epochs'] != 10 or state_hash(ck['head']) != ck['head_sha256']:
        raise ValueError('Wrong checkpoint/head')
    flow = load_arm(p / 'constant_teacher', 'constant_teacher')
    ref, lock, scope, hashes = validate_training_inputs({'constant_teacher': flow})
    if ck['head_sha256'] != hashes['fixed_head'] or ck['frozen_state_sha256'] != hashes['frozen_backbone']:
        raise ValueError('Different frozen state')
    bundle = torch.load(recipe['args']['bundle'], map_location='cpu', weights_only=False, mmap=True)
    cache = torch.load(recipe['args']['cache'], map_location='cpu', weights_only=False, mmap=True)
    curves = torch.load(curves_path, map_location='cpu', weights_only=False)
    cfg = yaml.safe_load(Path(recipe['args']['config']).read_text())
    names = {name: int(sid) for name, sid in zip(ref['q']['speaker'], ref['q']['speaker_id'])}
    diagnosis = delivery_diagnostic(ref, bundle['bundles']['external_dev'], curves, ck['head'],
        cfg['data']['emotion_classes'], names)
    train = cache['splits']['train']; q = train['q']
    mask = observed(q)
    residual = center(q['motion'] - train['base']['b0'] - train['identity']['baseline'][:, None], q['valid'].float())
    energy = torch.where(mask, residual, 0).square().sum((0, 1))
    count = mask.sum((0, 1))
    rms = (energy / count.clamp_min(1)).sqrt()
    tr = bundle['bundles']['internal']; head = RestoredFixedAudioHead(ck['head'])
    with torch.no_grad():
        audio = head(audio_features(tr), tr['weight']); teacher = head.teacher(tr['motion_bins'].float(), tr['weight'])
    w = tr['weight'].float()
    ar = ((audio.square()*w[...,None]).sum((0,1))/w.sum()).sqrt()
    mr = ((teacher.square()*w[...,None]).sum((0,1))/w.sum()).sqrt()
    result = {'schema': 'centered_same_clock_delivery_diagnosis_v1', 'script_sha256': sha(__file__),
        'source_report_sha256': sha(run/'report.json'), 'checkpoint_sha256': sha(checkpoint_path),
        'data_scope': scope, 'diagnosis': diagnosis,
        'training_only': {'dynamic_rms_per_channel': rms.tolist(), 'energy_per_channel': energy.tolist(),
            'observed_count_per_channel': count.tolist(), 'audio_control_rms': ar.tolist(),
            'teacher_control_rms': mr.tolist(), 'teacher_audio_rms_ratio': (mr/ar.clamp_min(1e-10)).tolist()},
        'outer280_loaded': False, 'new_identity439_loaded': False, 'test_loaded': False,
        'training_performed': False, 'default_replaced': False}
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'nonneutral': diagnosis['scores']['pooled/nonneutral'],
        'training_control_rms_ratio':result['training_only']['teacher_audio_rms_ratio'],
        'train_dynamic_rms':rms.tolist()}), flush=True)


if __name__ == '__main__': main()
