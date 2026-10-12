"""Server-only detached postprocessing for the first v4 MEAD experiment."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf8')
    tmp.replace(path)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--training-pid', type=int, required=True)
    a = p.parse_args()
    root, code = a.root.resolve(), a.code.resolve()
    train = root / 'mead_formal_seed47'
    state = root / 'postprocess_state.json'
    stage = 'waiting_for_training'
    try:
        write(state, {'status': stage, 'training_pid': a.training_pid,
                      'server_pid': os.getpid(), 'updated_unix': time.time()})
        deadline = time.monotonic() + 8 * 3600
        while not (train / 'complete.json').exists():
            if not alive(a.training_pid):
                raise RuntimeError('Training exited before complete.json; inspect training log')
            if time.monotonic() > deadline:
                raise TimeoutError('Training did not finish within eight hours')
            time.sleep(30)
        jobs = [
            ('development_metrics', 'evaluate_unified_v4.py', root / 'mead_eval_seed47', []),
            ('dynamic_audit', 'evaluate_v4_dynamics.py', root / 'mead_dynamics_seed47',
             ['--limit', '128', '--seeds', '47', '91', '137']),
        ]
        for stage, script, output, extra in jobs:
            write(state, {'status': stage, 'server_pid': os.getpid(), 'updated_unix': time.time()})
            command = [sys.executable, '-u', str(code / 'scripts' / script),
                       '--binding', str(root / 'binding.json'),
                       '--checkpoint', str(train / 'final.pt'), '--output', str(output)] + extra
            with (root / (stage + '.log')).open('w', encoding='utf8') as log:
                subprocess.run(command, cwd=code, check=True, stdout=log, stderr=subprocess.STDOUT)
        report = json.loads((root / 'mead_eval_seed47/report.json').read_text())
        rows = []
        metric_keys = ['arkit_mbe', 'arkit_lbe', 'lve_mean_mm_mean',
                       'eye_forehead_eve_sqrt_mean', 'jaw_closure_frame_f1',
                       'jawOpen/pred_q90_q10', 'jawOpen/centered_correlation']
        for method, result in report['results'].items():
            rows.append({'method': method, 'clips': report['clips'],
                         **{k: result['metrics'][k] for k in metric_keys},
                         **{'probe_%d_macro_f1' % (i + 1): probe['macro_f1']
                            for i, probe in enumerate(result['probes'])}})
        with (root / 'main_metrics.csv').open('w', encoding='utf8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        write(root / 'post_complete.json', {'status': 'complete', 'test_loaded': False,
            'metrics_csv': str(root / 'main_metrics.csv'),
            'metrics_report': str(root / 'mead_eval_seed47/report.json'),
            'dynamic_report': str(root / 'mead_dynamics_seed47/report.json'),
            'render_inputs': str(root / 'mead_eval_seed47/render_inputs'),
            'videos_rendered': False, 'paper_promotion': False})
        write(state, {'status': 'complete', 'updated_unix': time.time()})
    except Exception as exc:
        write(state, {'status': 'failed', 'stage': stage, 'error': repr(exc),
                      'updated_unix': time.time()})
        raise


if __name__ == '__main__':
    main()
