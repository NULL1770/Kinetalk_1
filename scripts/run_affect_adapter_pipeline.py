"""Run one immutable affect-adapter training/evaluation pipeline."""
from __future__ import annotations

import argparse, json, os, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_affect_adapter import parser as train_parser, train
from scripts.train_expression_response import sha, write
from scripts.evaluate_expression_response import evaluate


def main(root):
    root = Path(root)
    lock = root / 'pipeline.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w') as f:
        f.write(str(os.getpid()))
    state = root / 'pipeline_state.json'
    try:
        binding = json.loads((root / 'binding.json').read_text())
        contract = json.loads((root / 'launch_contract.json').read_text())
        if sha(root / 'binding.json') != contract['binding_sha256']:
            raise ValueError('Binding contract changed')
        if sha(Path(__file__)) != contract['pipeline_sha256']:
            raise ValueError('Pipeline source changed')
        for name, digest in binding['source_files'].items():
            if sha(root / 'code' / name) != digest:
                raise ValueError('Source changed: ' + name)
        write(state, dict(status='training', pid=os.getpid(), started_at=time.time(),
                          test_loaded=False, default_replaced=False))
        args = train_parser().parse_args([
            '--binding', str(root / 'binding.json'),
            '--output', str(root / 'seed47'), '--epochs', '8', '--seed', '47',
            '--batch-size', '16', '--lr', '0.0003', '--device', 'cuda'])
        runtime = train(args)
        write(state, dict(status='evaluating', pid=os.getpid(), test_loaded=False,
                          default_replaced=False))
        evaluate(binding, root / 'seed47', root / 'evaluation', runtime=runtime)
        write(state, dict(status='complete', pid=os.getpid(), finished_at=time.time(),
                          test_loaded=False, default_replaced=False))
    except Exception as exc:
        write(state, dict(status='failed', type=type(exc).__name__, message=str(exc),
                          traceback=traceback.format_exc(), test_loaded=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    main(p.parse_args().root)
