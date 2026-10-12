"""Continue HDTF train/val feature extraction as local ARKit sources finish.

Source files are immutable; password stays in memory. Every transferred file
and downloaded feature is checked against its receipt. Original videos and
test/quarantine artifacts never leave the local dataset.
"""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

import paramiko

PIN = '10fb2d83c8cd37ebb706cd49c903aa9417aadf0f9ae15ca7a0901ee3c627215d'


class PinnedHost(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        if hashlib.sha256(key.asbytes()).hexdigest() != PIN:
            raise RuntimeError('SSH host key differs from verified server')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def remote(client, code, timeout=60):
    command = "/root/miniconda3/bin/python - <<'HDTF_STREAM'\n" + code + '\nHDTF_STREAM'
    _, out, err = client.exec_command(command, timeout=timeout)
    result = out.read().decode('utf-8'); error = err.read().decode('utf-8')
    if out.channel.recv_exit_status() != 0:
        raise RuntimeError(error[-3000:] or result[-3000:])
    return result


def available(root):
    people = {}
    for r in rows(root / 'inventory.jsonl'):
        if r['speaker_group'] in people and people[r['speaker_group']] != r['split']:
            raise ValueError('An identity group crosses splits')
        people[r['speaker_group']] = r['split']
    result = []
    for r in rows(root / 'processed_sources.jsonl'):
        if r.get('processing_status') not in {'ok', 'cached'} or r.get('split') not in {'train', 'val'}:
            continue
        if r.get('n_training_eligible_windows', 0) < 1:
            continue
        if not re.fullmatch(r'[A-Za-z0-9_-]+', r['source_id']):
            raise ValueError('Unexpected source ID')
        if people[r['speaker_group']] != r['split']:
            raise ValueError('Source/inventory split mismatch')
        result.append(r)
    return result


def run(a):
    root = a.input_root.resolve(); local = a.output_root.resolve()
    if local == root or root.is_relative_to(local):
        raise ValueError('Use a dedicated feature output directory')
    server = a.remote_root.rstrip('/')
    if not server.startswith('/root/kinetalk_unified_v2_20261010/') or '..' in server:
        raise ValueError('Remote outputs must remain in the named v2 experiment')
    for folder in ('features', 'receipts', 'logs'):
        (local / folder).mkdir(parents=True, exist_ok=True)
    password = getpass.getpass('SSH password: ')
    client = paramiko.SSHClient(); client.set_missing_host_key_policy(PinnedHost())
    client.connect('connect.nmb1.seetacloud.com', port=11473, username='root', password=password,
                   look_for_keys=False, allow_agent=False, timeout=30)
    password = None
    sftp = client.open_sftp(); started = time.monotonic(); seen = set(); batches = 0
    extractor_sha = sha(Path(__file__).with_name('extract_hdtf_features_v2.py'))
    atomic_json(local / 'worker_binding.json', {'schema': 'hdtf-stream-worker-v2.1', 'pid': os.getpid(),
        'argv': sys.argv, 'script_sha256': sha(__file__), 'started_unix': time.time(),
        'extractor_sha256': extractor_sha,
        'test_transferred': False, 'quarantine_transferred': False, 'credentials_persisted': False})
    try:
        remote_sha = remote(client, "import pathlib,hashlib\nprint(hashlib.sha256(pathlib.Path('/root/kinetalk_unified_v2_20261010/code_stage2/scripts/extract_hdtf_features_v2.py').read_bytes()).hexdigest())").strip()
        if remote_sha != extractor_sha:
            raise ValueError('Local/remote feature recipes differ')
        remote(client, 'import pathlib\nr=pathlib.Path(' + repr(server) + ')\n'
               + "[ (r/d).mkdir(parents=True,exist_ok=True) for d in ['audio16k','motion_raw','features/features','features/receipts','logs'] ]")
        while time.monotonic() - started < a.timeout_hours * 3600:
            sources = available(root)
            window_snapshot = rows(root / 'training_windows.jsonl')
            window_source_ids = {r['source_id'] for r in window_snapshot}
            # Local cache receipts are checked before skipping a source.
            for r in sources:
                sid = r['source_id']; receipt = local / 'receipts' / (sid + '.json'); feature = local / 'features' / (sid + '.npz')
                if sid in seen or not receipt.exists() or not feature.exists():
                    continue
                done = json.loads(receipt.read_text(encoding='utf-8'))
                if done.get('audio_sha256') == r['audio_sha256'] and done.get('motion_sha256') == r['motion_sha256'] and done.get('provenance', {}).get('script_sha256') == extractor_sha and sha(feature) == done.get('features_sha256'):
                    seen.add(sid)
            pending = [r for r in sources if r['source_id'] not in seen and r['source_id'] in window_source_ids][:a.batch_sources]
            if not pending:
                state = json.loads((root / 'status.json').read_text(encoding='utf-8'))
                if state['state'] == 'complete' and all(r['source_id'] in seen for r in sources):
                    summary = {'status': 'complete', 'processed_sources': len(seen), 'available_sources': len(sources),
                               'source_extraction': state, 'batches': batches, 'elapsed_seconds': time.monotonic() - started}
                    atomic_json(local / 'complete.json', summary); atomic_json(local / 'status.json', summary)
                    print(json.dumps(summary, ensure_ascii=False), flush=True)
                    return
                if state['state'] != 'running':
                    raise RuntimeError('Source extraction stopped before completion: ' + state['state'])
                atomic_json(local / 'status.json', {'status': 'waiting_for_sources', 'downloaded_sources': len(seen), 'elapsed_seconds': time.monotonic() - started, 'updated_unix': time.time()})
                time.sleep(30); continue
            if shutil.disk_usage(local).free < 5 * 2**30:
                raise OSError('Local 5-GiB storage reserve reached')
            free = int(remote(client, "import shutil\nprint(shutil.disk_usage('/root').free)"))
            if free < 5 * 2**30:
                raise OSError('Remote 5-GiB storage reserve reached')
            selected_ids = {r['source_id'] for r in pending}
            entries = []
            for r in pending:
                sid = r['source_id']
                atomic_json(local / 'status.json', {'status': 'transferring', 'source_id': sid, 'downloaded_sources': len(seen), 'updated_unix': time.time()})
                for directory, suffix, key in [('audio16k', '.wav', 'audio_sha256'), ('motion_raw', '.npz', 'motion_sha256')]:
                    p = root / directory / (sid + suffix); target = server + '/' + directory + '/' + p.name
                    if sha(p) != r[key]:
                        raise ValueError('Local source artifact changed: ' + sid)
                    reuse = '/root/kinetalk_unified_v2_20261010/hdtf_batch20/' + directory + '/' + p.name
                    linked = json.loads(remote(client, 'import pathlib,os,json\np=pathlib.Path(' + repr(target) + ')\nq=pathlib.Path(' + repr(reuse) + ")\nif not p.exists() and q.exists():os.link(q,p)\nprint(json.dumps(p.exists()))"))
                    if not linked:
                        sftp.put(str(p), target + '.partial'); sftp.posix_rename(target + '.partial', target)
                    entries.append({'path': target, 'sha256': r[key]})
            code = 'import pathlib,hashlib,json\nitems=json.loads(' + repr(json.dumps(entries)) + ")\nfor e in items:\n p=pathlib.Path(e['path']);h=hashlib.sha256()\n with p.open('rb') as f:\n  for block in iter(lambda:f.read(4194304),b''):h.update(block)\n if h.hexdigest()!=e['sha256']:raise ValueError('Transfer hash mismatch')\nprint('hashes verified')"
            remote(client, code, timeout=120)
            # Snapshot only this batch; its source clocks/identity assignments
            # stay fixed throughout the remote feature process.
            manifest = [r for r in window_snapshot if r['source_id'] in selected_ids]
            if {r['source_id'] for r in manifest} != selected_ids:
                raise ValueError('Every transferred source must have eligible windows in the same snapshot')
            if any(r['split'] not in {'train', 'val'} for r in manifest):
                raise ValueError('Sealed target entered transfer')
            for r in manifest:
                r['motion_path'] = server + '/motion_raw/' + r['source_id'] + '.npz'
                r['audio_path'] = server + '/audio16k/' + r['source_id'] + '.wav'
            batches += 1; name = f'batch_{batches:04d}'
            manifest_local = local / 'logs' / (name + '.jsonl')
            manifest_local.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in manifest), encoding='utf-8')
            manifest_remote = server + '/logs/' + name + '.jsonl'
            sftp.put(str(manifest_local), manifest_remote)
            feature_root = server + '/features'
            command = ['/root/autodl-tmp/kinetalk_emotion2vec_venv/bin/python', 'scripts/extract_hdtf_features_v2.py',
                       '--input-root', server, '--manifest', manifest_remote, '--output-root', feature_root,
                       '--hubert-model', '/root/autodl-tmp/models/hubert-base-ls960',
                       '--emotion2vec-model', '/root/autodl-tmp/model_cache/models/iic--emotion2vec_plus_base/snapshots/master']
            # Separate audio model process uses at most two CPU threads.
            code = 'import subprocess,os,json\ncmd=json.loads(' + repr(json.dumps(command)) + ')\n'
            code += "with open(" + repr(server + '/logs/' + name + '.log') + ",'w') as f:\n p=subprocess.run(cmd,cwd='/root/kinetalk_unified_v2_20261010/code_stage2',stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})\nprint(json.dumps({'returncode':p.returncode}))\n"
            atomic_json(local / 'status.json', {'status': 'extracting_features', 'batch': batches, 'sources': sorted(selected_ids), 'downloaded_sources': len(seen), 'updated_unix': time.time()})
            result = json.loads(remote(client, code, timeout=3600))
            sftp.get(server + '/logs/' + name + '.log', str(local / 'logs' / (name + '.log')))
            if result['returncode'] != 0:
                raise RuntimeError('Feature batch failed; see ' + name + '.log')
            for sid in sorted(selected_ids):
                receipt = local / 'receipts' / (sid + '.json'); feature = local / 'features' / (sid + '.npz')
                sftp.get(feature_root + '/receipts/' + sid + '.json', str(receipt) + '.partial')
                done = json.loads(Path(str(receipt) + '.partial').read_text(encoding='utf-8'))
                if done.get('provenance', {}).get('script_sha256') != extractor_sha:
                    raise ValueError('Extracted feature recipe differs: ' + sid)
                sftp.get(feature_root + '/features/' + sid + '.npz', str(feature) + '.partial')
                if sha(str(feature) + '.partial') != done['features_sha256']:
                    raise ValueError('Downloaded feature checksum differs: ' + sid)
                Path(str(feature) + '.partial').replace(feature); Path(str(receipt) + '.partial').replace(receipt); seen.add(sid)
            atomic_json(local / 'status.json', {'status': 'batch_complete', 'batch': batches, 'downloaded_sources': len(seen), 'updated_unix': time.time(), 'elapsed_seconds': time.monotonic() - started})
            print(json.dumps({'batch': batches, 'downloaded_sources': len(seen), 'elapsed_seconds': time.monotonic() - started}), flush=True)
        raise TimeoutError('Feature stream exceeded configured timeout')
    except Exception as exc:
        atomic_json(local / 'status.json', {'status': 'error', 'error': type(exc).__name__ + ': ' + str(exc), 'downloaded_sources': len(seen), 'updated_unix': time.time()})
        raise
    finally:
        sftp.close(); client.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--batch-sources', type=int, default=8)
    p.add_argument('--timeout-hours', type=float, default=6)
    run(p.parse_args())
