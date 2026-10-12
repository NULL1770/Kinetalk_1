"""Bundle completed train/validation HDTF artifacts without original videos.

The snapshot is immutable and contains complete sources. Split metadata is
checked before touching artifacts; held-out and quarantined targets stay local.
"""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def choose(rows, split, limit, seed):
    groups = defaultdict(list)
    for row in rows:
        if row['split'] == split and row.get('n_training_eligible_windows', 0) >= 3:
            groups[row['speaker_group']].append(row)
    rng = random.Random(seed)
    names = sorted(groups)
    rng.shuffle(names)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while any(groups.values()):
        for name in names:
            if groups[name]:
                selected.append(groups[name].pop())
            if limit and len(selected) >= limit:
                return selected
    return selected


def run(a):
    root = Path(a.input_root)
    all_rows = read_rows(root / 'processed_sources.jsonl')
    available = [r for r in all_rows if r.get('processing_status') in {'ok', 'cached'}]
    # Verify group assignment using the full inventory, including sealed splits.
    assignments = defaultdict(set)
    for row in read_rows(root / 'inventory.jsonl'):
        assignments[row['speaker_group']].add(row['split'])
    if any(len(values) > 1 for values in assignments.values()):
        raise ValueError('An identity group crosses splits')
    selected = choose(available, 'train', a.train_sources, a.seed) + choose(available, 'val', a.val_sources, a.seed + 1)
    if not selected:
        raise ValueError('No completed training/validation sources with independent windows')
    source_ids = {r['source_id'] for r in selected}
    windows = [r for r in read_rows(root / 'training_windows.jsonl') if r['source_id'] in source_ids]
    if any(r['split'] not in {'train', 'val'} for r in windows):
        raise ValueError('Sealed target in transfer selection')
    remote = a.remote_root.rstrip('/')
    for row in windows:
        row['motion_path'] = f"{remote}/motion_raw/{row['source_id']}.npz"
        row['audio_path'] = f"{remote}/audio16k/{row['source_id']}.wav"
    entries = []
    for row in selected:
        for directory, suffix, key in [('audio16k', '.wav', 'audio_sha256'), ('motion_raw', '.npz', 'motion_sha256')]:
            path = root / directory / (row['source_id'] + suffix)
            sha = digest(path)
            if sha != row[key]:
                raise ValueError(f"Source artifact differs from receipt: {path.name}")
            entries.append({'local': str(path), 'archive': f'{directory}/{path.name}', 'sha256': sha, 'bytes': path.stat().st_size})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    binding = {'schema': 'hdtf-transfer-v2.1', 'source_ids': sorted(source_ids),
               'sources': len(selected), 'windows': len(windows), 'remote_root': remote,
               'splits': {s: sum(r['split'] == s for r in selected) for s in ('train', 'val')},
               'identities': len({r['speaker_group'] for r in selected}),
               'test_transferred': False, 'quarantine_transferred': False,
               'source_inventory_sha256': digest(root / 'inventory.jsonl'), 'entries': entries}
    with tarfile.open(a.output, 'w:gz', compresslevel=2) as archive:
        for entry in entries:
            archive.add(entry['local'], arcname=entry['archive'])
        for name, payload in [('training_windows.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in windows)),
                              ('source_receipts.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected)),
                              ('transfer_binding.json', json.dumps(binding, ensure_ascii=False, indent=2))]:
            raw = payload.encode('utf-8'); info = tarfile.TarInfo(name); info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    binding.update(bundle_sha256=digest(a.output), bundle_bytes=a.output.stat().st_size)
    a.output.with_suffix('.json').write_text(json.dumps(binding, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: binding[k] for k in ('sources', 'windows', 'splits', 'identities', 'bundle_sha256', 'bundle_bytes')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--train-sources', type=int, default=16, help='0 selects all available train sources')
    p.add_argument('--val-sources', type=int, default=4, help='0 selects all available validation sources')
    p.add_argument('--seed', type=int, default=47)
    run(p.parse_args())
