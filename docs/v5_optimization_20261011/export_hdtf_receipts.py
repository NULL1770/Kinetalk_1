from pathlib import Path
import json
root = Path('/root/kinetalk_unified_v2_20261010/hdtf_full/features/receipts')
rows = []
for p in sorted(root.glob('*.json')):
    r = json.loads(p.read_text())
    if r['split'] not in ('train', 'val'): continue
    rows.append({k: r[k] for k in ('source_id', 'split', 'speaker_group', 'n_frames',
        'motion_sha256', 'audio_sha256', 'features_sha256', 'motion_path', 'audio_path', 'features_path')})
out = Path('/root/kinetalk_v5_20261011/hdtf_available_sources.json')
out.write_text(json.dumps(rows, indent=2))
print(json.dumps({'sources': len(rows), 'path': str(out)}))
