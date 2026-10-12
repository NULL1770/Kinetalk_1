from pathlib import Path
from collections import Counter
import json

root = Path('/root/kinetalk_unified_v2_20261010/hdtf_full')
for directory in ('motion_raw', 'audio16k', 'features/features', 'features/receipts'):
    files = list((root / directory).glob('*'))
    print(json.dumps({'directory': directory, 'files': len(files),
                      'bytes': sum(p.stat().st_size for p in files if p.is_file())}))
receipts = []
for path in (root / 'features/receipts').glob('*.json'):
    row = json.loads(path.read_text())
    receipts.append(row)
print(json.dumps({'receipt_splits': dict(Counter(r.get('split') for r in receipts)),
                  'statuses': dict(Counter(r.get('status') for r in receipts)),
                  'total_receipt_frames': sum(r.get('n_frames', 0) for r in receipts),
                  'train_sources': len([r for r in receipts if r.get('split') == 'train']),
                  'example_fields': list(receipts[0]) if receipts else []}))
status = root / 'features/status.json'
if status.exists():
    row = json.loads(status.read_text())
    print(json.dumps({'feature_status': {k: v for k, v in row.items() if k not in ('provenance', 'last_receipt')}}))
diag = Path('/root/kinetalk_v5_20261011/diagnostic_seed47')
if (diag / 'state.json').exists():
    print('DIAGNOSTIC', (diag / 'state.json').read_text())
print('LOG_TAIL', '\n'.join(Path('/root/kinetalk_v5_20261011/diagnostic.log').read_text().splitlines()[-4:]))
