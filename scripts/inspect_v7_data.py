from pathlib import Path
import collections
import json
import numpy as np
import torch

root = Path('/root/autodl-tmp/kinetalk_data/packed_trainval_20260923')
meta = torch.load(root/'metadata.pt', weights_only=False, map_location='cpu')
print('fit_sids', meta['fit_sids'], 'dev_sids', meta['dev_sids'])
print('split_info', {k: list(v) for k, v in meta['split_meta'].items()})
info = meta['split_meta']['train']
for key in ('clip_id','sentence_id','speaker'):
    print(key, info[key][:8])
for role in ('train','validation'):
    s = np.load(root/role/'speaker_id.npy')
    e = np.load(root/role/'emotion_id.npy')
    print(role, 'rows', len(s), 'speakers', len(np.unique(s)),
          'emos', dict(collections.Counter(e.tolist())))
    for sid in np.unique(s)[:8]:
        print('S', sid, dict(collections.Counter(e[s == sid].tolist())))
for path in [Path('/root/kinetalk_v5_20261011/hdtf_binding_final/training_windows_verified.jsonl'),
             Path('/root/kinetalk_v5_20261011/hdtf_windows.jsonl')]:
    print('HDTF', path)
    for line in path.read_text().splitlines()[:2]:
        print(line[:5000])
print('stats', {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__)
                for k, v in meta['feature_stats'].items()})
