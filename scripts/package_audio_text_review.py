"""Compact, explicitly optimizer-free copy of completed audio/text pilot assets."""
from pathlib import Path
import argparse
import json
import hashlib
import zipfile
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    files, bindings = [], {}
    for arm in ('text', 'no_text'):
        run = a.root/arm
        inventory = json.loads((run/'output_hashes.json').read_text())
        original = run/'epoch008.pt'
        if sha(original) != inventory['epoch008.pt']: raise ValueError('Final checkpoint binding differs')
        full = torch.load(original, map_location='cpu', weights_only=False)
        required = ('schema', 'recipe', 'recipe_sha256', 'renderer', 'encoder', 'renderer_sha256',
                    'encoder_sha256', 'protected_sha256', 'completed_epochs', 'step', 'minibatch_sha256', 'noise_time_sha256')
        compact = {k: full[k] for k in required}
        compact['compact_copy'] = {'source_checkpoint_sha256': sha(original), 'optimizer_and_rng_omitted': True,
            'resumable_training_checkpoint': False, 'requires_original_bound_base_and_feature_pipeline': True}
        target = a.output/(arm+'_epoch008_inference.pt')
        torch.save(compact, target)
        bindings[arm] = {'original_checkpoint_sha256': sha(original), 'compact_sha256': sha(target)}
        files.append((target, target.name))
        for path in run.rglob('*'):
            if path.is_file() and path.suffix != '.pt': files.append((path, arm+'/'+path.relative_to(run).as_posix()))
    for name in ('audit.json', 'text_fit.json', 'no_text_fit.json', 'text_train.log', 'no_text_train.log', 'smoke.log',
                 'audio_extract.log', 'text_extract.log', 'target_extract.log'):
        path = a.root/name
        if path.is_file(): files.append((path, name))
    for sub in ('visual', 'data/source'):
        for path in (a.root/sub).rglob('*'):
            if path.is_file(): files.append((path, path.relative_to(a.root).as_posix()))
    for name in ('data/audio.json','data/text.json','data/intensity_targets/provenance.json','data/intensity_targets/targets.pt'):
        path = a.root/name
        if path.is_file(): files.append((path,name))
    manifest = {'schema':'audio_text_compact_review_v1','bindings':bindings,
        'files':{name:{'sha256':sha(path),'bytes':path.stat().st_size} for path,name in files},
        'full_training_and_curves_preserved_remote':str(a.root.resolve()),
        'omissions':'Full audio/text/renderer feature caches, all stochastic full-set curves, optimizer/RNG and intermediate checkpoints remain at original remote paths.'}
    manifest_path = a.output/'manifest.json'; manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    with zipfile.ZipFile(a.output/'audio_text_compact.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        for path,name in files: archive.write(path,name)
        archive.write(manifest_path,'manifest.json')
    digest = sha(a.output/'audio_text_compact.zip')
    (a.output/'zip_sha256.txt').write_text(digest+'\n')
    print(json.dumps({'zip':str(a.output/'audio_text_compact.zip'),'bytes':(a.output/'audio_text_compact.zip').stat().st_size,'sha256':digest}))


if __name__ == '__main__': main()
