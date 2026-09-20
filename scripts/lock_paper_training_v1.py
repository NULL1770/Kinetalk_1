"""Review identity-generalization metadata while honoring historical seals."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.lock_paper_manifests_20260920 import read_rows, build_identity, four_class, sha
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.prepare_paper_full_data import validate_manifest


def lock(metadata_root,reserved,output):
    rows,hashes=read_rows(metadata_root)
    sealed=[json.loads(l) for l in reserved.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    sentences={r['sentence'] for r in sealed}
    removed=[r for r in rows['train'] if four_class(r) and r['sentence'] in sentences]
    rows['train']=[r for r in rows['train'] if r['sentence'] not in sentences]
    manifest=build_identity(rows)
    manifest.update(schema='paper_training_manifest_v1',status='approved_train_val_only',
        authorization='User requested upload current snapshot then implement and train on 2026-09-20',
        source_metadata_sha256=hashes,reserved_metadata_sha256=sha(reserved),reserved_sentences=sorted(sentences),
        exclusions=[{'clip_id':r['clip_id'],'sentence':r['sentence'],'reason':'historical sealed sentence'} for r in removed],
        sealed_test_targets_loaded=False,
        test_policy='Do not materialize test motion or references before method freeze; old test15 remains sealed',
        initialization_policy='All KineTalk modules initialized afresh; frozen pretrained audio/content sources disclosed separately',
        exposure_policy='Historical development exposure persists; native val is development, not untouched evaluation; original B0 lineage not used',
        frame_policy='all native valid frames; no fixed central crop',
        validation_policy='fixed planned endpoint, validation diagnostics only; not final test',
        metric_policy='FaceDiffuser BEAT MBE/LBE/FDD plus declared Upper9 timing and distribution metrics')
    manifest['manifest_sha256']=canonical_hash(manifest);validate_manifest(manifest)
    if output.exists():raise FileExistsError('Fresh manifest required')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'manifest_sha256':manifest['manifest_sha256'],'excluded':len(removed),
        'roles':{r:{'query':len(v['query']),'refs':len(v['enrollment']),'valid_frames':sum(x['valid_frames'] for x in v['query'])} for r,v in manifest['roles'].items()}}))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--metadata-root',type=Path,required=True);p.add_argument('--reserved',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();lock(a.metadata_root,a.reserved,a.output)
