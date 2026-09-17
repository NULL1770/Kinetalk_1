"""Export only the trained projection and fixed audio head after hash checks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_predictable_renderer import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Fresh output required')
    report=json.loads((args.run/'summary.json').read_text())
    if report.get('adaptation')!='projection_only' or not all(report[k] for k in ('frozen_unchanged','renderer_unchanged','head_unchanged')):
        raise ValueError('Only strictly frozen projection-only runs may be compacted')
    delta=torch.load(args.run/'renderer_delta.pt',map_location='cpu',weights_only=False)
    if sha(args.checkpoint)!=delta['provenance']['hashes']['checkpoint']:raise ValueError('Original checkpoint hash mismatch')
    original=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    for key,value in delta['renderer'].items():
        if not torch.equal(value,original['model']['renderer.'+key]):raise ValueError('Renderer changed')
    result={k:v for k,v in delta.items() if k!='renderer'}
    result.update(schema='predictable_projection_adapter_v1',
        restore='Load exact original full checkpoint, then replace local_projection; evaluate audio head from frozen content/middle/prosody bins, not old local controls.',
        default_enabled=False,development_only=True)
    torch.save(result,args.output)
    print(json.dumps({'output':str(args.output),'bytes':args.output.stat().st_size,'sha256':sha(args.output)}))


if __name__=='__main__':main()
