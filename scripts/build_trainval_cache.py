"""Materialize one shared train/validation tensor cache for baselines.

The prepared-data directory is shard based.  Loading it independently in four
training processes causes each process to spend tens of minutes scanning and
deserializing the same 13k files.  This utility performs that read once and
writes a provenance-bound cache; it never selects or opens test artifacts.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.packed_trainval_cache import build_from_prepared

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    marker=build_from_prepared(a.data,a.output)
    meta=torch.load(a.output/'metadata.pt',map_location='cpu',weights_only=False)
    print(json.dumps({'output':str(a.output),'train':len(meta['split_meta']['train']['clip_id']),
                      'validation':len(meta['split_meta']['validation']['clip_id']),
                      'schema':marker['schema'],'test_loaded':False}),flush=True)
if __name__=='__main__': main()
