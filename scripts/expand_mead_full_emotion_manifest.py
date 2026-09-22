"""Expand an existing MEAD TRAIN/val protocol to all eight emotions.

Reads native train.jsonl and val.jsonl only. Existing speaker assignments,
two neutral enrollment clips per person, reserved TRAIN sentences and the
entire sealed test metadata role remain fixed. No native arrays, artifact
files or test metadata files are opened. Metadata preflight is not a human
review, dataset-quality validation, or authorization to unseal test data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_paper_full_data import validate_manifest
from scripts.train_formal_predictable_projection import canonical_hash, save_json
from scripts.extract_emotion2vec_pilot import sha

SCHEMA='paper_mead_eight_emotion_manifest_v1'
EMOTIONS=('neutral','angry','contempt','disgust','fear','happy','sad','surprise')
REQUIRED=('clip_id','dataset','speaker','sentence','emotion','intensity','artifact',
          'artifact_sha256','frames','valid_frames')


def read_metadata(root):
    rows={}; hashes={}
    for role in ('train','val'):
        path=Path(root)/f'{role}.jsonl'
        hashes[role]=sha(path)
        rows[role]=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    return rows,hashes


def verify_row(row,role):
    if any(k not in row for k in REQUIRED):
        raise ValueError('Native metadata missing required fields')
    for key in ('clip_id','dataset','speaker','sentence','artifact'):
        if not isinstance(row[key],str) or not row[key]:raise ValueError('Empty metadata '+key)
    if row.get('split',row.get('source_split'))!=role:
        raise ValueError('Native source split mismatch: '+row['clip_id'])
    if 'source_split' in row and row['source_split']!=role:
        raise ValueError('Conflicting source split: '+row['clip_id'])
    if 'split' in row and row['split']!=role:
        raise ValueError('Conflicting split: '+row['clip_id'])
    for key in ('emotion','intensity','frames','valid_frames'):
        if type(row[key]) is not int:raise ValueError('Integer metadata required: '+key)
    if not 0<=row['valid_frames']<=row['frames'] or row['frames']<1:
        raise ValueError('Invalid frame counts: '+row['clip_id'])
    if not isinstance(row['artifact_sha256'],str) or not re.fullmatch('[0-9a-fA-F]{64}',row['artifact_sha256']):
        raise ValueError('Invalid artifact hash: '+row['clip_id'])
    path=PurePosixPath(row['artifact'].replace('\\','/'))
    if path.is_absolute() or '..' in path.parts or ':' in str(path):
        raise ValueError('Unsafe native artifact path: '+row['clip_id'])
    if 'source_split' in row and row['source_split']!=role:raise ValueError('Conflicting source role')
    if row['dataset']=='mead':
        if row['emotion'] not in range(8):raise ValueError('Unknown MEAD emotion')
        expected={0} if row['emotion']==0 else {1,2,3}
        if row['intensity'] not in expected:raise ValueError('Unexpected MEAD intensity')


def summarize(rows):
    counts=Counter((r['emotion'],r['intensity']) for r in rows)
    return {'clips':len(rows),'speaker_count':len({r['speaker'] for r in rows}),
            'speakers':sorted({r['speaker'] for r in rows}),
            'sentence_count':len({r['sentence'] for r in rows}),
            'valid_frames':sum(r['valid_frames'] for r in rows),
            'hours_at_25fps':sum(r['valid_frames'] for r in rows)/25/3600,
            'emotion_counts':{str(i):sum(n for (e,_),n in counts.items() if e==i) for i in range(8)},
            'intensity_counts':dict(sorted(Counter(str(r['intensity']) for r in rows).items())),
            'emotion_intensity_counts':{f'{e}:{i}':n for (e,i),n in sorted(counts.items())}}


def build_manifest(base,rows,hashes,base_file_sha256):
    validate_manifest(base)
    original=copy.deepcopy(base)
    if set(rows)!={'train','val'} or set(hashes)!={'train','val'}:
        raise ValueError('Expansion accepts only train/val source metadata')
    reserved=set(base['reserved_sentences'])
    out_roles={'test':copy.deepcopy(base['roles']['test'])}
    exclusions=[];partitions={}; seen_ids=set(); native_by_role={}
    for role in ('train','val'):
        native_by_role[role]={}
        for row in rows[role]:
            verify_row(row,role)
            if row['clip_id'] in seen_ids:raise ValueError('Duplicate/cross-role native clip ID')
            seen_ids.add(row['clip_id']);native_by_role[role][row['clip_id']]=row
    for role in ('train','val'):
        old=base['roles'][role]
        people={r['speaker'] for r in old['query']+old['enrollment']}
        reference=copy.deepcopy(old['enrollment'])
        if any(len([r for r in reference if r['speaker']==s])!=2 for s in people):
            raise ValueError('Base protocol must have exactly two neutral references per identity')
        for ref in reference:
            native=native_by_role[role].get(ref['clip_id'])
            if native is None:raise ValueError('Base enrollment missing from native metadata')
            if any(ref[key]!=native[key] for key in REQUIRED):
                raise ValueError('Base enrollment metadata/hash changed: '+ref['clip_id'])
            if native['valid_frames']<32:raise ValueError('Base enrollment has fewer than 32 valid frames')
        ref_ids={r['clip_id'] for r in reference}
        ref_pairs={(r['speaker'],r['sentence']) for r in reference}
        queries=[];partition=Counter()
        for row in rows[role]:
            reasons=[]
            if row['dataset']!='mead':reasons.append('outside_mead_scope')
            elif row['speaker'] not in people:reasons.append('outside_fixed_base_speakers')
            else:
                if row['valid_frames']<32:reasons.append('fewer_than_32_valid_frames')
                if role=='train' and row['sentence'] in reserved:reasons.append('historical_reserved_sentence')
                if row['clip_id'] in ref_ids:
                    partition['enrollment']+=1
                    continue
                if (row['speaker'],row['sentence']) in ref_pairs:reasons.append('query_reference_speaker_sentence_overlap')
            if reasons:
                exclusions.append({'source_split':role,'clip_id':row['clip_id'],'reasons':reasons,
                                   'native_metadata':copy.deepcopy(row)})
                partition['excluded']+=1
                continue
            # Preserve every original metadata key/value; add only role fields.
            item=copy.deepcopy(row)
            for key,value in (('source_split',role),('manifest_role','query')):
                if key in item and item[key]!=value:raise ValueError('Conflicting native manifest field: '+key)
                item[key]=value
            queries.append(item);partition['query']+=1
        if not queries or {r['speaker'] for r in queries}!=people:
            raise ValueError('Expansion lost query coverage for a fixed base speaker')
        old_ids={r['clip_id'] for r in old['query']}
        if not old_ids<={r['clip_id'] for r in queries}:
            raise ValueError('Expansion would remove a previously included base query')
        if set(r['emotion'] for r in queries)!=set(range(8)):
            raise ValueError('TRAIN/val expansion does not cover all eight MEAD emotions')
        expected={(0,0)}|{(e,i) for e in range(1,8) for i in (1,2,3)}
        actual={(r['emotion'],r['intensity']) for r in queries}
        if not expected<=actual:raise ValueError('Missing emotion/intensity combination after exclusions')
        if sum(partition.values())!=len(rows[role]):raise RuntimeError('Metadata partition accounting failed')
        partitions[role]=dict(partition)
        out_roles[role]={'query':queries,'enrollment':reference,
                         'summary':{'query':summarize(queries),'enrollment':summarize(reference)}}
    manifest={'schema':SCHEMA,'status':'approved_train_val_only','dataset':'mead',
        'track':'fixed_identity_disjoint_mead_eight_emotion_train_val',
        'authorization':{'basis':'user_explicit_request_full_correct_dataset_training_20260922',
            'scope':'expand and prepare all MEAD emotions/intensities in existing TRAIN/val identities',
            'human_review_of_expanded_manifest_completed':False,
            'compatibility_status_note':'approved_train_val_only denotes existing user training authorization plus metadata preflight; not a new author sign-off'},
        'preflight':{'kind':'metadata_only','passed':True,'native_arrays_read':False,
            'native_artifact_hashes_recomputed':False,'test_metadata_file_opened':False,
            'preparation_must_verify_artifact_hash_clock_masks_and_finiteness':True},
        'base_manifest_sha256':base['manifest_sha256'],'base_manifest_file_sha256':base_file_sha256,
        'source_metadata_sha256':hashes,'source_files_read':['train.jsonl','val.jsonl'],
        'reserved_sentences':copy.deepcopy(base['reserved_sentences']),
        'reserved_metadata_sha256':base.get('reserved_metadata_sha256'),
        'roles':{role:out_roles[role] for role in ('train','val','test')},
        'filter':'all MEAD emotion IDs 0..7, neutral level0 and other levels1/2/3, >=32 native valid frames; fixed references/speakers and reserved TRAIN sentences',
        'emotion_names':list(EMOTIONS),'exclusions':exclusions,'source_partition_counts':partitions,
        'base_exclusions_inherited_as_history':copy.deepcopy(base.get('exclusions',[])),
        'query_enrollment_disjoint':True,'sealed_test_targets_loaded':False,
        'test_policy':'test role copied unchanged from four-emotion base metadata; no test metadata file or artifacts opened; not an eight-emotion test benchmark',
        'test_role_canonical_sha256':canonical_hash(base['roles']['test']),
        'exposure_policy':'validation remains historically exposed development; no untouched test claim',
        'frame_policy':'all native valid frames, no fixed crop',
        'full_raw_mead_dataset_claim':False,
        'coverage_scope':'all eligible native MEAD TRAIN/val metadata under fixed identities, references and historical exclusions; not every raw MEAD video',
        'builder_sha256':sha(Path(__file__))}
    manifest['manifest_sha256']=canonical_hash(manifest)
    validate_manifest(manifest)
    if manifest['roles']['test']!=base['roles']['test'] or base!=original:
        raise RuntimeError('Base or sealed test metadata changed')
    summary={'schema':SCHEMA,'manifest_sha256':manifest['manifest_sha256'],
        'base_manifest_sha256':base['manifest_sha256'],'source_metadata_sha256':hashes,
        'source_partition_counts':partitions,
        'roles':{r:manifest['roles'][r]['summary'] for r in ('train','val')},
        'excluded_reason_counts':dict(sorted(Counter(reason for row in exclusions for reason in row['reasons']).items())),
        'exclusion_count':len(exclusions),'test_metadata_unchanged':True,'test_metadata_file_opened':False,
        'test_is_eight_emotion_benchmark':False,'native_arrays_read':False,
        'human_review_claimed':False,'preflight':manifest['preflight']}
    return manifest,summary


def expand(native_root,base_manifest,output):
    native_root=Path(native_root);base_manifest=Path(base_manifest);output=Path(output)
    if output.exists():raise FileExistsError('Fresh independent output directory required')
    base=json.loads(base_manifest.read_text(encoding='utf-8-sig'))
    rows,hashes=read_metadata(native_root)
    manifest,summary=build_manifest(base,rows,hashes,sha(base_manifest))
    output.mkdir(parents=True)
    save_json(output/'manifest.json',manifest);save_json(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-root','base-manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(expand(a.native_root,a.base_manifest,a.output),ensure_ascii=False),flush=True)
