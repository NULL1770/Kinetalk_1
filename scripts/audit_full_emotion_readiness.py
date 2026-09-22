"""Read-only train/val native-data readiness inventory; never opens test data.

Inspect archive headers, masks, clocks and provenance for every train/val row.
This is not a training-readiness certification: audio existence does not prove
waveform alignment, and large motion/acoustic arrays are not decoded here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np


REQUIRED = ('motion', 'content', 'audio', 'times', 'mask', 'channel_mask', 'provenance')
UPPER = (41, 42, 43, 44, 45, 5, 6, 12, 13)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(1024*1024), b''):
            h.update(value)
    return h.hexdigest()


def _header_shapes(path):
    shapes = {}
    with zipfile.ZipFile(path) as archive:
        for field in ('motion', 'content', 'audio'):
            key = field+'.npy'
            if key not in archive.namelist():
                continue
            with archive.open(key) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1,0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2,0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported NPY header version '+str(version))
                shapes[field] = {'shape':list(shape), 'dtype':str(dtype)}
    return shapes


def audit_archive(path, row):
    issues, summary = [], {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            summary['fields'] = sorted(archive.files)
            missing = sorted(set(REQUIRED)-set(archive.files))
            if missing:
                return {'issues':['missing_fields:'+','.join(missing)], **summary}
            times = np.asarray(archive['times'])
            mask = np.asarray(archive['mask'])
            channels = np.asarray(archive['channel_mask'])
            item = archive['provenance'].item()
            if isinstance(item, bytes): item = item.decode('utf8')
            provenance = json.loads(str(item))
            if not isinstance(provenance, dict):
                raise ValueError('provenance is not a mapping')
        shapes = _header_shapes(path)
        summary['array_headers'] = shapes
        n = len(times) if times.ndim == 1 else 0
        summary.update(frames=n, provenance=provenance)
        monotonic = bool(n >= 2 and np.isfinite(times).all() and (np.diff(times)>0).all())
        regular25 = bool(monotonic and np.allclose(np.diff(times), .04, atol=1e-5, rtol=0))
        summary['regular_25fps_timestamps'] = regular25
        if not regular25: issues.append('timestamps_not_regular_25fps')
        if provenance.get('clock_evidence') != 'embedded_video': issues.append('clock_evidence_not_embedded_video')
        try: fps_ok = float(provenance.get('fps', 0)) == 25.
        except (ValueError,TypeError): fps_ok = False
        if not fps_ok: issues.append('provenance_fps_not_25')
        if mask.shape != (n,) or not np.isin(mask, [0,1]).all():
            issues.append('invalid_frame_mask')
        else:
            valid = mask.astype(bool)
            summary['valid_frames'] = int(valid.sum())
            summary['valid_fraction'] = float(valid.mean()) if n else 0.
            if valid.sum() < 32: issues.append('fewer_than_32_valid_frames')
            if row.get('valid_frames') is not None and int(row['valid_frames']) != int(valid.sum()):
                issues.append('metadata_valid_frames_mismatch')
        if channels.shape != (52,) or not np.isin(channels, [0,1]).all():
            issues.append('invalid_channel_mask')
        else:
            channels = channels.astype(bool)
            summary['observed_channel_indices'] = np.flatnonzero(channels).tolist()
            summary['unobserved_channel_indices'] = np.flatnonzero(~channels).tolist()
            summary['upper9_observed'] = bool(channels[list(UPPER)].all())
            if not channels.all(): issues.append('unobserved_arkit_channels')
            if not summary['upper9_observed']: issues.append('unobserved_upper9_channels')
        if row.get('frames') is not None and int(row['frames']) != n: issues.append('metadata_frames_mismatch')
        for field, width in (('motion',52),('content',768),('audio',83)):
            if shapes.get(field,{}).get('shape') != [n,width]: issues.append(field+'_shape_mismatch')
        for key in ('audio_path','audio_sha256','audio_offset_s'):
            if key not in provenance: issues.append('missing_provenance_'+key)
        if 'audio_offset_s' in provenance:
            try: offset_finite = np.isfinite(float(provenance['audio_offset_s']))
            except (TypeError,ValueError): offset_finite = False
            if not offset_finite: issues.append('invalid_audio_offset_s')
    except Exception as error:
        issues.append('archive_read_error:'+type(error).__name__+':'+str(error))
    return {**summary, 'issues':issues}


def wav_inventory(data_root, datasets):
    locations = {'mead_media_v1/wav':[data_root/'processed'/'mead_media_v1'/'wav',
        data_root/'mead_media_v1'/'wav',data_root.parent/'mead_media_v1'/'wav']}
    for dataset in datasets:
        locations['selected_audio/'+dataset] = [data_root/'selected_audio'/dataset]
        locations['interim/'+dataset+'/audio'] = [data_root/'interim'/dataset/'audio']
    result, indexes = {}, {}
    for label, candidates in locations.items():
        roots = list(dict.fromkeys(p.resolve() for p in candidates))
        paths = sorted({p.resolve() for root in roots if root.is_dir()
                        for p in root.rglob('*') if p.is_file() and p.suffix.lower()=='.wav'})
        by_stem = defaultdict(list)
        for path in paths: by_stem[path.stem].append(str(path))
        indexes[label] = by_stem
        result[label] = {'searched_roots':[str(p) for p in roots],
                         'existing_roots':[str(p) for p in roots if p.is_dir()],
                         'wav_files':len(paths), 'duplicate_stem_count':sum(len(v)>1 for v in by_stem.values()),
                         'sample_paths':[str(p) for p in paths[:5]]}
    return result, indexes


def audit(native_root, data_root):
    started = time.monotonic()
    native_root, data_root = Path(native_root).resolve(), Path(data_root).resolve()
    rows, manifests = [], {}
    # Deliberately enumerate two exact names; never glob or read test.jsonl.
    for role in ('train','val'):
        path = native_root/(role+'.jsonl')
        manifests[role] = {'path':str(path),'sha256':sha(path)}
        with path.open(encoding='utf8') as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row,dict): raise ValueError('Manifest rows must be objects')
                    rows.append((role,line_number,row))
    datasets = sorted({str(row.get('dataset','unknown')) for _,_,row in rows})
    inventory, indexes = wav_inventory(data_root,datasets)
    groups, anomalies, evidence = {}, [], defaultdict(list)
    speakers = defaultdict(set)
    clip_roles, artifact_roles = defaultdict(set), defaultdict(set)
    for number,(role,line,row) in enumerate(rows,1):
        dataset = str(row.get('dataset','unknown'))
        emotion, intensity = str(row.get('emotion',row.get('emotion_id','unknown'))), str(row.get('intensity',row.get('intensity_id','unknown')))
        speaker = str(row.get('speaker','unknown'))
        clip = str(row.get('clip_id',''))
        key = (role,dataset,emotion,intensity)
        if key not in groups:
            groups[key] = {'role':role,'dataset':dataset,'emotion':emotion,'intensity':intensity,
                'clips':0,'artifact_exists':0,'archives_inspected':0,'embedded_video_25fps':0,
                'upper9_observed':0,'all52_observed':0,'valid_frames':0,'frames':0,
                'provenance_audio_exists':0,'issues':Counter(),'clock_evidence':Counter(),
                'schemas':Counter(),'observed_channels':[0]*52,'speaker_counts':Counter(),
                'wav_matched_clips':Counter(),'wav_ambiguous_clips':Counter()}
        group=groups[key];group['clips']+=1;group['speaker_counts'][speaker]+=1
        speakers[(role,dataset)].add(speaker);clip_roles[clip].add(role)
        issues=[]
        # A row labelled test inside a train/val file is also forbidden.
        if row.get('source_split',row.get('split',role)) not in ('train','val','validation'):
            issues.append('forbidden_source_role_not_opened')
            report={}
        else:
            raw=row.get('artifact')
            path=(native_root/str(raw)).resolve() if raw else None
            if path is None or not path.is_relative_to(native_root):
                issues.append('artifact_missing_or_outside_native_root');report={}
            elif not path.is_file():
                issues.append('artifact_file_missing');report={}
            else:
                artifact_roles[str(path)].add(role)
                group['artifact_exists']+=1
                report=audit_archive(path,row);group['archives_inspected']+=1
                issues.extend(report.pop('issues'))
        provenance=report.get('provenance',{})
        group['clock_evidence'][str(provenance.get('clock_evidence','missing'))]+=1
        group['schemas'][str(provenance.get('schema','missing'))]+=1
        group['frames']+=report.get('frames',0);group['valid_frames']+=report.get('valid_frames',0)
        group['embedded_video_25fps']+=int(report.get('regular_25fps_timestamps',False)
            and provenance.get('clock_evidence')=='embedded_video' and str(provenance.get('fps')) in ('25','25.0'))
        observed=report.get('observed_channel_indices',[])
        group['all52_observed']+=int(len(observed)==52)
        group['upper9_observed']+=int(report.get('upper9_observed',False))
        for index in observed: group['observed_channels'][index]+=1
        declared=provenance.get('audio_path')
        if declared:
            audio_path=Path(str(declared))
            # Relative provenance is checked literally under the declared data
            # root and separately recorded; no arbitrary file substitution.
            checked=audio_path if audio_path.is_absolute() else data_root/audio_path
            present=checked.is_file();group['provenance_audio_exists']+=int(present)
            report['provenance_audio_checked_path']=str(checked)
            report['provenance_audio_exists']=present
            if not present: issues.append('declared_audio_path_missing')
        aliases={clip}
        if clip.startswith(dataset+'_'): aliases.add(clip[len(dataset)+1:])
        matches={}
        for label,index in indexes.items():
            if label=='mead_media_v1/wav' and dataset!='mead': continue
            if label!='mead_media_v1/wav' and label not in ('selected_audio/'+dataset,'interim/'+dataset+'/audio'): continue
            paths=sorted({p for alias in aliases for p in index.get(alias,[])})
            group['wav_matched_clips'][label]+=int(bool(paths))
            group['wav_ambiguous_clips'][label]+=int(len(paths)>1)
            matches[label]=paths[:3]
        group['issues'].update(issues)
        if issues: anomalies.append({'role':role,'line':line,'clip_id':clip,'dataset':dataset,'emotion':emotion,'issues':issues})
        evidence_key=(role,dataset,emotion)
        if len(evidence[evidence_key])<2:
            evidence[evidence_key].append({'clip_id':clip,'artifact':row.get('artifact'),
                                           'archive':report,'wav_matches':matches})
        if number%1000==0: print(json.dumps({'audited':number,'total':len(rows),'elapsed_seconds':time.monotonic()-started}),flush=True)
    groups_out=[]
    for key in sorted(groups):
        value=groups[key]
        groups_out.append({k:dict(v) if isinstance(v,Counter) else v for k,v in value.items()})
    return {'schema':'full_emotion_readiness_audit_v1','native_root':str(native_root),'data_root':str(data_root),
        'read_roles':['train','val'],'test_metadata_read':False,'test_arrays_read':False,'data_modified':False,
        'manifests':manifests,'total_rows':len(rows),'groups':groups_out,
        'dataset_role_speakers':[{'role':r,'dataset':d,'speakers':sorted(v),'count':len(v)} for (r,d),v in sorted(speakers.items())],
        'cross_role_clip_ids':[k for k,v in clip_roles.items() if len(v)>1],
        'cross_role_artifact_paths':[k for k,v in artifact_roles.items() if len(v)>1],
        'wav_inventory':inventory,'wav_match_policy':'exact clip_id stem or exact stem after removing dataset_ prefix; ambiguous matches reported; no contents/alignment inference',
        'anomalies':anomalies,'sample_evidence':[{'role':r,'dataset':d,'emotion':e,'samples':v} for (r,d,e),v in sorted(evidence.items())],
        'not_verified':['full motion/content/audio tensor finiteness','all artifact/audio hashes against prior provenance',
            'waveform equality between old caches and embedded-video audio','audio/motion temporal alignment',
            'independent enrollment construction','prepared full-emotion feature extraction and retraining'],
        'training_ready_established':False,'elapsed_seconds':time.monotonic()-started}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('native-root','data-root','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Use a fresh report file')
    report=audit(args.native_root,args.data_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'report':str(args.output),'rows':report['total_rows'],'anomaly_rows':len(report['anomalies'])}),flush=True)


if __name__=='__main__': main()
