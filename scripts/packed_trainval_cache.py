"""Shared variable-length train/validation cache.

Prepared data contains one torch file per clip. This cache performs that scan
once and stores frame arrays concatenated by split with an offsets index. The
large arrays are NumPy memmaps; a training batch is padded only to its own
longest clip.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA = "packed_trainval_memmap_v2"
FRAME_KEYS = ("audio_features", "motion", "times")
FIXED_KEYS = ("valid", "channel_mask", "anchors", "anchor_valid", "speaker_id",
              "emotion_id", "intensity_id", "intensity_valid", "dataset_id", "motion_valid")


def is_packed(path: str | Path) -> bool:
    return Path(path).is_dir() and (Path(path) / "packed_cache.json").is_file()


def _pad_shard(saved: dict[str, Any], length: int, sid: int, *, require_audio: bool) -> dict[str, Any]:
    row, n = saved["row"], len(saved["valid"])
    out: dict[str, Any] = {}
    for key in ("motion", "content", "valid"):
        value = saved[key]
        padded = torch.zeros((length, *value.shape[1:]), dtype=value.dtype)
        padded[:n] = value
        out[key] = padded
    times = saved["times"][0] + torch.arange(length, dtype=torch.float64) / 25
    times[:n] = saved["times"]
    out["times"] = times
    out.update(channel_mask=saved["channel_mask"].bool(), motion_valid=out["valid"].clone(),
               clip_id=row["clip_id"], sentence_id=row["sentence"], speaker=row["speaker"],
               speaker_id=torch.tensor(sid), emotion_id=torch.tensor(row["emotion"]),
               intensity_id=torch.tensor(row["intensity"]),
               intensity_valid=torch.tensor(row["intensity"] >= 0), dataset_id=torch.tensor(0))
    audio = torch.zeros((length, 1540), dtype=torch.float16)
    if "middle" in saved and "prosody" in saved:
        features = torch.cat((saved["content"].float(), saved["middle"].float(), saved["prosody"].float()), -1)
        if features.shape[-1] != 1540:
            raise ValueError("Prepared frame audio feature width differs from 1540")
        audio[:n] = features.to(torch.float16)
    elif require_audio:
        raise ValueError("Query shard lacks frame audio features")
    out["audio_features"] = torch.where(out["valid"][:, None], audio, torch.zeros_like(audio))
    return out


def _store_flat(root: Path, role: str, clips: list[dict[str, Any]]) -> dict[str, Any]:
    role_dir = root / role
    role_dir.mkdir()
    lengths = np.asarray([int(c["valid"].shape[0]) for c in clips], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    np.save(role_dir / "lengths.npy", lengths, allow_pickle=False)
    np.save(role_dir / "offsets.npy", offsets, allow_pickle=False)
    for key in FRAME_KEYS:
        value = np.concatenate([c[key].detach().cpu().numpy() for c in clips], axis=0)
        if key == "audio_features": value = value.astype(np.float16, copy=False)
        np.save(role_dir / f"{key}.npy", value, allow_pickle=False)
    max_len = int(lengths.max())
    fixed_specs = {
        "valid": (np.bool_, (len(clips), max_len)), "motion_valid": (np.bool_, (len(clips), max_len)),
        "channel_mask": (np.bool_, (len(clips), 52)), "anchors": (np.float32, (len(clips), 52)),
        "anchor_valid": (np.bool_, (len(clips), 52)), "speaker_id": (np.int64, (len(clips),)),
        "emotion_id": (np.int64, (len(clips),)), "intensity_id": (np.int64, (len(clips),)),
        "intensity_valid": (np.bool_, (len(clips),)), "dataset_id": (np.int64, (len(clips),)),
    }
    for key, (dtype, shape) in fixed_specs.items():
        values = []
        for c in clips:
            value = c[key]
            if key in ("valid", "motion_valid"):
                padded = torch.zeros(max_len, dtype=value.dtype); padded[:len(value)] = value; value = padded
            values.append(value.detach().cpu().numpy() if torch.is_tensor(value) else value)
        np.save(role_dir / f"{key}.npy", np.stack(values).astype(dtype, copy=False), allow_pickle=False)
    return {"lengths": lengths.tolist(), "max_len": max_len,
            "clip_id": [str(c["clip_id"]) for c in clips],
            "sentence_id": [str(c["sentence_id"]) for c in clips],
            "speaker": [str(c["speaker"]) for c in clips],
            "metadata": [{"packed": True} for _ in clips]}


def build_from_prepared(directory: str | Path, output: str | Path) -> dict[str, Any]:
    import yaml
    from scripts.prepare_paper_full_data import validate_manifest, sha as file_sha
    from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
    root, output = Path(directory).resolve(), Path(output)
    if output.exists(): raise FileExistsError(output)
    index = json.loads((root / "index.json").read_text(encoding="utf8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8")); validate_manifest(manifest)
    if index.get("test_loaded") is not False or manifest.get("sealed_test_targets_loaded") is not False:
        raise ValueError("Prepared source is not train/validation only")
    people = sorted({r["speaker"] for role in ("train", "val") for r in manifest["roles"][role]["query"]})
    sids = {speaker: i for i, speaker in enumerate(people)}
    records = {(r["clip_id"], r["role"], r["kind"]): r for r in index["records"]}
    expected = {r["clip_id"]: r for role in ("train", "val") for kind in ("query", "enrollment") for r in manifest["roles"][role][kind]}
    output.mkdir(parents=True)
    max_len = max(r["frames"] for r in index["records"])
    enrollment: dict[int, list[dict[str, Any]]] = {sids[s]: [] for s in people}
    for role in ("train", "val"):
        for row in manifest["roles"][role]["enrollment"]:
            rec = records[(row["clip_id"], role, "enrollment")]
            saved = torch.load(root / rec["path"], map_location="cpu", weights_only=False)
            if saved["row"] != expected[row["clip_id"]]: raise ValueError("Shard metadata differs")
            enrollment[sids[row["speaker"]]].append(_pad_shard(saved, max_len, sids[row["speaker"]], require_audio=False))
    refs: dict[int, dict[str, Any]] = {}; anchors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for sid, clips in enrollment.items():
        ref = {key: torch.stack([c[key] for c in clips]) for key in ("motion", "content", "audio_features", "valid", "times", "channel_mask", "motion_valid")}
        for key in ("clip_id", "sentence_id", "speaker"): ref[key] = [c[key] for c in clips]
        ref["speaker_id"] = torch.full((len(clips),), sid, dtype=torch.long); refs[sid] = ref
        observed = ref["valid"][..., None] & ref["channel_mask"][:, None]; count = observed.sum(1)
        means = torch.where(observed, ref["motion"], 0.).sum(1) / count.clamp_min(1)
        available = count.gt(0).all(0); median = means.quantile(.5, dim=0)
        anchors[sid] = (torch.where(available, median, torch.zeros_like(median)), available)

    def make_role(role: str, source_role: str) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, int]:
        rows = manifest["roles"][source_role]["query"]
        lengths = np.asarray([records[(r["clip_id"], source_role, "query")]["frames"] for r in rows], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64))); total = int(offsets[-1]); max_role = int(lengths.max())
        rd = output / role; rd.mkdir()
        np.save(rd / "lengths.npy", lengths, allow_pickle=False); np.save(rd / "offsets.npy", offsets, allow_pickle=False)
        flat_audio = np.lib.format.open_memmap(rd / "audio_features.npy", mode="w+", dtype=np.float16, shape=(total,1540))
        flat_motion = np.lib.format.open_memmap(rd / "motion.npy", mode="w+", dtype=np.float32, shape=(total,52))
        flat_times = np.lib.format.open_memmap(rd / "times.npy", mode="w+", dtype=np.float64, shape=(total,))
        fixed = {
            "valid": np.lib.format.open_memmap(rd / "valid.npy", mode="w+", dtype=np.bool_, shape=(len(rows),max_role)),
            "motion_valid": np.lib.format.open_memmap(rd / "motion_valid.npy", mode="w+", dtype=np.bool_, shape=(len(rows),max_role)),
            "channel_mask": np.lib.format.open_memmap(rd / "channel_mask.npy", mode="w+", dtype=np.bool_, shape=(len(rows),52)),
            "anchors": np.lib.format.open_memmap(rd / "anchors.npy", mode="w+", dtype=np.float32, shape=(len(rows),52)),
            "anchor_valid": np.lib.format.open_memmap(rd / "anchor_valid.npy", mode="w+", dtype=np.bool_, shape=(len(rows),52)),
            "speaker_id": np.lib.format.open_memmap(rd / "speaker_id.npy", mode="w+", dtype=np.int64, shape=(len(rows),)),
            "emotion_id": np.lib.format.open_memmap(rd / "emotion_id.npy", mode="w+", dtype=np.int64, shape=(len(rows),)),
            "intensity_id": np.lib.format.open_memmap(rd / "intensity_id.npy", mode="w+", dtype=np.int64, shape=(len(rows),)),
            "intensity_valid": np.lib.format.open_memmap(rd / "intensity_valid.npy", mode="w+", dtype=np.bool_, shape=(len(rows),)),
            "dataset_id": np.lib.format.open_memmap(rd / "dataset_id.npy", mode="w+", dtype=np.int64, shape=(len(rows),)),
        }
        fs = torch.zeros(1540,dtype=torch.float64); fq=torch.zeros(1540,dtype=torch.float64); fc=0; ss=torch.zeros(52,dtype=torch.float64); sc=torch.zeros(52,dtype=torch.float64)
        clip_ids=[]; sentence_ids=[]; speakers=[]
        for i,row in enumerate(rows):
            rec=records[(row["clip_id"],source_role,"query")]; saved=torch.load(root/rec["path"],map_location="cpu",weights_only=False)
            if saved["row"] != expected[row["clip_id"]]: raise ValueError("Shard metadata differs")
            n=len(saved["valid"]); sid=sids[row["speaker"]]; anchor,anchor_valid=anchors[sid]
            feat=torch.cat((saved["content"].float(),saved["middle"].float(),saved["prosody"].float()),-1)
            if feat.shape[-1]!=1540: raise ValueError("Query feature width differs")
            lo,hi=int(offsets[i]),int(offsets[i+1]); valid=saved["valid"].bool(); obs=valid[:,None]&saved["channel_mask"].bool()[None,:]&anchor_valid[None,:]
            flat_audio[lo:hi]=torch.where(valid[:,None],feat,torch.zeros_like(feat)).numpy().astype(np.float16,copy=False)
            flat_motion[lo:hi]=saved["motion"].numpy(); flat_times[lo:hi]=saved["times"].numpy()
            fixed["valid"][i]=False; fixed["valid"][i,:n]=valid.numpy(); fixed["motion_valid"][i]=fixed["valid"][i]; fixed["channel_mask"][i]=saved["channel_mask"].numpy(); fixed["anchors"][i]=anchor.numpy(); fixed["anchor_valid"][i]=anchor_valid.numpy(); fixed["speaker_id"][i]=sid; fixed["emotion_id"][i]=int(row["emotion"]); fixed["intensity_id"][i]=int(row["intensity"]); fixed["intensity_valid"][i]=int(row["intensity"]>=0); fixed["dataset_id"][i]=0
            if source_role=="train":
                x=feat[valid]; fs+=x.double().sum(0); fq+=x.double().square().sum(0); fc+=int(valid.sum()); delta=torch.where(obs,saved["motion"]-anchor[None,:],torch.zeros_like(saved["motion"])); ss+=delta.double().square().sum(0); sc+=obs.sum(0).double()
            clip_ids.append(row["clip_id"]); sentence_ids.append(row["sentence"]); speakers.append(row["speaker"])
        for value in [flat_audio,flat_motion,flat_times,*fixed.values()]: value.flush()
        info={"lengths":lengths.tolist(),"max_len":max_role,"clip_id":clip_ids,"sentence_id":sentence_ids,"speaker":speakers,"metadata":[{"packed":True} for _ in rows]}
        return info,fs,fq,fc,ss,sc

    split_meta={}; train_info,fs,fq,fc,scale_sum,scale_count=make_role("train","train"); split_meta["train"]=train_info
    val_info,_,_,_,_,_=make_role("validation","val"); split_meta["validation"]=val_info
    mean=fs/fc; variance=(fq/fc-mean.square()).clamp_min(0.)
    feature_stats={"mean":mean.float(),"std":variance.sqrt().clamp_min(1e-3).float(),"count":fc,"ddof":0,"std_floor":1e-3,"fit_clip_ids":split_meta["train"]["clip_id"],"source":"train_valid_native_frames_only","application":"(raw_features - fit_mean)/fit_std then mask"}
    count=scale_count.clamp_min(1); target_scales=(scale_sum/count).sqrt().clamp_min(.02).float()
    config=yaml.safe_load((root/"config.yaml").read_text(encoding="utf8")); NeutralAffectSystem(config).eval()
    provenance={"schema":"paper_full_native_data_v1","index_sha256":file_sha(root/"index.json"),"manifest_sha256":manifest["manifest_sha256"],"config_sha256":file_sha(root/"config.yaml"),"fit_clips":len(split_meta["train"]["clip_id"]),"development_clips":len(split_meta["validation"]["clip_id"]),"fit_valid_frames":fc,"fit_sids":sorted({sids[r["speaker"]] for r in manifest["roles"]["train"]["query"]}),"dev_sids":sorted({sids[r["speaker"]] for r in manifest["roles"]["val"]["query"]}),"test_loaded":False,"frame_policy":"complete native sequences, dynamic batch trimming only"}
    torch.save(refs,output/"refs.pt"); torch.save({"schema":SCHEMA,"source":str(root),"source_provenance":provenance,"config":config,"fit_sids":provenance["fit_sids"],"dev_sids":provenance["dev_sids"],"feature_stats":feature_stats,"target_scales":target_scales,"split_meta":split_meta,"test_loaded":False},output/"metadata.pt")
    marker={"schema":SCHEMA,"metadata":"metadata.pt","refs":"refs.pt","source":str(root),"test_loaded":False}; (output/"packed_cache.json").write_text(json.dumps(marker,indent=2),encoding="utf8"); return marker


class PackedSplit(dict):
    """Index-backed split that only pads the clips in the requested batch.

    Extra frame-aligned tensors (for example the B0/H0 cache produced after
    articulation) are kept as a list of native-length tensors.  This lets the
    staged runner reuse the shared cache without rebuilding a multi-gigabyte
    padded torch object.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._extras: dict[str, list[torch.Tensor]] = {}

    def set_extra(self, key: str, values: list[torch.Tensor]) -> None:
        if len(values) != len(self["_lengths"]):
            raise ValueError(f"Extra {key} length differs from packed split")
        self._extras[key] = [v.detach().cpu() for v in values]

    def __setitem__(self, key, value):
        # Frame caches are deliberately kept out of the dict's tensor backing
        # arrays.  Assignment remains supported for compatibility with older
        # callers, while batch() serves the native-length values lazily.
        if key in {"b0", "h0", "residual"} and torch.is_tensor(value):
            n = len(self["_lengths"])
            if value.shape[0] != n:
                raise ValueError(f"Extra {key} has wrong clip dimension")
            self.set_extra(key, [value[i, :int(self["_lengths"][i])] for i in range(n)])
            return
        super().__setitem__(key, value)

    def batch(self, ids: torch.Tensor, device: str | torch.device = "cpu", *, keys=None) -> dict[str, Any]:
        if ids.numel() == 0:
            raise ValueError("Cannot materialize an empty packed batch")
        ids = ids.detach().cpu().tolist(); lengths = self["_lengths"][ids]; width = int(max(lengths)); out: dict[str, torch.Tensor] = {}
        for key in FRAME_KEYS:
            if keys is not None and key not in keys and not (key == 'audio_features' and 'content' in keys):
                continue
            flat, offsets = self[f"_{key}_flat"], self["_offsets"]
            chunks = [flat[int(offsets[i]):int(offsets[i + 1])] for i in ids]
            shape = (len(ids), width) if flat.ndim == 1 else (len(ids), width, flat.shape[-1])
            value = torch.zeros(shape, dtype=flat.dtype)
            for j, chunk in enumerate(chunks): value[j, :len(chunk)] = chunk
            if key == 'times':
                for j, chunk in enumerate(chunks):
                    value[j, len(chunk):] = chunk[-1] + torch.arange(1, width-len(chunk)+1, dtype=value.dtype) / 25
            out[key] = value
        for key in FIXED_KEYS:
            if keys is not None and key not in keys: continue
            value = self[key][ids]; out[key] = value[:, :width] if key in ("valid", "motion_valid") else value
        if "audio_features" in out:
            out["content"] = out["audio_features"][..., :768]
        for key, values in self._extras.items():
            if keys is not None and key not in keys: continue
            chunks = [values[int(i)] for i in ids]
            shape = (len(ids), width) if chunks[0].ndim == 1 else (len(ids), width, chunks[0].shape[-1])
            value = torch.zeros(shape, dtype=chunks[0].dtype)
            for j, chunk in enumerate(chunks): value[j, :len(chunk)] = chunk
            out[key] = value
        result = {key: value.to(device=device, dtype=torch.float32 if value.is_floating_point() and key != 'times' else value.dtype) for key, value in out.items()}
        for key in ('clip_id', 'sentence_id', 'speaker', 'metadata'):
            if keys is None or key in keys:
                result[key] = [self[key][int(i)] for i in ids]
        return result


def _open_split(root: Path, role: str, info: dict[str, Any]) -> PackedSplit:
    # Copy-on-write maps share clean OS pages and protect the cache on disk
    # while avoiding PyTorch's undefined writes to read-only NumPy mappings.
    rd = root / role; split = PackedSplit(); split["_lengths"] = torch.from_numpy(np.load(rd / "lengths.npy", mmap_mode="c", allow_pickle=False)); split["_offsets"] = torch.from_numpy(np.load(rd / "offsets.npy", mmap_mode="c", allow_pickle=False))
    for key in FRAME_KEYS: split[f"_{key}_flat"] = torch.from_numpy(np.load(rd / f"{key}.npy", mmap_mode="c", allow_pickle=False))
    for key in FIXED_KEYS: split[key] = torch.from_numpy(np.load(rd / f"{key}.npy", mmap_mode="c", allow_pickle=False))
    for key in ("clip_id", "sentence_id", "speaker", "metadata"): split[key] = info[key]
    split["audio_features"], split["motion"], split["times"] = split["_audio_features_flat"], split["_motion_flat"], split["_times_flat"]
    split["content"] = split["audio_features"][..., :768]; return split


def _materialize_split(split: PackedSplit) -> dict[str, Any]:
    batch = split.batch(torch.arange(len(split["_lengths"])))
    result = {key: value.cpu() if torch.is_tensor(value) else value for key, value in batch.items()}
    for key in ("clip_id", "sentence_id", "speaker", "metadata"): result[key] = split[key]
    return result


def load_packed(path: str | Path, *, materialize: bool = False, with_refs: bool = False) -> dict[str, Any]:
    root = Path(path); marker = json.loads((root / "packed_cache.json").read_text(encoding="utf8"))
    if marker.get("schema") != SCHEMA or marker.get("test_loaded") is not False: raise ValueError("Invalid packed cache marker")
    meta = torch.load(root / marker["metadata"], map_location="cpu", weights_only=False)
    if meta.get("schema") != SCHEMA or meta.get("test_loaded") is not False: raise ValueError("Invalid packed cache metadata")
    from scripts.prepare_paper_full_data import sha, validate_manifest
    source = Path(meta['source'])
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf8'))
    validate_manifest(manifest)
    if meta['source_provenance']['manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Packed source manifest binding differs')
    for key, filename in [('index_sha256','index.json'), ('config_sha256','config.yaml')]:
        current = sha(source/filename)
        recorded = meta['source_provenance'].get(key)
        if recorded and recorded != current: raise ValueError('Packed source hash differs: '+key)
        meta['source_provenance'][key] = current
    meta['source_provenance']['audio_storage_dtype'] = 'float16; converted to float32 per batch'
    meta['source_provenance']['metadata_sha256'] = sha(root/marker['metadata'])
    splits = {role: _open_split(root, role, meta["split_meta"][role]) for role in ("train", "validation")}
    if materialize: splits = {role: _materialize_split(split) for role, split in splits.items()}
    refs = torch.load(root / marker["refs"], map_location="cpu", weights_only=False)
    # The first cache finalizer intentionally omitted enrollment tensors for
    # the three baselines.  KineTalk still needs those small neutral reference
    # clips for identity encoding.  Reconstruct only enrollment shards here;
    # query shards remain exclusively in the shared memmap and are never
    # scanned a second time.
    if with_refs and not refs:
        refs = _load_enrollment_refs(Path(meta["source"]), meta["fit_sids"], meta["dev_sids"])
    provenance = dict(meta["source_provenance"]); provenance.update({"packed_cache_schema": SCHEMA, "packed_cache_root": str(root.resolve()), "test_loaded": False, "data_loading": "shared variable-length memmap"})
    from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
    system = NeutralAffectSystem(meta["config"]).eval()
    return {"system": system, "config": meta["config"], "splits": splits, "refs": refs, "fit_sids": meta["fit_sids"], "dev_sids": meta["dev_sids"], "ref_groups": {}, "feature_stats": meta["feature_stats"], "target_scales": meta["target_scales"], "provenance": provenance}


def _load_enrollment_refs(source: Path, fit_sids: list[int], dev_sids: list[int]) -> dict[int, dict[str, Any]]:
    """Read only approved train/val enrollment shards for identity fitting."""
    from scripts.prepare_paper_full_data import validate_manifest, sha
    index = json.loads((source / "index.json").read_text(encoding="utf8"))
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf8"))
    validate_manifest(manifest)
    if index.get('test_loaded') is not False or index['recipe']['manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Enrollment source index/manifest differs')
    rows = {r["clip_id"]: r for role in ("train", "val")
            for r in manifest["roles"][role]["enrollment"]}
    records = {(r["clip_id"], r["role"], r["kind"]): r for r in index["records"]}
    people = sorted({r["speaker"] for role in ("train", "val")
                     for r in manifest["roles"][role]["query"]})
    sids = {speaker: i for i, speaker in enumerate(people)}
    max_len = max(r["frames"] for r in index["records"])
    grouped: dict[int, list[dict[str, Any]]] = {sid: [] for sid in sids.values()}
    for role in ("train", "val"):
        for row in manifest["roles"][role]["enrollment"]:
            rec = records[(row["clip_id"], role, "enrollment")]
            path = (source / rec['path']).resolve()
            if not path.is_relative_to(source.resolve()) or sha(path) != rec['sha256']:
                raise ValueError('Enrollment source path/hash differs')
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["row"] != row:
                raise ValueError("Enrollment metadata differs from approved manifest")
            grouped[sids[row["speaker"]]].append(_pad_shard(saved, max_len, sids[row["speaker"]], require_audio=False))
    refs = {}
    for sid, clips in grouped.items():
        if len(clips) < 2: raise ValueError('At least two neutral references required')
        refs[sid] = {key: torch.stack([c[key] for c in clips])
                     for key in ("motion", "content", "audio_features", "valid", "times", "channel_mask", "motion_valid")}
        for key in ("clip_id", "sentence_id", "speaker"):
            refs[sid][key] = [c[key] for c in clips]
        refs[sid]["speaker_id"] = torch.full((len(clips),), sid, dtype=torch.long)
    if set(refs) != set(fit_sids) | set(dev_sids):
        raise ValueError('Enrollment speaker membership differs')
    return refs
