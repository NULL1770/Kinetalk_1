"""Prepare auditable lip-evidence candidates; never releases production labels.

``prepare`` audits legacy ASR consensus rows. Their expected/asr tokens can
screen content, but have no acoustic timestamps and therefore cannot activate a
bilabial closure interval. ``calibrate`` accepts separately reviewed external
core/release segments and TRAIN arrays, then calls the same calibration contract
used by ``kinetalk_v2.phonetic``. It refuses jawOpen/diagnostic labels and always
writes ``production_ready: false`` until a human review gate is performed.

No ASR or forced-aligner model is downloaded by this script. The external timing
JSONL contract is intentionally small and can be generated later by a CTC/forced
aligner or a human reviewer:
``{"clip_id":"...", "source":"external_closure_release",
"provenance":"tool/version/reviewer", "segments":[{"phone":"p",
"phase":"core", "start":0.40, "end":0.48, "confidence":0.92}]}``.
Whole-phone ASR intervals and missing confidence are ignored by the parser.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from difflib import SequenceMatcher

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_v2.phonetic import (  # noqa: E402
    EXTERNAL_SOURCES, PhoneticEvidence, LipConstraintAsset,
    align_bilabial_phases, calibrate_intervals,
)


def rows(path: Path):
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
    obj = json.loads(path.read_text(encoding="utf8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        return obj["results"]
    if isinstance(obj, dict) and isinstance(obj.get("clips"), list):
        return obj["clips"]
    raise ValueError(f"Expected JSONL/list/results/clips: {path}")


def toks(value):
    if isinstance(value, list):
        return [str(x).lower() for x in value if str(x).strip()]
    # Match the exact legacy ASR content-audit tokenizer. Apostrophes were
    # split there; changing normalization would invent new disagreement.
    return re.findall(r"[a-z]+", str(value or "").lower())


def sim(a, b):
    aa, bb = toks(a), toks(b)
    return SequenceMatcher(None, aa, bb, autojunk=False).ratio() if aa or bb else 0.0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in values) + "\n",
                    encoding="utf8")


def manifest_index(path):
    """Read ordinary JSONL, formal roles or exported packed split metadata."""
    if path is None: return {}
    if path.suffix == ".jsonl": source = rows(path)
    else:
        doc = json.loads(path.read_text(encoding="utf8"))
        source = []
        if isinstance(doc,list): source = doc
        elif "roles" in doc:
            for split,role in doc["roles"].items():
                for kind in ("query","enrollment"):
                    source.extend(dict(x,split=split,role=kind) for x in role.get(kind,[]))
        elif "split_meta" in doc:
            for split, meta in doc["split_meta"].items():
                for i,clip in enumerate(meta["clip_id"]):
                    source.append({"clip_id":clip,"split":split,"speaker":meta.get("speaker",[None]*len(meta["clip_id"]))[i],"frames":meta.get("lengths",[None]*len(meta["clip_id"]))[i]})
        elif "train" in doc and isinstance(doc["train"],list):
            for split in ("train","val","validation","test"):
                source.extend({"clip_id":x,"split":split} if isinstance(x,str) else dict(x,split=split) for x in doc.get(split,[]))
        else: raise ValueError("Unrecognized split/media manifest")
    result = {}
    for item in source:
        key = str(item.get("clip_id",""))
        if key in result and result[key].get("split") != item.get("split"):
            raise ValueError(f"Clip has inconsistent split: {key}")
        # Formal query metadata takes precedence over enrollment duplicates.
        if key not in result or item.get("role") != "enrollment": result[key] = item
    return result


def lexicon(path):
    """CMU-style lexicon. Alternative pronunciations remain alternatives."""
    result = {}
    if path is None: return result
    for line in path.read_text(encoding="utf8").splitlines():
        if not line.strip() or line.startswith(";;;") or line.startswith("#"): continue
        bits = line.split()
        if len(bits)<2: continue
        word = re.sub(r"\(\d+\)$","",bits[0].lower())
        result.setdefault(word,[]).append([re.sub(r"[012]$","",x.upper()) for x in bits[1:]])
    return result


def bilabial_words(tokens, dictionary):
    found, unknown = [], []
    for index,word in enumerate(tokens):
        pronunciations = dictionary.get(word)
        if pronunciations is None:
            unknown.append(word); continue
        candidates = [[x for x in phones if x in ("P","B","M")] for phones in pronunciations]
        if any(candidates):
            found.append({"word_index":index,"word":word,"pronunciation_alternatives":pronunciations,
                          "bilabial_phone_alternatives":candidates,"timing_available":False,
                          "candidate_only":True})
    return found,unknown


def runtime_inventory():
    names = ("transformers","torchaudio","faster_whisper","whisper","phonemizer",
             "cmudict","ctc_segmentation","montreal_forced_aligner")
    available = {}
    for name in names:
        try: available[name] = importlib.util.find_spec(name) is not None
        except (ValueError,ImportError): available[name] = False
    cache = Path.home()/".cache"/"huggingface"/"hub"
    return {"python":sys.executable,"packages_available":available,
            "huggingface_model_directories":[x.name for x in cache.glob("models--*")] if cache.is_dir() else [],
            "scan_scope":"current Python and conventional HF cache only; remote/runtime environments not inspected",
            "aligner_status":"No model loaded or downloaded; availability alone does not validate timing."}


def review_html(candidates,path):
    """Portable review tool: local audio selected by reviewer; no remote fetch."""
    payload = json.dumps(candidates,ensure_ascii=False).replace("<","\\u003c")
    template = r'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Phonetic evidence review</title>
<style>body{font:16px system-ui;margin:32px;max-width:1100px;background:#f7f8fb;color:#172036}button,input,select{font:inherit;padding:6px;margin:5px}pre{white-space:pre-wrap;background:white;padding:16px}table{border-collapse:collapse;width:100%}td,th{padding:5px;border:1px solid #ccc}td input{width:85px}canvas{width:100%;height:140px;background:white}small{display:block;color:#456}label{display:inline-block}#error{color:#a21e25}</style>
<h1>External phonetic evidence review</h1><p>This tool creates review records. It does not activate model constraints. Transcript/word timing agreement is not lip closure evidence. For /p b/ acoustic silence and release can be candidate phases; /m/ can remain voiced. State the actual timing evidence and uncertainty; do not infer seal from B0/jawOpen.</p>
<label>Find clip <input id="filter" placeholder="clip ID / transcript"></label><select id="clips"></select>
<pre id="meta"></pre><label>Load matching local audio or video <input id="audioFile" type="file" accept="audio/*,video/*"></label><video id="audio" controls style="display:block;max-width:650px;max-height:350px"></video><canvas id="wave" width="1000" height="140"></canvas><small id="clock">Waveform uses the selected media clock; confirm it matches native motion times.</small>
<p>Reviewer <input id="reviewer" placeholder="name/id"> Evidence basis <input id="basis" placeholder="tool/version + observed acoustic/visual landmark" size="50"></p>
<button id="add">Add reviewed phase</button><button id="now">Copy current playback time to last start</button><table><thead><tr><th>Phone</th><th>Phase</th><th>Start (s)</th><th>End (s)</th><th>Confidence</th><th>Remove</th></tr></thead><tbody id="labels"></tbody></table>
<label><input id="approve" type="checkbox">I independently reviewed the matching media and boundaries; these are external phase candidates with explicit uncertainty.</label>
<p id="error"></p><button id="save">Save reviewed clip</button><button id="export">Export JSONL review records</button><pre id="status"></pre>
<script>const data=__PAYLOAD__, stored={}, $=x=>document.getElementById(x);let selected=null, objectURL=null;
function displayOptions(){const q=$('filter').value.toLowerCase();$('clips').textContent='';data.filter(x=>(x.clip_id+' '+x.expected_text).toLowerCase().includes(q)).slice(0,500).forEach(x=>{const o=document.createElement('option');o.value=x.clip_id;o.textContent=x.clip_id+' | '+x.status;$('clips').appendChild(o)});show()}
function show(){selected=data.find(x=>x.clip_id===$('clips').value);$('meta').textContent=selected?JSON.stringify(selected,null,2):'No clips';$('labels').textContent='';$('approve').checked=false;if(selected&&stored[selected.clip_id]) stored[selected.clip_id].segments.forEach(addRow);$('error').textContent=''}
function addRow(s={phone:'p',phase:'core',start:0,end:0,confidence:.8}){const tr=document.createElement('tr');for(const key of ['phone','phase','start','end','confidence']){const td=document.createElement('td'), input=document.createElement(key==='phase'?'select':'input');input.dataset.key=key;if(key==='phase'){for(const v of ['core','release']){const o=document.createElement('option');o.value=v;o.textContent=v;input.appendChild(o)}}else input.type=['start','end','confidence'].includes(key)?'number':'text';input.value=s[key];if(input.type==='number')input.step='.01';td.appendChild(input);tr.appendChild(td)}const td=document.createElement('td'),button=document.createElement('button');button.textContent='×';button.onclick=()=>tr.remove();td.appendChild(button);tr.appendChild(td);$('labels').appendChild(tr)}
$('filter').oninput=displayOptions;$('clips').onchange=show;$('add').onclick=()=>addRow();$('now').onclick=()=>{const rows=$('labels').querySelectorAll('tr');if(rows.length)rows[rows.length-1].querySelector('[data-key=start]').value=$('audio').currentTime.toFixed(3)};
$('audioFile').onchange=async()=>{const file=$('audioFile').files[0];if(!file)return;if(objectURL)URL.revokeObjectURL(objectURL);objectURL=URL.createObjectURL(file);$('audio').src=objectURL;$('clock').textContent='Selected '+file.name+'; confirm clip identity and native motion clock. Audio has not been sent anywhere.';try{const ctx=new AudioContext(),buffer=await ctx.decodeAudioData(await file.arrayBuffer()),samples=buffer.getChannelData(0),c=$('wave').getContext('2d');c.clearRect(0,0,1000,140);c.strokeStyle='#2864ae';c.beginPath();for(let x=0;x<1000;x++){const start=Math.floor(x*samples.length/1000),end=Math.floor((x+1)*samples.length/1000),view=samples.subarray(start,end);let lo=0,hi=0;for(const a of view){lo=Math.min(lo,a);hi=Math.max(hi,a)}c.moveTo(x,70-65*hi);c.lineTo(x,70-65*lo)}c.stroke();ctx.close()}catch(e){$('error').textContent='Waveform decode failed: '+e.message}};
function save(){try{if(!selected||!$('reviewer').value.trim()||!$('basis').value.trim()||!$('approve').checked||!$('audioFile').files[0])throw Error('Select clip/media and supply reviewer, evidence basis, and review acknowledgment.');const segments=Array.from($('labels').querySelectorAll('tr')).map(tr=>{const s={};tr.querySelectorAll('[data-key]').forEach(i=>s[i.dataset.key]=['start','end','confidence'].includes(i.dataset.key)?Number(i.value):i.value.toLowerCase());if(!['p','b','m'].includes(s.phone)||!['core','release'].includes(s.phase)||!Number.isFinite(s.start)||!Number.isFinite(s.end)||s.start<0||s.end<=s.start||s.confidence<0||s.confidence>1)throw Error('Invalid p/b/m stage or interval/confidence');return s});if(!segments.length)throw Error('At least one reviewed interval required.');stored[selected.clip_id]={clip_id:selected.clip_id,source:'manual_closure_release',provenance:$('reviewer').value.trim()+'; '+$('basis').value.trim(),reviewed:true,production_ready:false,media_filename:$('audioFile').files[0].name,clock_unit:'seconds_from_native_audio_start',segments};$('status').textContent='Saved '+Object.keys(stored).length+' reviewed clips locally. Export is still required.';$('error').textContent=''}catch(e){$('error').textContent=e.message}}
$('save').onclick=save;$('export').onclick=()=>{const blob=new Blob([Object.values(stored).map(x=>JSON.stringify(x)).join('\n')+'\n'],{type:'application/jsonl'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='reviewed_phonetic_candidates.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};displayOptions();</script></html>'''
    path.write_text(template.replace("__PAYLOAD__",payload),encoding="utf8")


def prepare(args):
    source = rows(args.consensus)
    media = manifest_index(args.media_manifest)
    split = manifest_index(args.split_manifest)
    dictionary = lexicon(args.lexicon)
    candidates, counts = [], Counter()
    seen = set()
    for index, row in enumerate(source):
        clip_id = str(row.get("clip_id", row.get("source_clip_id", "")))
        if not clip_id:
            counts["missing_clip_id"] += 1; continue
        duplicate = clip_id in seen; seen.add(clip_id)
        expected = row.get("expected_text", row.get("text_normalized", row.get("text", "")))
        asr = row.get("asr_tokens", row.get("asr_text", ""))
        expected_tokens, asr_tokens = toks(expected), toks(asr)
        agreement = sim(expected_tokens, asr_tokens)
        words = row.get("words") or row.get("word_timestamps") or []
        segments = row.get("phoneme_segments") or row.get("segments") or []
        has_phone_timing = bool(segments and all(isinstance(x, dict) and
                                    "start" in x and "end" in x and "phone" in x for x in segments))
        has_words = bool(words and all(isinstance(x, dict) and "start" in x and "end" in x for x in words))
        if duplicate: status = "duplicate_clip"
        elif not expected_tokens: status = "missing_expected_text"
        elif not asr_tokens: status = "missing_asr_tokens"
        elif agreement < args.min_asr_similarity: status = "asr_disagreement_quarantine"
        elif has_phone_timing: status = "phone_timing_present_review_required"
        else: status = "text_only_no_phoneme_timing"
        counts[status] += 1
        lexical, unknown = bilabial_words(re.findall(r"[a-z]+(?:'[a-z]+)?",str(expected or "").lower()),dictionary)
        mapping = {**media.get(clip_id,{}),**split.get(clip_id,{})}
        authoritative_split = split.get(clip_id,{}).get("split") if args.split_manifest else mapping.get("split",row.get("split"))
        audio = mapping.get("audio",row.get("audio"))
        if not audio and args.media_root:
            audio = str(args.media_root / mapping.get("audio_rel",f"audio/{clip_id}.wav"))
        candidates.append({
            "clip_id": clip_id, "dataset": row.get("dataset"), "speaker": row.get("speaker"),
            "split": authoritative_split, "content_id": row.get("content_id"),
            "legacy_media_split":media.get(clip_id,{}).get("split"),
            "expected_text": str(expected or ""), "expected_tokens": expected_tokens,
            "asr_text": str(row.get("asr_text", "")), "asr_tokens": asr_tokens,
            "asr_annotation_similarity": agreement,
            "has_word_timestamps": has_words, "has_phoneme_timestamps": has_phone_timing,
            "asr_logprob": row.get("avg_logprob", row.get("avg_log_prob")),
            "status": status, "production_ready": False,
            "audio":audio,"audio_sha256":mapping.get("audio_sha256"),
            "audio_path_verified":bool(audio and Path(audio).is_file()),
            "split_manifest_matched":clip_id in split,
            "lexical_bilabial_candidates":lexical,"lexicon_unknown_words":unknown,
            "phonetic_quality_mask":0,
            "review_segments":[],"review_required":True,
            "source_index": index,
        })
    summary = {
        "schema": "kinetalk_phonetic_evidence_candidates_v2",
        "input": str(args.consensus.resolve()), "input_sha256": digest(args.consensus),
        "rows": len(candidates), "unique_clip_ids": len(seen), "statuses": dict(counts),
        "min_asr_similarity": args.min_asr_similarity,
        "lexicon":str(args.lexicon.resolve()) if args.lexicon else None,
        "lexicon_sha256":digest(args.lexicon) if args.lexicon else None,
        "clips_with_lexical_bilabials":sum(bool(x["lexical_bilabial_candidates"]) for x in candidates),
        "split_manifest_matched":sum(x["split_manifest_matched"] for x in candidates),
        "split_counts":dict(Counter(str(x["split"]) for x in candidates)),
        "verified_local_media_paths":sum(x["audio_path_verified"] for x in candidates),
        "review_html_eligible_clips":sum(x["split"] in ("train","val","validation") or args.split_manifest is None for x in candidates),
        "active_phonetic_frames":0,
        "bilabial_phase_evidence_available":False,
        "media_manifest_sha256":digest(args.media_manifest) if args.media_manifest else None,
        "split_manifest_sha256":digest(args.split_manifest) if args.split_manifest else None,
        "runtime_inventory":runtime_inventory(),
        "production_ready": False,
        "interpretation": "ASR expected/asr token agreement screens content only; it is not phoneme timing or bilabial closure evidence.",
        "missing_capability": "legacy consensus was generated with word_timestamps=False and contains no validated core/release timings",
    }
    write_jsonl(args.output, candidates)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    if args.review_html:
        args.review_html.parent.mkdir(parents=True,exist_ok=True)
        review_html([x for x in candidates if x["split"] in ("train","val","validation") or args.split_manifest is None],args.review_html)
    print(json.dumps(summary, ensure_ascii=False))


def load_arrays(path):
    with np.load(path, allow_pickle=False) as d:
        required = ("motion", "base", "valid", "times", "fit_confidence", "channel_mask", "clip_ids")
        missing = [x for x in required if x not in d]
        if missing: raise ValueError(f"Training NPZ missing fields: {missing}")
        arrays = {x: d[x].copy() for x in required}
    arrays["clip_ids"] = [str(x) for x in arrays["clip_ids"]]
    return arrays


def calibrate(args):
    if args.source == "diagnostic_proxy_b0_jaw":
        raise ValueError("jawOpen diagnostic labels cannot be used by this CLI")
    if args.source not in EXTERNAL_SOURCES:
        raise ValueError(f"source must be one of {sorted(EXTERNAL_SOURCES)}")
    asset = LipConstraintAsset.load(args.asset)
    arrays = load_arrays(args.train_npz)
    splits = manifest_index(args.split_manifest)
    verified_train_ids = {key for key,row in splits.items() if row.get("split") == "train"}
    if not verified_train_ids or not set(arrays["clip_ids"]).issubset(verified_train_ids):
        raise ValueError("Calibration arrays must all belong to TRAIN in an independent split manifest")
    timing = {str(r.get("clip_id")): r for r in rows(args.timing)}
    if len(timing) != len(rows(args.timing)):
        raise ValueError("Duplicate timing clip_id")
    B, T, C = arrays["motion"].shape
    if C != 52 or arrays["base"].shape != (B,T,C) or arrays["valid"].shape != (B,T) or arrays["times"].shape != (B,T):
        raise ValueError("Training arrays are not aligned [B,T,52]/[B,T]")
    if len(arrays["clip_ids"]) != B or len(set(arrays["clip_ids"])) != B:
        raise ValueError("Training clip_ids must be unique")
    phases, confidence = [], []
    quality = {"missing_timing": 0, "alignment_invalid": 0, "timed": 0, "timed_with_bilabial": 0}
    for i, clip_id in enumerate(arrays["clip_ids"]):
        row = timing.get(clip_id)
        if row is None:
            phases.append(np.zeros(T,np.int64)); confidence.append(np.zeros(T,np.float32)); quality["missing_timing"] += 1; continue
        if row.get("source", args.source) != args.source or str(row.get("provenance", "")) == "" or row.get("reviewed") is not True:
            raise ValueError(f"Timing provenance/source invalid for {clip_id}")
        try:
            evidence = align_bilabial_phases(arrays["times"][i], row.get("segments", []),
                                              source=args.source, provenance=str(row["provenance"]),
                                              min_confidence=args.min_confidence)
        except ValueError:
            quality["alignment_invalid"] += 1; raise
        phases.append(np.asarray(evidence.phase)); confidence.append(np.asarray(evidence.confidence))
        quality["timed"] += 1
        quality["timed_with_bilabial"] += int(np.count_nonzero(evidence.phase) > 0)
    evidence = PhoneticEvidence(np.stack(phases), np.stack(confidence), args.source, args.provenance)
    if quality["timed_with_bilabial"] == 0:
        raise ValueError("No reviewed p/b/m core/release evidence survived confidence/time checks")
    ids = list(arrays["clip_ids"])
    cal = calibrate_intervals(asset, arrays["motion"], arrays["base"], arrays["valid"], evidence,
        split="train", training_ids=ids, fit_confidence=arrays["fit_confidence"],
        channel_mask=arrays["channel_mask"], min_samples=args.min_samples,
        min_confidence=args.min_confidence, margin_mm=args.margin_mm)
    cal.save(args.output)
    summary = {"schema":"kinetalk_phonetic_calibration_report_v2", "asset_sha256":asset.digest,
        "calibration":str(args.output.resolve()), "calibration_sha256":digest(args.output),
        "training_npz":str(args.train_npz.resolve()), "training_npz_sha256":digest(args.train_npz),
        "timing":str(args.timing.resolve()), "timing_sha256":digest(args.timing), "quality":quality,
        "split_manifest":str(args.split_manifest.resolve()),"split_manifest_sha256":digest(args.split_manifest),
        "sample_count":cal.sample_count.tolist(), "reliability_max":cal.reliability.max(axis=1).tolist(),
        "production_ready":False,
        "release_gate":"Human review and held-out evaluation are required before wiring constraints into inference.",
        "source":args.source,"provenance":args.provenance}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(summary,ensure_ascii=False))


def ctc_phone_candidates(logits, times, phones, token_ids, blank_id):
    """Viterbi alignment of externally computed *phone* CTC emissions.

    This is an aligner, not an acoustic model. Whole-phone nonblank spans are
    candidates only; CTC blank durations cannot be relabelled as oral closure.
    Raw token posterior is deliberately not called calibrated phase confidence.
    Repeated phones require intervening blanks under the ordinary CTC topology.
    """
    logits,times = np.asarray(logits,dtype=np.float64),np.asarray(times,dtype=np.float64)
    ids = np.asarray(token_ids,dtype=np.int64)
    if logits.ndim!=2 or len(times)!=len(logits) or len(logits)<1 or len(phones)!=len(ids) or not len(ids):
        raise ValueError("Phone CTC logits [T,V], times [T] and nonempty phone IDs required")
    if not np.isfinite(logits).all() or not np.isfinite(times).all() or (np.diff(times)<=0).any():
        raise ValueError("Finite emissions and strictly increasing clock required")
    if blank_id<0 or blank_id>=logits.shape[1] or (ids<0).any() or (ids>=logits.shape[1]).any() or (ids==blank_id).any():
        raise ValueError("Invalid phone/blank token IDs")
    shifts=logits-logits.max(1,keepdims=True)
    logp=shifts-np.log(np.exp(shifts).sum(1,keepdims=True))
    sequence=np.full(2*len(ids)+1,blank_id,np.int64);sequence[1::2]=ids
    S=len(sequence);scores=np.full((len(times),S),-np.inf);back=np.full((len(times),S),-1,np.int64)
    scores[0,0]=logp[0,blank_id];scores[0,1]=logp[0,ids[0]]
    for t in range(1,len(times)):
        for state in range(S):
            choices=[state]
            if state>0: choices.append(state-1)
            if state>1 and sequence[state]!=blank_id and sequence[state]!=sequence[state-2]: choices.append(state-2)
            previous=max(choices,key=lambda x:scores[t-1,x])
            scores[t,state]=scores[t-1,previous]+logp[t,sequence[state]];back[t,state]=previous
    state=max((S-1,S-2),key=lambda x:scores[-1,x])
    if not np.isfinite(scores[-1,state]): raise ValueError("No feasible complete CTC path")
    path=np.empty(len(times),np.int64)
    for t in range(len(times)-1,-1,-1):
        path[t]=state
        if t: state=back[t,state]
    boundaries=np.empty(len(times)+1)
    if len(times)>1:
        boundaries[1:-1]=(times[:-1]+times[1:])/2
        boundaries[0]=max(0,times[0]-(times[1]-times[0])/2)
        boundaries[-1]=times[-1]+(times[-1]-times[-2])/2
    else: raise ValueError("At least two CTC clock steps required")
    segments=[]
    for i,phone in enumerate(phones):
        frames=np.flatnonzero(path==2*i+1)
        if not len(frames):raise ValueError("Phone missing from complete CTC path")
        segments.append({"phone":str(phone),"start":float(boundaries[frames[0]]),
            "end":float(boundaries[frames[-1]+1]),"token_posterior_mean":float(np.exp(logp[frames,ids[i]]).mean()),
            "phase":None,"phase_confidence":None,"requires_independent_boundary_review":True})
    return segments


def ctc_candidates(args):
    with np.load(args.emissions,allow_pickle=False) as d:
        for key in ("logits","times","phones","phone_ids"):
            if key not in d:raise ValueError(f"CTC input lacks {key}")
        segments=ctc_phone_candidates(d["logits"],d["times"],d["phones"],d["phone_ids"],args.blank_id)
    row={"clip_id":args.clip_id,"source":"phone_ctc_alignment_candidates",
        "provenance":args.model_provenance,"emissions_sha256":digest(args.emissions),
        "segments":segments,"production_ready":False,"reviewed":False,
        "phonetic_quality_mask":0,"interpretation":"Whole-phone CTC spans are not oral closure/release labels; blanks are not oral closures."}
    write_jsonl(args.output,[row]);print(json.dumps(row,ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest="command", required=True)
    a = sp.add_parser("prepare"); a.add_argument("--consensus",type=Path,required=True); a.add_argument("--output",type=Path,required=True); a.add_argument("--summary",type=Path,required=True); a.add_argument("--min-asr-similarity",type=float,default=.75)
    a.add_argument("--media-manifest",type=Path);a.add_argument("--split-manifest",type=Path);a.add_argument("--media-root",type=Path);a.add_argument("--lexicon",type=Path);a.add_argument("--review-html",type=Path)
    b = sp.add_parser("calibrate"); b.add_argument("--asset",type=Path,required=True); b.add_argument("--train-npz",type=Path,required=True); b.add_argument("--split-manifest",type=Path,required=True); b.add_argument("--timing",type=Path,required=True); b.add_argument("--output",type=Path,required=True); b.add_argument("--report",type=Path,required=True); b.add_argument("--source",default="external_closure_release"); b.add_argument("--provenance",required=True); b.add_argument("--min-confidence",type=float,default=.8); b.add_argument("--min-samples",type=int,default=32); b.add_argument("--margin-mm",type=float,default=.25)
    c = sp.add_parser("ctc-candidates"); c.add_argument("--emissions",type=Path,required=True);c.add_argument("--blank-id",type=int,required=True);c.add_argument("--clip-id",required=True);c.add_argument("--model-provenance",required=True);c.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    {"prepare":prepare,"calibrate":calibrate,"ctc-candidates":ctc_candidates}[args.command](args)


if __name__ == "__main__": main()
