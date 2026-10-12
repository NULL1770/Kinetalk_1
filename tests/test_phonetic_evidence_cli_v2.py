import json
import subprocess
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = __import__("sys").executable
SCRIPT = ROOT / "scripts" / "prepare_phonetic_evidence_v2.py"


def run(*args):
    return subprocess.run([PYTHON, str(SCRIPT), *map(str,args)], cwd=ROOT, check=True, capture_output=True, text=True)


def test_prepare_marks_legacy_asr_text_only(tmp_path):
    source = tmp_path / "consensus.jsonl"
    source.write_text("\n".join([
        json.dumps({"clip_id":"good","expected_text":"please be","asr_tokens":["please","be"],"words":[]}),
        json.dumps({"clip_id":"bad","expected_text":"please be","asr_tokens":["green"]}),
    ])+"\n")
    out, summary = tmp_path/"candidates.jsonl", tmp_path/"summary.json"
    run("prepare","--consensus",source,"--output",out,"--summary",summary)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert rows[0]["status"] == "text_only_no_phoneme_timing"
    assert rows[1]["status"] == "asr_disagreement_quarantine"
    assert json.loads(summary.read_text())["production_ready"] is False


def make_train(tmp_path):
    B,T=2,40; motion=np.zeros((B,T,52),np.float32); motion[:,:,1]=.05; motion[:,:,2]=.15
    valid=np.ones((B,T),bool); times=np.arange(T,dtype=np.float64)[None].repeat(B,0)*.04
    fit=np.ones((B,T),np.float32); channels=np.ones((B,52),bool)
    path=tmp_path/"train.npz"; np.savez(path,motion=motion,base=motion,valid=valid,times=times,fit_confidence=fit,channel_mask=channels,clip_ids=np.array(["a","b"]))
    timing=tmp_path/"timing.jsonl"; timing.write_text("\n".join(json.dumps({"clip_id":x,"source":"external_closure_release","provenance":"manual-v1","reviewed":True,"segments":[{"phone":"p","phase":"core","start":.2,"end":.8,"confidence":.95}]}) for x in ["a","b"])+'\n')
    split=tmp_path/"split.json";split.write_text(json.dumps({"train":["a","b"],"val":["v"]}))
    return path,timing,split


def test_calibrate_writes_contract_but_never_production_ready(tmp_path):
    train,timing,split=make_train(tmp_path); asset=ROOT/"final_experiment/rig/lip_constraints_v2_20261010.npz"
    # Lower min samples only for a tiny fixture; this remains TRAIN-only.
    output,report=tmp_path/"cal.npz",tmp_path/"report.json"
    run("calibrate","--asset",asset,"--train-npz",train,"--split-manifest",split,"--timing",timing,"--output",output,"--report",report,"--provenance","manual-review-v1","--min-samples","2")
    result=json.loads(report.read_text()); assert result["production_ready"] is False
    with np.load(output,allow_pickle=False) as d: assert d["schema"].item()=="kinetalk_lip_intervals_v2.1"


def test_jaw_proxy_and_missing_bilabial_timing_are_rejected(tmp_path):
    train,timing,split=make_train(tmp_path); asset=ROOT/"final_experiment/rig/lip_constraints_v2_20261010.npz"
    timing.write_text(json.dumps({"clip_id":"a","source":"diagnostic_proxy_b0_jaw","provenance":"jaw threshold","segments":[]})+'\n')
    with pytest.raises(subprocess.CalledProcessError):
        run("calibrate","--asset",asset,"--train-npz",train,"--split-manifest",split,"--timing",timing,"--output",tmp_path/"x.npz","--report",tmp_path/"x.json","--source","diagnostic_proxy_b0_jaw","--provenance","jaw")


def test_lexicon_candidate_is_not_a_timing_label_and_formal_split_wins(tmp_path):
    source=tmp_path/"asr.jsonl";source.write_text(json.dumps({"clip_id":"a","expected_text":"please be","asr_tokens":["please","be"],"split":"val"})+'\n')
    media=tmp_path/"media.jsonl";media.write_text(json.dumps({"clip_id":"a","split":"val","audio":"audio.wav"})+'\n')
    split=tmp_path/"split.json";split.write_text(json.dumps({"roles":{"train":{"query":[{"clip_id":"a"}],"enrollment":[]}}}))
    lex=tmp_path/"dictionary.txt";lex.write_text("please P L IY1 Z\nbe B IY1\n")
    out,summary,review=tmp_path/"out.jsonl",tmp_path/"summary.json",tmp_path/"review.html"
    run("prepare","--consensus",source,"--media-manifest",media,"--split-manifest",split,"--lexicon",lex,"--output",out,"--summary",summary,"--review-html",review)
    row=json.loads(out.read_text());assert row["split"]=="train"
    assert len(row["lexical_bilabial_candidates"])==2 and row["phonetic_quality_mask"]==0
    assert row["production_ready"] is False and row["review_segments"]==[]
    assert "external phase candidates" in review.read_text()


def test_calibration_rejects_validation_ids_and_unreviewed_segments(tmp_path):
    train,timing,split=make_train(tmp_path);asset=ROOT/"final_experiment/rig/lip_constraints_v2_20261010.npz"
    split.write_text(json.dumps({"train":["a"],"val":["b"]}))
    with pytest.raises(subprocess.CalledProcessError):
        run("calibrate","--asset",asset,"--train-npz",train,"--split-manifest",split,"--timing",timing,"--output",tmp_path/"bad.npz","--report",tmp_path/"bad.json","--provenance","test")


def test_phone_ctc_alignment_keeps_whole_phones_as_review_candidates(tmp_path):
    logits=np.full((8,3),-5.)
    path=[0,1,1,0,2,2,0,0]
    logits[np.arange(8),path]=5.
    emissions=tmp_path/"emissions.npz"
    np.savez(emissions,logits=logits,times=np.arange(8)*.02+.01,phones=np.array(["P","M"]),phone_ids=np.array([1,2]))
    out=tmp_path/"ctc.jsonl"
    run("ctc-candidates","--emissions",emissions,"--blank-id","0","--clip-id","a","--model-provenance","synthetic fixture","--output",out)
    row=json.loads(out.read_text())
    assert [x["phone"] for x in row["segments"]]==["P","M"]
    assert row["production_ready"] is False and row["phonetic_quality_mask"]==0
    assert all(x["phase"] is None and x["phase_confidence"] is None for x in row["segments"])


def test_ctc_repeated_phone_needs_blank_and_invalid_path_fails():
    import sys
    sys.path.insert(0,str(ROOT))
    from scripts.prepare_phonetic_evidence_v2 import ctc_phone_candidates
    logits=np.full((3,2),-5.);logits[[0,2],1]=5.;logits[1,0]=5.
    segments=ctc_phone_candidates(logits,np.array([.01,.03,.05]),["P","P"],[1,1],0)
    assert segments[0]["end"] <= segments[1]["start"]
    with pytest.raises(ValueError,match="feasible"):
        ctc_phone_candidates(logits[:2],np.array([.01,.03]),["P","P"],[1,1],0)
