"""Matched geometry receiver pilot, gated on completed posture experiments."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT=Path('/root/kinetalk_v5_20261011')
CODE=ROOT/'code_acoustic_geometry'
PY='/root/miniconda3/bin/python'
SOURCE=ROOT/'mead_matched/final.pt'
EXPECTED='c9189be8050a1b536054bfc11df265d0c0898488a21791638228f9682b937c83'
ARMS=('geometry_coarse','geometry_acoustic','geometry_acoustic_mean')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def main():
    (ROOT/'acoustic_geometry_pilot.lock').open('x').write(str(os.getpid()))
    status=ROOT/'acoustic_geometry_pilot_state.json';stage='waiting_posture';child=None
    try:
        deadline=time.monotonic()+4*3600
        while not (ROOT/'posture_pilot_complete.json').is_file():
            prior=json.loads((ROOT/'posture_pilot_state.json').read_text())
            if prior['status']=='failed':raise RuntimeError('Posture pilot failed: '+prior.get('error',''))
            if prior['status']!='complete':os.kill(prior['pid'],0)
            if time.monotonic()>deadline:raise TimeoutError('Posture wait exceeded four hours')
            write(status,dict(status='waiting',stage=stage,pid=os.getpid(),updated_unix=time.time()))
            time.sleep(15)
        if sha(SOURCE)!=EXPECTED:raise ValueError('Source checkpoint differs')
        sources=json.loads((ROOT/'acoustic_geometry_source_hashes.json').read_text())
        if any(sha(CODE/name)!=digest for name,digest in sources.items()):
            raise ValueError('Source snapshot differs from launch')
        binding=ROOT/'binding.json'
        common=[PY,'-u','scripts/train_v5_mixed.py','--binding',str(binding),
            '--checkpoint',str(SOURCE),'--seed','47','--epochs','1','--hdtf-weight','0',
            '--mead-batch-size','12','--rollout-mean-weight','.5','--teacher-guidance-mode','frame']
        jobs=[('geometry_acoustic_smoke',common+['--geometry-audio-mode','hubert',
            '--geometry-rollout-weight','.2','--steps','16','--eval-limit','32',
            '--output',str(ROOT/'geometry_acoustic_smoke')])]
        for arm,mode,weight in zip(ARMS,('coarse','hubert','hubert'),('0','0','.2')):
            jobs.append((arm+'_train',common+['--geometry-audio-mode',mode,
                '--geometry-rollout-weight',weight,'--steps','256','--eval-limit','128',
                '--output',str(ROOT/arm)]))
            checkpoint=ROOT/arm/'final.pt'
            for suffix,script,extra in (
                ('subset_eval','evaluate_v5_candidate.py',['--selection-json',str(ROOT/'diagnostic_seed47/selection.json')]),
                ('full_eval','evaluate_v5_candidate.py',[]),
                ('dynamics','evaluate_v4_dynamics.py',['--limit','128','--seeds','47','91','137']),
                ('controls','audit_v5_controls.py',['--limit','32']),
                ('geometry','audit_v5_geometry.py',['--limit','128','--device','cpu'])):
                jobs.append((arm+'_'+suffix,[PY,'-u','scripts/'+script,'--binding',str(binding),
                    '--checkpoint',str(checkpoint),'--output',str(ROOT/(arm+'_'+suffix))]+extra))
        write(ROOT/'acoustic_geometry_jobs.json',dict(jobs=jobs,source_checkpoint_sha256=EXPECTED,
            source_snapshot=str(CODE),control='existing posture_control256updates',test_loaded=False,
            matched_updates=256,training_seeds=[47],sampling_seeds=[47,91,137],promotion=False,
            hypotheses=['coarse vs acoustic isolates information with same receiver parameters/objective',
                'acoustic vs acoustic_mean isolates sampled geometry-mean supervision'],
            limits=['One continuation seed, not three independent train seeds.',
                'Geometry receivers and mean objectives are not established novelty.',
                'No-receiver control has different capacity; score all modalities.']))
        env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        control=json.loads((ROOT/'posture_control/complete.json').read_text())
        expected_match={key:control['protocol'][key] for key in ('training_references_sha256','fit_indices_sha256')}
        receipts=[];initial_receivers=[];initial_common=[]
        for stage,command in jobs:
            start=time.monotonic()
            with (ROOT/(stage+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    write(status,dict(status='running',stage=stage,pid=os.getpid(),child_pid=child.pid,
                        stage_elapsed_seconds=time.monotonic()-start,updated_unix=time.time()))
                    if time.monotonic()-start>45*60:
                        child.terminate()
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        raise TimeoutError('Owned stage exceeded45minute hard timeout: '+stage)
                    time.sleep(15)
                if child.returncode:raise RuntimeError('Stage failed: '+stage+' exit '+str(child.returncode))
                child=None
            if stage.endswith('_train') or stage.endswith('_smoke'):
                arm=stage.removesuffix('_train')
                result=json.loads((ROOT/arm/'complete.json').read_text())
                p=result['protocol']
                if p['source_checkpoint_sha256']!=EXPECTED or p['test_loaded'] or p['hdtf_windows']:
                    raise ValueError('Geometry source or data scope differs')
                if not result['geometry_head_gradient_verified']:raise ValueError('Receiver gradient gate failed')
                if result['frozen_b0_sha256']!=control['frozen_b0_sha256']:raise ValueError('Frozen B0 changed')
                if stage.endswith('_smoke') and result['updates']!=16:raise ValueError('Smoke budget differs')
                if stage.endswith('_train'):
                    if result['updates']!=256 or any(p[k]!=v for k,v in expected_match.items()):
                        raise ValueError('Geometry matching scope differs')
                    if result['mead_training_sequence_sha256']!=control['mead_training_sequence_sha256']:
                        raise ValueError('Training sequence differs from control')
                    initial_receivers.append(p['geometry_receiver_initial_sha256'])
                    initial_common.append(p['common_initial_parameters_sha256'])
                    if len(set(initial_receivers))!=1 or len(set(initial_common))!=1:
                        raise ValueError('Matched receiver/common initial parameters differ')
                    receipts.append(dict(arm=arm,updates=result['updates'],
                        common_initial=p['common_initial_parameters_sha256'],
                        receiver_initial=p['geometry_receiver_initial_sha256'],
                        train_sequence=result['mead_training_sequence_sha256']))
            if sha(SOURCE)!=EXPECTED:raise ValueError('Source checkpoint mutated')
        write(ROOT/'acoustic_geometry_complete.json',dict(status='complete',receipts=receipts,
            test_loaded=False,promotion=False,paper_result=False))
        write(status,dict(status='complete',stage=stage,pid=os.getpid(),updated_unix=time.time()))
    except Exception as error:
        write(status,dict(status='failed',stage=stage,error=repr(error),pid=os.getpid(),
            child_pid=child.pid if child else None,updated_unix=time.time()))
        raise


if __name__=='__main__':main()
