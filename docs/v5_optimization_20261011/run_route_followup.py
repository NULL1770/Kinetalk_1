"""Score the routed candidate after prior full scoring releases the GPU."""
from pathlib import Path
import json,os,subprocess,time
ROOT=Path('/root/kinetalk_v5_20261011');CODE=ROOT/'code_content_style_route';PY='/root/miniconda3/bin/python'
STATE=ROOT/'route_followup_state.json'
def write(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def main():
 with (ROOT/'route_followup.lock').open('x') as f:f.write(str(os.getpid()))
 stage='waiting_for_route_and_prior_scoring'
 try:
  deadline=time.monotonic()+3*3600
  while not all((ROOT/f).exists() for f in ['content_style_route_complete.json','pilot_followup_complete.json']):
   for f in ['content_style_route_state.json','pilot_followup_state.json']:
    if (ROOT/f).exists():
     s=json.loads((ROOT/f).read_text())
     if s['status']=='failed':raise RuntimeError(f+':'+s.get('error','failed'))
     if s['status']!='complete':
      try:os.kill(s['pid'],0)
      except ProcessLookupError:raise RuntimeError(f+': worker exited without completion')
   if time.monotonic()>deadline:raise TimeoutError('route follow-up wait exceeded three hours')
   write(STATE,dict(status='waiting',stage=stage,pid=os.getpid(),updated_unix=time.time()));time.sleep(30)
  complete=json.loads((ROOT/'content_style_route/complete.json').read_text())
  if not complete['hdtf_affect_gradient_preservation_verified']:raise RuntimeError('HDTF gradient preservation was not verified')
  for stage,script,suffix,extra in [('full_development','evaluate_v5_candidate.py','full_eval',[]),('factor_controls','audit_v5_controls.py','controls',['--limit','32'])]:
   write(STATE,dict(status='running',stage=stage,pid=os.getpid(),updated_unix=time.time()))
   cmd=[PY,'-u','scripts/'+script,'--binding',str(ROOT/'binding.json'),'--checkpoint',str(ROOT/'content_style_route/final.pt'),'--output',str(ROOT/('content_style_route_'+suffix))]+extra
   with (ROOT/('route_'+stage+'.log')).open('x') as log:subprocess.run(cmd,cwd=CODE,check=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  write(ROOT/'route_followup_complete.json',dict(status='complete',test_loaded=False,promotion=False,pid=os.getpid()))
  write(STATE,dict(status='complete',stage=stage,updated_unix=time.time()))
 except Exception as e:write(STATE,dict(status='failed',stage=stage,error=repr(e),updated_unix=time.time()));raise
if __name__=='__main__':main()
