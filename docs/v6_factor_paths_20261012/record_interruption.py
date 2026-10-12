"""Record an owned experiment stop while preserving completed evidence."""
import json
from pathlib import Path
from remote_transport import transport

code='''from pathlib import Path
import json,time,torch,hashlib,os
r=Path('/root/kinetalk_v6_20261012_paths')
queue=json.loads((r/'pilot_queue/queue_state.json').read_text())
for pid in [81449,82426]:
 if Path('/proc/'+str(pid)).exists():
  stat=Path('/proc/'+str(pid)+'/stat').read_text()
  if stat.split()[2]!='Z':raise RuntimeError('owned process still alive '+str(pid))
d=r/'runs/posterior_seed47';resume=d/'resume.pt'
c=torch.load(resume,map_location='cpu',weights_only=False)
receipt=dict(schema='v6_manual_interruption_v1',status='stopped',time=time.time(),
 reason='direct_seed47 completed development results regress lip/affect/dynamics; stop remaining exploratory queue for architecture reconsideration requested in conversation',
 completed_jobs=[j['name'] for j in queue['jobs'] if j['status']=='complete'],
 interrupted_run='posterior_seed47',last_checkpoint_step=c['step'],planned_updates=c['protocol']['updates'],
 resume_sha256=hashlib.sha256(resume.read_bytes()).hexdigest(),
 stopped_owned_pids=[81449,82426],test_loaded=False,checkpoints_preserved=True,automatic_resume=False)
(r/'manual_interruption.json').write_text(json.dumps(receipt,indent=2))
queue['status']='stopped';queue['stop_receipt']=str(r/'manual_interruption.json');queue['finished']=time.time()
for j in queue['jobs']:
 if j['status']=='running':j.update(status='stopped',finished=time.time(),reason=receipt['reason'])
(r/'pilot_queue/queue_state.json').write_text(json.dumps(queue,indent=2))
print(json.dumps(receipt))
'''
result=transport({'command':"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY"})
Path(__file__).with_name('manual_interruption.json').write_text(result.strip(),encoding='utf8',newline='\n')
print(result)
