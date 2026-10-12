"""Stage an explicit source/evidence/video archive; exclude weights/private data."""
from pathlib import Path
import json
import re
import subprocess

PROJECT=Path(__file__).resolve().parents[2]
paths=[]
for folder in ('kinetalk_v2','kinetalk_v3','kinetalk_v4','kinetalk_v5'):
    paths.extend((PROJECT/folder).rglob('*.py'))
for folder in ('scripts','tests'):
    raw=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',folder],cwd=PROJECT)
    paths.extend(PROJECT/p for p in raw.decode('utf8').splitlines() if p.endswith('.py'))
docs=PROJECT/'docs/v5_optimization_20261011'
# Current complete evidence, supporting source exporters, prior compact tables,
# and recovery records. Older large diagnostic dumps remain locally preserved.
paths.extend(p for p in docs.iterdir() if p.is_file() and (
    p.suffix in ('.py','.csv') or p.name in ('progress.md','task_plan.md',
    'dual_timescale_review_20261012.md','dual_timescale_protocol_20261012.md',
    'current_four_factor_status_20261012.md','dual_tables_provenance.json',
    'dual_new_parameter_diagnostics.json','dual_timescale_smoke_numeric_only.json')))
paths.extend((docs/'dual_evidence').rglob('*.json'))
paths.append(PROJECT/'docs/CURRENT_OPTIMIZATION_STATE.md')
paths.extend((PROJECT/'final_experiment/rig').glob('lip_constraints_v2_20261010*'))
art=PROJECT/'artifacts/v5_dual_timescale_review_20261012'
paths.append(art/'native_review_receipt.json')
for emotion in ('angry','happy'):
    paths.append(art/(emotion+'_inputs.npz'))
    paths.extend(art/(emotion+'_render')/name for name in
        ('comparison.mp4','display_report.json','rig_audit.json','uniform_time_inspection.png'))
paths=sorted(set(p for p in paths if p.is_file()))
secret=[]
pattern=re.compile(r'''(?i)(?:password|passwd|token|api_key)\s*[=:]\s*['"]([^'"\r\n]{8,})['"]''')
for p in paths:
    if p.suffix not in ('.py','.json','.md','.csv'):continue
    content=p.read_text(encoding='utf-8-sig')
    for match in pattern.finditer(content):
        value=match.group(1)
        if value.lower() in ('ssh password: ',):continue
        secret.append(str(p.relative_to(PROJECT)));break
if secret:raise SystemExit(json.dumps(dict(status='secret_scan_requires_review',paths=secret)))
too_large=[str(p.relative_to(PROJECT)) for p in paths if p.stat().st_size>50*1024*1024]
if too_large:raise SystemExit(json.dumps(dict(status='large_file_review',paths=too_large)))
relative=[p.relative_to(PROJECT).as_posix() for p in paths]
# Remove only archive additions that an earlier, broader pass staged under
# this evidence directory; this changes the Git index, never local evidence.
staged=subprocess.check_output(['git','diff','--cached','--name-only','--diff-filter=A','-z'],cwd=PROJECT).decode('utf8').split('\0')
extra=[p for p in staged if p.startswith('docs/v5_optimization_20261011/') and p not in relative]
for start in range(0,len(extra),60):
    subprocess.run(['git','restore','--staged','--',*extra[start:start+60]],cwd=PROJECT,check=True)
for start in range(0,len(relative),60):
    subprocess.run(['git','add','-f','--',*relative[start:start+60]],cwd=PROJECT,check=True)
print(json.dumps(dict(status='staged',files=len(paths),bytes=sum(p.stat().st_size for p in paths))))
