"""Archive one explicitly retired run; reclaim only after local verification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ALLOWED_ROOT = Path('/root/kinetalk_temporal_source_20261005')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*2**20),b''):h.update(block)
    return h.hexdigest()


def inventory(root):
    files={};directories=[]
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('Archive root contains a link')
        rel=str(p.relative_to(root))
        if p.is_file():files[rel]={'bytes':p.stat().st_size,'sha256':sha(p)}
        elif p.is_dir():directories.append(rel)
        else:raise ValueError('Special file')
    return files,directories


def safe_root(root):
    if root!=ALLOWED_ROOT or root.resolve()!=ALLOWED_ROOT:
        raise ValueError('Only the explicitly retired absolute run is allowed')
    if json.loads((root/'postprocess_state.json').read_text())['status']!='complete':
        raise ValueError('Run not complete')
    processes=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines()
    for line in processes:
        pid=int(line.split(None,1)[0])
        if pid!=os.getpid() and str(root) in line and any(k in line for k in ('python','torchrun')):
            raise ValueError('Run still in use by process')


def pack(root,out):
    safe_root(root)
    out.mkdir(parents=True,exist_ok=True)
    if out.resolve()!=Path('/dev/shm/kinetalk_cleanup_20261007'):
        raise ValueError('Unexpected temporary archive directory')
    archive=out/(root.name+'.tar.zst');mp=out/(root.name+'.manifest.json')
    if archive.exists() or mp.exists():raise FileExistsError('Fresh archive required')
    files,dirs=inventory(root)
    partial=archive.with_suffix(archive.suffix+'.part')
    with partial.open('xb') as target:
        process=subprocess.Popen(['zstd','-q','-3','-T2','-c'],stdin=subprocess.PIPE,stdout=target)
        try:
            with tarfile.open(fileobj=process.stdin,mode='w|') as tf:
                for name in dirs:tf.add(root/name,arcname=name,recursive=False)
                for name in files:tf.add(root/name,arcname=name,recursive=False)
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('Compression failed')
        except BaseException:
            process.kill();process.wait();raise
        target.flush();os.fsync(target.fileno())
    partial.rename(archive)
    seen=set();seen_dirs=set()
    process=subprocess.Popen(['zstd','-q','-d','-c',str(archive)],stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout,mode='r|') as tf:
        for member in tf:
            if member.isdir():
                if member.name not in dirs or member.name in seen_dirs:raise ValueError('Directory differs')
                seen_dirs.add(member.name);continue
            if not member.isfile() or member.name not in files or member.name in seen:
                raise ValueError('Archive contains unexpected entry')
            h=hashlib.sha256();count=0
            with tf.extractfile(member) as f:
                for block in iter(lambda:f.read(4*2**20),b''):h.update(block);count+=len(block)
            if {'bytes':count,'sha256':h.hexdigest()}!=files[member.name]:raise ValueError('Member differs')
            seen.add(member.name)
    if process.wait()!=0 or seen!=set(files) or seen_dirs!=set(dirs):raise ValueError('Archive incomplete')
    after,after_dirs=inventory(root)
    if after!=files or after_dirs!=dirs:raise ValueError('Original changed')
    report=dict(schema='retired_run_archive_v1',remote_root=str(root),archive=str(archive),
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,files=files,directories=dirs,
        total_bytes=sum(v['bytes'] for v in files.values()),every_entry_verified=True,
        originals_unchanged=True,training_performed=False,test_loaded=False,created_at=time.time())
    mp.write_text(json.dumps(report,indent=2))
    return dict(status='packed',manifest=str(mp),archive=str(archive),
                archive_sha256=report['archive_sha256'],original_bytes=report['total_bytes'])


def reclaim(root,manifest,receipt):
    safe_root(root)
    report=json.loads(manifest.read_text());proof=json.loads(receipt.read_text())
    if not (report['remote_root']==proof['remote_root']==str(root)
            and proof['passed'] and proof['every_entry_verified']
            and proof['manifest_sha256']==sha(manifest)
            and proof['archive_sha256']==report['archive_sha256']):
        raise ValueError('Persistent local verification differs')
    if sha(report['archive'])!=report['archive_sha256']:raise ValueError('Temporary archive changed')
    files,dirs=inventory(root)
    if files!=report['files'] or dirs!=report['directories']:raise ValueError('Original changed before reclaim')
    # Only this validated exact root is removed; unique bytes are preserved in
    # the verified persistent local archive. No current data/model is touched.
    shutil.rmtree(root);root.mkdir()
    pointer=dict(schema='verified_local_archive_pointer_v1',**proof,reclaimed_bytes=report['total_bytes'],
        note='Retired run archived locally, every member SHA verified. Restore archive before using original paths.')
    (root/'ARCHIVED.json').write_text(json.dumps(pointer,indent=2))
    return dict(status='reclaimed',root=str(root),bytes=report['total_bytes'],
                free_GiB=shutil.disk_usage('/root').free/2**30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['pack','reclaim'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output-dir',type=Path)
    p.add_argument('--manifest',type=Path);p.add_argument('--receipt',type=Path)
    a=p.parse_args();result=pack(a.root,a.output_dir) if a.mode=='pack' else reclaim(a.root,a.manifest,a.receipt)
    print(json.dumps(result),flush=True)
