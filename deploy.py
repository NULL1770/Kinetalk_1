"""One-shot deployment script for the Stage2/3 upgrade experiment.

Steps:
  1. SFTP-upload the four locally-modified files (idempotent overwrite).
  2. On the remote host, back up existing Stage-2/3/4 checkpoints to a
     timestamped folder under ``outputs_b0_residual/backup_<stamp>/``.
  3. Verify no training process is currently running (abort if there is).
  4. Launch ``run_stages_234.sh`` in the background via ``nohup``, write
     stdout/stderr to ``logs/pipeline_<stamp>.log``, and print the PID so
     you can attach with ``tail -f`` later.

Usage:
  python deploy.py                     # prompts for SSH password
  SSHPASS='...' python deploy.py       # non-interactive with env var
"""
from __future__ import annotations

import getpass
import hashlib
import os
import posixpath
import sys
import time
from pathlib import Path

import paramiko


HOST = "connect.nmb1.seetacloud.com"
PORT = 31948
USER = "root"
REMOTE_ROOT = "/root/autodl-tmp/kinetalk_b0_residual_train"
LOCAL_ROOT = Path(__file__).resolve().parent

MODIFIED_FILES = [
    "kinetalk_b0/data.py",
    "kinetalk_b0/losses.py",
    "kinetalk_b0/models/model.py",
    "kinetalk_b0/models/encoders.py",
    "train.py",
    "configs/train.yaml",
    "scripts/02_eval_all_stages.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(client: paramiko.SSHClient, command: str, *, timeout: int = 60) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    return exit_code, stdout.read().decode(errors="replace"), stderr.read().decode(errors="replace")


def upload(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    parent = posixpath.dirname(remote)
    if parent:
        try:
            sftp.stat(parent)
        except FileNotFoundError:
            parts = parent.split("/")
            current = ""
            for piece in parts:
                if not piece:
                    current = ""
                    continue
                current = f"{current}/{piece}" if current else piece
                path = "/" + current if parent.startswith("/") else current
                try:
                    sftp.stat(path)
                except FileNotFoundError:
                    sftp.mkdir(path)
    sftp.put(str(local), remote)


def main() -> int:
    password = os.environ.get("SSHPASS") or getpass.getpass(f"SSH password for {USER}@{HOST}: ")

    print(f"Connecting to {USER}@{HOST}:{PORT} ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=password, timeout=30)
    sftp = client.open_sftp()

    try:
        # Match "train.py --" so this pgrep line does not match itself.
        code, out, _ = run(client, "pgrep -af 'train\\.py --' | grep -v pgrep || true")
        if out.strip():
            print("ERROR: training process is still running:")
            print(out)
            print("Kill it manually first (see PIDs above), then re-run this script.")
            return 2

        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{REMOTE_ROOT}/outputs_b0_residual/backup_{stamp}"
        print(f"Backing up existing stage2/3/4 checkpoints to {backup_dir}/ ...")
        run(client, f"mkdir -p {backup_dir}")
        run(client, f"for f in stage2_factors.pt stage3_audio_emotion.pt stage4_generator.pt; do "
                    f"src={REMOTE_ROOT}/outputs_b0_residual/$f; [ -f $src ] && cp -v $src {backup_dir}/; done")

        print("\nUploading modified files:")
        for rel in MODIFIED_FILES:
            local_path = LOCAL_ROOT / rel
            remote_path = f"{REMOTE_ROOT}/{rel}"
            local_hash = sha256(local_path)
            upload(sftp, local_path, remote_path)
            code, out, _ = run(client, f"sha256sum {remote_path}")
            remote_hash = out.split()[0] if out else "?"
            match = "OK" if remote_hash == local_hash else "MISMATCH"
            print(f"  {match}  {rel}  ({local_hash[:12]})")

        # Fresh log per run so a repeated deploy does not concatenate output.
        log_path = f"{REMOTE_ROOT}/logs/pipeline_{stamp}.log"
        run(client, f"mkdir -p {REMOTE_ROOT}/logs")
        print(f"\nStarting run_stages_234.sh in background -> {log_path}")
        launch_cmd = (
            f"cd {REMOTE_ROOT} && "
            f"nohup bash run_stages_234.sh > {log_path} 2>&1 & "
            f"echo $! > /tmp/kinetalk_pipeline.pid && "
            f"cat /tmp/kinetalk_pipeline.pid"
        )
        code, out, err = run(client, launch_cmd, timeout=15)
        print("Pipeline PID:", out.strip())
        if err.strip():
            print("STDERR:", err.strip())

        print("\nDone. Follow progress with:")
        print(f"  ssh -p {PORT} {USER}@{HOST} 'tail -f {log_path}'")
        print(f"Stop the run with:")
        print(f"  ssh -p {PORT} {USER}@{HOST} 'kill $(cat /tmp/kinetalk_pipeline.pid)'")
    finally:
        sftp.close()
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
