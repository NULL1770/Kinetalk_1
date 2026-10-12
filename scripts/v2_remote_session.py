"""Interactive, pinned-host SSH bridge. Password is never stored."""
import getpass
import hashlib
import json
from pathlib import Path
import shlex
import sys
import paramiko

PIN = '10fb2d83c8cd37ebb706cd49c903aa9417aadf0f9ae15ca7a0901ee3c627215d'

class PinnedHost(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        if hashlib.sha256(key.asbytes()).hexdigest() != PIN:
            raise RuntimeError('SSH host key differs from verified server')

client = paramiko.SSHClient()
client.set_missing_host_key_policy(PinnedHost())
client.connect('connect.nmb1.seetacloud.com', port=11473, username='root',
               password=getpass.getpass('SSH password: '), look_for_keys=False,
               allow_agent=False, timeout=30)
print(json.dumps({'connected': True}), flush=True)
try:
    for line in sys.stdin:
        try:
            q = json.loads(line)
            action = q['action']
            if action == 'close':
                break
            if action in ('get', 'put'):
                with client.open_sftp() as sftp:
                    if action == 'get':
                        sftp.get(q['remote'], q['local'])
                    else:
                        sftp.put(q['local'], q['remote'])
                result = {'action': action, 'remote': q['remote']}
            else:
                if action == 'python':
                    source = Path(q['script']).read_text(encoding='utf-8-sig')
                    cmd = "/root/miniconda3/bin/python - <<'CODEX_V2'\n" + source + '\nCODEX_V2'
                elif action == 'command':
                    cmd = q['cmd']
                else:
                    raise ValueError('Unknown action')
                _, out, err = client.exec_command(cmd, timeout=q.get('timeout', 240))
                result = {'stdout': out.read().decode('utf8', errors='replace'),
                          'stderr': err.read().decode('utf8', errors='replace'),
                          'exit_status': out.channel.recv_exit_status()}
            if q.get('output'):
                Path(q['output']).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as e:
            print(json.dumps({'error': str(e)}), flush=True)
finally:
    client.close()
