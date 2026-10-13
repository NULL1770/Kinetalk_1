"""Archive the author-supplied manuscript before revising its current sources."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('D:/科研/小论文')
DEST = ROOT / 'paper_history/20261014_before_v5_revision'
SKIP = {'node_modules', '.architecture_build', 'ppt_refactor_build', '__pycache__'}
KEEP = {'.md', '.bib', '.tex', '.docx', '.pptx', '.csv', '.json', '.png', '.jpg', '.pdf'}

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    entries = []
    for src in sorted(SOURCE.rglob('*')):
        rel = src.relative_to(SOURCE)
        if not src.is_file() or any(p in SKIP for p in rel.parts):
            continue
        if src.name.startswith('~$') or src.suffix.lower() not in KEEP:
            continue
        dst = DEST / rel
        data = src.read_bytes()
        if dst.exists() and dst.read_bytes() != data:
            raise RuntimeError('Immutable archive already differs: ' + str(rel))
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        entries.append({'path': rel.as_posix(), 'bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest()})
    manifest = {'schema': 'author_manuscript_archive_v1', 'source': str(SOURCE),
                'purpose': 'Original documents before frozen-v5 rewrite',
                'excluded': sorted(SKIP), 'files': entries}
    (DEST / 'archive_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'destination': str(DEST), 'files': len(entries),
                      'bytes': sum(x['bytes'] for x in entries)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
