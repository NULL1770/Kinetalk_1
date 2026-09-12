from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def stable_key(speaker: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{speaker}".encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--val-speakers", type=int, default=4)
    p.add_argument("--test-speakers", type=int, default=5)
    args = p.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    speakers = sorted({str(row.get("source_clip_id", "")).split("_")[1] for row in rows})
    ordered = sorted(speakers, key=lambda x: stable_key(x, args.seed))
    test = set(ordered[: args.test_speakers])
    val = set(ordered[args.test_speakers : args.test_speakers + args.val_speakers])
    train = set(speakers) - val - test
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for name, chosen in (("train", train), ("val", val), ("test", test)):
        path = out / f"teacher_manifest_{name}_speaker.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                speaker = str(row.get("source_clip_id", "")).split("_")[1]
                if speaker in chosen:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(name, len(chosen), sorted(chosen), sum(str(r.get("source_clip_id", "")).split("_")[1] in chosen for r in rows), path)


if __name__ == "__main__":
    main()
