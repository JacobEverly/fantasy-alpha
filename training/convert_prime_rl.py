"""Convert our SFT chat JSONL into the layout prime-rl's SFT trainer loads.

prime-rl (`SFTDataConfig.name` -> HF `datasets.load_dataset(name)`) accepts a
local directory of split-named JSONL files: `train.jsonl` / `validation.jsonl`
inside one directory, each line `{"messages": [...]}`. Extra columns are
carried along but unused; we strip `meta` down to a JSON string to keep arrow
schema inference happy (nested heterogenous dicts can break it).

Loss masking is prime-rl's default `LossMaskConfig` (assistant=True, all other
roles False) — evidence text lives in user messages and is therefore never a
completion target (training/README.md licensing requirement). Verify in the
run's resolved-config dump that `data.loss_mask` shows user=false.

T0 usage (pilot corpus, deterministic split):
    python3 -m training.convert_prime_rl data/processed/sft/pilot_v0.jsonl out_dir --val-frac 0.08

T1 note: replace the random split with a season split (train 2015-2022,
validation 2024) using meta.season — pilot_v0 has no 2024 traces, so T0 uses a
random holdout purely to exercise the val loop.
"""

import argparse
import json
import random
from pathlib import Path


def convert(src: Path, out_dir: Path, val_frac: float, seed: int = 20260808) -> tuple[int, int]:
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    for r in rows:
        assert set(m["role"] for m in r["messages"]) <= {"system", "user", "assistant"}, r.keys()
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac))
    val, train = rows[:n_val], rows[n_val:]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train.jsonl", train), ("validation.jsonl", val)):
        with open(out_dir / name, "w") as f:
            for r in split:
                f.write(json.dumps({"messages": r["messages"], "meta": json.dumps(r.get("meta", {}))}) + "\n")
    return len(train), len(val)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--val-frac", type=float, default=0.08)
    args = ap.parse_args()
    n_train, n_val = convert(args.src, args.out_dir, args.val_frac)
    print(f"wrote {n_train} train / {n_val} validation -> {args.out_dir}")
