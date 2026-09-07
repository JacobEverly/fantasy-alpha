"""Dedicated Tinker CLI for the high-confidence override experiment.

Kept separate from ``tinker_sft.py`` because completed T1.2 experiments pin
that historical entry point by hash.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.override_dataset import CORPUS_PATH, load_override_corpus
from training.tinker_backend import (
    ROOT, RunConfig, corpus_manifest, estimate_run_cost,
    grouped_development_split, run_sft, safe_text,
)

DEFAULT_OUT = ROOT / "artifacts/high-confidence-override-v1"


def config() -> RunConfig:
    return RunConfig(
        run_name="fantasy-alpha-override-qwen35-9b-r16",
        experiment="high-confidence-override-v1",
        lora_rank=16,
        learning_rate=1e-4,
        batch_size=32,
        epochs=1,
        development_fraction=0.15,
        checkpoint_at_fraction=None,
    )


def preflight(out: Path) -> dict:
    rows = load_override_corpus()
    cfg = config()
    train_rows, dev_rows = grouped_development_split(
        rows, fraction=cfg.development_fraction, seed=cfg.seed
    )
    train_episodes = {row["meta"]["episode_id"] for row in train_rows}
    dev_episodes = {row["meta"]["episode_id"] for row in dev_rows}
    payload = {
        "corpus": corpus_manifest(rows, cfg, source_path=CORPUS_PATH),
        "split": {
            "train_rows": len(train_rows),
            "development_rows": len(dev_rows),
            "train_episodes": len(train_episodes),
            "development_episodes": len(dev_episodes),
            "episode_overlap": len(train_episodes & dev_episodes),
        },
        "estimate": estimate_run_cost(train_rows, dev_rows, cfg),
        "incremental_hard_cap_usd": 10.0,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--out", type=Path, default=DEFAULT_OUT / "preflight.json")
    train = sub.add_parser("train")
    train.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "tinker-run")
    args = parser.parse_args()
    if args.command == "preflight":
        payload = preflight(args.out)
    else:
        payload = run_sft(
            load_override_corpus(), config(), args.out_dir,
            hard_cap_usd=10.0, source_path=CORPUS_PATH,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - redact provider errors at CLI boundary
        raise SystemExit(f"Tinker override operation failed: {safe_text(exc)}") from None
