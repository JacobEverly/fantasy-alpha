"""CLI for the Fantasy Alpha Tinker SFT milestone."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.t11_dataset import (
    CORPUS_PATH as T11_CORPUS,
)
from training.t11_dataset import (
    canary_subset as t11_canary_subset,
)
from training.t11_dataset import (
    load_t11_corpus,
)
from training.t12_dataset import (
    CORPUS_PATH as T12_CORPUS,
)
from training.t12_dataset import (
    canary_subset as t12_canary_subset,
)
from training.t12_dataset import (
    load_t12_corpus,
)
from training.tinker_backend import (
    DEFAULT_CORPUS,
    ROOT,
    T0_CORPUS,
    RunConfig,
    billing_snapshot,
    billing_window_for_today,
    corpus_manifest,
    estimate_run_cost,
    grouped_development_split,
    load_corpus,
    load_t0_corpus,
    recover_sft_finalization,
    run_sft,
    safe_text,
    smoke_subset,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fantasy Alpha SFT through Tinker")
    sub = ap.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="validate and count the frozen corpus (no API)")
    pre.add_argument("--out", type=Path,
                     default=ROOT / "artifacts/tinker-sft-v1/preflight.json")
    smoke = sub.add_parser("smoke", help="paid T0 lifecycle smoke test")
    smoke.add_argument("--out-dir", type=Path,
                       default=ROOT / "artifacts/tinker-sft-v1/t0-smoke")
    legacy = sub.add_parser("smoke-existing-t0", help="run the exact legacy T0 corpus")
    legacy.add_argument("--out-dir", type=Path,
                        default=ROOT / "artifacts/tinker-sft-v1/t0-existing-smoke")
    train = sub.add_parser("train", help="paid full T1 SFT run")
    train.add_argument("--out-dir", type=Path,
                       default=ROOT / "artifacts/tinker-sft-v1/t1-full")
    recover = sub.add_parser("recover-smoke", help="resume post-training T0 checks")
    recover.add_argument("--out-dir", type=Path,
                         default=ROOT / "artifacts/tinker-sft-v1/t0-smoke")
    bill = sub.add_parser("billing", help="snapshot provider billing events")
    bill.add_argument("--out", type=Path,
                      default=ROOT / "artifacts/tinker-sft-v1/billing.json")
    t11_pre = sub.add_parser(
        "t11-preflight", help="validate/count the targeted T1.1 corpus (no API)"
    )
    t11_pre.add_argument(
        "--out", type=Path,
        default=ROOT / "artifacts/tinker-sft-t11/preflight.json",
    )
    t11_canary = sub.add_parser(
        "t11-canary", help="paid final-format T1.1 lifecycle canary"
    )
    t11_canary.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "artifacts/tinker-sft-t11/canary",
    )
    t11_train = sub.add_parser("t11-train", help="paid targeted T1.1 SFT run")
    t11_train.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "artifacts/tinker-sft-t11/t11-full",
    )
    t12_pre = sub.add_parser(
        "t12-preflight", help="validate/count the T1.2 weighted mixture (no API)"
    )
    t12_pre.add_argument(
        "--out", type=Path,
        default=ROOT / "artifacts/tinker-sft-t12/preflight.json",
    )
    t12_canary = sub.add_parser(
        "t12-canary", help="paid representative T1.2 lifecycle canary"
    )
    t12_canary.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "artifacts/tinker-sft-t12/canary-v2",
    )
    t12_train = sub.add_parser("t12-train", help="paid T1.2 weighted SFT run")
    t12_train.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "artifacts/tinker-sft-t12/t12-full",
    )
    args = ap.parse_args(argv)

    if args.command.startswith("t12-"):
        rows = load_t12_corpus()
    elif args.command.startswith("t11-"):
        rows = load_t11_corpus()
    else:
        rows = load_corpus(DEFAULT_CORPUS)
    if args.command == "preflight":
        cfg = RunConfig(run_name="fantasy-alpha-t1-full")
        train_rows, dev_rows = grouped_development_split(rows)
        payload = {
            "corpus": corpus_manifest(rows, cfg),
            "split": {
                "train_rows": len(train_rows),
                "development_rows": len(dev_rows),
                "question_group_overlap": 0,
            },
            "estimate": estimate_run_cost(train_rows, dev_rows, cfg),
        }
        _write(args.out, payload)
    elif args.command == "smoke":
        cfg = RunConfig(
            run_name="fantasy-alpha-t0-tinker-smoke", batch_size=8, epochs=6,
            development_fraction=0.25, checkpoint_every_epochs=3,
        )
        print(json.dumps(run_sft(smoke_subset(rows), cfg, args.out_dir), indent=2))
    elif args.command == "smoke-existing-t0":
        cfg = RunConfig(
            run_name="fantasy-alpha-t0-existing-pilot", batch_size=16, epochs=2,
            checkpoint_every_epochs=1,
        )
        print(json.dumps(run_sft(
            load_t0_corpus(), cfg, args.out_dir, source_path=T0_CORPUS
        ), indent=2))
    elif args.command == "train":
        cfg = RunConfig(run_name="fantasy-alpha-t1-qwen35-9b-r32")
        print(json.dumps(run_sft(rows, cfg, args.out_dir), indent=2))
    elif args.command == "recover-smoke":
        cfg = RunConfig(
            run_name="fantasy-alpha-t0-tinker-smoke", batch_size=8, epochs=6,
            development_fraction=0.25, checkpoint_every_epochs=3,
        )
        print(json.dumps(
            recover_sft_finalization(smoke_subset(rows), cfg, args.out_dir), indent=2
        ))
    elif args.command == "billing":
        start, end = billing_window_for_today()
        _write(args.out, billing_snapshot(start, end))
    elif args.command == "t11-preflight":
        cfg = RunConfig(
            run_name="fantasy-alpha-t11-qwen35-9b-r32",
            experiment="tinker-sft-t11",
        )
        train_rows, dev_rows = grouped_development_split(rows)
        payload = {
            "corpus": corpus_manifest(rows, cfg, source_path=T11_CORPUS),
            "split": {
                "train_rows": len(train_rows),
                "development_rows": len(dev_rows),
                "question_group_overlap": 0,
            },
            "estimate": estimate_run_cost(train_rows, dev_rows, cfg),
            "incremental_budget_usd": 10.0,
        }
        _write(args.out, payload)
    elif args.command == "t11-canary":
        cfg = RunConfig(
            run_name="fantasy-alpha-t11-final-format-canary",
            experiment="tinker-sft-t11",
            batch_size=8,
            epochs=3,
            development_fraction=0.25,
        )
        print(json.dumps(run_sft(
            t11_canary_subset(rows), cfg, args.out_dir,
            hard_cap_usd=10.0, source_path=T11_CORPUS,
        ), indent=2))
    elif args.command == "t11-train":
        cfg = RunConfig(
            run_name="fantasy-alpha-t11-qwen35-9b-r32",
            experiment="tinker-sft-t11",
        )
        print(json.dumps(run_sft(
            rows, cfg, args.out_dir, hard_cap_usd=10.0, source_path=T11_CORPUS,
        ), indent=2))
    elif args.command == "t12-preflight":
        cfg = RunConfig(
            run_name="fantasy-alpha-t12-qwen35-9b-r32",
            experiment="tinker-sft-t12", learning_rate=1e-4, epochs=1,
            checkpoint_at_fraction=0.5,
        )
        train_rows, dev_rows = grouped_development_split(rows)
        payload = {
            "corpus": corpus_manifest(rows, cfg, source_path=T12_CORPUS),
            "split": {
                "train_rows": len(train_rows),
                "development_rows": len(dev_rows),
                "question_group_overlap": 0,
                "development_schemas": sorted({
                    row["meta"]["t12_schema"] for row in dev_rows
                }),
            },
            "estimate": estimate_run_cost(train_rows, dev_rows, cfg),
            "incremental_budget_target_usd": 7.0,
            "incremental_hard_cap_usd": 10.0,
        }
        _write(args.out, payload)
    elif args.command == "t12-canary":
        cfg = RunConfig(
            run_name="fantasy-alpha-t12-representative-canary-v2",
            experiment="tinker-sft-t12", learning_rate=1e-4,
            batch_size=18, epochs=1, development_fraction=0.25,
            checkpoint_at_fraction=0.5,
        )
        print(json.dumps(run_sft(
            t12_canary_subset(rows), cfg, args.out_dir,
            hard_cap_usd=10.0, source_path=T12_CORPUS,
        ), indent=2))
    elif args.command == "t12-train":
        cfg = RunConfig(
            run_name="fantasy-alpha-t12-qwen35-9b-r32",
            experiment="tinker-sft-t12", learning_rate=1e-4, epochs=1,
            checkpoint_at_fraction=0.5,
        )
        print(json.dumps(run_sft(
            rows, cfg, args.out_dir, hard_cap_usd=10.0, source_path=T12_CORPUS,
        ), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI boundary redacts provider errors
        raise SystemExit(f"Tinker operation failed: {safe_text(exc)}") from None
