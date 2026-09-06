"""Tinker training/sampling backend for Fantasy Alpha.

This module deliberately keeps provider mechanics separate from the corpus,
environment, and evaluation code.  The same frozen JSONL remains usable by
prime-rl; Tinker is an additional backend, not a migration of the scientific
contract.

Only the assistant response receives loss.  The Qwen3.5 no-thinking renderer
is used for both training and inference so the chat template is identical at
both ends of the experiment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "data/processed/sft/t1_corpus_v1.jsonl"
T0_CORPUS = ROOT / "data/processed/sft/pilot_v0.jsonl"
BASE_MODEL = "Qwen/Qwen3.5-9B"
RENDERER_NAME = "qwen3_5_disable_thinking"
TINKER_VERSION = "0.27.1"
TINKER_COOKBOOK_VERSION = "0.5.7"
CORPUS_SHA256 = "b68020f1209c78c085d492a486d0905fa5e74631c33755c6a5bfd3f68a5d0f2f"
T0_CORPUS_SHA256 = "3ebcf9ea68c46d750afd2855764cb2f2c1d8fd257703bc27f8fd025ce1679e4a"
EXPECTED_CORPUS_ROWS = 811
TRAIN_SEASONS = frozenset(range(2008, 2023)) - {2013, 2018}
SEALED_SEASONS = frozenset({2013, 2018, 2023, 2024, 2025})

# Published Tinker prices for Qwen3.5-9B, checked 2026-09-04.
# https://tinker-docs.thinkingmachines.ai/tinker/models/
TRAIN_USD_PER_MTOK = 1.463
PREFILL_USD_PER_MTOK = 0.660
CACHED_PREFILL_USD_PER_MTOK = 0.132
SAMPLE_USD_PER_MTOK = 1.995
STORAGE_USD_PER_GB_MONTH = 0.10

_SECRET_RE = re.compile(r"(?:tml|sk|dtn)_[A-Za-z0-9_-]{12,}|tml-[A-Za-z0-9_-]{12,}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_text(value: object) -> str:
    """Redact token-like strings before an exception reaches an artifact."""
    return _SECRET_RE.sub("[REDACTED]", str(value))


def require_api_key() -> None:
    key = os.environ.get("TINKER_API_KEY", "")
    if not key:
        raise RuntimeError("TINKER_API_KEY is required")
    if not key.startswith("tml-"):
        raise RuntimeError("TINKER_API_KEY has an unexpected format")


def _group_key(row: dict) -> tuple[str, str, int, str]:
    meta = row["meta"]
    return (
        str(meta["bench"]), str(meta["family"]), int(meta["season"]),
        str(meta["question_id"]),
    )


def load_corpus(path: Path = DEFAULT_CORPUS, *, exact: bool = True) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing frozen corpus: {path}")
    if exact and sha256_file(path) != CORPUS_SHA256:
        raise ValueError("frozen T1 corpus SHA-256 mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if exact and len(rows) != EXPECTED_CORPUS_ROWS:
        raise ValueError(f"expected {EXPECTED_CORPUS_ROWS} rows, found {len(rows)}")
    trace_ids: set[str] = set()
    for i, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or [m.get("role") for m in messages] != [
            "system", "user", "assistant"
        ]:
            raise ValueError(f"row {i}: expected system/user/assistant messages")
        if not all(isinstance(m.get("content"), str) and m["content"] for m in messages):
            raise ValueError(f"row {i}: empty or non-text message")
        meta = row.get("meta") or {}
        season = int(meta.get("season", -1))
        if season not in TRAIN_SEASONS or season in SEALED_SEASONS:
            raise ValueError(f"row {i}: season {season} violates the frozen train split")
        tid = str(meta.get("trace_id", ""))
        if not tid or tid in trace_ids:
            raise ValueError(f"row {i}: missing or duplicate trace_id")
        trace_ids.add(tid)
        if meta.get("track") not in {"anonymized", "named_pilot"}:
            raise ValueError(f"row {i}: unexpected track {meta.get('track')!r}")
    return rows


def load_t0_corpus(path: Path = T0_CORPUS) -> list[dict]:
    """Load the exact 162-row subset used by the original Prime T0 run."""
    if sha256_file(path) != T0_CORPUS_SHA256:
        raise ValueError("existing T0 corpus SHA-256 mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 162:
        raise ValueError(f"expected 162 T0 rows, found {len(rows)}")
    for i, row in enumerate(rows):
        if [m.get("role") for m in row.get("messages", [])] != [
            "system", "user", "assistant"
        ]:
            raise ValueError(f"T0 row {i}: expected system/user/assistant messages")
    return rows


def grouped_development_split(
    rows: Sequence[dict], *, fraction: float = 0.10, seed: int = 20260808
) -> tuple[list[dict], list[dict]]:
    """Stable, family-stratified split with question variants kept together."""
    if not 0.0 < fraction < 0.5:
        raise ValueError("development fraction must be between 0 and 0.5")
    strata: dict[tuple[str, str], dict[tuple[str, str, int, str], list[dict]]] = {}
    for row in rows:
        meta = row["meta"]
        stratum = (str(meta["bench"]), str(meta["family"]))
        strata.setdefault(stratum, {}).setdefault(_group_key(row), []).append(row)
    train: list[dict] = []
    development: list[dict] = []
    for stratum in sorted(strata):
        groups = list(strata[stratum].items())
        groups.sort(key=lambda item: hashlib.sha256(
            f"{seed}:{item[0]}".encode()
        ).hexdigest())
        target = max(1, round(sum(len(v) for _, v in groups) * fraction))
        n = 0
        for _, members in groups:
            if n < target:
                development.extend(members)
                n += len(members)
            else:
                train.extend(members)
    train.sort(key=lambda r: r["meta"]["trace_id"])
    development.sort(key=lambda r: r["meta"]["trace_id"])
    tg = {_group_key(r) for r in train}
    dg = {_group_key(r) for r in development}
    if not train or not development or tg & dg:
        raise AssertionError("invalid grouped development split")
    return train, development


def smoke_subset(rows: Sequence[dict], per_family: int = 6) -> list[dict]:
    out: list[dict] = []
    by_family: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        meta = row["meta"]
        by_family.setdefault((meta["bench"], meta["family"]), []).append(row)
    for key in sorted(by_family):
        out.extend(sorted(by_family[key], key=lambda r: r["meta"]["trace_id"])[:per_family])
    return out


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    experiment: str = "tinker-sft-v1"
    model: str = BASE_MODEL
    renderer: str = RENDERER_NAME
    lora_rank: int = 32
    learning_rate: float = 3e-4
    batch_size: int = 32
    epochs: int = 2
    max_length: int = 4096
    development_fraction: float = 0.10
    seed: int = 20260808
    checkpoint_every_epochs: int = 1
    checkpoint_at_fraction: float | None = None


@dataclass
class CostLedger:
    training_tokens: int = 0
    prefill_tokens: int = 0
    cached_prefill_tokens: int = 0
    sample_tokens: int = 0
    checkpoint_count: int = 0

    @property
    def usd(self) -> float:
        return (
            self.training_tokens * TRAIN_USD_PER_MTOK
            + self.prefill_tokens * PREFILL_USD_PER_MTOK
            + self.cached_prefill_tokens * CACHED_PREFILL_USD_PER_MTOK
            + self.sample_tokens * SAMPLE_USD_PER_MTOK
        ) / 1_000_000

    def add_training(self, datums: Sequence[Any]) -> None:
        self.training_tokens += sum(d.model_input.length for d in datums)

    def add_sample(self, prompt_tokens: int, response: Any) -> None:
        cached = min(int(getattr(response, "prompt_cache_hit_tokens", 0) or 0), prompt_tokens)
        self.cached_prefill_tokens += cached
        self.prefill_tokens += prompt_tokens - cached
        self.sample_tokens += sum(len(s.tokens) for s in response.sequences)


def _deps() -> dict[str, Any]:
    try:
        import tinker
        import tinker_cookbook
        from tinker_cookbook import renderers
        from tinker_cookbook.supervised.common import compute_mean_nll
        from tinker_cookbook.supervised.data import conversation_to_datum
        from tinker_cookbook.tokenizer_utils import get_tokenizer
    except ImportError as exc:  # pragma: no cover - exercised by CLI installations
        raise RuntimeError("install the project with the 'tinker' extra") from exc
    if getattr(tinker, "__version__", None) != TINKER_VERSION:
        raise RuntimeError(f"expected tinker {TINKER_VERSION}")
    if getattr(tinker_cookbook, "__version__", None) != TINKER_COOKBOOK_VERSION:
        raise RuntimeError(f"expected tinker-cookbook {TINKER_COOKBOOK_VERSION}")
    return {
        "tinker": tinker,
        "renderers": renderers,
        "compute_mean_nll": compute_mean_nll,
        "conversation_to_datum": conversation_to_datum,
        "get_tokenizer": get_tokenizer,
    }


def render_rows(rows: Sequence[dict], config: RunConfig) -> tuple[list[Any], Any, Any]:
    d = _deps()
    tokenizer = d["get_tokenizer"](config.model)
    renderer = d["renderers"].get_renderer(
        config.renderer, tokenizer, model_name=config.model
    )
    train_on = d["renderers"].TrainOnWhat.LAST_ASSISTANT_MESSAGE
    datums = [
        d["conversation_to_datum"](
            row["messages"], renderer, config.max_length, train_on
        )
        for row in rows
    ]
    for row, datum in zip(rows, datums, strict=True):
        loss_weight = float(row.get("meta", {}).get("loss_weight", 1.0))
        if not math.isfinite(loss_weight) or loss_weight <= 0:
            raise ValueError("loss_weight must be a positive finite number")
        if loss_weight != 1.0:
            weights = datum.loss_fn_inputs["weights"]
            datum.loss_fn_inputs["weights"] = d["tinker"].TensorData(
                data=[float(value) * loss_weight for value in weights.data],
                dtype=weights.dtype,
                shape=weights.shape,
            )
    return datums, tokenizer, renderer


def corpus_manifest(
    rows: Sequence[dict], config: RunConfig, *, source_path: Path = DEFAULT_CORPUS
) -> dict[str, Any]:
    datums, _, _ = render_rows(rows, config)
    lengths = [d.model_input.length for d in datums]
    targets = [sum(float(x) > 0 for x in d.loss_fn_inputs["weights"].data) for d in datums]
    weighted_targets = [sum(float(x) for x in d.loss_fn_inputs["weights"].data) for d in datums]
    if max(lengths) > config.max_length:
        raise ValueError("at least one rendered example exceeds max_length")
    return {
        "created_at": utc_now(),
        "corpus_path": str(source_path.relative_to(ROOT)),
        "corpus_sha256": sha256_file(source_path),
        "rows": len(rows),
        "model": config.model,
        "renderer": config.renderer,
        "max_length": config.max_length,
        "total_rendered_tokens": sum(lengths),
        "trainable_assistant_tokens": sum(targets),
        "weighted_trainable_token_mass": sum(weighted_targets),
        "sequence_tokens": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "p95": sorted(lengths)[math.floor(0.95 * (len(lengths) - 1))],
            "max": max(lengths),
        },
        "truncated_examples": sum(x >= config.max_length for x in lengths),
    }


def estimate_run_cost(
    train_rows: Sequence[dict], development_rows: Sequence[dict], config: RunConfig
) -> dict[str, float | int]:
    train_datums, _, _ = render_rows(train_rows, config)
    dev_datums, _, _ = render_rows(development_rows, config)
    training_tokens = sum(d.model_input.length for d in train_datums) * config.epochs
    # Development NLL is measured before training and after every epoch.
    validation_passes = config.epochs + 1 + int(config.checkpoint_at_fraction is not None)
    validation_tokens = sum(d.model_input.length for d in dev_datums) * validation_passes
    total = training_tokens + validation_tokens
    return {
        "training_tokens": training_tokens,
        "validation_forward_tokens": validation_tokens,
        "billable_training_tokens": total,
        "estimated_training_usd": total * TRAIN_USD_PER_MTOK / 1_000_000,
    }


def _batches(items: Sequence[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield list(items[start:start + size])


def _batch_nll(result: Any, datums: Sequence[Any], compute_mean_nll: Any) -> float:
    logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
    weights = [d.loss_fn_inputs["weights"] for d in datums]
    return float(compute_mean_nll(logprobs, weights))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _loss_summary(metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    train = [float(row["mean_nll"]) for row in metrics if row["type"] == "train"]
    development = [
        float(row["mean_nll"]) for row in metrics if row["type"] == "development"
    ]
    if not train or not development:
        raise ValueError("run metrics do not contain both train and development losses")
    return {
        "initial_development_nll": development[0],
        "final_development_nll": development[-1],
        "development_nll_history": development,
        "first_training_batch_nll": train[0],
        "final_training_batch_nll": train[-1],
        "training_loss_improved": train[-1] < train[0],
        "development_loss_improved": development[-1] < development[0],
    }


def _sample(
    sampling_client: Any, renderer: Any, messages: list[dict], *, max_tokens: int,
    ledger: CostLedger, temperature: float = 0.0, seed: int = 20260808,
) -> dict[str, Any]:
    d = _deps()
    prompt = renderer.build_generation_prompt(messages)
    response = sampling_client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=d["tinker"].SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=renderer.get_stop_sequences(),
            seed=seed,
        ),
    ).result()
    ledger.add_sample(prompt.length, response)
    sequence = response.sequences[0]
    parsed, termination = renderer.parse_response(sequence.tokens)
    content = parsed.get("content", "")
    if isinstance(content, list):
        content = "".join(str(getattr(x, "text", x)) for x in content)
    return {
        "content": str(content),
        "prompt_tokens": prompt.length,
        "output_tokens": len(sequence.tokens),
        "stop_reason": sequence.stop_reason,
        "parse_termination": str(termination),
    }


class TinkerChatSampler:
    """Small provider adapter shared by the SFT lifecycle and eval harness."""

    def __init__(
        self, *, model_path: str | None = None, model: str = BASE_MODEL,
        experiment: str = "tinker-sft-v1",
    ):
        require_api_key()
        d = _deps()
        self.service = d["tinker"].ServiceClient(user_metadata={
            "project": "fantasy-alpha", "experiment": experiment,
            "operation": "frozen-evaluation",
        })
        validate_service_model(self.service, model)
        self.client = self.service.create_sampling_client(
            model_path=model_path
        ) if model_path else self.service.create_sampling_client(base_model=model)
        tokenizer = d["get_tokenizer"](model)
        self.renderer = d["renderers"].get_renderer(
            RENDERER_NAME, tokenizer, model_name=model
        )
        self.ledger = CostLedger()

    def chat(
        self, messages: list[dict], *, max_tokens: int, temperature: float, seed: int
    ) -> dict[str, Any]:
        return _sample(
            self.client, self.renderer, messages, max_tokens=max_tokens,
            ledger=self.ledger, temperature=temperature, seed=seed,
        )


def validate_service_model(service_client: Any, model: str = BASE_MODEL) -> dict[str, Any]:
    capabilities = service_client.get_server_capabilities()
    matches = [m for m in capabilities.supported_models if m.model_name == model]
    if len(matches) != 1:
        raise RuntimeError(f"{model} is not uniquely present in Tinker capabilities")
    m = matches[0]
    if not m.trainable or not m.sampleable:
        raise RuntimeError(f"{model} must be trainable and sampleable")
    return {
        "model": m.model_name,
        "trainable": m.trainable,
        "sampleable": m.sampleable,
        "max_context_length": m.max_context_length,
    }


def run_sft(
    rows: Sequence[dict], config: RunConfig, out_dir: Path, *, hard_cap_usd: float = 100.0,
    source_path: Path = DEFAULT_CORPUS,
) -> dict[str, Any]:
    """Run a recoverable LoRA experiment and export the final adapter archive."""
    require_api_key()
    d = _deps()
    tinker = d["tinker"]
    out_dir.mkdir(parents=True, exist_ok=True)
    completion_path = out_dir / "run-summary.json"
    if completion_path.exists():
        existing = json.loads(completion_path.read_text())
        if existing.get("status") == "complete":
            return existing
        raise RuntimeError(f"refusing to overwrite incomplete run directory: {out_dir}")

    train_rows, dev_rows = grouped_development_split(
        rows, fraction=config.development_fraction, seed=config.seed
    )
    estimate = estimate_run_cost(train_rows, dev_rows, config)
    if float(estimate["estimated_training_usd"]) >= hard_cap_usd:
        raise RuntimeError("estimated training cost reaches the hard cap")
    train_datums, _, renderer = render_rows(train_rows, config)
    dev_datums, _, _ = render_rows(dev_rows, config)
    metrics_path = out_dir / "metrics.jsonl"
    config_path = out_dir / "config.json"
    started = datetime.now(timezone.utc)
    config_payload = {
        **asdict(config),
        "tinker_version": TINKER_VERSION,
        "tinker_cookbook_version": TINKER_COOKBOOK_VERSION,
        "corpus_path": str(source_path.relative_to(ROOT)),
        "corpus_sha256": sha256_file(source_path),
        "train_rows": len(train_rows),
        "development_rows": len(dev_rows),
        "train_groups": len({_group_key(r) for r in train_rows}),
        "development_groups": len({_group_key(r) for r in dev_rows}),
        "estimate": estimate,
        "credential_source": "TINKER_API_KEY environment variable",
        "started_at": started.isoformat(),
    }
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n")

    ledger = CostLedger()
    user_metadata = {
        "project": "fantasy-alpha",
        "experiment": config.experiment,
        "run_name": config.run_name,
        "renderer_name": config.renderer,
        "corpus_sha256": sha256_file(source_path)[:16],
    }
    service = tinker.ServiceClient(user_metadata=user_metadata)
    capability = validate_service_model(service, config.model)
    training_client = service.create_lora_training_client(
        base_model=config.model,
        rank=config.lora_rank,
        seed=config.seed,
        user_metadata=user_metadata,
    )
    training_run_id = str(training_client.get_info().model_id)

    def evaluate(step: int, epoch: int) -> float:
        values: list[tuple[float, int]] = []
        for batch in _batches(dev_datums, config.batch_size):
            result = training_client.forward(batch, loss_fn="cross_entropy").result()
            ledger.add_training(batch)
            values.append((_batch_nll(result, batch, d["compute_mean_nll"]), len(batch)))
        nll = sum(v * n for v, n in values) / sum(n for _, n in values)
        _append_jsonl(metrics_path, {
            "type": "development", "step": step, "epoch": epoch,
            "mean_nll": nll, "billable_tokens": ledger.training_tokens,
            "estimated_cost_usd": ledger.usd, "timestamp": utc_now(),
        })
        return nll

    initial_dev_nll = evaluate(0, 0)
    total_steps = config.epochs * math.ceil(len(train_datums) / config.batch_size)
    fraction_step = None
    if config.checkpoint_at_fraction is not None:
        if not 0.0 < config.checkpoint_at_fraction < 1.0:
            raise ValueError("checkpoint_at_fraction must be between zero and one")
        fraction_step = max(1, min(total_steps - 1, round(
            total_steps * config.checkpoint_at_fraction
        )))
    step = 0
    checkpoints: list[dict[str, Any]] = []
    dev_history = [initial_dev_nll]
    for epoch in range(1, config.epochs + 1):
        order = list(range(len(train_datums)))
        random.Random(config.seed + epoch).shuffle(order)
        shuffled = [train_datums[i] for i in order]
        for batch in _batches(shuffled, config.batch_size):
            step += 1
            # A short linear decay keeps the last update meaningful instead of hitting zero.
            progress = (step - 1) / max(1, total_steps - 1)
            lr = config.learning_rate * (1.0 - 0.9 * progress)
            fb = training_client.forward_backward(batch, loss_fn="cross_entropy")
            opt = training_client.optim_step(tinker.AdamParams(
                learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8,
                grad_clip_norm=1.0,
            ))
            result = fb.result()
            optim_result = opt.result()
            ledger.add_training(batch)
            _append_jsonl(metrics_path, {
                "type": "train", "step": step, "epoch": epoch,
                "mean_nll": _batch_nll(result, batch, d["compute_mean_nll"]),
                "learning_rate": lr, "num_sequences": len(batch),
                "num_tokens": sum(x.model_input.length for x in batch),
                "optimizer_metrics": optim_result.metrics or {},
                "billable_tokens": ledger.training_tokens,
                "estimated_cost_usd": ledger.usd, "timestamp": utc_now(),
            })
            if fraction_step is not None and step == fraction_step:
                fraction_dev_nll = evaluate(step, epoch)
                dev_history.append(fraction_dev_nll)
                name = f"{config.run_name}-step-{step}"
                state = training_client.save_state(
                    name, ttl_seconds=7 * 24 * 3600
                ).result()
                sampler = training_client.save_weights_for_sampler(
                    f"{name}-sampler", ttl_seconds=7 * 24 * 3600,
                    user_metadata=user_metadata,
                ).result()
                ledger.checkpoint_count += 2
                checkpoints.append({
                    "fraction": config.checkpoint_at_fraction,
                    "step": step,
                    "development_nll": fraction_dev_nll,
                    "state_path": state.path,
                    "sampler_path": sampler.path,
                })
        dev_nll = evaluate(step, epoch)
        dev_history.append(dev_nll)
        if epoch % config.checkpoint_every_epochs == 0 and epoch < config.epochs:
            name = f"{config.run_name}-epoch-{epoch}"
            state = training_client.save_state(name, ttl_seconds=7 * 24 * 3600).result()
            sampler = training_client.save_weights_for_sampler(
                f"{name}-sampler", ttl_seconds=7 * 24 * 3600,
                user_metadata=user_metadata,
            ).result()
            ledger.checkpoint_count += 2
            checkpoints.append({
                "epoch": epoch, "step": step, "development_nll": dev_nll,
                "state_path": state.path, "sampler_path": sampler.path,
            })

    final_state = training_client.save_state(
        f"{config.run_name}-final-state", ttl_seconds=None
    ).result()
    final_sampler = training_client.save_weights_for_sampler(
        f"{config.run_name}-final-sampler", ttl_seconds=None,
        user_metadata=user_metadata,
    ).result()
    ledger.checkpoint_count += 2

    # Reloading through a new client proves the saved state is operational.
    reloaded = service.create_training_client_from_state_with_optimizer(final_state.path)
    info = reloaded.get_info()
    reloaded_model = info.model_data.model_name
    reloaded_rank = info.lora_rank
    if reloaded_model != config.model or int(reloaded_rank) != config.lora_rank:
        raise RuntimeError("reloaded checkpoint metadata does not match the run")

    prompt_messages = list(dev_rows[0]["messages"][:-1])
    base_sampling = service.create_sampling_client(base_model=config.model)
    adapter_sampling = service.create_sampling_client(model_path=final_sampler.path)
    sample_base = _sample(
        base_sampling, renderer, prompt_messages, max_tokens=420, ledger=ledger
    )
    sample_adapter = _sample(
        adapter_sampling, renderer, prompt_messages, max_tokens=420, ledger=ledger
    )
    (out_dir / "sample-base.json").write_text(
        json.dumps(sample_base, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "sample-adapter.json").write_text(
        json.dumps(sample_adapter, indent=2, sort_keys=True) + "\n"
    )

    archive_info = service.create_rest_client().get_checkpoint_archive_url_from_tinker_path(
        final_sampler.path
    ).result()
    archive_path = out_dir / "final-adapter.tar.gz"
    urllib.request.urlretrieve(archive_info.url, archive_path)

    loss = _loss_summary(_read_metrics(metrics_path))
    summary = {
        "status": "complete",
        "completed_at": utc_now(),
        "config": config_payload,
        "capability": capability,
        "training_run_id": training_run_id,
        "steps": step,
        **loss,
        "checkpoints": checkpoints,
        "final_state_path": final_state.path,
        "final_sampler_path": final_sampler.path,
        "reload_verified": True,
        "reloaded_model": reloaded_model,
        "reloaded_lora_rank": reloaded_rank,
        "archive": {
            "path": str(archive_path.relative_to(ROOT)),
            "sha256": sha256_file(archive_path),
            "bytes": archive_path.stat().st_size,
        },
        "cost": {**asdict(ledger), "computed_usd": ledger.usd},
        "runtime_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "sample_prompt_trace_id": dev_rows[0]["meta"]["trace_id"],
        "secret_persisted": False,
    }
    completion_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def recover_sft_finalization(
    rows: Sequence[dict], config: RunConfig, out_dir: Path
) -> dict[str, Any]:
    """Finish sampling/export after a post-training client-side failure.

    Recovery is deliberately read-mostly: it discovers the named remote run and
    reuses its already-saved final checkpoints, so optimizer work is never repeated.
    """
    require_api_key()
    d = _deps()
    tinker = d["tinker"]
    completion_path = out_dir / "run-summary.json"
    if completion_path.exists():
        existing = json.loads(completion_path.read_text())
        if existing.get("status") == "complete":
            return existing
        raise RuntimeError(f"refusing to overwrite incomplete summary: {completion_path}")
    config_path = out_dir / "config.json"
    metrics_path = out_dir / "metrics.jsonl"
    if not config_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("recovery requires the original config and metrics")
    config_payload = json.loads(config_path.read_text())
    for key, expected in asdict(config).items():
        if config_payload.get(key) != expected:
            raise ValueError(f"recovery config mismatch for {key}")

    service = tinker.ServiceClient(user_metadata={
        "project": "fantasy-alpha", "experiment": "tinker-sft-v1",
        "operation": "post-training-recovery",
    })
    rest = service.create_rest_client()
    candidates = [
        run for run in rest.list_training_runs(limit=50).result().training_runs
        if (run.user_metadata or {}).get("run_name") == config.run_name
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one remote training run named {config.run_name}, found {len(candidates)}"
        )
    remote_run = candidates[0]
    checkpoints = rest.list_checkpoints(remote_run.training_run_id).result().checkpoints
    state_name = f"weights/{config.run_name}-final-state"
    sampler_name = f"sampler_weights/{config.run_name}-final-sampler"
    state = next((x for x in checkpoints if x.checkpoint_id == state_name), None)
    sampler = next((x for x in checkpoints if x.checkpoint_id == sampler_name), None)
    if state is None or sampler is None:
        raise RuntimeError("remote run is missing a final state or sampler checkpoint")

    reloaded = service.create_training_client_from_state_with_optimizer(state.tinker_path)
    info = reloaded.get_info()
    if info.model_data.model_name != config.model or int(info.lora_rank or -1) != config.lora_rank:
        raise RuntimeError("reloaded checkpoint metadata does not match the run")

    _, dev_rows = grouped_development_split(
        rows, fraction=config.development_fraction, seed=config.seed
    )
    _, _, renderer = render_rows(dev_rows[:1], config)
    ledger = CostLedger()
    metrics = _read_metrics(metrics_path)
    ledger.training_tokens = max(int(row["billable_tokens"]) for row in metrics)
    ledger.checkpoint_count = len(checkpoints)
    prompt_messages = list(dev_rows[0]["messages"][:-1])
    sample_base = _sample(
        service.create_sampling_client(base_model=config.model), renderer,
        prompt_messages, max_tokens=420, ledger=ledger,
    )
    sample_adapter = _sample(
        service.create_sampling_client(model_path=sampler.tinker_path), renderer,
        prompt_messages, max_tokens=420, ledger=ledger,
    )
    (out_dir / "sample-base.json").write_text(
        json.dumps(sample_base, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "sample-adapter.json").write_text(
        json.dumps(sample_adapter, indent=2, sort_keys=True) + "\n"
    )

    archive_info = rest.get_checkpoint_archive_url_from_tinker_path(
        sampler.tinker_path
    ).result()
    archive_path = out_dir / "final-adapter.tar.gz"
    urllib.request.urlretrieve(archive_info.url, archive_path)
    started = datetime.fromisoformat(config_payload["started_at"])
    finished = datetime.now(timezone.utc)
    final_paths = {state.tinker_path, sampler.tinker_path}
    intermediate = [
        {"checkpoint_type": x.checkpoint_type, "path": x.tinker_path,
         "expires_at": x.expires_at.isoformat() if x.expires_at else None}
        for x in checkpoints if x.tinker_path not in final_paths
    ]
    loss = _loss_summary(metrics)
    summary = {
        "status": "complete",
        "completed_at": finished.isoformat(),
        "config": config_payload,
        "capability": validate_service_model(service, config.model),
        "training_run_id": str(remote_run.training_run_id),
        "steps": max(int(row["step"]) for row in metrics),
        **loss,
        "checkpoints": intermediate,
        "final_state_path": state.tinker_path,
        "final_sampler_path": sampler.tinker_path,
        "reload_verified": True,
        "reloaded_model": info.model_data.model_name,
        "reloaded_lora_rank": info.lora_rank,
        "archive": {
            "path": str(archive_path.relative_to(ROOT)),
            "sha256": sha256_file(archive_path),
            "bytes": archive_path.stat().st_size,
        },
        "cost": {**asdict(ledger), "computed_usd": ledger.usd},
        "runtime_seconds": (finished - started).total_seconds(),
        "sample_prompt_trace_id": dev_rows[0]["meta"]["trace_id"],
        "recovery": {
            "reason": "Tinker SDK 0.27.1 exposes lora_rank on GetInfoResponse",
            "optimizer_steps_repeated": 0,
            "finalization_attempts": 2,
        },
        "secret_persisted": False,
    }
    completion_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def billing_snapshot(starting_on: str, ending_before: str) -> dict[str, Any]:
    """Provider-authoritative token events; dollar amounts are computed separately."""
    require_api_key()
    d = _deps()
    service = d["tinker"].ServiceClient(
        user_metadata={"project": "fantasy-alpha", "operation": "billing-audit"}
    )
    response = service.create_rest_client().get_billing_usage(
        starting_on, ending_before
    ).result()
    events = []
    for event in response.data:
        info = event.event_info
        row = {
            "bucket_start": event.bucket_start.isoformat(),
            "base_model": event.base_model,
            "session_id": event.session_id,
            "type": info.type,
        }
        for key in ("token_count", "gigabyte_hours", "count", "cached"):
            if hasattr(info, key):
                row[key] = getattr(info, key)
        events.append(row)
    return {"queried_at": utc_now(), "events": events}


def billing_window_for_today() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return start.isoformat(), end.isoformat()
