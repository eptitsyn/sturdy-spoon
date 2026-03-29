#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

from dataset_loader import (
    DatasetConfig,
    Label,
    _SOURCE_REGISTRY,
    _canonical_source_name,
    _clean_text,
    _first_present,
    _iter_local_rows,
    _materialize_source,
    _parse_label,
    _raw_source_dir,
    _resolve_local_source_path,
)
from ui import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    RICH_AVAILABLE,
    SpinnerColumn,
    Table,
    TextColumn,
    TimeElapsedColumn,
    box,
    console,
    print_info,
    print_section,
    print_warning,
)


INTERESTING_RAW_FIELDS = (
    "label",
    "#label",
    "source",
    "model",
    "generator",
    "class",
    "target",
    "is_ai",
    "generated",
    "domain",
    "human_text",
    "ai_text",
    "wiki_intro",
    "generated_intro",
    "Human_story",
    "text",
    "abstract",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug dataset loading and label assignment")
    parser.add_argument("--sources", nargs="+", default=None, help="Subset of dataset sources to inspect")
    parser.add_argument("--source-path", action="append", default=[], help="source=/abs/path")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max loaded samples per source")
    parser.add_argument("--raw-preview", type=int, default=5, help="How many raw rows to preview")
    parser.add_argument("--sample-preview", type=int, default=3, help="How many loaded samples to preview per label")
    parser.add_argument("--cache-dir", default=".cache/datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-processed-cache", action="store_true", help="Reuse processed source cache")
    parser.add_argument("--allow-kaggle-download", action="store_true", help="Allow Kaggle auto-download for local-only datasets")
    return parser.parse_args()


def parse_source_paths(items: list[str]) -> dict[str, str]:
    source_paths: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --source-path value: {item}. Expected source=/path.")
        source_name, path = item.split("=", 1)
        source_name = source_name.strip()
        path = path.strip()
        if not source_name or not path:
            raise ValueError(f"Invalid --source-path value: {item}. Expected source=/path.")
        source_paths[source_name] = path
    return source_paths


def _label_name(value: int | Label) -> str:
    return "human" if int(value) == int(Label.HUMAN) else "ai"


def _shorten(value: object, limit: int = 100) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _sample_issue(source_name: str, sample) -> str | None:
    canonical = _canonical_source_name(source_name)
    generator = _clean_text(sample.generator).lower()

    if canonical in {"raid", "ai_pile", "gpt_wiki", "human_vs_ai", "ai_human_mixed", "hc3", "coat", "gsingh_train"}:
        if generator == "human" and sample.label != Label.HUMAN:
            return f"generator=human but label={_label_name(sample.label)}"
        if generator != "human" and sample.label != Label.AI:
            return f"generator={sample.generator} but label={_label_name(sample.label)}"

    return None


def _infer_expected_label_from_raw(source_name: str, row: dict) -> Label | None:
    canonical = _canonical_source_name(source_name)
    if canonical == "raid":
        model = _clean_text(row.get("model")).lower()
        if not model:
            return None
        return Label.HUMAN if model == "human" else Label.AI
    if canonical == "ai_pile":
        source = _clean_text(row.get("source")).lower()
        if not source:
            return None
        return Label.HUMAN if source == "human" else Label.AI
    if canonical == "ai_human_mixed":
        return _parse_label(row.get("label"))
    if canonical in {"coat", "ruatd"}:
        raw_label = _clean_text(row.get("label"))
        if not raw_label:
            return None
        return Label.HUMAN if raw_label.strip("\"' ").lower() == "human" else Label.AI
    if canonical in {"daigt_proper", "daigt_v2", "m_daigt"}:
        return _parse_label(_first_present(row, "#label", "label", "generated", "is_ai", "target", "class"))
    return None


def _find_raw_path(source_name: str, cfg: DatasetConfig) -> Path | None:
    canonical = _canonical_source_name(source_name)
    if canonical == "coat":
        authorship_path = _raw_source_dir(canonical, cfg) / "authorship"
        if authorship_path.exists():
            return authorship_path
    if canonical in {"daigt_proper", "daigt_v2", "m_daigt"}:
        try:
            return _resolve_local_source_path(canonical, cfg)
        except FileNotFoundError:
            return None

    candidate = _raw_source_dir(source_name, cfg)
    if candidate.exists():
        return candidate

    if canonical != source_name:
        canonical_candidate = _raw_source_dir(canonical, cfg)
        if canonical_candidate.exists():
            return canonical_candidate
    return None


def _print_counter_table(title: str, counter: Counter, limit: int = 10) -> None:
    items = counter.most_common(limit)
    if not items:
        print_info(f"{title}: none")
        return

    if RICH_AVAILABLE:
        table = Table(title=title, box=box.SIMPLE_HEAVY)
        table.add_column("Value")
        table.add_column("Count", justify="right")
        for value, count in items:
            table.add_row(str(value), f"{count:,}")
        console.print(table)
        return

    print_info(title)
    for value, count in items:
        print_info(f"  {value}: {count:,}")


def _print_sample_preview(source_name: str, samples: list, sample_preview: int) -> None:
    by_label = {
        Label.HUMAN: [s for s in samples if s.label == Label.HUMAN][:sample_preview],
        Label.AI: [s for s in samples if s.label == Label.AI][:sample_preview],
    }
    for label, label_samples in by_label.items():
        if not label_samples:
            print_info(f"Preview {_label_name(label)}: none")
            continue
        print_info(f"Preview {_label_name(label)}:")
        for idx, sample in enumerate(label_samples, start=1):
            print_info(
                f"  {idx}. generator={sample.generator!r} domain={sample.domain!r} text={_shorten(sample.text, 140)!r}"
            )


def _inspect_raw_rows(source_name: str, cfg: DatasetConfig, raw_preview: int) -> None:
    raw_path = _find_raw_path(source_name, cfg)
    if raw_path is None:
        print_warning("Raw source path is not available.")
        return

    print_info(f"Raw path: {raw_path}")
    raw_files = [path for path in raw_path.rglob("*") if path.is_file()] if raw_path.is_dir() else [raw_path]
    if not raw_files:
        print_warning("Raw cache exists but contains no files.")
        return

    try:
        rows = list(islice(_iter_local_rows(raw_path), raw_preview))
    except Exception as exc:
        print_warning(f"Failed to inspect raw rows: {exc}")
        return

    if not rows:
        print_warning("No raw rows found.")
        return

    field_presence: Counter[str] = Counter()
    field_values: dict[str, Counter[str]] = {field: Counter() for field in INTERESTING_RAW_FIELDS}
    raw_expected_labels: Counter[str] = Counter()

    for row, _ in rows:
        for field in INTERESTING_RAW_FIELDS:
            value = _first_present(row, field)
            if value is None:
                continue
            field_presence[field] += 1
            cleaned = _shorten(value, 60)
            if cleaned:
                field_values[field][cleaned] += 1

        expected = _infer_expected_label_from_raw(source_name, row)
        if expected is not None:
            raw_expected_labels[_label_name(expected)] += 1

    _print_counter_table("Raw field presence", field_presence)
    if raw_expected_labels:
        _print_counter_table("Expected labels from raw preview", raw_expected_labels)

    for field in INTERESTING_RAW_FIELDS:
        if field_values[field]:
            _print_counter_table(f"Raw values: {field}", field_values[field], limit=5)

    print_info("Raw row preview:")
    for idx, (row, file_path) in enumerate(rows, start=1):
        preview = {}
        for field in INTERESTING_RAW_FIELDS:
            value = _first_present(row, field)
            if value is None:
                continue
            preview[field] = _shorten(value, 80)
        if not preview:
            preview = {key: _shorten(value, 80) for key, value in list(row.items())[:8]}
        print_info(f"  {idx}. file={file_path.name} {preview}")


def _load_samples_for_source(source_name: str, cfg: DatasetConfig, max_samples: int) -> list:
    loader = _SOURCE_REGISTRY[source_name]
    if cfg.cache_sources:
        return _materialize_source(source_name, max_samples, cfg, loader)
    return list(loader(max_samples, cfg))


def inspect_source(source_name: str, cfg: DatasetConfig, args: argparse.Namespace) -> None:
    print_section(f"Source: {source_name}")

    try:
        samples = _load_samples_for_source(source_name, cfg, args.max_samples)
    except Exception as exc:
        print_warning(f"Failed to load {source_name}: {exc}")
        if isinstance(exc, ModuleNotFoundError) and exc.name == "datasets":
            print_warning("Install the 'datasets' package in the training environment to enable live loading.")
        _inspect_raw_rows(source_name, cfg, args.raw_preview)
        return

    if not samples:
        print_warning("No samples loaded.")
        return

    label_counts = Counter(_label_name(sample.label) for sample in samples)
    generator_counts = Counter(_clean_text(sample.generator) or "unknown" for sample in samples)
    domain_counts = Counter(_clean_text(sample.domain) or "unknown" for sample in samples)
    text_lengths = [len(sample.text) for sample in samples]
    issues = []
    progress_context = (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        if RICH_AVAILABLE
        else nullcontext()
    )
    with progress_context as progress:
        task_id = progress.add_task(f"Validating {source_name}", total=len(samples)) if RICH_AVAILABLE else None
        for sample in samples:
            issue = _sample_issue(source_name, sample)
            if issue is not None:
                issues.append((issue, sample))
                if len(issues) >= 10:
                    if RICH_AVAILABLE:
                        progress.advance(task_id)
                    break
            if RICH_AVAILABLE:
                progress.advance(task_id)

    print_info(f"Loaded {len(samples):,} samples")
    print_info(
        f"Text length: min={min(text_lengths):,} avg={sum(text_lengths) / len(text_lengths):.1f} max={max(text_lengths):,}"
    )
    _print_counter_table("Loaded label counts", label_counts)
    _print_counter_table("Top generators", generator_counts)
    _print_counter_table("Top domains", domain_counts)

    if issues:
        print_warning(f"Found {len(issues)} invariant violations in previewed samples:")
        for idx, (issue, sample) in enumerate(issues, start=1):
            print_warning(
                f"  {idx}. {issue}; generator={sample.generator!r}; text={_shorten(sample.text, 140)!r}"
            )
    else:
        print_info("No sample-level invariant violations found.")

    _print_sample_preview(source_name, samples, args.sample_preview)
    _inspect_raw_rows(source_name, cfg, args.raw_preview)


def main() -> None:
    args = parse_args()
    source_paths = parse_source_paths(args.source_path)
    cfg = DatasetConfig(
        sources=args.sources,
        max_per_source=args.max_samples,
        max_total=args.max_samples,
        min_text_length=1,
        max_text_length=20_000,
        balance_labels=False,
        balance_sources=False,
        seed=args.seed,
        source_paths=source_paths,
        cache_dir=args.cache_dir,
        auto_download_kaggle=args.allow_kaggle_download,
        cache_sources=args.use_processed_cache,
    )

    requested_sources = args.sources or cfg.available_sources
    seen: set[str] = set()
    sources: list[str] = []
    for source_name in requested_sources:
        if source_name not in _SOURCE_REGISTRY:
            print_warning(f"Unknown source: {source_name}")
            continue
        canonical = _canonical_source_name(source_name)
        if canonical in seen:
            print_warning(f"Skipping duplicate alias {source_name} -> {canonical}")
            continue
        seen.add(canonical)
        sources.append(source_name)

    if not sources:
        raise SystemExit("No valid sources selected.")

    progress_context = (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        if RICH_AVAILABLE
        else nullcontext()
    )
    with progress_context as progress:
        task_id = progress.add_task("Validating datasets", total=len(sources)) if RICH_AVAILABLE else None
        for source_name in sources:
            inspect_source(source_name, cfg, args)
            if RICH_AVAILABLE:
                progress.advance(task_id)


if __name__ == "__main__":
    main()
