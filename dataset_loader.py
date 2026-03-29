"""
Загрузчик комбинированного датасета из нескольких источников.

Поддерживаемые датасеты:
  • RAID           — liamdugan/raid (6M+ примеров, 11 LLM, 8 доменов)
  • AI-Pile        — artem9k/ai-text-detection-pile (GPT-2/3/J/ChatGPT + human)
  • GPT-Wiki       — aadityaubhat/GPT-wiki-intro (Wikipedia vs GPT-3)
  • Human-vs-AI    — dmitva/human_ai_generated_text (human vs AI essays)
  • AI-Human-Mixed — Ateeqq/AI-and-Human-Generated-Text (GPT-4 + BARD)
  • HC3            — Hello-SimpleAI/HC3 (Human vs ChatGPT, multi-domain)
  • CoAT / RuATD   — RussianNLP/coat (ruatd = compatibility alias)
  • gsingh train   — gsingh1-py/train (human stories + multiple LLM outputs)
  • DAIGT Proper   — Kaggle DAIGT v2 / local export fallback
  • M-DAIGT        — local export of the shared-task dataset

Использование:
    from dataset_loader import load_combined_dataset, DatasetConfig
    texts, labels, meta = load_combined_dataset(DatasetConfig(max_per_source=5000))
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator

from datasets import load_dataset


# ─── Types ────────────────────────────────────────────────────────────────────

class Label(int, Enum):
    HUMAN = 0
    AI = 1


@dataclass
class Sample:
    text: str
    label: Label
    source_dataset: str
    generator: str = "unknown"  # модель-генератор (для AI) или "human"
    domain: str = "unknown"


@dataclass
class DatasetConfig:
    """Конфигурация загрузки."""

    # Какие датасеты включить (None = все доступные)
    sources: list[str] | None = None

    # Лимиты
    max_per_source: int = 10_000      # макс. сэмплов с одного датасета
    max_total: int = 100_000          # макс. сэмплов итого
    min_text_length: int = 50         # минимальная длина текста (символов)
    max_text_length: int = 5_000      # максимальная длина текста (символов)

    # Балансировка
    balance_labels: bool = True        # выровнять human/ai 50/50
    balance_sources: bool = False      # выровнять между датасетами

    # Воспроизводимость
    seed: int = 42

    # Пути к локальным источникам, если датасет не доступен напрямую через HF.
    # Пример: {"daigt_proper": "/path/to/train.csv", "m_daigt": "/path/to/data_dir"}
    source_paths: dict[str, str] = field(default_factory=dict)
    cache_dir: str = ".cache/datasets"
    auto_download_kaggle: bool = True
    cache_sources: bool = True

    # Список всех доступных источников
    available_sources: list[str] = field(default_factory=lambda: [
        "raid",
        "ai_pile",
        "gpt_wiki",
        "human_vs_ai",
        "ai_human_mixed",
        "hc3",
        "coat",
        "ruatd",
        "daigt_proper",
        "daigt_v2",
        "m_daigt",
        "gsingh_train",
    ])


@dataclass
class LoadResult:
    """Результат загрузки."""
    texts: list[str]
    labels: list[int]
    samples: list[Sample]
    stats: dict


_LOCAL_SOURCE_ENV_VARS = {
    "daigt_proper": "DAIGT_PROPER_PATH",
    "daigt_v2": "DAIGT_PROPER_PATH",
    "m_daigt": "M_DAIGT_PATH",
}

_SUPPORTED_LOCAL_SUFFIXES = {".csv", ".tsv", ".jsonl", ".json", ".parquet"}
_KAGGLE_DATASETS = {
    "daigt_proper": "thedrcat/daigt-v2-train-dataset",
    "daigt_v2": "thedrcat/daigt-v2-train-dataset",
}
_SOURCE_CACHE_SCHEMA_VERSION = 4
_SOURCE_ALIASES = {
    "ruatd": "coat",
    "daigt_v2": "daigt_proper",
    "gsingh1_train": "gsingh_train",
}


def _clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "\n".join(parts)
    return str(value).strip()


def _canonical_source_name(source_name: str) -> str:
    return _SOURCE_ALIASES.get(source_name, source_name)


def _normalized_row(row: dict) -> dict[str, object]:
    return {
        str(key).strip().lower(): value
        for key, value in row.items()
        if key is not None
    }


def _first_present(row: dict, *candidates: str):
    normalized = _normalized_row(row)
    for candidate in candidates:
        value = normalized.get(candidate.lower())
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _parse_label(value) -> Label | None:
    if isinstance(value, Label):
        return value
    if isinstance(value, bool):
        return Label.AI if value else Label.HUMAN
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if int(value) == 1:
            return Label.AI
        if int(value) == 0:
            return Label.HUMAN
        return None

    text = _clean_text(value).lower()
    if not text:
        return None

    if text in {"1", "ai", "generated", "machine-generated", "ai-generated", "m", "true", "yes"}:
        return Label.AI
    if text in {"0", "human", "written", "human-written", "original", "h", "false", "no"}:
        return Label.HUMAN
    return None


def _sample_to_row(sample: Sample) -> dict[str, object]:
    return {
        "text": sample.text,
        "label": int(sample.label),
        "source_dataset": sample.source_dataset,
        "generator": sample.generator,
        "domain": sample.domain,
    }


def _row_to_sample(row: dict) -> Sample:
    return Sample(
        text=_clean_text(row.get("text")),
        label=Label(int(row.get("label", 0))),
        source_dataset=_clean_text(row.get("source_dataset")) or "unknown",
        generator=_clean_text(row.get("generator")) or "unknown",
        domain=_clean_text(row.get("domain")) or "unknown",
    )


def _source_cache_file(source_name: str, max_samples: int, cfg: DatasetConfig) -> Path:
    source_path = cfg.source_paths.get(source_name) or ""
    cache_key = {
        "schema_version": _SOURCE_CACHE_SCHEMA_VERSION,
        "source": source_name,
        "max_samples": max_samples,
        "source_path": str(Path(source_path).expanduser()) if source_path else "",
    }
    digest = hashlib.sha256(
        json.dumps(cache_key, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_root = Path(cfg.cache_dir).expanduser() / "source_samples"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"{source_name}_{digest}.jsonl"


def _load_cached_samples(cache_file: Path) -> list[Sample]:
    samples: list[Sample] = []
    with cache_file.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(_row_to_sample(json.loads(line)))
    return samples


def _write_cached_samples(cache_file: Path, samples: list[Sample]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(_sample_to_row(sample), ensure_ascii=False) + "\n")


def _materialize_source(
    source_name: str,
    max_samples: int,
    cfg: DatasetConfig,
    loader_fn,
) -> list[Sample]:
    if not cfg.cache_sources:
        return list(loader_fn(max_samples, cfg))

    cache_file = _source_cache_file(source_name, max_samples, cfg)
    if cache_file.exists():
        return _load_cached_samples(cache_file)

    samples = list(loader_fn(max_samples, cfg))
    _write_cached_samples(cache_file, samples)
    return samples


def _download_kaggle_dataset(source_name: str, cfg: DatasetConfig) -> Path:
    dataset_slug = _KAGGLE_DATASETS[source_name]
    cache_root = Path(cfg.cache_dir).expanduser()
    target_dir = cache_root / source_name
    target_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(target_dir.rglob("*"))
    if any(path.is_file() and path.suffix.lower() in _SUPPORTED_LOCAL_SUFFIXES for path in existing_files):
        return target_dir

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError:
        KaggleApi = None

    if KaggleApi is not None:
        try:
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(dataset_slug, path=str(target_dir), unzip=True, quiet=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download the Kaggle dataset. "
                "Set KAGGLE_USERNAME and KAGGLE_KEY or provide a local source path."
            ) from exc
        return target_dir

    kaggle_exe = shutil.which("kaggle")
    if kaggle_exe:
        try:
            subprocess.run(
                [
                    kaggle_exe,
                    "datasets",
                    "download",
                    "-d",
                    dataset_slug,
                    "-p",
                    str(target_dir),
                    "--unzip",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Failed to download the Kaggle dataset with the CLI. "
                "Check your Kaggle credentials or provide a local source path."
            ) from exc
        return target_dir

    raise RuntimeError(
        "Kaggle auto-download requested but neither the kaggle package nor CLI is installed."
    )


def _resolve_local_source_path(source_name: str, cfg: DatasetConfig) -> Path:
    env_name = _LOCAL_SOURCE_ENV_VARS.get(source_name)
    raw_path = cfg.source_paths.get(source_name) or (os.getenv(env_name) if env_name else None)
    if not raw_path and cfg.auto_download_kaggle and source_name in _KAGGLE_DATASETS:
        return _download_kaggle_dataset(source_name, cfg)
    if not raw_path:
        raise FileNotFoundError(
            f"{source_name} requires a local export. "
            f"Pass DatasetConfig(source_paths={{'{source_name}': '/path'}})"
            + (f" or set ${env_name}." if env_name else ".")
        )

    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Configured path for {source_name} does not exist: {path}")
    return path


def _iter_local_rows(path: Path) -> Iterator[tuple[dict, Path]]:
    if path.is_file():
        files = [path]
    else:
        files = sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in _SUPPORTED_LOCAL_SUFFIXES
        )

    if not files:
        raise FileNotFoundError(f"No supported data files found in {path}")

    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with file_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                for row in reader:
                    yield row, file_path
            continue

        if suffix == ".jsonl":
            with file_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row, file_path
            continue

        if suffix == ".json":
            with file_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                for row in payload:
                    if isinstance(row, dict):
                        yield row, file_path
            elif isinstance(payload, dict):
                for value in payload.values():
                    if isinstance(value, list):
                        for row in value:
                            if isinstance(row, dict):
                                yield row, file_path
            continue

        if suffix == ".parquet":
            ds = load_dataset("parquet", data_files=str(file_path), split="train", streaming=True)
            for row in ds:
                yield row, file_path


def _infer_domain(source_name: str, row: dict, file_path: Path) -> str:
    domain = _first_present(
        row,
        "domain",
        "subtask",
        "task",
        "category",
        "dataset",
        "source_dataset",
        "source_domain",
    )
    if domain is not None:
        return _clean_text(domain) or "unknown"

    path_hint = file_path.as_posix().lower()
    if "academic" in path_hint or "paper" in path_hint:
        return "academic"
    if "news" in path_hint or "article" in path_hint:
        return "news"
    if source_name == "daigt_proper":
        return "essays"
    if source_name == "m_daigt":
        return "multi_domain"
    return "unknown"


def _iter_local_labeled_dataset(
    source_name: str,
    max_samples: int,
    cfg: DatasetConfig,
    *,
    text_fields: tuple[str, ...],
    label_fields: tuple[str, ...],
    generator_fields: tuple[str, ...] = (),
    balance_binary_labels: bool = False,
) -> Iterator[Sample]:
    path = _resolve_local_source_path(source_name, cfg)
    if balance_binary_labels:
        rng = random.Random(cfg.seed)
        target_human = max_samples // 2
        target_ai = max_samples - target_human
        humans: list[Sample] = []
        ais: list[Sample] = []
    else:
        count = 0

    for row, file_path in _iter_local_rows(path):
        text = _clean_text(_first_present(row, *text_fields))
        if not text:
            continue

        label = _parse_label(_first_present(row, *label_fields))
        if label is None:
            continue

        generator = _clean_text(_first_present(row, *generator_fields))
        if not generator:
            generator = "human" if label == Label.HUMAN else "unknown"

        sample = Sample(
            text=text,
            label=label,
            source_dataset=source_name,
            generator=generator,
            domain=_infer_domain(source_name, row, file_path),
        )
        if balance_binary_labels:
            if sample.label == Label.HUMAN:
                if len(humans) < target_human:
                    humans.append(sample)
            else:
                if len(ais) < target_ai:
                    ais.append(sample)

            if len(humans) >= target_human and len(ais) >= target_ai:
                break
            continue

        yield sample
        count += 1
        if count >= max_samples:
            return

    if balance_binary_labels:
        samples = humans + ais
        rng.shuffle(samples)
        for sample in samples[:max_samples]:
            yield sample


# ─── Адаптеры для каждого датасета ────────────────────────────────────────────

def _iter_raid(max_samples: int, cfg: DatasetConfig | None = None) -> Iterator[Sample]:
    """
    RAID: liamdugan/raid
    Самый большой бенчмарк: 11 LLM, 8 доменов, атаки на детекторы.
    model='human' → human, иначе → AI.
    """
    ds = load_dataset("liamdugan/raid", split="train", streaming=True)
    count = 0
    for row in ds:
        text = (row.get("generation") or "").strip()
        if not text:
            continue
        model = row.get("model", "unknown")
        yield Sample(
            text=text,
            label=Label.HUMAN if model == "human" else Label.AI,
            source_dataset="raid",
            generator=model,
            domain=row.get("domain", "unknown"),
        )
        count += 1
        if count >= max_samples:
            return


def _iter_ai_pile(max_samples: int, cfg: DatasetConfig | None = None) -> Iterator[Sample]:
    """
    artem9k/ai-text-detection-pile
    Long-form essays: human, GPT-2, GPT-3, ChatGPT, GPT-J.
    """
    seed = (cfg.seed if cfg is not None else 42)
    ds = load_dataset("artem9k/ai-text-detection-pile", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=min(max_samples * 8, 50_000))
    rng = random.Random(seed)
    target_human = max_samples // 2
    target_ai = max_samples - target_human
    humans: list[Sample] = []
    ais: list[Sample] = []

    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        source = _clean_text(row.get("source")).lower() or "unknown"
        sample = Sample(
            text=text,
            label=Label.HUMAN if source == "human" else Label.AI,
            source_dataset="ai_pile",
            generator=source,
            domain="essays",
        )

        if sample.label == Label.HUMAN:
            if len(humans) < target_human:
                humans.append(sample)
        else:
            if len(ais) < target_ai:
                ais.append(sample)

        if len(humans) >= target_human and len(ais) >= target_ai:
            break

    samples = humans + ais
    rng.shuffle(samples)
    for sample in samples[:max_samples]:
        yield sample


def _iter_gpt_wiki(max_samples: int, cfg: DatasetConfig | None = None) -> Iterator[Sample]:
    """
    aadityaubhat/GPT-wiki-intro
    Paired: Wikipedia intro (human) vs GPT-3 generated intro.
    Каждый ряд даёт 2 сэмпла.
    """
    ds = load_dataset("aadityaubhat/GPT-wiki-intro", split="train", streaming=True)
    count = 0
    for row in ds:
        wiki = (row.get("wiki_intro") or "").strip()
        gen = (row.get("generated_intro") or "").strip()

        if wiki:
            yield Sample(
                text=wiki,
                label=Label.HUMAN,
                source_dataset="gpt_wiki",
                generator="human",
                domain="wikipedia",
            )
            count += 1
            if count >= max_samples:
                return

        if gen:
            yield Sample(
                text=gen,
                label=Label.AI,
                source_dataset="gpt_wiki",
                generator="gpt-3",
                domain="wikipedia",
            )
            count += 1
            if count >= max_samples:
                return


def _iter_human_vs_ai(max_samples: int, cfg: DatasetConfig | None = None) -> Iterator[Sample]:
    """
    dmitva/human_ai_generated_text
    Paired: human essays vs AI-generated essays.
    """
    ds = load_dataset("dmitva/human_ai_generated_text", split="train", streaming=True)
    count = 0
    for row in ds:
        human = (row.get("human_text") or "").strip()
        ai = (row.get("ai_text") or "").strip()

        if human:
            yield Sample(
                text=human,
                label=Label.HUMAN,
                source_dataset="human_vs_ai",
                generator="human",
                domain="essays",
            )
            count += 1
            if count >= max_samples:
                return

        if ai:
            yield Sample(
                text=ai,
                label=Label.AI,
                source_dataset="human_vs_ai",
                generator="ai",
                domain="essays",
            )
            count += 1
            if count >= max_samples:
                return


def _iter_ai_human_mixed(max_samples: int, cfg: DatasetConfig | None = None) -> Iterator[Sample]:
    """
    Ateeqq/AI-and-Human-Generated-Text
    GPT-4, BARD и human тексты разных жанров.
    label: 0 = human, 1 = AI.
    """
    ds = load_dataset("Ateeqq/AI-and-Human-Generated-Text", split="train", streaming=True)
    count = 0
    for row in ds:
        text = (row.get("abstract") or "").strip()
        if not text:
            continue
        label_val = row.get("label", 0)
        yield Sample(
            text=text,
            label=Label.AI if label_val == 1 else Label.HUMAN,
            source_dataset="ai_human_mixed",
            generator="gpt4/bard" if label_val == 1 else "human",
            domain="mixed",
        )
        count += 1
        if count >= max_samples:
            return


def _iter_hc3(max_samples: int, cfg: DatasetConfig | None = None) -> Iterator[Sample]:
    """
    HC3: Hello-SimpleAI/HC3
    Human vs ChatGPT answers across domains (ELI5, finance, medicine, wiki, open QA).
    Загрузка через huggingface_hub напрямую (loading script сломан).
    """
    import json
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset")
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if count >= max_samples:
                return
            row = json.loads(line)
            domain = row.get("source", "unknown")

            # Human answers
            for answer in row.get("human_answers", []):
                text = answer.strip()
                if text:
                    yield Sample(
                        text=text,
                        label=Label.HUMAN,
                        source_dataset="hc3",
                        generator="human",
                        domain=domain,
                    )
                    count += 1
                    if count >= max_samples:
                        return

            # ChatGPT answers
            for answer in row.get("chatgpt_answers", []):
                text = answer.strip()
                if text:
                    yield Sample(
                        text=text,
                        label=Label.AI,
                        source_dataset="hc3",
                        generator="chatgpt",
                        domain=domain,
                    )
                    count += 1
                    if count >= max_samples:
                        return


def _iter_coat(max_samples: int, cfg: DatasetConfig | None = None, source_name: str = "coat") -> Iterator[Sample]:
    """
    CoAT: RussianNLP/coat
    Public binary subset for Russian AI-text detection.
    """
    ds = load_dataset("RussianNLP/coat", "binary", split="train", streaming=True)
    count = 0
    for row in ds:
        text = _clean_text(row.get("text"))
        label = _parse_label(row.get("label"))
        if not text or label is None:
            continue

        yield Sample(
            text=text,
            label=label,
            source_dataset=source_name,
            generator="human" if label == Label.HUMAN else "unknown_ru_model",
            domain=_clean_text(row.get("domain")) or "russian",
        )
        count += 1
        if count >= max_samples:
            return


def _iter_ruatd(max_samples: int, cfg: DatasetConfig | None = None) -> Iterator[Sample]:
    """
    RuATD now points to CoAT as the public extended corpus, so this source is
    provided as a compatibility alias over the CoAT binary split.
    """
    yield from _iter_coat(max_samples, cfg, source_name="ruatd")


def _iter_gsingh_train(max_samples: int, cfg: DatasetConfig | None = None) -> Iterator[Sample]:
    """
    gsingh1-py/train
    One human story plus several model-generated variants per prompt.
    """
    ds = load_dataset("gsingh1-py/train", split="train", streaming=True)
    count = 0
    ai_columns = [
        "gemma-2-9b",
        "mistral-7B",
        "qwen-2-72B",
        "llama-8B",
        "accounts/yi-01-ai/models/yi-large",
        "GPT_4-o",
    ]

    for row in ds:
        human_text = _clean_text(row.get("Human_story"))
        if human_text:
            yield Sample(
                text=human_text,
                label=Label.HUMAN,
                source_dataset="gsingh_train",
                generator="human",
                domain="stories",
            )
            count += 1
            if count >= max_samples:
                return

        for column in ai_columns:
            ai_text = _clean_text(row.get(column))
            if not ai_text:
                continue
            yield Sample(
                text=ai_text,
                label=Label.AI,
                source_dataset="gsingh_train",
                generator=column,
                domain="stories",
            )
            count += 1
            if count >= max_samples:
                return


def _iter_daigt_proper(max_samples: int, cfg: DatasetConfig | None = None) -> Iterator[Sample]:
    """
    thedrcat/daigt-v2-train-dataset
    Kaggle dataset, auto-downloaded when credentials are available, or read from
    a local CSV/JSON/Parquet export.
    """
    yield from _iter_local_labeled_dataset(
        "daigt_proper",
        max_samples,
        cfg or DatasetConfig(),
        text_fields=("text", "essay", "essay_text", "content"),
        label_fields=("#label", "label", "generated", "is_ai", "target", "generated_text"),
        generator_fields=("source", "model", "generator"),
        balance_binary_labels=True,
    )


def _iter_daigt_v2(max_samples: int, cfg: DatasetConfig | None = None) -> Iterator[Sample]:
    yield from _iter_local_labeled_dataset(
        "daigt_v2",
        max_samples,
        cfg or DatasetConfig(),
        text_fields=("text", "essay", "essay_text", "content"),
        label_fields=("#label", "label", "generated", "is_ai", "target", "generated_text"),
        generator_fields=("source", "model", "generator"),
        balance_binary_labels=True,
    )


def _iter_m_daigt(max_samples: int, cfg: DatasetConfig | None = None) -> Iterator[Sample]:
    """
    M-DAIGT shared-task dataset.
    The public repo contains documentation, while the actual rows are expected
    from a local export directory or file.
    """
    yield from _iter_local_labeled_dataset(
        "m_daigt",
        max_samples,
        cfg or DatasetConfig(),
        text_fields=("text", "article", "content", "body", "snippet", "document"),
        label_fields=("label", "generated", "is_ai", "target", "class"),
        generator_fields=("generator", "model", "llm", "source_model"),
    )


# ─── Registry ─────────────────────────────────────────────────────────────────

_SOURCE_REGISTRY: dict[str, callable] = {
    "raid": _iter_raid,
    "ai_pile": _iter_ai_pile,
    "gpt_wiki": _iter_gpt_wiki,
    "human_vs_ai": _iter_human_vs_ai,
    "ai_human_mixed": _iter_ai_human_mixed,
    "hc3": _iter_hc3,
    "coat": _iter_coat,
    "ruatd": _iter_ruatd,
    "daigt_proper": _iter_daigt_proper,
    "daigt_v2": _iter_daigt_v2,
    "m_daigt": _iter_m_daigt,
    "gsingh_train": _iter_gsingh_train,
    "gsingh1_train": _iter_gsingh_train,
}


# ─── Main loader ──────────────────────────────────────────────────────────────

def load_combined_dataset(config: DatasetConfig | None = None) -> LoadResult:
    """
    Загрузка и объединение нескольких датасетов.

    Returns:
        LoadResult с texts, labels, samples, stats
    """
    cfg = config or DatasetConfig()
    rng = random.Random(cfg.seed)

    requested_sources = cfg.sources or cfg.available_sources
    filtered_sources = [s for s in requested_sources if s in _SOURCE_REGISTRY]
    sources: list[str] = []
    seen_canonical_sources: set[str] = set()
    for source_name in filtered_sources:
        canonical_name = _canonical_source_name(source_name)
        if canonical_name in seen_canonical_sources:
            print(
                f"Skipping {source_name} because it duplicates "
                f"{canonical_name}."
            )
            continue
        seen_canonical_sources.add(canonical_name)
        sources.append(source_name)

    if not sources:
        raise ValueError(f"Нет валидных источников. Доступные: {list(_SOURCE_REGISTRY)}")

    # ── Загрузка ─────────────────────────────────────────────────────────
    all_samples: list[Sample] = []
    source_stats: dict[str, dict] = {}

    for src_name in sources:
        print(f"Loading {src_name}...")
        loader = _SOURCE_REGISTRY[src_name]
        src_samples: list[Sample] = []

        try:
            loaded_samples = _materialize_source(src_name, cfg.max_per_source, cfg, loader)
            for sample in loaded_samples:
                # Фильтр по длине
                if len(sample.text) < cfg.min_text_length:
                    continue
                if len(sample.text) > cfg.max_text_length:
                    sample.text = sample.text[: cfg.max_text_length]
                src_samples.append(sample)
        except Exception as e:
            print(f"  ⚠ Failed to load {src_name}: {e}")
            continue

        n_human = sum(1 for s in src_samples if s.label == Label.HUMAN)
        n_ai = len(src_samples) - n_human
        source_stats[src_name] = {"total": len(src_samples), "human": n_human, "ai": n_ai}
        print(f"  ✓ {len(src_samples):,} samples (human={n_human:,}, ai={n_ai:,})")

        all_samples.extend(src_samples)

    if not all_samples:
        raise RuntimeError("Не удалось загрузить ни одного сэмпла.")

    # ── Балансировка между источниками ────────────────────────────────────
    if cfg.balance_sources and len(sources) > 1:
        min_per_source = min(source_stats[s]["total"] for s in source_stats)
        balanced = []
        for src_name in source_stats:
            src = [s for s in all_samples if s.source_dataset == src_name]
            rng.shuffle(src)
            balanced.extend(src[:min_per_source])
        all_samples = balanced

    # ── Балансировка human/ai ─────────────────────────────────────────────
    if cfg.balance_labels:
        humans = [s for s in all_samples if s.label == Label.HUMAN]
        ais = [s for s in all_samples if s.label == Label.AI]
        min_count = min(len(humans), len(ais))
        rng.shuffle(humans)
        rng.shuffle(ais)
        all_samples = humans[:min_count] + ais[:min_count]

    # ── Обрезка до max_total и шафл ──────────────────────────────────────
    rng.shuffle(all_samples)
    all_samples = all_samples[: cfg.max_total]

    # ── Финальная статистика ──────────────────────────────────────────────
    n_human = sum(1 for s in all_samples if s.label == Label.HUMAN)
    n_ai = len(all_samples) - n_human
    generators = {}
    for s in all_samples:
        generators[s.generator] = generators.get(s.generator, 0) + 1

    stats = {
        "total": len(all_samples),
        "human": n_human,
        "ai": n_ai,
        "per_source": source_stats,
        "generators": dict(sorted(generators.items(), key=lambda x: -x[1])),
    }

    print(f"\n{'='*50}")
    print(f"Combined dataset: {len(all_samples):,} samples")
    print(f"  Human: {n_human:,} | AI: {n_ai:,}")
    print(f"  Sources: {list(source_stats)}")
    print(f"  Generators: {stats['generators']}")
    print(f"{'='*50}\n")

    texts = [s.text for s in all_samples]
    labels = [s.label.value for s in all_samples]

    return LoadResult(texts=texts, labels=labels, samples=all_samples, stats=stats)


# ─── Быстрый доступ ──────────────────────────────────────────────────────────

def load_quick(n: int = 10_000, sources: list[str] | None = None) -> tuple[list[str], list[int]]:
    """Быстрая загрузка — возвращает только (texts, labels)."""
    result = load_combined_dataset(DatasetConfig(
        sources=sources,
        max_per_source=n,
        max_total=n,
        balance_labels=True,
    ))
    return result.texts, result.labels


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Загрузка комбинированного датасета")
    parser.add_argument("--sources", nargs="+", default=None, help="Источники данных")
    parser.add_argument(
        "--source-path",
        action="append",
        default=[],
        help="Локальный путь в формате source=/abs/path. Нужен для daigt_proper и m_daigt.",
    )
    parser.add_argument("--max-per-source", type=int, default=5_000)
    parser.add_argument("--max-total", type=int, default=20_000)
    parser.add_argument("--cache-dir", default=".cache/datasets")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_paths = {}
    for item in args.source_path:
        if "=" not in item:
            raise ValueError(f"Неверный --source-path: {item}. Ожидается source=/path")
        source_name, path = item.split("=", 1)
        source_paths[source_name.strip()] = path.strip()

    cfg = DatasetConfig(
        sources=args.sources,
        max_per_source=args.max_per_source,
        max_total=args.max_total,
        balance_labels=not args.no_balance,
        seed=args.seed,
        source_paths=source_paths,
        cache_dir=args.cache_dir,
        cache_sources=not args.no_cache,
    )

    result = load_combined_dataset(cfg)

    # Показать примеры
    for label_name, label_val in [("HUMAN", 0), ("AI", 1)]:
        print(f"\n--- {label_name} examples ---")
        examples = [s for s in result.samples if s.label.value == label_val][:3]
        for s in examples:
            print(f"  [{s.source_dataset}/{s.generator}] {s.text[:120]}...")
