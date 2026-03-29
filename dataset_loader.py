"""
Загрузчик комбинированного датасета из нескольких источников.

Поддерживаемые датасеты:
  • RAID           — liamdugan/raid (6M+ примеров, 11 LLM, 8 доменов)
  • AI-Pile        — artem9k/ai-text-detection-pile (GPT-2/3/J/ChatGPT + human)
  • GPT-Wiki       — aadityaubhat/GPT-wiki-intro (Wikipedia vs GPT-3)
  • Human-vs-AI    — dmitva/human_ai_generated_text (human vs AI essays)
  • AI-Human-Mixed — Ateeqq/AI-and-Human-Generated-Text (GPT-4 + BARD)

Использование:
    from dataset_loader import load_combined_dataset, DatasetConfig
    texts, labels, meta = load_combined_dataset(DatasetConfig(max_per_source=5000))
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
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

    # Список всех доступных источников
    available_sources: list[str] = field(default_factory=lambda: [
        "raid", "ai_pile", "gpt_wiki", "human_vs_ai", "ai_human_mixed",
    ])


@dataclass
class LoadResult:
    """Результат загрузки."""
    texts: list[str]
    labels: list[int]
    samples: list[Sample]
    stats: dict


# ─── Адаптеры для каждого датасета ────────────────────────────────────────────

def _iter_raid(max_samples: int) -> Iterator[Sample]:
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


def _iter_ai_pile(max_samples: int) -> Iterator[Sample]:
    """
    artem9k/ai-text-detection-pile
    Long-form essays: human, GPT-2, GPT-3, ChatGPT, GPT-J.
    """
    ds = load_dataset("artem9k/ai-text-detection-pile", split="train", streaming=True)
    count = 0
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        source = row.get("source", "unknown")
        yield Sample(
            text=text,
            label=Label.HUMAN if source == "human" else Label.AI,
            source_dataset="ai_pile",
            generator=source,
            domain="essays",
        )
        count += 1
        if count >= max_samples:
            return


def _iter_gpt_wiki(max_samples: int) -> Iterator[Sample]:
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


def _iter_human_vs_ai(max_samples: int) -> Iterator[Sample]:
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


def _iter_ai_human_mixed(max_samples: int) -> Iterator[Sample]:
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


# ─── Registry ─────────────────────────────────────────────────────────────────

_SOURCE_REGISTRY: dict[str, callable] = {
    "raid": _iter_raid,
    "ai_pile": _iter_ai_pile,
    "gpt_wiki": _iter_gpt_wiki,
    "human_vs_ai": _iter_human_vs_ai,
    "ai_human_mixed": _iter_ai_human_mixed,
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

    sources = cfg.sources or cfg.available_sources
    sources = [s for s in sources if s in _SOURCE_REGISTRY]

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
            for sample in loader(cfg.max_per_source):
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
    parser.add_argument("--max-per-source", type=int, default=5_000)
    parser.add_argument("--max-total", type=int, default=20_000)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = DatasetConfig(
        sources=args.sources,
        max_per_source=args.max_per_source,
        max_total=args.max_total,
        balance_labels=not args.no_balance,
        seed=args.seed,
    )

    result = load_combined_dataset(cfg)

    # Показать примеры
    for label_name, label_val in [("HUMAN", 0), ("AI", 1)]:
        print(f"\n--- {label_name} examples ---")
        examples = [s for s in result.samples if s.label.value == label_val][:3]
        for s in examples:
            print(f"  [{s.source_dataset}/{s.generator}] {s.text[:120]}...")
