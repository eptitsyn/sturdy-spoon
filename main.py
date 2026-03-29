#!/usr/bin/env python3
"""
Полный пайплайн: загрузка → тренировка → оценка с метриками.

Быстрый тест:
    python main.py --sources hc3 gpt_wiki --max-per-source 1000 --max-total 2000 --epochs 5

Полная тренировка:
    python main.py --max-per-source 20000 --max-total 50000 --epochs 15

Только RAID + HC3:
    python main.py --sources raid hc3 --max-total 30000 --epochs 10
"""

from __future__ import annotations

import argparse
from collections import Counter

from sklearn.model_selection import train_test_split

from dataset_loader import load_combined_dataset, DatasetConfig
from train import train, TrainConfig
from inference import Detector
from evaluate import evaluate_model, evaluate_per_source, print_eval_report, print_source_breakdown
from ui import RICH_AVAILABLE, Table, box, console, print_info, print_section


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Text Detector — train & evaluate")

    # Data
    p.add_argument("--sources", nargs="+", default=None,
                   help="raid, ai_pile, gpt_wiki, human_vs_ai, ai_human_mixed, hc3, coat, ruatd, daigt_proper, daigt_v2, m_daigt, gsingh_train")
    p.add_argument(
        "--source-path",
        action="append",
        default=[],
        help="Optional local source path in the form source=/abs/path. Needed for daigt_proper and m_daigt.",
    )
    p.add_argument("--max-per-source", type=int, default=100_000)
    p.add_argument("--max-total", type=int, default=200_000)
    p.add_argument("--min-text-len", type=int, default=35)
    p.add_argument("--max-text-len", type=int, default=3_000)
    p.add_argument("--test-ratio", type=float, default=0.15, help="Доля тестовой выборки")
    p.add_argument("--cache-dir", default=".cache/datasets")
    p.add_argument("--no-dataset-cache", action="store_true", help="Disable on-disk source caching")

    # Model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dim-ff", type=int, default=512)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=50_000)

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


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


def split_data(result, test_ratio: float, seed: int):
    """Train/test split с сохранением метаданных сэмплов."""
    indices = list(range(len(result.texts)))
    stratify = None

    source_label_groups = [
        f"{result.samples[i].source_dataset}:{result.labels[i]}"
        for i in indices
    ]
    group_counts = Counter(source_label_groups)
    if group_counts and min(group_counts.values()) >= 2:
        stratify = source_label_groups
    else:
        label_counts = Counter(result.labels)
        if label_counts and min(label_counts.values()) >= 2:
            stratify = result.labels

    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed,
            stratify=None,
        )

    train_texts = [result.texts[i] for i in train_idx]
    train_labels = [result.labels[i] for i in train_idx]
    test_texts = [result.texts[i] for i in test_idx]
    test_labels = [result.labels[i] for i in test_idx]
    test_samples = [result.samples[i] for i in test_idx]

    return train_texts, train_labels, test_texts, test_labels, test_samples


def main():
    args = parse_args()
    source_paths = parse_source_paths(args.source_path)

    # ── 1. Загрузка данных ───────────────────────────────────────────────
    print_section("STEP 1: Loading combined dataset")

    data_cfg = DatasetConfig(
        sources=args.sources,
        max_per_source=args.max_per_source,
        max_total=args.max_total,
        min_text_length=args.min_text_len,
        max_text_length=args.max_text_len,
        balance_labels=True,
        seed=args.seed,
        source_paths=source_paths,
        cache_dir=args.cache_dir,
        cache_sources=not args.no_dataset_cache,
    )

    result = load_combined_dataset(data_cfg)

    # Train/test split
    train_texts, train_labels, test_texts, test_labels, test_samples = split_data(
        result, args.test_ratio, args.seed
    )
    print_info(f"Train: {len(train_texts):,} | Test: {len(test_texts):,}")

    # ── 2. Тренировка ────────────────────────────────────────────────────
    print_section("STEP 2: Training")

    train_cfg = TrainConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        max_len=args.max_seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_vocab=args.vocab_size,
        seed=args.seed,
    )

    model, tokenizer, stylometric_vectorizer, threshold = train(train_texts, train_labels, train_cfg)

    # ── 3. Оценка на тестовой выборке ────────────────────────────────────
    print_section("STEP 3: Evaluation on held-out test set")

    eval_result = evaluate_model(
        model, tokenizer, stylometric_vectorizer, test_texts, test_labels,
        batch_size=args.batch_size,
        threshold=threshold,
    )
    print_eval_report(eval_result, title="Test Set Evaluation")

    # ── 4. Per-source breakdown ──────────────────────────────────────────
    detector = Detector(model, tokenizer, stylometric_vectorizer, threshold=threshold)
    breakdowns = evaluate_per_source(detector, test_samples)
    if breakdowns:
        print_source_breakdown(breakdowns)

    # ── 5. Живые примеры ─────────────────────────────────────────────────
    print_section("STEP 4: Live inference examples")

    examples = [
        ("Ну блин, опять забыл зонт и промок как собака.", "Human"),
        ("Just grabbed tacos, best decision I've made all week lol", "Human"),
        ("spent 3 hours debugging only to find a missing comma", "Human"),
        ("The implementation of machine learning algorithms in healthcare has "
         "demonstrated significant potential for improving diagnostic accuracy.", "AI"),
        ("In conclusion, the comprehensive analysis reveals a multifaceted "
         "landscape that necessitates careful consideration.", "AI"),
        ("Furthermore, it is important to note that the integration of these "
         "systems requires a holistic understanding of the underlying frameworks.", "AI"),
    ]

    correct = 0
    if RICH_AVAILABLE:
        table = Table(title="Live inference examples", box=box.SIMPLE_HEAVY)
        table.add_column("OK", justify="center")
        table.add_column("Pred", justify="center")
        table.add_column("Conf", justify="right")
        table.add_column("Text")
        for text, expected in examples:
            r = detector.predict(text)
            ok = "✓" if r.label == expected else "✗"
            correct += ok == "✓"
            table.add_row(ok, r.label, f"{r.confidence:.1%}", text[:95] + ("..." if len(text) > 95 else ""))
        console.print(table)
    else:
        print()
        for text, expected in examples:
            r = detector.predict(text)
            ok = "✓" if r.label == expected else "✗"
            correct += ok == "✓"
            bar = "█" * int(r.confidence * 20) + "░" * (20 - int(r.confidence * 20))
            print(f"  {ok} [{r.label:>5}] {r.confidence:.1%} {bar}  {text[:75]}...")

    print_info(f"Live accuracy: {correct}/{len(examples)}")
    print_info("Model saved to: checkpoints/")
    print_info(f"Threshold: {threshold:.3f}")
    print_info("Usage:")
    print_info("  from inference import Detector")
    print_info("  detector = Detector.from_checkpoint('checkpoints/')")
    print_info("  result = detector.predict('some text')")
    print_info("  print(result.label, result.confidence)")


if __name__ == "__main__":
    main()
