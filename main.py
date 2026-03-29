#!/usr/bin/env python3
"""
Полный пайплайн: загрузка комбинированного датасета → тренировка → инференс.

Быстрый запуск (маленький сэмпл):
    python main.py --max-per-source 2000 --max-total 5000 --epochs 5

Полная тренировка:
    python main.py --max-per-source 20000 --max-total 50000 --epochs 15

Только конкретные датасеты:
    python main.py --sources raid ai_pile --max-total 10000
"""

from __future__ import annotations

import argparse

from dataset_loader import load_combined_dataset, DatasetConfig
from train import train, TrainConfig
from inference import Detector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Text Detector — train & evaluate")

    # Data
    p.add_argument("--sources", nargs="+", default=None,
                   help="Источники: raid, ai_pile, gpt_wiki, human_vs_ai, ai_human_mixed")
    p.add_argument("--max-per-source", type=int, default=5_000)
    p.add_argument("--max-total", type=int, default=20_000)
    p.add_argument("--min-text-len", type=int, default=50)
    p.add_argument("--max-text-len", type=int, default=3_000)

    # Model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dim-ff", type=int, default=512)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=30_000)

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Загрузка данных ───────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading combined dataset")
    print("=" * 60)

    data_cfg = DatasetConfig(
        sources=args.sources,
        max_per_source=args.max_per_source,
        max_total=args.max_total,
        min_text_length=args.min_text_len,
        max_text_length=args.max_text_len,
        balance_labels=True,
        seed=args.seed,
    )

    result = load_combined_dataset(data_cfg)
    texts, labels = result.texts, result.labels

    # ── 2. Тренировка ────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 2: Training")
    print("=" * 60)

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

    model, tokenizer = train(texts, labels, train_cfg)

    # ── 3. Инференс ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Inference demo")
    print("=" * 60)

    detector = Detector(model, tokenizer)

    test_texts = [
        # Human-like
        "Ну блин, опять забыл зонт и промок как собака. День не задался.",
        "Just grabbed tacos for lunch, honestly best decision I've made all week lol",
        "My cat knocked over my coffee this morning. I'm not even mad anymore, it's just what she does.",
        "spent 3 hours debugging only to find a missing comma. classic monday vibes",
        # AI-like
        "The implementation of machine learning algorithms in healthcare has demonstrated significant potential "
        "for improving diagnostic accuracy and patient outcomes across various medical specialties.",
        "In conclusion, the comprehensive analysis of the aforementioned factors reveals a multifaceted "
        "landscape that necessitates careful consideration of both quantitative and qualitative metrics.",
        "This paper presents a novel approach to addressing the challenges associated with natural language "
        "processing in low-resource settings, leveraging transfer learning methodologies.",
        "Furthermore, it is important to note that the integration of these systems requires a holistic "
        "understanding of the underlying computational frameworks and their practical implications.",
    ]

    expected = ["Human"] * 4 + ["AI"] * 4

    print()
    correct = 0
    for text, exp in zip(test_texts, expected):
        r = detector.predict(text)
        ok = "✓" if r.label == exp else "✗"
        correct += ok == "✓"
        bar = "█" * int(r.confidence * 20) + "░" * (20 - int(r.confidence * 20))
        print(f"  {ok} [{r.label:>5}] {r.confidence:.1%} {bar}")
        print(f"    {text[:90]}...")
        print()

    print(f"Demo accuracy: {correct}/{len(test_texts)} ({correct / len(test_texts):.0%})")


if __name__ == "__main__":
    main()
