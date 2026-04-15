"""
Полноценная оценка модели: F1, ROC-AUC, confusion matrix, classification report,
per-source breakdown, threshold tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from model import AITextDetector
from data import BPETokenizer, TextClassificationDataset, collate_fn
from inference import Detector
from ui import RICH_AVAILABLE, Table, box, console, print_info, print_section


# ─── Основные метрики ─────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    accuracy: float
    f1: float
    f1_human: float
    f1_ai: float
    precision_human: float
    precision_ai: float
    recall_human: float
    recall_ai: float
    roc_auc: float
    confusion: np.ndarray          # 2×2
    threshold_used: float
    report: str                    # sklearn classification_report


def evaluate_model(
    model: AITextDetector,
    tokenizer: BPETokenizer,
    texts: Sequence[str],
    labels: Sequence[int],
    batch_size: int = 32,
    device: str = "auto",
    threshold: float = 0.5,
) -> EvalResult:
    """
    Полная оценка модели на тестовых данных.

    Args:
        model: обученная модель
        tokenizer: токенизатор
        texts: тестовые тексты
        labels: 0=human, 1=AI
        batch_size: размер батча
        device: "auto", "cpu", "cuda"
        threshold: порог для положительного класса (AI)

    Returns:
        EvalResult с полным набором метрик
    """
    dev = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(dev).eval()

    dataset = TextClassificationDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    all_logits: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for input_ids, targets in loader:
            input_ids = input_ids.to(dev)
            with autocast(dev.type):
                logits = model(input_ids)
            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(targets.int().tolist())

    logits_arr = np.array(all_logits)
    labels_arr = np.array(all_labels)
    probs = 1 / (1 + np.exp(-logits_arr))  # sigmoid

    # ── Predictions at configured threshold ───────────────────────────────
    preds_threshold = (probs >= threshold).astype(int)

    # ── Метрики ──────────────────────────────────────────────────────────
    target_names = ["Human", "AI"]

    report = classification_report(
        labels_arr,
        preds_threshold,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(labels_arr, preds_threshold)

    f1_per_class = f1_score(labels_arr, preds_threshold, average=None)
    from sklearn.metrics import precision_score, recall_score
    prec_per_class = precision_score(labels_arr, preds_threshold, average=None, zero_division=0)
    rec_per_class = recall_score(labels_arr, preds_threshold, average=None, zero_division=0)

    try:
        auc = roc_auc_score(labels_arr, probs)
    except ValueError:
        auc = 0.0  # если только один класс

    return EvalResult(
        accuracy=accuracy_score(labels_arr, preds_threshold),
        f1=f1_score(labels_arr, preds_threshold, average="macro"),
        f1_human=float(f1_per_class[0]),
        f1_ai=float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
        precision_human=float(prec_per_class[0]),
        precision_ai=float(prec_per_class[1]) if len(prec_per_class) > 1 else 0.0,
        recall_human=float(rec_per_class[0]),
        recall_ai=float(rec_per_class[1]) if len(rec_per_class) > 1 else 0.0,
        roc_auc=auc,
        confusion=cm,
        threshold_used=threshold,
        report=report,
    )


# ─── Per-source breakdown ─────────────────────────────────────────────────────

@dataclass
class SourceBreakdown:
    source: str
    n_samples: int
    accuracy: float
    f1: float
    roc_auc: float


def evaluate_per_source(
    detector: Detector,
    samples: list,  # list[Sample] from dataset_loader
) -> list[SourceBreakdown]:
    """Оценка модели отдельно по каждому источнику данных."""
    from collections import defaultdict

    by_source: dict[str, tuple[list[str], list[int]]] = defaultdict(lambda: ([], []))
    for s in samples:
        texts, labels = by_source[s.source_dataset]
        texts.append(s.text)
        labels.append(s.label.value if hasattr(s.label, "value") else int(s.label))

    results = []
    for source, (texts, labels) in sorted(by_source.items()):
        if len(set(labels)) < 2:
            continue

        preds = []
        probs = []
        for t in texts:
            r = detector.predict(t)
            preds.append(1 if r.label == "AI" else 0)
            probs.append(r.confidence if r.label == "AI" else 1 - r.confidence)

        preds_arr = np.array(preds)
        labels_arr = np.array(labels)
        probs_arr = np.array(probs)

        try:
            auc = roc_auc_score(labels_arr, probs_arr)
        except ValueError:
            auc = 0.0

        results.append(SourceBreakdown(
            source=source,
            n_samples=len(texts),
            accuracy=accuracy_score(labels_arr, preds_arr),
            f1=f1_score(labels_arr, preds_arr, average="macro"),
            roc_auc=auc,
        ))

    return results


# ─── Pretty print ─────────────────────────────────────────────────────────────

def print_eval_report(result: EvalResult, title: str = "Evaluation Report"):
    """Красивый вывод результатов."""
    if RICH_AVAILABLE:
        print_section(title)

        summary = Table(box=box.SIMPLE_HEAVY)
        summary.add_column("Metric")
        summary.add_column("Value", justify="right")
        summary.add_row("Accuracy", f"{result.accuracy:.4f}")
        summary.add_row("F1 (macro)", f"{result.f1:.4f}")
        summary.add_row("ROC-AUC", f"{result.roc_auc:.4f}")
        summary.add_row("Threshold", f"{result.threshold_used:.3f}")
        console.print(summary)

        class_table = Table(title="Per-class metrics", box=box.SIMPLE_HEAVY)
        class_table.add_column("Class")
        class_table.add_column("Precision", justify="right")
        class_table.add_column("Recall", justify="right")
        class_table.add_column("F1", justify="right")
        class_table.add_row("Human", f"{result.precision_human:.4f}", f"{result.recall_human:.4f}", f"{result.f1_human:.4f}")
        class_table.add_row("AI", f"{result.precision_ai:.4f}", f"{result.recall_ai:.4f}", f"{result.f1_ai:.4f}")
        console.print(class_table)

        confusion = Table(title="Confusion matrix", box=box.SIMPLE_HEAVY)
        confusion.add_column("")
        confusion.add_column("Pred Human", justify="right")
        confusion.add_column("Pred AI", justify="right")
        confusion.add_row("True Human", str(result.confusion[0][0]), str(result.confusion[0][1]))
        confusion.add_row("True AI", str(result.confusion[1][0]), str(result.confusion[1][1]))
        console.print(confusion)
        print_info("Full report:")
        console.print(result.report.rstrip())
        return

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"\n  Accuracy:    {result.accuracy:.4f}")
    print(f"  F1 (macro):  {result.f1:.4f}")
    print(f"  ROC-AUC:     {result.roc_auc:.4f}")
    print(f"  Threshold:   {result.threshold_used:.3f}")
    print(f"\n  {'':15s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print(f"  {'─'*45}")
    print(f"  {'Human':15s} {result.precision_human:10.4f} {result.recall_human:10.4f} {result.f1_human:10.4f}")
    print(f"  {'AI':15s} {result.precision_ai:10.4f} {result.recall_ai:10.4f} {result.f1_ai:10.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {'':15s} {'Pred Human':>12s} {'Pred AI':>12s}")
    print(f"  {'True Human':15s} {result.confusion[0][0]:12d} {result.confusion[0][1]:12d}")
    print(f"  {'True AI':15s} {result.confusion[1][0]:12d} {result.confusion[1][1]:12d}")
    print(f"\n  Full report:\n{result.report}")
    print(f"{'='*60}\n")


def print_source_breakdown(breakdowns: list[SourceBreakdown]):
    """Вывод per-source метрик."""
    if RICH_AVAILABLE:
        table = Table(title="Per-Source Breakdown", box=box.SIMPLE_HEAVY)
        table.add_column("Source")
        table.add_column("N", justify="right")
        table.add_column("Acc", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("AUC", justify="right")
        for b in breakdowns:
            table.add_row(b.source, str(b.n_samples), f"{b.accuracy:.3f}", f"{b.f1:.3f}", f"{b.roc_auc:.3f}")
        console.print(table)
        return

    print(f"\n{'─'*60}")
    print(f"  Per-Source Breakdown")
    print(f"{'─'*60}")
    print(f"  {'Source':20s} {'N':>6s} {'Acc':>8s} {'F1':>8s} {'AUC':>8s}")
    print(f"  {'─'*50}")
    for b in breakdowns:
        print(f"  {b.source:20s} {b.n_samples:6d} {b.accuracy:8.3f} {b.f1:8.3f} {b.roc_auc:8.3f}")
    print()
