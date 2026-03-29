#!/usr/bin/env python3
"""Run saved detector on texts from test.txt."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on texts from a file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("test.txt"),
        help="Path to the input text file. Defaults to test.txt.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory with tokenizer/ and best_model.pt.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Inference device: "auto", "cpu", or "cuda".',
    )
    return parser.parse_args()


def load_texts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"Input file is empty: {path}")

    paragraph_chunks = [
        chunk.strip()
        for chunk in re.split(r"\n\s*\n", raw_text)
        if chunk.strip()
    ]

    if len(paragraph_chunks) > 1:
        texts = [
            chunk
            for chunk in paragraph_chunks
            if not chunk.lstrip().startswith("#")
        ]
    else:
        texts = [
            line.strip()
            for line in raw_text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if not texts:
        raise ValueError(f"No texts to process in: {path}")

    return texts


def validate_checkpoint_dir(checkpoint_dir: Path) -> None:
    required_files = [
        checkpoint_dir / "best_model.pt",
    ]
    tokenizer_dir = checkpoint_dir / "tokenizer"
    if not tokenizer_dir.exists():
        required_files.append(tokenizer_dir)
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Inference-ready checkpoint files are missing:\n"
            f"{missing_list}\n"
            "Run training until it exports tokenizer/ and best_model.pt to the checkpoint directory."
        )


def main() -> int:
    args = parse_args()

    try:
        validate_checkpoint_dir(args.checkpoint_dir)
        texts = load_texts(args.input)
        from inference import Detector
        detector = Detector.from_checkpoint(args.checkpoint_dir, device=args.device)
        results = detector.predict_batch(texts)
    except ModuleNotFoundError as exc:
        print(
            f"Error: missing dependency '{exc.name}'. Install the project dependencies before running inference.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Processed {len(texts)} text(s) from {args.input}")
    print("-" * 80)
    for idx, (text, result) in enumerate(zip(texts, results), start=1):
        preview = " ".join(text.split())
        if len(preview) > 120:
            preview = f"{preview[:117]}..."
        print(
            f"[{idx}] {result.label:<5} "
            f"confidence={result.confidence:.2%} "
            f"logit={result.logit:.4f}"
        )
        print(f"    {preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
