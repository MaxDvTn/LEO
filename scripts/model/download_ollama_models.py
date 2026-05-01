#!/usr/bin/env python3
"""Download Ollama models used by the synthetic data benchmark suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    group: str
    note: str


MODELS = [
    ModelSpec(
        "mistral-nemo",
        "recommended",
        "12B Mistral/NVIDIA model; strong quality/speed tradeoff.",
    ),
    ModelSpec(
        "aya-expanse:8b",
        "recommended",
        "Multilingual-first model; useful for translation-style synthetic data.",
    ),
    ModelSpec(
        "mistral-small3.2",
        "recommended",
        "24B Mistral model; stronger instruction following, needs more VRAM.",
    ),
    ModelSpec(
        "gemma3:27b",
        "recommended",
        "Largest Gemma 3 local target for the RTX 4090 when training is stopped.",
    ),
    ModelSpec(
        "gemma3:12b",
        "light",
        "Smaller Gemma 3 baseline; useful if VRAM is busy.",
    ),
    ModelSpec(
        "mistral:latest",
        "light",
        "Small Mistral baseline; not the main quality target.",
    ),
    ModelSpec(
        "phi4",
        "structured",
        "Strong structured-output baseline; useful for JSON/table-like responses.",
    ),
]

PRESETS = {
    "recommended": ["aya-expanse:8b", "mistral-nemo", "mistral-small3.2", "gemma3:27b"],
    "light": ["aya-expanse:8b", "mistral-nemo", "gemma3:12b", "mistral:latest"],
    "structured": ["phi4"],
    "all": [model.name for model in MODELS],
}


def run(cmd: list[str], *, dry_run: bool = False) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, check=False, text=True)


def installed_models() -> set[str]:
    result = subprocess.run(
        ["ollama", "list"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Unable to run 'ollama list'. Check that Ollama is installed and running."
        )

    installed: set[str] = set()
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            installed.add(parts[0])
    return installed


def resolve_models(args: argparse.Namespace) -> list[str]:
    selected: list[str] = []

    for preset in args.preset:
        selected.extend(PRESETS[preset])

    selected.extend(args.model)

    seen: set[str] = set()
    deduped: list[str] = []
    for model in selected:
        if model not in seen:
            seen.add(model)
            deduped.append(model)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Ollama models for local synthetic data generation tests."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        action="append",
        default=["recommended"],
        help="Model preset to download. Can be passed multiple times. Default: recommended.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Extra Ollama model to pull, for example qwen3:32b.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pull even if the model already appears in 'ollama list'.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with the next model if a pull fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without downloading anything.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = resolve_models(args)

    if not models:
        print("No models selected.")
        return 0

    print("Selected models:")
    notes = {model.name: model.note for model in MODELS}
    for model in models:
        suffix = f" - {notes[model]}" if model in notes else ""
        print(f"  - {model}{suffix}")

    installed = set() if args.force or args.dry_run else installed_models()

    for model in models:
        if not args.force and model in installed:
            print(f"\nSkipping {model}: already installed.")
            continue

        print(f"\nPulling {model}...")
        result = run(["ollama", "pull", model], dry_run=args.dry_run)
        if result.returncode != 0:
            print(f"Failed to pull {model}.", file=sys.stderr)
            if not args.continue_on_error:
                return result.returncode

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
