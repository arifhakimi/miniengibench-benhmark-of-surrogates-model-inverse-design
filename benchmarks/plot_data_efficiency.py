"""Plot metrics vs. training fraction for the data-efficiency benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

METRICS = ("MAE", "RMSE", "R2")
SPLITS = ("test", "val")


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        help="Task name matching <task>_data_efficiency.json.",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="test",
        help="Which split metrics to visualize (test or val).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing the JSON results.",
    )
    return parser.parse_args(list(args) if args is not None else None)


def load_results(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return json.loads(path.read_text())


def sorted_fractions(results: dict) -> list[tuple[float, str]]:
    keyed = []
    for key in results["fractions"].keys():
        keyed.append((float(key), key))
    return sorted(keyed, key=lambda item: item[0])


def plot_metric(
    task: str,
    results: dict,
    metric: str,
    split: str,
    fractions: list[tuple[float, str]],
    models: list[str],
    out_dir: Path,
) -> None:
    plt.figure()
    x_percent = [frac * 100 for frac, _ in fractions]
    for model in models:
        y = [
            results["fractions"][key]["models"][model][f"metrics_{split}"][metric]
            for _, key in fractions
        ]
        plt.plot(x_percent, y, marker="o", label=model)

    direction = "higher" if metric == "R2" else "lower"
    plt.title(f"{task} – {metric} vs. data fraction ({split} set, {direction} is better)")
    plt.xlabel("Training data used (%)")
    plt.ylabel(metric)
    plt.xticks(x_percent)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{task}_data_eff_{metric.lower()}_{split}.png"
    plt.savefig(out_dir / suffix, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    path = args.results_dir / f"{args.task}_data_efficiency.json"
    results = load_results(path)

    fraction_pairs = sorted_fractions(results)
    fractions_float = [pair[0] for pair in fraction_pairs]
    if not fractions_float:
        raise ValueError("No fractions found in results JSON.")

    models = list(next(iter(results["fractions"].values()))["models"].keys())
    out_dir = args.results_dir

    for metric in METRICS:
        plot_metric(
            task=args.task,
            results=results,
            metric=metric,
            split=args.split,
            fractions=fraction_pairs,
            models=models,
            out_dir=out_dir,
        )

    print(
        "Saved figures to "
        + ", ".join(
            str(out_dir / f"{args.task}_data_eff_{metric.lower()}_{args.split}.png")
            for metric in METRICS
        )
    )


if __name__ == "__main__":
    main()
