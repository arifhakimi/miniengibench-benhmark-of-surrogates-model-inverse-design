"""Plot inverse design benchmark trajectories from saved JSON logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        help="Task name used in the JSON filename (e.g., airfoil).",
    )
    parser.add_argument(
        "--quantity",
        choices=["best", "regret"],
        default="best",
        help="Whether to plot best objective trajectories or regret.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        default=None,
        help="Optional subset of fractions (e.g., 0.1 0.5).",
    )
    parser.add_argument(
        "--surrogates",
        nargs="+",
        default=None,
        help="Optional subset of surrogate names to plot.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to inverse design JSON (defaults to results/<task>_inverse_design.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where figures will be saved.",
    )
    return parser.parse_args(None if argv is None else list(argv))


def normalise_fraction(fractions: Sequence[str] | None, available: Sequence[str]) -> list[str]:
    if fractions is None:
        return list(available)
    target = []
    avail_lower = {key.lower(): key for key in available}
    for fraction in fractions:
        key = fraction.lower()
        if key in avail_lower:
            target.append(avail_lower[key])
            continue
        # allow raw numeric strings without formatting
        numeric = format_fraction_str(float(fraction))
        if numeric in avail_lower:
            target.append(avail_lower[numeric])
        elif numeric in available:
            target.append(numeric)
        else:
            valid = ", ".join(available)
            raise ValueError(f"Fraction '{fraction}' not found. Available: {valid}")
    return target


def format_fraction_str(value: float) -> str:
    text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".")


def filter_surrogates(requested: Sequence[str] | None, available: Sequence[str]) -> list[str]:
    if requested is None:
        return list(available)
    available_map = {name.lower(): name for name in available}
    selected = []
    for name in requested:
        key = name.lower()
        if key not in available_map:
            valid = ", ".join(available)
            raise ValueError(f"Surrogate '{name}' not found. Available: {valid}")
        selected.append(available_map[key])
    return selected


def load_results(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Could not find inverse design log at {path}")
    return json.loads(path.read_text())


def plot_quantity(
    results: dict,
    quantity: str,
    fractions: Sequence[str],
    surrogates: Sequence[str],
    output_path: Path,
) -> None:
    num_fractions = len(fractions)
    if num_fractions == 0:
        raise ValueError("No fractions selected for plotting.")

    iterations = None
    fig, axes = plt.subplots(1, num_fractions, figsize=(5 * num_fractions, 4), sharey=True)
    if num_fractions == 1:
        axes = [axes]

    for ax, frac_key in zip(axes, fractions):
        for surrogate_name in surrogates:
            surrogate_entry = results["surrogates"][surrogate_name]
            frac_payload = surrogate_entry["fractions"][frac_key]
            summary = frac_payload["summary"]
            if quantity == "best":
                mean = np.array(summary["mean_best_history"])
                std = np.array(summary["std_best_history"])
                ylabel = results.get("objective_label", "Objective")
            else:
                mean = np.array(summary["mean_regret_history"])
                std = np.array(summary["std_regret_history"])
                ylabel = "Regret"
            if iterations is None:
                iterations = np.arange(len(mean))
            ax.plot(iterations, mean, label=surrogate_name)
            ax.fill_between(iterations, mean - std, mean + std, alpha=0.2)

        ax.set_title(f"Initial fraction: {frac_key}")
        ax.set_xlabel("Design iteration")
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="best")

    direction_note = results.get("direction_note")
    if direction_note:
        fig.suptitle(direction_note)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = (
        args.input
        if args.input is not None
        else Path("results") / f"{args.task}_inverse_design.json"
    )
    results = load_results(input_path)

    available_fractions = list(results["surrogates"].values())[0]["fractions"].keys()
    fractions = normalise_fraction(args.fractions, list(available_fractions))

    available_surrogates = list(results["surrogates"].keys())
    surrogates = filter_surrogates(args.surrogates, available_surrogates)

    suffix = "best" if args.quantity == "best" else "regret"
    output_path = args.output_dir / f"{args.task}_inverse_design_{suffix}.png"

    plot_quantity(results, args.quantity, fractions, surrogates, output_path)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
