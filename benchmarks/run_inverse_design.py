"""Benchmark inverse design performance across surrogate models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from datasets.airfoil import load_airfoil
from datasets.concrete import load_concrete
from inverse_design import (
    SurrogateFactory,
    flatten_dataset,
    make_default_surrogates,
    run_inverse_design_benchmark,
)

TASKS = {
    "airfoil": {
        "loader": load_airfoil,
        "objective": "minimize",
        "objective_label": "Sound Pressure Level (dB)",
        "direction_note": "Lower SPL is better (quieter airfoil).",
    },
    "concrete": {
        "loader": load_concrete,
        "objective": "maximize",
        "objective_label": "Compressive Strength (MPa)",
        "direction_note": "Higher strength is better (stronger concrete).",
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=sorted(TASKS.keys()),
        required=True,
        help="Which dataset to run (airfoil or concrete).",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.5],
        help="Initial design set fractions to evaluate.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=20,
        help="Number of inverse design iterations (additional evaluations).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="How many random restarts per configuration.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon-greedy probability for random exploration.",
    )
    parser.add_argument(
        "--ucb-beta",
        type=float,
        default=1.0,
        help="Exploration weight for UCB when uncertainty is available.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for dataset splits and run seeds.",
    )
    parser.add_argument(
        "--surrogates",
        nargs="+",
        default=None,
        help="Optional subset of surrogate names to run (defaults to all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for JSON logs.",
    )
    return parser.parse_args(None if argv is None else list(argv))


def validate_fractions(fractions: Sequence[float]) -> list[float]:
    cleaned = []
    for frac in fractions:
        if frac <= 0 or frac > 1:
            raise ValueError(f"Fractions must lie in (0, 1], got {frac}")
        cleaned.append(float(frac))
    return cleaned


def filter_surrogates(
    requested: Sequence[str] | None, all_factories: list[SurrogateFactory]
) -> list[SurrogateFactory]:
    if requested is None:
        return all_factories
    name_map = {factory.name.lower(): factory for factory in all_factories}
    selected = []
    for name in requested:
        key = name.lower()
        if key not in name_map:
            valid = ", ".join(sorted(factory.name for factory in all_factories))
            raise ValueError(f"Unknown surrogate '{name}'. Available: {valid}")
        selected.append(name_map[key])
    return selected


def summarize(results: dict) -> str:
    lines = []
    for surrogate_name, surrogate_entry in results["surrogates"].items():
        for frac_key, payload in surrogate_entry["fractions"].items():
            summary = payload["summary"]
            best_mean = summary["final_best_mean"]
            best_std = summary["final_best_std"]
            regret = summary["final_regret_mean"]
            lines.append(
                f"{surrogate_name} (frac={frac_key}): best={best_mean:.3f}±{best_std:.3f}, regret={regret:.3f}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    task_cfg = TASKS[args.task]
    fractions = validate_fractions(args.fractions)

    all_surrogates = make_default_surrogates()
    surrogate_factories = filter_surrogates(args.surrogates, all_surrogates)

    loader = task_cfg["loader"]
    dataset_parts = loader(random_state=args.random_state)
    X, y = flatten_dataset(dataset_parts)

    results = run_inverse_design_benchmark(
        X=X,
        y=y,
        objective=task_cfg["objective"],
        surrogates=surrogate_factories,
        fractions=fractions,
        budget=args.budget,
        num_runs=args.num_runs,
        base_seed=args.random_state,
        epsilon=args.epsilon,
        ucb_beta=args.ucb_beta,
    )

    results.update(
        {
            "task": args.task,
            "objective_label": task_cfg["objective_label"],
            "direction_note": task_cfg["direction_note"],
            "random_state": int(args.random_state),
            "selected_surrogates": [factory.name for factory in surrogate_factories],
            "surrogates": results["surrogates"],
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.task}_inverse_design.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(f"Saved inverse design benchmark to {out_path}")
    print("\nSummary (final best ± std, mean regret):")
    print(summarize(results))


if __name__ == "__main__":
    main()
