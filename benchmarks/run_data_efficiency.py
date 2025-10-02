"""Data efficiency benchmark for surrogate models on engineering datasets."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from datasets.airfoil import load_airfoil
from datasets.concrete import load_concrete
from models.mlp import MLPRegressorTorch

DATASETS = {
    "airfoil": load_airfoil,
    "concrete": load_concrete,
}


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return common regression metrics."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_models(input_dim: int) -> dict[str, object]:
    """Instantiate the benchmark model zoo for a given input dimension."""
    return {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=4,
            random_state=42,
        ),
        "MLP": MLPRegressorTorch(in_dim=input_dim, hidden=(64, 64), epochs=300, lr=1e-3),
    }


def sample_subset(
    X: np.ndarray,
    y: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a deterministic subset of the training set for the requested fraction."""
    if not 0 < fraction <= 1:
        raise ValueError(f"Fraction must be in (0, 1], got {fraction}")
    n_samples = X.shape[0]
    take = max(1, int(np.ceil(n_samples * fraction)))
    idx = rng.choice(n_samples, size=take, replace=False)
    return X[idx], y[idx]


def format_fraction(frac: float) -> str:
    """Format a fraction for dictionary keys in a stable way."""
    return f"{frac:.2f}".rstrip("0").rstrip(".")


def run_data_efficiency(
    task: str,
    fractions: Sequence[float],
    random_state: int,
    output_dir: Path,
) -> dict[str, object]:
    """Execute the benchmark for the provided task and list of training fractions."""
    loader = DATASETS[task]
    X_train, y_train, X_val, y_val, X_test, y_test = loader(random_state=random_state)

    results: dict[str, object] = {
        "task": task,
        "train_size_full": int(X_train.shape[0]),
        "fractions": {},
        "random_state": int(random_state),
    }

    for i, frac in enumerate(fractions):
        frac_key = format_fraction(frac)
        subset_rng = np.random.default_rng(random_state + i)
        X_sub, y_sub = sample_subset(X_train, y_train, frac, subset_rng)

        models = build_models(X_train.shape[1])

        frac_entry: dict[str, object] = {
            "train_fraction": frac,
            "train_samples": int(X_sub.shape[0]),
            "models": {},
        }

        for name, model in models.items():
            t0 = time.time()
            model.fit(X_sub, y_sub)
            train_time = time.time() - t0

            y_pred_test, inf_time = time_predict(model, X_test)
            y_pred_val = model.predict(X_val)

            frac_entry["models"][name] = {
                "metrics_test": eval_metrics(y_test, y_pred_test),
                "metrics_val": eval_metrics(y_val, y_pred_val),
                "train_time_s": float(train_time),
                "inference_time_s": float(inf_time),
            }

        results["fractions"][frac_key] = frac_entry

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{task}_data_efficiency.json"
    path.write_text(json.dumps(results, indent=2))
    return results


def time_predict(model: object, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Wrapper to time inference while returning predictions."""
    t0 = time.time()
    preds = model.predict(X)
    return preds, time.time() - t0


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the data efficiency benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=sorted(DATASETS.keys()),
        required=True,
        help="Which dataset to benchmark.",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.5],
        help="Training fractions to evaluate (0-1].",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for splits and subsampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where the JSON results will be saved.",
    )
    return parser.parse_args(list(args) if args is not None else None)


def main() -> None:
    args = parse_args()
    fractions = sorted(set(args.fractions))
    if not fractions:
        raise ValueError("Provide at least one training fraction.")

    for frac in fractions:
        if frac <= 0 or frac > 1:
            raise ValueError(f"Fractions must be within (0, 1], got {fractions}")

    results = run_data_efficiency(
        task=args.task,
        fractions=fractions,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
