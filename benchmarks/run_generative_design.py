"""Train a conditional VAE and sample high-performing designs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from datasets.airfoil import load_airfoil
from datasets.concrete import load_concrete
from models.cvae import CVAEConfig, ConditionalVAE
from models.mlp import MLPRegressorTorch

TASKS = {
    "airfoil": {
        "loader": load_airfoil,
        "objective": "minimize",
        "objective_label": "Sound Pressure Level (dB)",
        "default_percentile": 0.1,
    },
    "concrete": {
        "loader": load_concrete,
        "objective": "maximize",
        "objective_label": "Compressive Strength (MPa)",
        "default_percentile": 0.9,
    },
}


def parse_hidden_dims(values: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(int(v) for v in values)
    if not dims:
        raise argparse.ArgumentTypeError("At least one hidden dimension is required.")
    if any(v <= 0 for v in dims):
        raise argparse.ArgumentTypeError("Hidden dimensions must be positive integers.")
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS.keys()), required=True)
    parser.add_argument("--latent-dim", type=int, default=8, help="Latent dimensionality.")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=(128, 128),
        help="Hidden layer sizes for encoder/decoder (space-separated).",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs.")
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight on KL divergence term (beta-VAE coefficient).",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Desired performance value. Overrides percentile-based target when set.",
    )
    parser.add_argument(
        "--target-percentile",
        type=float,
        default=None,
        help="Percentile used to derive target when --target is not provided.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="How many designs to sample from the trained decoder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for numpy/torch and dataset split.",
    )
    parser.add_argument(
        "--eval-surrogate",
        action="store_true",
        help="Also train a fast MLP surrogate to score generated designs.",
    )
    parser.add_argument(
        "--surrogate-epochs",
        type=int,
        default=400,
        help="Epochs for the auxiliary surrogate scorer when --eval-surrogate is set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for JSON output.",
    )
    parser.add_argument("--device", type=str, default=None, help="Force computation device.")
    args = parser.parse_args()
    args.hidden_dims = parse_hidden_dims(args.hidden_dims)
    if args.latent_dim <= 0:
        parser.error("--latent-dim must be positive.")
    if args.batch_size <= 0 or args.epochs <= 0:
        parser.error("--batch-size and --epochs must be positive.")
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive.")
    if args.beta <= 0:
        parser.error("--beta must be positive.")
    return args


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_target(
    y: np.ndarray,
    objective: str,
    explicit_target: float | None,
    percentile: float | None,
    default_percentile: float,
) -> tuple[float, float | None]:
    if explicit_target is not None:
        return float(explicit_target), None
    pct = default_percentile if percentile is None else float(percentile)
    if not 0 < pct < 1:
        raise ValueError("Percentile must lie in (0, 1).")
    if objective == "maximize":
        target = float(np.quantile(y, pct))
    else:
        target = float(np.quantile(y, 1 - pct))
    return target, pct


def train_cvae(
    args: argparse.Namespace,
    splits: tuple[np.ndarray, ...],
) -> tuple[ConditionalVAE, dict]:
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    config = CVAEConfig(
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        beta=args.beta,
        device=args.device,
    )

    model = ConditionalVAE(x_dim=X_fit.shape[1], cond_dim=1, config=config)
    model.fit(X_fit, y_fit)

    recon_test = model.reconstruct(X_test, y_test)
    recon_mse = float(np.mean((recon_test - X_test) ** 2))

    metrics = {
        "train_loss_last": model.training_history_[-1] if model.training_history_ else None,
        "train_loss_first": model.training_history_[0] if model.training_history_ else None,
        "reconstruction_mse_test": recon_mse,
    }
    return model, metrics


def evaluate_with_surrogate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    generated: np.ndarray,
    epochs: int,
) -> np.ndarray:
    surrogate = MLPRegressorTorch(in_dim=X_train.shape[1], epochs=epochs)
    surrogate.fit(X_train, y_train)
    preds = surrogate.predict(generated)
    return preds


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    task_cfg = TASKS[args.task]
    loader = task_cfg["loader"]
    splits = loader(random_state=args.seed)

    model, metrics = train_cvae(args, splits)

    X_train, y_train, X_val, y_val, _, _ = splits
    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    target_value, target_percentile = resolve_target(
        y_fit,
        objective=task_cfg["objective"],
        explicit_target=args.target,
        percentile=args.target_percentile,
        default_percentile=task_cfg["default_percentile"],
    )

    targets = np.full(shape=(args.num_samples,), fill_value=target_value, dtype=np.float32)
    generated = model.generate(targets, num_samples=args.num_samples)

    surrogate_scores = None
    if args.eval_surrogate:
        surrogate_scores = evaluate_with_surrogate(
            X_train=X_fit,
            y_train=y_fit,
            generated=generated,
            epochs=args.surrogate_epochs,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.task}_generative_cvae.json"
    payload = {
        "task": args.task,
        "objective_label": task_cfg["objective_label"],
        "config": {
            "latent_dim": args.latent_dim,
            "hidden_dims": list(args.hidden_dims),
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "beta": args.beta,
            "device": str(model.device),
        },
        "seed": int(args.seed),
        "target_value": target_value,
        "target_percentile": target_percentile,
        "num_samples": int(args.num_samples),
        "metrics": metrics,
        "generated_designs": generated.tolist(),
        "surrogate_scores": surrogate_scores.tolist() if surrogate_scores is not None else None,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"Saved generative designs to {out_path}")
    if metrics["train_loss_last"] is not None:
        print(
            "Final train loss:",
            f"{metrics['train_loss_last']:.4f}",
            "(first epoch:",
            f"{metrics['train_loss_first']:.4f})",
        )
    print(f"Target value used: {target_value:.3f}")
    if surrogate_scores is not None:
        print(
            "Surrogate score stats -- mean:",
            f"{float(np.mean(surrogate_scores)):.3f}",
            "std:",
            f"{float(np.std(surrogate_scores)):.3f}",
        )


if __name__ == "__main__":
    main()
