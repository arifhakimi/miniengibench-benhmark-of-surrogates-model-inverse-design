"""Reusable utilities for benchmarking surrogate-driven inverse design."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from models.mlp import MLPDropoutRegressorTorch, MLPRegressorTorch

ObjectiveKind = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class SurrogateFactory:
    """Callable factory that builds fresh surrogate adapters per run."""

    name: str
    build: Callable[[int, int], "SurrogateAdapter"]


class SurrogateAdapter:
    """Thin wrapper that exposes a consistent API across libraries."""

    name: str = "surrogate"
    supports_uncertainty: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SurrogateAdapter":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class LinearSurrogate(SurrogateAdapter):
    def __init__(self) -> None:
        self.model = LinearRegression()
        self.name = "Linear"
        self.supports_uncertainty = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSurrogate":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class RandomForestSurrogate(SurrogateAdapter):
    def __init__(self, random_state: int) -> None:
        self.model = RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )
        self.name = "RandomForest"
        self.supports_uncertainty = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestSurrogate":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        preds = np.stack([tree.predict(X) for tree in self.model.estimators_], axis=0)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std


class XGBSurrogate(SurrogateAdapter):
    def __init__(self, random_state: int) -> None:
        self.model = XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=4,
            random_state=random_state,
            objective="reg:squarederror",
        )
        self.name = "XGBoost"
        self.supports_uncertainty = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBSurrogate":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class MLPSurrogate(SurrogateAdapter):
    def __init__(self, input_dim: int) -> None:
        self.model = MLPRegressorTorch(in_dim=input_dim, hidden=(64, 64), epochs=300)
        self.name = "MLP"
        self.supports_uncertainty = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPSurrogate":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class DropoutMLPSurrogate(SurrogateAdapter):
    def __init__(self, input_dim: int) -> None:
        self.model = MLPDropoutRegressorTorch(
            in_dim=input_dim,
            hidden=(128, 128),
            dropout=0.1,
            epochs=400,
            mc_samples=40,
        )
        self.name = "MLPDropout"
        self.supports_uncertainty = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DropoutMLPSurrogate":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.model.predict_with_uncertainty(X)


class GaussianProcessSurrogate(SurrogateAdapter):
    def __init__(self, random_state: int) -> None:
        kernel = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(
            length_scale=1.0, nu=2.5
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=2,
            normalize_y=True,
            random_state=random_state,
        )
        self.name = "GaussianProcess"
        self.supports_uncertainty = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessSurrogate":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean, _ = self.model.predict(X, return_std=True)
        return mean

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.model.predict(X, return_std=True)


@dataclass(frozen=True)
class InverseDesignSpec:
    """Configuration of an inverse design benchmark run."""

    objective: ObjectiveKind
    budget: int
    initial_fraction: float
    epsilon: float = 0.1
    ucb_beta: float = 1.0


@dataclass
class RunTrace:
    iteration: int
    candidate_index: int
    objective_value: float
    best_so_far: float
    fit_time_s: float
    acquisition_time_s: float


@dataclass
class RunResult:
    run_id: int
    seed: int
    initial_indices: list[int]
    initial_objectives: list[float]
    best_history: list[float]
    regret_history: list[float]
    best_value: float
    best_iteration: int
    final_regret: float
    traces: list[RunTrace]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": int(self.run_id),
            "seed": int(self.seed),
            "initial_indices": [int(i) for i in self.initial_indices],
            "initial_objectives": [float(v) for v in self.initial_objectives],
            "best_history": [float(v) for v in self.best_history],
            "regret_history": [float(v) for v in self.regret_history],
            "best_value": float(self.best_value),
            "best_iteration": int(self.best_iteration),
            "final_regret": float(self.final_regret),
            "traces": [
                {
                    "iteration": int(t.iteration),
                    "candidate_index": int(t.candidate_index),
                    "objective_value": float(t.objective_value),
                    "best_so_far": float(t.best_so_far),
                    "fit_time_s": float(t.fit_time_s),
                    "acquisition_time_s": float(t.acquisition_time_s),
                }
                for t in self.traces
            ],
        }


def make_default_surrogates() -> list[SurrogateFactory]:
    """Return the default surrogate roster used across benchmarks."""

    def build_linear(input_dim: int, _: int) -> SurrogateAdapter:
        _ = input_dim  # unused, but keeps signature consistent
        return LinearSurrogate()

    def build_rf(_: int, seed: int) -> SurrogateAdapter:
        return RandomForestSurrogate(seed)

    def build_xgb(_: int, seed: int) -> SurrogateAdapter:
        return XGBSurrogate(seed)

    def build_mlp(input_dim: int, _: int) -> SurrogateAdapter:
        return MLPSurrogate(input_dim)

    def build_gp(_: int, seed: int) -> SurrogateAdapter:
        return GaussianProcessSurrogate(seed)

    def build_mlp_dropout(input_dim: int, _: int) -> SurrogateAdapter:
        return DropoutMLPSurrogate(input_dim)

    return [
        SurrogateFactory("Linear", build_linear),
        SurrogateFactory("RandomForest", build_rf),
        SurrogateFactory("XGBoost", build_xgb),
        SurrogateFactory("MLP", build_mlp),
        SurrogateFactory("GaussianProcess", build_gp),
        SurrogateFactory("MLPDropout", build_mlp_dropout),
    ]


def format_fraction(frac: float) -> str:
    text = f"{frac:.2f}"
    return text.rstrip("0").rstrip(".")


def compute_regret(
    best_history: Sequence[float],
    global_optimum: float,
    objective: ObjectiveKind,
) -> list[float]:
    best_history_arr = np.asarray(best_history)
    if objective == "maximize":
        regret = global_optimum - best_history_arr
    else:
        regret = best_history_arr - global_optimum
    regret = np.maximum(regret, 0.0)
    return regret.astype(float).tolist()


def select_candidate(
    surrogate: SurrogateAdapter,
    X_pool: np.ndarray,
    available_indices: np.ndarray,
    objective: ObjectiveKind,
    rng: np.random.Generator,
    epsilon: float,
    ucb_beta: float,
) -> Tuple[int, float]:
    if available_indices.size == 0:
        raise ValueError("No candidates remaining to evaluate.")

    subset = X_pool[available_indices]
    t0 = time.time()
    if surrogate.supports_uncertainty:
        mean, std = surrogate.predict_with_uncertainty(subset)
        if objective == "maximize":
            scores = mean + ucb_beta * std
        else:
            scores = -mean + ucb_beta * std
    else:
        mean = surrogate.predict(subset)
        scores = mean if objective == "maximize" else -mean
    acq_time = time.time() - t0

    if epsilon > 0 and rng.random() < epsilon:
        choice = rng.choice(len(available_indices))
    else:
        choice = int(np.argmax(scores))
    candidate = int(available_indices[choice])
    return candidate, acq_time


def run_single_inverse_design(
    X: np.ndarray,
    y: np.ndarray,
    surrogate_factory: SurrogateFactory,
    spec: InverseDesignSpec,
    global_optimum: float,
    run_id: int,
    seed: int,
) -> RunResult:
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    init = max(2, int(np.ceil(spec.initial_fraction * n_samples)))
    if init + spec.budget > n_samples:
        raise ValueError(
            "Initial sample size plus budget exceeds dataset size; reduce budget or fraction."
        )
    initial_indices = rng.choice(n_samples, size=init, replace=False)
    available_mask = np.ones(n_samples, dtype=bool)
    available_mask[initial_indices] = False

    observed_idx = list(int(i) for i in initial_indices)
    observed_y = [float(y[i]) for i in observed_idx]
    best_value = min(observed_y) if spec.objective == "minimize" else max(observed_y)
    best_history: list[float] = [best_value]

    traces: list[RunTrace] = []
    total_iterations = spec.budget
    surrogate = surrogate_factory.build(X.shape[1], seed)

    for iteration in range(1, total_iterations + 1):
        fit_start = time.time()
        observed_array = np.array(observed_idx, dtype=int)
        observed_targets = np.array(observed_y, dtype=float)
        surrogate.fit(X[observed_array], observed_targets)
        fit_time = time.time() - fit_start

        available_indices = np.where(available_mask)[0]
        candidate, acq_time = select_candidate(
            surrogate,
            X,
            available_indices,
            spec.objective,
            rng,
            spec.epsilon,
            spec.ucb_beta,
        )
        available_mask[candidate] = False
        observed_idx.append(int(candidate))
        objective_value = float(y[candidate])
        observed_y.append(objective_value)

        if spec.objective == "maximize":
            best_value = max(best_value, objective_value)
        else:
            best_value = min(best_value, objective_value)
        best_history.append(best_value)

        traces.append(
            RunTrace(
                iteration=iteration,
                candidate_index=int(candidate),
                objective_value=objective_value,
                best_so_far=best_value,
                fit_time_s=fit_time,
                acquisition_time_s=acq_time,
            )
        )

    regret_history = compute_regret(best_history, global_optimum, spec.objective)
    final_best = best_history[-1]
    final_regret = regret_history[-1]
    tol = 1e-8
    best_iteration = next(
        (idx for idx, value in enumerate(best_history) if abs(value - final_best) <= tol),
        len(best_history) - 1,
    )

    return RunResult(
        run_id=run_id,
        seed=seed,
        initial_indices=list(int(i) for i in initial_indices),
        initial_objectives=[float(y[i]) for i in initial_indices],
        best_history=best_history,
        regret_history=regret_history,
        best_value=final_best,
        best_iteration=best_iteration,
        final_regret=final_regret,
        traces=traces,
    )


def aggregate_runs(runs: Sequence[RunResult]) -> dict[str, Any]:
    best_matrix = np.array([run.best_history for run in runs])
    regret_matrix = np.array([run.regret_history for run in runs])
    return {
        "mean_best_history": best_matrix.mean(axis=0).tolist(),
        "std_best_history": best_matrix.std(axis=0).tolist(),
        "mean_regret_history": regret_matrix.mean(axis=0).tolist(),
        "std_regret_history": regret_matrix.std(axis=0).tolist(),
        "final_best_mean": float(np.mean([run.best_value for run in runs])),
        "final_best_std": float(np.std([run.best_value for run in runs])),
        "final_regret_mean": float(np.mean([run.final_regret for run in runs])),
        "final_regret_std": float(np.std([run.final_regret for run in runs])),
        "best_iteration_mean": float(np.mean([run.best_iteration for run in runs])),
        "best_iteration_std": float(np.std([run.best_iteration for run in runs])),
        "area_under_regret_mean": float(
            np.mean([np.trapz(run.regret_history) for run in runs])
        ),
    }


def run_inverse_design_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    objective: ObjectiveKind,
    surrogates: Sequence[SurrogateFactory],
    fractions: Sequence[float],
    budget: int,
    num_runs: int,
    base_seed: int = 42,
    epsilon: float = 0.1,
    ucb_beta: float = 1.0,
) -> dict[str, Any]:
    if budget <= 0:
        raise ValueError("Budget must be positive.")
    if num_runs <= 0:
        raise ValueError("num_runs must be positive.")

    n_samples = X.shape[0]
    if objective == "maximize":
        global_optimum = float(np.max(y))
    else:
        global_optimum = float(np.min(y))

    fractions_sorted = sorted(set(fractions))
    results: dict[str, Any] = {
        "budget": int(budget),
        "num_runs": int(num_runs),
        "epsilon": float(epsilon),
        "ucb_beta": float(ucb_beta),
        "fractions": [float(f) for f in fractions_sorted],
        "surrogates": {},
        "objective": objective,
        "global_optimum": global_optimum,
        "dataset_size": int(n_samples),
    }

    for factory in surrogates:
        surrogate_entry: dict[str, Any] = {
            "name": factory.name,
            "supports_uncertainty": bool(
                factory.build(X.shape[1], base_seed).supports_uncertainty
            ),
            "fractions": {},
        }

        for frac in fractions_sorted:
            spec = InverseDesignSpec(
                objective=objective,
                budget=budget,
                initial_fraction=frac,
                epsilon=epsilon,
                ucb_beta=ucb_beta,
            )
            runs: List[RunResult] = []
            for run_id in range(num_runs):
                seed = base_seed + run_id
                result = run_single_inverse_design(
                    X,
                    y,
                    surrogate_factory=factory,
                    spec=spec,
                    global_optimum=global_optimum,
                    run_id=run_id,
                    seed=seed,
                )
                runs.append(result)
            surrogate_entry["fractions"][format_fraction(frac)] = {
                "runs": [run.to_dict() for run in runs],
                "summary": aggregate_runs(runs),
            }
        results["surrogates"][factory.name] = surrogate_entry
    return results


def flatten_dataset(
    loader_output: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if len(loader_output) != 6:
        raise ValueError("Dataset loader must return 6 items: Xtr, ytr, Xval, yval, Xte, yte")
    Xtr, ytr, Xval, yval, Xte, yte = loader_output
    X = np.vstack([Xtr, Xval, Xte])
    y = np.concatenate([ytr, yval, yte])
    return X, y
