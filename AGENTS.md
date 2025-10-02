# Repository Guidelines

## Project Structure & Module Organization
mini_engibench is organised to keep datasets, models, and experiments decoupled. `datasets/` hosts loaders such as `airfoil.py` and `concrete.py`; each exposes `load_<task>()` that returns train/val/test NumPy arrays with `StandardScaler` applied. `models/` contains model wrappers like `mlp.py` with a `.fit/.predict` API that mirrors scikit-learn. Benchmark entry points live in `benchmarks/` (e.g. `run_airfoil.py`, `plot_results.py`) and write JSON plus figures into `results/`. Keep exploratory notebooks inside `notebooks/`, and store generated assets under `results/`.

## Build, Test, and Development Commands
Use Python 3.11; install dependencies via `pip install -r requirements.txt` or the conda recipe in `README.md`. Key workflows:
- `python -m benchmarks.run_airfoil` — train all configured models on the airfoil task and log metrics to `results/airfoil.json`.
- `python -m benchmarks.run_concrete` — mirror of the above for the concrete dataset.
- `python -m benchmarks.plot_results --task airfoil` (or `concrete`) — regenerate comparison plots in `results/*.png`.
Re-run commands after code changes so artefacts match source.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents, snake_case modules, and descriptive function names (`load_<dataset>` for loaders). Prefer small, composable helpers over monoliths. When adding Torch modules, expose the same `.fit/.predict` interface and keep defaults CPU-friendly. Annotate public function signatures with lightweight type hints and include concise docstrings that explain reasoning, mirroring existing files.

## Testing Guidelines
There is no dedicated pytest suite yet; treat benchmark scripts as integration tests. Before committing, execute the relevant `python -m benchmarks.run_*` command and confirm JSON metrics, timings, and plots regenerate. For new datasets or models, compare validation metrics against baselines and document any expected variance. Ensure data loaders remain deterministic by seeding random splits (use the `random_state` pattern already in place).

## Commit & Pull Request Guidelines
The repo ships without Git history, so adopt conventional, imperative commit subjects (e.g., `Add ridge baseline for airfoil`). Group related code, data, and result updates together, and note when regenerated artefacts are intentional. Pull requests should summarise scope, list reproducibility steps (`python -m ...` commands run), mention environment details, and include updated metrics or figure paths. Link issues if applicable and add screenshots only when visuals change.
