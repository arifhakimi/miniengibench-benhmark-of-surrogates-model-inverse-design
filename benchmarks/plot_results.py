import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt

def plot_task(task: str):
    path = Path(f"results/{task}.json")
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}. Run the benchmark first.")
    data = json.loads(path.read_text())

    models = list(data.keys())
    mae = [data[m]["metrics"]["MAE"] for m in models]
    rmse = [data[m]["metrics"]["RMSE"] for m in models]
    r2 = [data[m]["metrics"]["R2"] for m in models]

    # Plot MAE
    plt.figure()
    plt.bar(models, mae)
    plt.title(f"{task} – MAE (lower is better)")
    plt.ylabel("MAE")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(f"results/fig_{task}_mae.png", dpi=200)

    # Plot RMSE
    plt.figure()
    plt.bar(models, rmse)
    plt.title(f"{task} – RMSE (lower is better)")
    plt.ylabel("RMSE")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(f"results/fig_{task}_rmse.png", dpi=200)

    # Plot R2
    plt.figure()
    plt.bar(models, r2)
    plt.title(f"{task} – R² (higher is better)")
    plt.ylabel("R²")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(f"results/fig_{task}_r2.png", dpi=200)

    print(f"Saved figures to results/fig_{task}_*.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["airfoil", "concrete"], help="Which results to plot")
    args = ap.parse_args()
    plot_task(args.task)
