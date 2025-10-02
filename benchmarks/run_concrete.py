import json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from datasets.concrete import load_concrete
from models.mlp import MLPRegressorTorch

def eval_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred))
    }

def time_predict(model, X):
    t0 = time.time(); y = model.predict(X); return y, time.time() - t0

def main():
    Xtr, ytr, Xval, yval, Xte, yte = load_concrete()
    results = {}

    models = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, n_jobs=4
        ),
        "MLP": MLPRegressorTorch(in_dim=Xtr.shape[1], hidden=(64, 64), epochs=300, lr=1e-3),
    }

    for name, model in models.items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        train_time = time.time() - t0
        y_pred, inf_time = time_predict(model, Xte)
        metrics = eval_metrics(yte, y_pred)
        results[name] = {"metrics": metrics, "train_time_s": float(train_time), "inference_time_s": float(inf_time)}

    Path("results").mkdir(exist_ok=True, parents=True)
    Path("results/concrete.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
