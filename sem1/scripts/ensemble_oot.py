"""Evalúa el ensemble de 42 variables en el test OOT del capítulo.

Mismo bloque que 05_modelling/experimentos/pipeline.oot_split (últimos 4 meses,
gap 6) y mismos modelos tuneados de models/*_tuned.joblib como referencia.
Hiperparámetros fijos (reports/master_features_v1/reconstructed_tuning); el OOT
no participó en su búsqueda (observaciones <= 2025-01).

Uso: uv run python -m scripts.ensemble_oot
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from scripts.master_features_ablation import make_component
from scripts.recent_windows import recent_training_indices
from scripts.temporal_optuna import load_raw

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "reports/master_features_v1/reconstructed_tuning"


def metrics(y, p, months):
    top = np.argsort(-p)[: len(y) // 10]
    monthly = [roc_auc_score(y[months == m], p[months == m]) for m in np.unique(months)]
    return {"auc": roc_auc_score(y, p), "auc_std_mes": np.std(monthly),
            "pr_auc": average_precision_score(y, p),
            "recall@0.5": recall_score(y, p >= 0.5), "precision@0.5": precision_score(y, p >= 0.5),
            "lift_decil": y[top].mean() / y.mean()}


def main():
    raw = load_raw()
    start = raw.mes_rank.max() - 3
    train_idx = np.flatnonzero(raw.mes_rank <= start - 7)
    test = raw[raw.mes_rank >= start]
    y, months = test.churn.to_numpy(), test.mes_obs.to_numpy()

    scores = {}
    for name, window in (("logreg", 36), ("xgboost", None)):
        params = json.loads((PARAMS / f"{name}_best.json").read_text())["params"]
        a = raw.iloc[recent_training_indices(raw, train_idx, window)]
        model = make_component(a, name, (), params, 4).fit(a, a.churn)
        scores[f"ens_{name}"] = model.predict_proba(test)[:, 1]
    scores["ensemble_42"] = (scores["ens_logreg"] + scores["ens_xgboost"]) / 2

    # Modelos del capítulo (reentrenados por retrain_tuned.py sobre el CSV vigente).
    feats = pd.read_csv(ROOT / "data/processed/churn_dataset_features.csv", parse_dates=["mes_obs"])
    feats = test[["id_vendedor", "mes_obs"]].merge(feats, on=["id_vendedor", "mes_obs"], how="left")
    assert len(feats) == len(test) and (feats.churn.to_numpy() == y).all()
    for name in ("xgboost", "rf", "logreg"):
        model = joblib.load(ROOT / f"models/{name}_tuned.joblib")
        scores[f"cap_{name}"] = model.predict_proba(feats[model.feature_names_in_])[:, 1]

    table = pd.DataFrame({k: metrics(y, p, months) for k, p in scores.items()}).T
    print(f"OOT {test.mes_obs.min().date()}..{test.mes_obs.max().date()} | n={len(y)} | prevalencia={y.mean():.3f}")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
