"""Reentrena los modelos tuneados (XGBoost / RF / LogReg) SIN correr Optuna.

Los notebooks 02..04_tuning_*_optuna.ipynb guardan sus hiperparámetros en
05_modelling/*_best_params.json y el modelo fit sobre el train del split OOT en
models/*_tuned.joblib. Cuando cambia el set de features (p. ej. al agregar las
features de campañas al SQL) esos binarios quedan desactualizados; este script
los regenera con los mismos hiperparámetros y el mismo fit (train OOT) que la
sección final de cada notebook de tuning, sobre el churn_dataset_features.csv
vigente. Los notebooks de 06_evaluation/ cargan models/xgboost_tuned.joblib.

Uso:  uv run python 05_modelling/retrain_tuned.py
Lee:  data/processed/churn_dataset_features.csv (salida de 04_feature_engineering)
Guarda: models/{xgboost,rf,logreg}_tuned.joblib
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "05_modelling" / "experimentos"))
from pipeline import ID, TARGET, make_model, oot_split  # noqa: E402

FILES = {"XGBoost": "xgboost_tuned.joblib", "RandomForest": "rf_tuned.joblib",
         "LogReg": "logreg_tuned.joblib"}

df = pd.read_csv(BASE / "data" / "processed" / "churn_dataset_features.csv", parse_dates=["mes_obs"])
feats = [c for c in df.columns if c not in ID + [TARGET]]
X, y = df[feats], df[TARGET].values
train_mask, test_mask = oot_split(df["mes_rank"])
models_dir = BASE / "models"
models_dir.mkdir(exist_ok=True)
print(f"{len(feats)} features | train {train_mask.sum():,} | test OOT {test_mask.sum():,}")

for name, fname in FILES.items():
    model = make_model(name, y[train_mask]).fit(X[train_mask], y[train_mask])
    auc = roc_auc_score(y[test_mask], model.predict_proba(X[test_mask])[:, 1])
    joblib.dump(model, models_dir / fname)
    print(f"{name:13s} AUC OOT {auc:.4f} -> models/{fname}")
