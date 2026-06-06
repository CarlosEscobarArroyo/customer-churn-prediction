"""Baseline lineal: Regresión Logística.

LogReg es sensible a la escala → se envuelve en Pipeline(StandardScaler, LogReg).
El scaler se ajusta SOLO en el train (dentro de cada fold de la CV y en el split
OOT) gracias al Pipeline → sin fuga. class_weight='balanced' para el desbalance,
igual que el resto del baseline. Se evalúa con los dos protocolos (GroupKFold OOF
+ OOT) y las mismas métricas que 01_modelos_baseline para que la fila sea
directamente comparable con Random Forest y XGBoost.

Uso:  uv run python 05_modelling/logreg_baseline.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- datos y split (idénticos al baseline) -----------------------------------
base = Path.cwd()
while not (base / "data" / "processed").exists() and base != base.parent:
    base = base.parent
df = pd.read_csv(base / "data" / "processed" / "churn_dataset_features.csv",
                 parse_dates=["mes_obs"])
ID = ["id_vendedor", "mes_obs", "mes_rank"]; TARGET = "churn"
FEATS = [c for c in df.columns if c not in ID + [TARGET]]
X = df[FEATS]; y = df[TARGET].values; groups = df["id_vendedor"].values


def oot_split(df, v=6, test_months=4, rank_col="mes_rank"):
    rank = df[rank_col]
    test_start = rank.max() - test_months + 1
    return (rank <= test_start - 1 - v).values, (rank >= test_start).values


train_mask, test_mask = oot_split(df)


def top_decile_lift(yt, p):
    k = max(len(yt) // 10, 1)
    return yt[np.argsort(-p)[:k]].mean() / yt.mean()


def make_logreg():
    # StandardScaler dentro del pipeline → se fitea por fold, sin leakage
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))


# --- Protocolo 1: GroupKFold por vendedora (OOF) ------------------------------
oof = cross_val_predict(make_logreg(), X, y, cv=GroupKFold(5), groups=groups,
                        method="predict_proba")[:, 1]
gkf_AUC = roc_auc_score(y, oof)
gkf_PRAUC = average_precision_score(y, oof)
gkf_lift = top_decile_lift(y, oof)

# --- Protocolo 2: split OOT ---------------------------------------------------
m = make_logreg(); m.fit(X[train_mask], y[train_mask])
p = m.predict_proba(X[test_mask])[:, 1]; yt = y[test_mask]; pr = (p >= 0.5).astype(int)
dft = df[test_mask].copy(); dft["p"] = p
aucs = [roc_auc_score(g["churn"], g["p"]) for _, g in
        dft.groupby(dft["mes_obs"].dt.strftime("%Y-%m")) if g["churn"].nunique() > 1]

row = {
    "gkf_AUC": gkf_AUC, "gkf_PRAUC": gkf_PRAUC, "gkf_lift10": gkf_lift,
    "oot_AUC": roc_auc_score(yt, p), "oot_AUCstd": np.std(aucs),
    "oot_F1": f1_score(yt, pr), "oot_prec": precision_score(yt, pr),
    "oot_rec": recall_score(yt, pr), "oot_lift10": top_decile_lift(yt, p),
}
print("Regresión Logística (baseline lineal) — los dos protocolos:\n")
print(pd.Series(row).round(4).to_string())
