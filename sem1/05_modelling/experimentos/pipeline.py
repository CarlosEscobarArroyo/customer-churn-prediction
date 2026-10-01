"""Pipeline 02→04 como funciones reutilizables (split OOT, preprocessing fit-en-train,
features derivadas, selección por permutación) + modelos tuneados y métricas de los
dos protocolos. Réplica exacta de los notebooks 02, 03, 04 y de 05/01_modelos_baseline;
lo usan los scripts de experimentos de 05_modelling/experimentos/.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE = Path(__file__).resolve().parents[2]  # raíz del repo
SQL_PATH = BASE / "00_dataset_construction" / "qry_churn.sql"
PROC = BASE / "data" / "processed"
REPORTS = BASE / "05_modelling" / "experimentos" / "reports"
TEST_MONTHS = 4
RS = 42
ID = ["id_vendedor", "mes_obs", "mes_rank"]
TARGET = "churn"
MODELS = ["LogReg", "RandomForest", "XGBoost"]
PARAMS = {m: json.loads((BASE / "05_modelling" / f"{m}_best_params.json").read_text())["best_params"]
          for m in ("logreg", "rf", "xgboost")}

# --- 03_preprocessing (mismas constantes) -------------------------------------
DROP = ["edad", "provincia"]
FILL_ZERO = ["tend_monto_u3_vs_prev3", "tend_nped_u3_vs_prev3", "monto_cv_u12",
             "monto_ult_vs_media", "monto_por_prod_acum",
             *[f"d_{m}_m{n}" for m in ("monto", "nped") for n in (1, 3, 6, 9, 12)]]
CAT_OHE = ["sexo", "tipo_vendedor", "departamento"]


def bq_client():
    from google.cloud import bigquery
    return bigquery.Client(project="glamour-peru-dw")


# --- 02: split ----------------------------------------------------------------
def oot_split(rank, v=6, test_months=TEST_MONTHS):
    test_start = rank.max() - test_months + 1
    return (rank <= test_start - 1 - v).values, (rank >= test_start).values


# --- 03: preprocessing fit-en-train + 04: features derivadas ------------------
def preprocess(df, train_mask):
    tr = df[train_mask]
    d = df.drop(columns=DROP).copy()
    d[FILL_ZERO] = d[FILL_ZERO].fillna(0)
    d["antiguedad_meses"] = d["antiguedad_meses"].fillna(tr["antiguedad_meses"].median())
    for c in CAT_OHE:
        d[c] = pd.Categorical(d[c].fillna("DESCONOCIDO"),
                              categories=sorted(tr[c].fillna("DESCONOCIDO").unique()))
    return pd.get_dummies(d, columns=CAT_OHE, prefix=CAT_OHE, dtype=int)


def engineer(d):
    div = lambda a, b: (a / b.replace(0, np.nan)).fillna(0)  # noqa: E731
    d["ticket_prom_u12"] = div(d["monto_u12"], d["n_ped_u12"])
    d["ticket_prom_u3"] = div(d["monto_u3"], d["n_ped_u3"])
    d["intensidad_u3"] = div(d["n_ped_u3"], d["meses_activos_u3"])
    d["basket_size_u12"] = div(d["n_prod_u12"], d["n_ped_u12"])
    d["recencia_norm"] = d["meses_desde_compra_previa"] * (d["meses_activos_u12"] / 12)
    d["tasa_act_reciente_vs_hist"] = div(d["meses_activos_u3"] / 3, d["meses_activos_u12"] / 12)
    return d


def prepare(df, train_mask):
    """preprocess + engineer; devuelve (d, lista de features)."""
    d = engineer(preprocess(df, train_mask))
    return d, [c for c in d.columns if c not in ID + [TARGET]]


# --- 04: selección por permutación en train -----------------------------------
def select_features(X, y, train_mask):
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                n_jobs=-1, random_state=RS)
    rf.fit(X[train_mask], y[train_mask])
    imp = permutation_importance(rf, X[train_mask], y[train_mask], n_repeats=5,
                                 random_state=RS, n_jobs=-1)
    return [f for f, v in zip(X.columns, imp.importances_mean) if v > 0]


# --- 05: modelos tuneados (sin Optuna) y métricas ------------------------------
def make_model(name, y_tr):
    if name == "LogReg":
        return make_pipeline(StandardScaler(), LogisticRegression(
            **PARAMS["logreg"], class_weight="balanced", solver="saga",
            max_iter=5000, random_state=RS))
    if name == "RandomForest":
        return RandomForestClassifier(**PARAMS["rf"], class_weight="balanced",
                                      n_jobs=-1, random_state=RS)
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    return XGBClassifier(**PARAMS["xgboost"], scale_pos_weight=spw, eval_metric="logloss",
                         tree_method="hist", n_jobs=-1, random_state=RS)


def lift10(yt, p):
    n = max(len(yt) // 10, 1)
    return yt[np.argsort(-p)[:n]].mean() / yt.mean()


def evaluate_fn(fit_predict, y, groups, mes, train_mask, test_mask):
    """Dos protocolos para cualquier `fit_predict(tr_idx, te_idx) -> p_te`.

    Sirve tanto para un modelo sklearn como para pipelines que ajustan pasos
    no supervisados o redes por fold (todo fit-en-train dentro de cada fold).
    """
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(np.zeros(len(y)), y, groups):
        oof[va] = fit_predict(tr, va)
    p = fit_predict(np.where(train_mask)[0], np.where(test_mask)[0])
    yt, mt = y[test_mask], mes[test_mask]
    pr = (p >= 0.5).astype(int)
    aucs = [roc_auc_score(yt[mt == mm], p[mt == mm]) for mm in np.unique(mt)
            if 0 < yt[mt == mm].mean() < 1]
    return {
        "gkf_AUC": roc_auc_score(y, oof), "gkf_PRAUC": average_precision_score(y, oof),
        "gkf_liftPR": average_precision_score(y, oof) / y.mean(), "gkf_lift10": lift10(y, oof),
        "oot_AUC": roc_auc_score(yt, p), "oot_AUCstd": float(np.std(aucs)),
        "oot_PRAUC": average_precision_score(yt, p),
        "oot_liftPR": average_precision_score(yt, p) / yt.mean(),
        "oot_F1": f1_score(yt, pr, zero_division=0),
        "oot_prec": precision_score(yt, pr, zero_division=0),
        "oot_rec": recall_score(yt, pr), "oot_lift10": lift10(yt, p),
    }


def evaluate(name, X, y, groups, mes, train_mask, test_mask):
    def fit_predict(tr, te):
        m = make_model(name, y[tr]).fit(X.iloc[tr], y[tr])
        return m.predict_proba(X.iloc[te])[:, 1]
    return evaluate_fn(fit_predict, y, groups, mes, train_mask, test_mask)
