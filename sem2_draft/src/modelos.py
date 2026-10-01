"""Constructores de los estimadores y ensemble por promedio (metodología §4.3).

`build(algo, params)` crea el estimador a partir del dict de hiperparámetros que produce la
búsqueda de `04_modelado/tuning.py` (incluye `w`, el peso de la clase positiva).
`EnsemblePromedio` promedia las probabilidades de varios miembros, cada uno con sus propias
variables y ventana de entrenamiento; `desde_spec` lo arma desde `04_modelado/modelo_final.json`.
"""

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
N_JOBS = 16


def build(algo, p):
    """Estimador sklearn-compatible a partir de un dict de hiperparámetros (incluye `w`)."""
    p = dict(p)
    w = p.pop("w", 1.0)
    if algo == "logreg":
        return Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(solver="liblinear", max_iter=2000,
                                                   class_weight={0: 1.0, 1: w},
                                                   random_state=SEED, **p))])
    if algo == "rf":
        return RandomForestClassifier(class_weight={0: 1.0, 1: w}, n_jobs=N_JOBS,
                                      random_state=SEED, **p)
    if algo == "xgboost":
        return XGBClassifier(scale_pos_weight=w, tree_method="hist", n_jobs=N_JOBS,
                             random_state=SEED, verbosity=0, **p)
    if algo == "lightgbm":
        return LGBMClassifier(scale_pos_weight=w, subsample_freq=1, n_jobs=N_JOBS,
                              random_state=SEED, verbose=-1, **p)
    if algo == "catboost":
        return CatBoostClassifier(scale_pos_weight=w, thread_count=N_JOBS,
                                  random_seed=SEED, verbose=0, allow_writing_files=False, **p)
    raise ValueError(algo)


class EnsemblePromedio:
    """Promedio simple de probabilidades. `miembros`: {nombre: (feats, estimador, ventana_meses)}.

    `fit(df, y)` recibe el DataFrame completo (necesita `mes_rank` para aplicar la ventana de
    cada miembro); `predict_proba(df)` devuelve la matriz (n, 2) habitual.
    """

    def __init__(self, miembros):
        self.miembros = miembros

    def fit(self, df, y):
        y = np.asarray(y)
        for feats, est, ventana in self.miembros.values():
            m = np.ones(len(df), dtype=bool)
            if ventana:
                m = (df.mes_rank > df.mes_rank.max() - ventana).to_numpy()
            est.fit(df[feats][m], y[m])
        return self

    def proba_miembros(self, df):
        return {n: est.predict_proba(df[feats])[:, 1] for n, (feats, est, _) in self.miembros.items()}

    def predict_proba(self, df):
        p = np.mean(list(self.proba_miembros(df).values()), axis=0)
        return np.c_[1 - p, p]

    @property
    def feats(self):
        out = []
        for feats, _, _ in self.miembros.values():
            out += [f for f in feats if f not in out]
        return out


def desde_spec(spec):
    """EnsemblePromedio con los miembros de `spec['final']` (04_modelado/modelo_final.json)."""
    miembros = {}
    for a in spec["final"]["miembros"]:
        c = spec["candidatos"][a]
        miembros[a] = (c["variables"], build(c["algoritmo"], c["params"]), c["ventana_meses"])
    return EnsemblePromedio(miembros)
