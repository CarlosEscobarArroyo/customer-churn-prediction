"""Evaluación sobre los bloques temporales (metodología §4.3).

`evaluar` entrena un modelo nuevo por bloque con toda la historia previa (menos la
brecha) y devuelve el AUC de cada bloque, su media y el lift del decil superior
calculado mes a mes, que es como se usaría la lista de retención.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import particion


def lift_decil_mensual(mes, y, score, frac=0.10):
    """Lift medio del top `frac` dentro de cada mes: precisión del top / prevalencia del mes."""
    d = pd.DataFrame({"mes": mes, "y": y, "s": score})
    lifts = []
    for _, g in d.groupby("mes"):
        k = max(1, int(len(g) * frac))
        top = g.nlargest(k, "s")
        if g.y.mean() > 0:
            lifts.append(top.y.mean() / g.y.mean())
    return float(np.mean(lifts))


def evaluar(make_model, df, feats, folds=None, ventana_meses=None):
    """Evalúa `make_model(y_train) -> estimador` sobre los bloques de `df` (pool de desarrollo).

    `ventana_meses` recorta las filas de entrenamiento a los últimos N meses (§4.3, ventanas).
    Devuelve dict con aucs por bloque, media, std, lift top-10 % mensual y predicciones OOF.
    """
    folds = folds or particion.temporal_folds(df.mes_rank)
    oof = np.full(len(df), np.nan)
    aucs = []
    for tr, va in folds:
        if ventana_meses:
            lim = df.mes_rank.iloc[tr].max() - ventana_meses
            tr = tr[df.mes_rank.iloc[tr].to_numpy() > lim]
        y_tr = df.churn.iloc[tr].to_numpy()
        m = make_model(y_tr).fit(df[feats].iloc[tr], y_tr)
        oof[va] = m.predict_proba(df[feats].iloc[va])[:, 1]
        aucs.append(roc_auc_score(df.churn.iloc[va], oof[va]))
    val = ~np.isnan(oof)
    return {
        "aucs": [round(a, 4) for a in aucs],
        "auc": float(np.mean(aucs)), "std": float(np.std(aucs)),
        "lift10": lift_decil_mensual(df.mes_obs[val], df.churn[val], oof[val]),
        "oof": oof,
    }


def scale_pos_weight(y):
    return float((y == 0).sum() / max(1, (y == 1).sum()))
