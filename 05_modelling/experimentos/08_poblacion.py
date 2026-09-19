"""Sensibilidad al filtro de población: ¿a qué vendedoras entran al dataset?

La regla vigente (qry_churn.sql) es: compró en t Y compras_hist >= 1. Acá se
endurece de dos formas, sin tocar el SQL (todas las columnas ya están):
  * historia mínima:     compras_hist >= N
  * regularidad reciente: meses_activos_u12 >= M  (M = 12 es la regla de
    Gattermann-Itschert: activo en cada uno de los últimos 12 meses)

Para cada regla se reporta el tamaño de la población, la prevalencia y dos AUC:
  * "modelo actual": XGBoost entrenado con TODA la población, evaluado solo en
    las filas que cumplen la regla (¿es un subconjunto más fácil o más difícil?);
  * "reentrenado":   XGBoost entrenado y evaluado solo en esa población
    (¿especializar el modelo aporta algo?).
Si ambos coinciden, el filtro cambia el alcance del modelo, no su calidad.

Uso:    uv run python 05_modelling/experimentos/08_poblacion.py
Salida: 05_modelling/experimentos/reports/poblacion_filtros.{csv,md}
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import PROC, REPORTS, TARGET, lift10, make_model, oot_split, prepare  # noqa: E402

REGLAS = [("compras_hist", n) for n in (1, 2, 3, 6, 12)] + \
         [("meses_activos_u12", m) for m in (2, 3, 6, 9, 12)]


def oof_y_oot(X, y, g, tr, te):
    oof = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, g):
        oof[b] = make_model("XGBoost", y[a]).fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]
    p = make_model("XGBoost", y[tr]).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof, p


def main():
    df = pd.read_csv(PROC / "churn_dataset.csv", parse_dates=["mes_obs"])
    tr, te = oot_split(df["mes_rank"])
    d, feats = prepare(df, tr)
    X, y, g = d[feats], d[TARGET].values, d["id_vendedor"].values
    ultimo = d["mes_rank"] == d["mes_rank"].max()
    oof_full, p_full = oof_y_oot(X, y, g, tr, te)
    print(f"población vigente: {len(d):,} filas | {d['id_vendedor'].nunique():,} vendedoras", flush=True)

    rows = []
    for col, umbral in REGLAS:
        t0 = time.time()
        m = (d[col] >= umbral).values
        mt = m[te]
        sub = d[m]
        oof_s, p_s = oof_y_oot(X[m].reset_index(drop=True), y[m], g[m], tr[m], te[m])
        rows.append({
            "regla": f"{col} >= {umbral}", "n_rows": int(m.sum()), "n_vend": sub["id_vendedor"].nunique(),
            "% filas": m.mean(), "vend_ultimo_mes": int((m & ultimo).sum()), "prevalencia": y[m].mean(),
            "gkf_AUC_modelo_actual": roc_auc_score(y[m], oof_full[m]),
            "gkf_AUC_reentrenado": roc_auc_score(y[m], oof_s),
            "gkf_liftPR_reentrenado": average_precision_score(y[m], oof_s) / y[m].mean(),
            "oot_AUC_modelo_actual": roc_auc_score(y[te][mt], p_full[mt]),
            "oot_AUC_reentrenado": roc_auc_score(y[te][mt], p_s),
            "oot_lift10_reentrenado": lift10(y[te][mt], p_s), "n_test": int(mt.sum()),
        })
        print(pd.DataFrame(rows[-1:]).round(4).to_string(index=False), f"\n  listo en {time.time() - t0:.0f}s\n", flush=True)
    write_report(pd.DataFrame(rows))


def write_report(res):
    REPORTS.mkdir(exist_ok=True)
    res.to_csv(REPORTS / "poblacion_filtros.csv", index=False)
    r = res.copy()
    r["% filas"] = (r["% filas"] * 100).round(0).astype(int).astype(str) + "%"
    md = [
        "# Sensibilidad al filtro de población — XGBoost tuneado",
        f"\n> Generado por `05_modelling/08_poblacion.py` el {time.strftime('%Y-%m-%d %H:%M')}. "
        "Regla vigente = `compras_hist >= 1` (primera fila). `vend_ultimo_mes` = vendedoras en "
        "alcance en el último mes observado del dataset.",
        "\n## Tamaño de la población por regla\n",
        r[["regla", "n_rows", "% filas", "n_vend", "vend_ultimo_mes", "prevalencia"]].round(3).to_markdown(index=False),
        "\n## AUC: modelo actual evaluado en el subconjunto vs. modelo reentrenado en él\n",
        r[["regla", "gkf_AUC_modelo_actual", "gkf_AUC_reentrenado", "oot_AUC_modelo_actual",
           "oot_AUC_reentrenado", "n_test"]].round(4).to_markdown(index=False),
        "\n## Lift (reentrenado)\n",
        r[["regla", "prevalencia", "gkf_liftPR_reentrenado", "oot_lift10_reentrenado"]].round(3).to_markdown(index=False),
        "\n## Notas\n",
        "- Las AUC entre reglas NO son comparables como calidad del modelo: cada regla cambia la "
        "población y la prevalencia. La comparación válida es por fila: modelo actual vs reentrenado.",
        "- Filtrar excluye vendedoras del alcance del modelo; en producción esas vendedoras no reciben score.",
    ]
    (REPORTS / "poblacion_filtros.md").write_text("\n".join(md) + "\n")
    print(f"\nreporte -> {REPORTS / 'poblacion_filtros.md'}")


if __name__ == "__main__":
    main()
