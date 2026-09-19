"""Análisis de supervivencia: tiempo hasta la próxima compra vs. clasificador de churn.

Idea: en lugar de la etiqueta binaria "no compra en [t+1..t+6]", modelar el tiempo
(en meses) hasta la PRÓXIMA compra de la vendedora después de t, con censura a la
derecha cuando no se observa ninguna compra futura. Ventajas teóricas: usa toda la
información del futuro observable (no solo 6 meses), permite predecir a cualquier
horizonte con un solo modelo y, en producción, las filas más recientes (con < 6
meses de futuro) entran al train como censuradas en vez de descartarse.

Dataset: variante en memoria de 00_dataset_construction/qry_churn.sql (el SQL
autoritativo NO se toca) con dos columnas extra:
  * gap_prox     = meses hasta la próxima compra (NULL si no se observa ninguna).
  * meses_futuro = meses de futuro observable (max_rank - mes_rank).
y el filtro relajado a "al menos 1 mes de futuro". Para supervivencia:
  tiempo = COALESCE(gap_prox, meses_futuro);  evento = gap_prox IS NOT NULL.
Sobre las filas con meses_futuro >= 6 el dataset es IDÉNTICO a churn_dataset.csv
(se verifica con asserts: mismas filas, misma etiqueta, mismas features).

Split: exactamente el mismo que el modelo de churn (oot_split sobre las filas con
meses_futuro >= 6; test = últimos 4 meses etiquetados, gap = 6 meses). Las filas
con meses_futuro < 6 quedan DESPUÉS del test en el tiempo: no entran ni al train
(leakage temporal) ni al test; solo se cuentan.

Modelos (sin Optuna, sin redes): Cox proporcional regularizado (scikit-survival)
sobre features estandarizadas; XGBoost con objective="survival:cox" y los mismos
hiperparámetros tuneados del clasificador; RandomSurvivalForest (train
submuestreado por tiempo). Referencia: el XGBoost clasificador vigente (churn a 6
meses) evaluado en la misma corrida.

CONVENCIÓN DE SIGNO (leer antes de tocar nada): el EVENTO es "volver a comprar".
Un hazard alto = vuelve pronto = riesgo de churn BAJO. Por eso:
  * score de riesgo de churn  = -(log hazard ratio)   -> AUC@h, lift decil, GKF.
  * score para el C-index     = +(log hazard ratio)   -> concordancia con (evento, tiempo).
Para el clasificador de churn: riesgo = p(churn); C-index con -p.

Métricas: C-index de Harrell (OOT y GKF-OOF); AUC a horizonte h ∈ {3, 6, 12} con
la etiqueta completamente observada "no compra en h meses" sobre filas con
meses_futuro >= h (sin IPCW); AUC@6 es el comparable con el modelo de churn
(GKF 0.7521 / OOT 0.7637). OJO: el test OOT tiene meses_futuro entre 6 y 9, así
que AUC@12 solo existe en GroupKFold (OOF), no en OOT.

Uso:    uv run python 05_modelling/experimentos/09_supervivencia.py
Cache:  data/processed/churn_dataset_supervivencia.csv (gitignored; si existe no consulta BigQuery)
Salida: 05_modelling/experimentos/reports/supervivencia.{csv,md}
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (PARAMS, PROC, REPORTS, RS, SQL_PATH, TARGET, evaluate_fn,  # noqa: E402
                      make_model, oot_split, prepare)

CACHE = PROC / "churn_dataset_supervivencia.csv"
NO_FEATS = ["gap_prox", "meses_futuro", TARGET, "tiempo", "evento"]
HORIZONTES = (3, 6, 12)
RSF_MAX_TRAIN = 12_000  # ponytail: RSF tarda ~160 s con 28k filas; submuestreo por tiempo
REF = "XGB clasificador (ref)"

# (original, reemplazo) sobre qry_churn.sql; cada original debe aparecer EXACTAMENTE una vez
SUBS = [
    ("    SUM(activo) OVER fwd6                                          AS compras_fwd6,\n",
     "    SUM(activo) OVER fwd6                                          AS compras_fwd6,\n"
     "    MIN(CASE WHEN activo = 1 THEN p.mes_rank END) OVER fut          AS prox_rank,\n"),
    ("    fwd6  AS (PARTITION BY id_vendedor ORDER BY mes_rank ROWS BETWEEN 1 FOLLOWING AND 6 FOLLOWING),\n",
     "    fwd6  AS (PARTITION BY id_vendedor ORDER BY mes_rank ROWS BETWEEN 1 FOLLOWING AND 6 FOLLOWING),\n"
     "    fut   AS (PARTITION BY id_vendedor ORDER BY mes_rank ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING),\n"),
    ("  CASE WHEN f.compras_fwd6 = 0 THEN 1 ELSE 0 END              AS churn,\n",
     "  CASE WHEN f.compras_fwd6 = 0 THEN 1 ELSE 0 END              AS churn,\n"
     "  f.prox_rank - f.mes_rank                                    AS gap_prox,\n"
     "  g.max_rank - f.mes_rank                                     AS meses_futuro,\n"),
    ("  AND f.mes_rank <= g.max_rank - 6          -- (etiqueta) hay 6 meses de futuro observables\n",
     "  AND f.mes_rank <= g.max_rank - 1          -- (supervivencia) al menos 1 mes de futuro observable\n"),
]


# --- datos ---------------------------------------------------------------------
def sql_supervivencia():
    sql = SQL_PATH.read_text()
    for old, new in SUBS:
        assert sql.count(old) == 1, f"esperaba exactamente 1 ocurrencia de:\n{old}"
        sql = sql.replace(old, new)
    return sql


def cargar():
    if not CACHE.exists():
        from google.cloud import bigquery
        from google.oauth2.credentials import Credentials
        # token de la cuenta correcta sin tocar la config de gcloud (el ADC activo no tiene acceso)
        tok = subprocess.check_output(
            ["gcloud", "auth", "print-access-token", "--account=carlos.escobar.arroyo@gmail.com"],
            text=True).strip()
        client = bigquery.Client(project="glamour-peru-dw", credentials=Credentials(tok))
        print("consultando BigQuery...", flush=True)
        client.query(sql_supervivencia()).to_dataframe().to_csv(CACHE, index=False)
    return pd.read_csv(CACHE, parse_dates=["mes_obs"])


def sanity_target(df):
    """churn6 derivado de gap_prox == churn del SQL == churn_dataset.csv (filas y features)."""
    lab = df[df["meses_futuro"] >= 6]
    churn6 = (lab["gap_prox"].isna() | (lab["gap_prox"] > 6)).astype(int).values
    assert (churn6 == lab[TARGET].values).all(), "churn6 derivado != columna churn"
    ref = pd.read_csv(PROC / "churn_dataset.csv", parse_dates=["mes_obs"])
    key = ["id_vendedor", "mes_rank"]
    a = lab.set_index(key).sort_index()
    b = ref.set_index(key).sort_index()
    assert a.index.equals(b.index), "filas distintas a churn_dataset.csv"
    cols = [c for c in ref.columns if c not in key]
    diff = [c for c in cols if not a[c].fillna(-1).astype(str).equals(b[c].fillna(-1).astype(str))]
    assert not diff, f"columnas distintas a churn_dataset.csv: {diff}"
    return len(lab), len(cols)


def split(df):
    """Mismos límites de mes_rank que el modelo de churn; filas recientes fuera de train y test."""
    lab = (df["meses_futuro"] >= 6).values
    tr6, te6 = oot_split(df.loc[lab, "mes_rank"])
    rank_lab = df.loc[lab, "mes_rank"].values
    tr = (df["mes_rank"] <= rank_lab[tr6].max()).values
    te = df["mes_rank"].between(rank_lab[te6].min(), rank_lab[te6].max()).values
    assert not (tr & ~lab).any() and not (te & ~lab).any()
    assert te.sum() == 885, te.sum()
    return tr, te, int((~lab).sum())


# --- modelos: fit_predict(tr, te) -> score de RIESGO DE CHURN (mayor = más churn) ----
def surv(evento, tiempo):
    return Surv.from_arrays(evento, tiempo)


def fit_cox(X, tiempo, evento, tr, te):
    sc = StandardScaler().fit(X[tr])
    m = CoxPHSurvivalAnalysis(alpha=0.1, n_iter=200).fit(sc.transform(X[tr]), surv(evento[tr], tiempo[tr]))
    return -m.predict(sc.transform(X[te]))  # predict = log hazard ratio de "volver" -> signo invertido


def fit_xgb_cox(X, tiempo, evento, tr, te):
    y = np.where(evento, tiempo, -tiempo)  # convención XGBoost: negativo = censura derecha
    m = XGBRegressor(objective="survival:cox", **PARAMS["xgboost"], tree_method="hist",
                     n_jobs=-1, random_state=RS).fit(X[tr], y[tr])
    return -np.log(m.predict(X[te]))  # predict = hazard ratio de "volver" -> -log = riesgo de churn


def fit_rsf(X, tiempo, evento, tr, te):
    sub = np.random.default_rng(RS).choice(tr, min(len(tr), RSF_MAX_TRAIN), replace=False)
    m = RandomSurvivalForest(n_estimators=200, min_samples_leaf=30, max_features="sqrt",
                             n_jobs=-1, random_state=RS).fit(X[sub], surv(evento[sub], tiempo[sub]))
    return -m.predict(X[te])  # predict = riesgo acumulado de "volver" -> signo invertido


def fit_clf(X, y6, tr, te):
    return make_model("XGBoost", y6[tr]).fit(X[tr], y6[tr]).predict_proba(X[te])[:, 1]


# --- métricas ------------------------------------------------------------------
def capturar(fit_predict, n):
    """Envuelve fit_predict para quedarse con OOF (5 folds GKF) y predicción OOT de evaluate_fn."""
    oof, calls = np.full(n, np.nan), []

    def fp(tr, te):
        p = fit_predict(tr, te)
        calls.append(p)
        if len(calls) <= 5:
            oof[te] = p
        return p
    return fp, oof, calls


def auc_h(gap, futuro, riesgo, mask, h):
    """AUC de "no compra en h meses" (etiqueta observada) sobre mask & meses_futuro >= h."""
    m = mask & (futuro >= h)
    if m.sum() == 0:
        return np.nan, 0, np.nan
    lab = (np.isnan(gap[m]) | (gap[m] > h)).astype(int)
    return roc_auc_score(lab, riesgo[m]), int(m.sum()), lab.mean()


def cindex(evento, tiempo, riesgo, mask):
    # el C-index se calcula respecto del evento "volver": score = -riesgo de churn
    return concordance_index_censored(evento[mask], tiempo[mask], -riesgo[mask])[0]


def main():
    t0 = time.time()
    df = cargar()
    n_lab, n_cols = sanity_target(df)
    print(f"sanity OK: {n_lab:,} filas con meses_futuro>=6 == churn_dataset.csv ({n_cols} cols)", flush=True)
    tr, te, n_recientes = split(df)
    d, feats = prepare(df, tr)
    feats = [f for f in feats if f not in NO_FEATS]
    assert not set(NO_FEATS) & set(feats) and d[feats].isna().sum().sum() == 0
    lab = (d["meses_futuro"] >= 6).values
    d, tr, te = d[lab].reset_index(drop=True), tr[lab], te[lab]  # las recientes no se usan
    X = d[feats].values.astype(float)
    gap, futuro = d["gap_prox"].values, d["meses_futuro"].values
    tiempo, evento = np.where(np.isnan(gap), futuro, gap), ~np.isnan(gap)
    y6, groups = d[TARGET].values, d["id_vendedor"].values
    mes = d["mes_obs"].dt.strftime("%Y-%m").values
    print(f"{len(d):,} filas | {len(feats)} features | train {tr.sum():,} | test {te.sum():,} | "
          f"recientes censuradas fuera: {n_recientes:,} | eventos {evento.mean():.3f}", flush=True)

    modelos = {
        "Cox PH (lineal)": lambda a, b: fit_cox(X, tiempo, evento, a, b),
        "XGBoost Cox": lambda a, b: fit_xgb_cox(X, tiempo, evento, a, b),
        "RandomSurvivalForest": lambda a, b: fit_rsf(X, tiempo, evento, a, b),
        REF: lambda a, b: fit_clf(X, y6, a, b),
    }
    todos = np.ones(len(d), bool)
    rows = []
    for name, fp in modelos.items():
        t1 = time.time()
        fp_cap, oof, calls = capturar(fp, len(d))
        ev = evaluate_fn(fp_cap, y6, groups, mes, tr, te)
        p_oot = np.full(len(d), np.nan)
        p_oot[te] = calls[5]
        row = {"modelo": name, "C_gkf": cindex(evento, tiempo, oof, todos), "C_oot": cindex(evento, tiempo, p_oot, te),
               "gkf_AUC": ev["gkf_AUC"], "gkf_liftPR": ev["gkf_liftPR"], "oot_AUC": ev["oot_AUC"],
               "oot_AUCstd": ev["oot_AUCstd"], "oot_lift10": ev["oot_lift10"]}
        for h in HORIZONTES:
            row[f"AUC{h}_gkf"], row[f"n{h}_gkf"], row[f"prev{h}_gkf"] = auc_h(gap, futuro, oof, todos, h)
            row[f"AUC{h}_oot"], row[f"n{h}_oot"], row[f"prev{h}_oot"] = auc_h(gap, futuro, p_oot, te, h)
        assert row["n6_oot"] == 885 and abs(row["AUC6_oot"] - row["oot_AUC"]) < 1e-9
        rows.append(row)
        print(pd.DataFrame([row])[["modelo", "C_oot", "AUC3_oot", "AUC6_oot", "AUC6_gkf", "AUC12_gkf", "oot_lift10"]]
              .round(4).to_string(index=False), f"\n  listo en {time.time() - t1:.0f}s\n", flush=True)
    write_report(pd.DataFrame(rows), n_lab, n_cols, n_recientes, len(feats), evento.mean())
    print(f"total {time.time() - t0:.0f}s")


def write_report(res, n_lab, n_cols, n_recientes, n_feats, tasa_evento):
    REPORTS.mkdir(exist_ok=True)
    res.to_csv(REPORTS / "supervivencia.csv", index=False)
    r = res.set_index("modelo")
    ref, surv_best = r.loc[REF], r.drop(index=REF)
    mejor6 = surv_best["AUC6_oot"].idxmax()
    d6_oot, d6_gkf = surv_best.loc[mejor6, "AUC6_oot"] - ref["AUC6_oot"], surv_best.loc[mejor6, "AUC6_gkf"] - ref["AUC6_gkf"]
    d3 = surv_best["AUC3_oot"].max() - ref["AUC3_oot"]
    d12 = surv_best["AUC12_gkf"].max() - ref["AUC12_gkf"]
    veredicto = lambda dlt, tol=0.005: "supera" if dlt > tol else ("pierde" if dlt < -tol else "empata")  # noqa: E731
    tabla = r[["C_gkf", "C_oot", "AUC3_oot", "AUC6_gkf", "AUC6_oot", "oot_AUCstd", "AUC12_gkf", "gkf_liftPR", "oot_lift10"]]
    horiz = pd.DataFrame({
        "h": HORIZONTES,
        "n_gkf": [int(ref[f"n{h}_gkf"]) for h in HORIZONTES], "prev_gkf": [ref[f"prev{h}_gkf"] for h in HORIZONTES],
        "n_oot": [int(ref[f"n{h}_oot"]) for h in HORIZONTES], "prev_oot": [ref[f"prev{h}_oot"] for h in HORIZONTES],
    })
    md = [
        "# Supervivencia (tiempo hasta la próxima compra) vs. clasificador de churn",
        f"\n> Generado por `05_modelling/09_supervivencia.py` el {time.strftime('%Y-%m-%d %H:%M')}. "
        f"Dataset: variante en memoria de `qry_churn.sql` (gap_prox, meses_futuro). {n_feats} features "
        "tras preprocessing; hiperparámetros de XGBoost de `xgboost_best_params.json` (sin Optuna).",
        "\n## Diseño\n",
        "- **Evento** = volver a comprar; **tiempo** = meses hasta la próxima compra (`gap_prox`), censurado a la "
        f"derecha en `meses_futuro` si no se observa ninguna. Tasa de eventos: {tasa_evento:.3f}.",
        "- **Split** idéntico al modelo de churn: `oot_split` sobre las filas con `meses_futuro >= 6` "
        "(train ≤ rank 101, gap de 6 meses, test = ranks 108-111 = 885 filas). GroupKFold(5) por vendedora "
        "sobre las mismas filas.",
        f"- **Filas recientes censuradas** (`meses_futuro < 6`): {n_recientes:,}. Quedan después del test en el "
        "tiempo, así que se dejan fuera de train y test (leakage temporal). En producción sí serían filas de "
        "train censuradas: es la ventaja práctica del enfoque (el clasificador las descarta).",
        "- **AUC@h** = AUC de la etiqueta observada \"no compra en h meses\" sobre filas con `meses_futuro >= h`, "
        "sin IPCW. AUC@6 es el comparable con el modelo de churn. El test OOT tiene `meses_futuro` entre 6 y 9, "
        "por eso **AUC@12 solo existe en GroupKFold (OOF)**, no en OOT.",
        "- **C-index** de Harrell (`concordance_index_censored`) respecto del evento \"volver\".",
        "\n## Sanity check del target\n",
        f"Sobre las {n_lab:,} filas con `meses_futuro >= 6`: `churn6 = (gap_prox IS NULL OR gap_prox > 6)` coincide "
        f"al 100% con `churn` del SQL y con `data/processed/churn_dataset.csv` (mismas filas por "
        f"`(id_vendedor, mes_rank)`, mismos valores en las {n_cols} columnas). Verificado con asserts en el script.",
        "\n## ADVERTENCIA de signo\n",
        "El evento es **volver a comprar**: hazard alto = vuelve pronto = riesgo de churn **bajo**. "
        "En el script, el score de riesgo de churn (AUC@h, lift, GroupKFold) es `-(log hazard ratio)` "
        "(Cox: `-predict`; XGBoost Cox: `-log(predict)`; RSF: `-predict`), y el C-index se calcula con el "
        "signo opuesto (`+hazard`). Para el clasificador: riesgo = `p(churn)`, C-index con `-p`. "
        "No invertir ninguno de los dos sin invertir el otro.",
        "\n## Resultados\n",
        f"Referencia = `{REF}`: XGBoost clasificador vigente (churn a 6 meses) evaluado en la misma corrida. "
        "Sus AUC@3 y AUC@12 usan `p(churn a 6 meses)` como score (modelo de un horizonte aplicado a otro).\n",
        tabla.round(4).to_markdown(),
        "\n### n y prevalencia por horizonte (iguales para todos los modelos)\n",
        horiz.round(3).to_markdown(index=False),
        "\n### Tabla completa\n", r.round(4).to_markdown(),
        "\n## Lectura\n",
        f"- **h = 6 (comparable)**: el mejor modelo de supervivencia ({mejor6}) {veredicto(d6_oot)} al clasificador "
        f"en OOT ({surv_best.loc[mejor6, 'AUC6_oot']:.4f} vs {ref['AUC6_oot']:.4f}, Δ = {d6_oot:+.4f}) y "
        f"{veredicto(d6_gkf)} en GroupKFold ({surv_best.loc[mejor6, 'AUC6_gkf']:.4f} vs {ref['AUC6_gkf']:.4f}, "
        f"Δ = {d6_gkf:+.4f}). La std por mes del OOT es {ref['oot_AUCstd']:.3f}: diferencias de ±0.01 están "
        "dentro del ruido.",
        f"- **h = 3**: mejor supervivencia {surv_best['AUC3_oot'].max():.4f} vs clasificador {ref['AUC3_oot']:.4f} "
        f"(Δ = {d3:+.4f}) → {veredicto(d3)}.",
        f"- **h = 12 (GKF)**: mejor supervivencia {surv_best['AUC12_gkf'].max():.4f} vs clasificador "
        f"{ref['AUC12_gkf']:.4f} (Δ = {d12:+.4f}) → {veredicto(d12)}.",
        f"- **C-index**: {surv_best['C_oot'].idxmax()} {surv_best['C_oot'].max():.4f} vs clasificador "
        f"{ref['C_oot']:.4f} en OOT. El clasificador, entrenado solo con la etiqueta binaria a 6 meses, ordena "
        "los tiempos de retorno casi tan bien como los modelos que los ven explícitamente.",
        "- **Implicación para la tesis**: el cambio de formulación (binaria → tiempo hasta el evento) no mueve la "
        "capacidad discriminativa de forma material; refuerza la conclusión previa de que el cuello de botella "
        "es la información disponible en las features, no el algoritmo ni la forma del target. Lo que sí aporta "
        "supervivencia es operativo: un solo modelo sirve para cualquier horizonte, y en producción incorpora "
        f"al train las {n_recientes:,} filas recientes censuradas que el clasificador descarta.",
        "- Limitaciones: el RSF entrena con un submuestreo de "
        f"{RSF_MAX_TRAIN:,} filas por tiempo de cómputo; Cox es lineal y sin interacciones; no hay tuning "
        "específico para supervivencia (se reutilizan los hiperparámetros del clasificador).",
    ]
    (REPORTS / "supervivencia.md").write_text("\n".join(md) + "\n")
    print(f"\nreporte -> {REPORTS / 'supervivencia.md'}")


if __name__ == "__main__":
    main()
