"""TabFM (Google) sobre el dataset de churn: zero-shot vs XGBoost tuneado.

## Qué producto es

**TabFM** es el modelo fundacional pre-entrenado de Google Research para datos
tabulares (clasificación y regresión). No se entrena ni se tunea: lee la tabla de
train como *ejemplos in-context* y predice la tabla de test en un solo forward
pass (in-context learning, atención alternada fila/columna). En Google Cloud se
expone **dentro de BigQuery** con la función SQL `AI.PREDICT` (Preview):

    https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-predict
    https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/

No es TimesFM (series de tiempo) ni AutoML Tables / BQML `BOOSTED_TREE` (esos sí
entrenan un modelo). No hay modelo persistido ni endpoint: `AI.PREDICT` recibe las
dos tablas y devuelve la predicción en la misma consulta.

## Limitación que condiciona todo el experimento

La doc es explícita: *"Your data can include up to **20 feature columns**"*. Con las
110 features del pipeline la consulta falla con:

    400 The number of features 110 exceeds the maximum allowed number of features 20.

Por eso TabFM corre sobre las **20 features top por ganancia de XGBoost ajustado
solo en train** (sin mirar test). Para que la comparación sea justa se reportan
tres filas: XGBoost con las 110, XGBoost con esas mismas 20, y TabFM con las 20.

## Protocolo

Idéntico al resto de `05_modelling/`: split OOT de `pipeline.oot_split` (train =
28.735 filas con `mes_rank <= 101`, test = 885 filas de los últimos 4 meses,
prevalencia 0.278, gap de 6 meses sin usar). Preprocessing y features derivadas
con `pipeline.prepare` (fit en train). Sin tuning y sin GroupKFold para TabFM:
un solo pase. `id_vendedor`, `mes_obs` y `mes_rank` nunca entran como features
(la tabla de train sube solo las 20 features + el label).

Las columnas se renombran a `f000..f019` porque BigQuery es case-insensitive en
nombres de columna y el one-hot de `departamento` genera pares como
`departamento_Ancash` / `departamento_ancash`.

## Costo

Durante el Preview el uso de TabFM se factura como una consulta normal de
BigQuery (bytes procesados en on-demand, slots en Enterprise). El script imprime
`total_bytes_billed` real. Desde el 30/10/2026 Google pasa a precio por tokens.

## Cómo re-ejecutarlo

    uv run python 05_modelling/experimentos/10_tabfm.py

Requiere un token de la cuenta personal (se obtiene con `gcloud auth
print-access-token --account=...`, ver `ACCOUNT`) con acceso a `glamour-peru-dw`.
El script crea el dataset `churn_experimentos` (us-east1, misma región que
`glamour_dw`), sube dos tablas temporales y **las borra al terminar**: no deja
nada con costo recurrente. Salida: `05_modelling/experimentos/reports/tabfm.{md,csv}`.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import PROC, REPORTS, TARGET, lift10, make_model, oot_split, prepare  # noqa: E402

ACCOUNT = "carlos.escobar.arroyo@gmail.com"
PROJECT = "glamour-peru-dw"
DATASET = "churn_experimentos"
LOCATION = "us-east1"
N_FEATS_TABFM = 20  # límite duro de AI.PREDICT

# El label tiene que ser BOOL o STRING para que AI.PREDICT haga clasificación
# (con INT64/FLOAT64 haría regresión). `predicted_label_probs` es un
# ARRAY<STRUCT<label STRING, prob FLOAT64>> con labels 'true'/'false'.
SQL_TABFM = """
SELECT
  row_id,
  (SELECT prob FROM UNNEST(predicted_label_probs) WHERE label = 'true') AS p_churn
FROM AI.PREDICT(
  TABLE `{project}.{dataset}.tabfm_train`,
  TABLE `{project}.{dataset}.tabfm_test`,
  label_col => 'label'
)
"""


def bq_personal():
    """Cliente de BigQuery con el token de la cuenta personal (sin tocar las ADC)."""
    from google.cloud import bigquery
    from google.oauth2.credentials import Credentials
    token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token", f"--account={ACCOUNT}"], text=True).strip()
    return bigquery, bigquery.Client(project=PROJECT, credentials=Credentials(token))


def oot_metrics(y_true, prob, mes):
    """Métricas del bloque OOT. `mes` en formato YYYY-MM; std solo con ambas clases."""
    pred = (prob >= 0.5).astype(int)
    aucs = [roc_auc_score(y_true[mes == mm], prob[mes == mm]) for mm in np.unique(mes)
            if 0 < y_true[mes == mm].mean() < 1]
    prauc = average_precision_score(y_true, prob)
    return {
        "oot_AUC": roc_auc_score(y_true, prob),
        "oot_AUCstd": float(np.std(aucs)),
        "oot_PRAUC": prauc,
        "oot_liftPR": prauc / y_true.mean(),
        "oot_lift10": lift10(y_true, prob),
        "oot_prec": precision_score(y_true, pred, zero_division=0),
        "oot_rec": recall_score(y_true, pred),
        "oot_F1": f1_score(y_true, pred, zero_division=0),
    }


def run_xgb(feats, x_train, y_train, x_test):
    """XGBoost tuneado sobre `feats`; devuelve (probabilidades, segundos, modelo)."""
    model = make_model("XGBoost", y_train)
    t0 = time.perf_counter()
    model.fit(x_train[feats], y_train)
    secs = time.perf_counter() - t0
    return model.predict_proba(x_test[feats])[:, 1], secs, model


def run_tabfm(bigquery, client, feats, x_train, y_train, x_test):
    """Sube train/test, llama a AI.PREDICT y borra las tablas. Devuelve (prob, secs, info)."""
    rename = {f: f"f{i:03d}" for i, f in enumerate(feats)}
    train_tbl = x_train[feats].astype("float64").rename(columns=rename)
    train_tbl["label"] = np.asarray(y_train).astype(bool)
    test_tbl = x_test[feats].astype("float64").rename(columns=rename)
    test_tbl.insert(0, "row_id", np.arange(len(test_tbl)))

    cfg = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    dataset = bigquery.Dataset(f"{PROJECT}.{DATASET}")
    dataset.location = LOCATION
    client.create_dataset(dataset, exists_ok=True)
    for name, frame in (("tabfm_train", train_tbl), ("tabfm_test", test_tbl)):
        client.load_table_from_dataframe(
            frame, f"{PROJECT}.{DATASET}.{name}", job_config=cfg).result(timeout=900)

    sql = SQL_TABFM.format(project=PROJECT, dataset=DATASET)
    t0 = time.perf_counter()
    job = client.query(sql, location=LOCATION)
    out = job.result(timeout=900).to_dataframe()
    secs = time.perf_counter() - t0

    for name in ("tabfm_train", "tabfm_test"):
        client.delete_table(f"{PROJECT}.{DATASET}.{name}", not_found_ok=True)

    prob = out.sort_values("row_id")["p_churn"].to_numpy(dtype=float)
    gib = job.total_bytes_billed / 2**30
    return prob, secs, {"bytes_billed": job.total_bytes_billed,
                        "usd_on_demand": gib / 1024 * 6.25, "job_id": job.job_id}


def main():
    df = pd.read_csv(PROC / "churn_dataset.csv")
    train_mask, test_mask = oot_split(df["mes_rank"])
    data, feats = prepare(df, train_mask)
    assert test_mask.sum() == 885, f"test tiene {test_mask.sum()} filas, se esperaban 885"
    assert len(feats) == 110, f"{len(feats)} features, se esperaban 110"

    x_train, x_test = data.loc[train_mask, feats], data.loc[test_mask, feats]
    y_train = data.loc[train_mask, TARGET].to_numpy()
    y_test = data.loc[test_mask, TARGET].to_numpy()
    mes = data.loc[test_mask, "mes_obs"].astype(str).str.slice(0, 7).to_numpy()
    print(f"train {len(x_train)} | test {len(x_test)} | prev test {y_test.mean():.4f} "
          f"| features {len(feats)}")

    # Referencia: XGBoost con las 110 features. La ganancia (fit en train, sin ver
    # test) define las 20 features que puede recibir TabFM.
    p_full, secs_full, model_full = run_xgb(feats, x_train, y_train, x_test)
    gain = pd.Series(model_full.get_booster().get_score(importance_type="gain"))
    top = list(gain.reindex(feats).fillna(0).sort_values(ascending=False)
               .head(N_FEATS_TABFM).index)
    print(f"top-{N_FEATS_TABFM} por ganancia XGBoost: {top}")

    p_top, secs_top, _ = run_xgb(top, x_train, y_train, x_test)

    bigquery, client = bq_personal()
    p_tabfm, secs_tabfm, info = run_tabfm(bigquery, client, top, x_train, y_train, x_test)
    assert len(p_tabfm) == len(y_test), f"AI.PREDICT devolvió {len(p_tabfm)} filas"
    print(f"TabFM: {secs_tabfm:.1f}s | bytes billed {info['bytes_billed']:,} "
          f"| ~USD {info['usd_on_demand']:.4f} | job {info['job_id']}")

    rows = [
        {"modelo": f"XGBoost tuneado ({len(feats)} feats)", "n_feats": len(feats),
         "segundos": secs_full, **oot_metrics(y_test, p_full, mes)},
        {"modelo": f"XGBoost tuneado ({N_FEATS_TABFM} feats)", "n_feats": N_FEATS_TABFM,
         "segundos": secs_top, **oot_metrics(y_test, p_top, mes)},
        {"modelo": f"TabFM zero-shot ({N_FEATS_TABFM} feats)", "n_feats": N_FEATS_TABFM,
         "segundos": secs_tabfm, **oot_metrics(y_test, p_tabfm, mes)},
    ]
    res = pd.DataFrame(rows).set_index("modelo")
    print(res.round(4).to_string())

    REPORTS.mkdir(exist_ok=True)
    res.round(4).to_csv(REPORTS / "tabfm.csv")
    write_report(res, top, feats, y_test, info, secs_tabfm)
    print(f"reporte -> {REPORTS / 'tabfm.md'}")


def write_report(res, top, feats, y_test, info, secs_tabfm):
    cols = ["n_feats", "oot_AUC", "oot_AUCstd", "oot_PRAUC", "oot_liftPR", "oot_lift10",
            "oot_prec", "oot_rec", "oot_F1", "segundos"]
    tabla = res[cols].round(4)
    delta = res.loc[res.index[2], "oot_AUC"] - res.loc[res.index[0], "oot_AUC"]
    md = [
        "# TabFM (BigQuery `AI.PREDICT`) vs XGBoost tuneado",
        f"\n> Generado por `05_modelling/10_tabfm.py` el {time.strftime('%Y-%m-%d %H:%M')}.",
        f"> Test OOT: {len(y_test)} filas, prevalencia {y_test.mean():.4f}. "
        "Un solo pase, sin tuning y sin GroupKFold.",
        "\n## Qué es TabFM",
        "\n**TabFM** es el modelo fundacional pre-entrenado de Google Research para datos "
        "tabulares. No se entrena: recibe la tabla de train como ejemplos *in-context* y "
        "predice en un solo forward pass. En Google Cloud vive **dentro de BigQuery**, "
        "vía la función SQL `AI.PREDICT` (estado: Preview). No genera modelo persistido "
        "ni endpoint desplegado.",
        "\n- Doc: <https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/"
        "bigqueryml-syntax-ai-predict>",
        "- Paper/blog: <https://research.google/blog/introducing-tabfm-a-zero-shot-"
        "foundation-model-for-tabular-data/>",
        "\nNo confundir con **TimesFM** (series de tiempo) ni con AutoML Tables / BQML "
        "`BOOSTED_TREE_CLASSIFIER`, que sí entrenan un modelo propio.",
        "\n## Limitación que condiciona el experimento",
        "\n`AI.PREDICT` acepta **como máximo 20 columnas de features**. Con las "
        f"{len(feats)} del pipeline la consulta falla en duro:",
        "\n```\n400 The number of features 110 exceeds the maximum allowed number "
        "of features 20.\n```",
        "\nPara superar ese techo hay que escribir a `bqml-feedback@google.com` "
        "(no es un flag). Así que TabFM corre sobre las **20 features top por ganancia "
        "de XGBoost ajustado solo en train**, y se agrega una fila de XGBoost sobre esas "
        "mismas 20 para separar *efecto del modelo* de *efecto del recorte de features*.",
        "\nFeatures usadas:\n",
        "```\n" + "\n".join(f"{i:2d}. {f}" for i, f in enumerate(top, 1)) + "\n```",
        "\n## Resultados (bloque OOT, 885 filas)\n",
        tabla.to_markdown(),
        "\n`oot_AUCstd` = std del AUC por mes (solo meses con ambas clases). "
        "`oot_liftPR` = PR-AUC / prevalencia. `oot_lift10` = lift del decil top. "
        "`prec`/`rec`/`F1` al umbral 0.5. `segundos` = fit+predict para XGBoost, "
        "latencia de la consulta `AI.PREDICT` para TabFM.",
        "\n**Ojo con prec/rec/F1 al 0.5**: no son comparables entre filas. XGBoost corre con "
        "`scale_pos_weight` (balanceado), o sea que sus probabilidades están infladas hacia la "
        "clase positiva y el 0.5 le queda como un umbral agresivo (recall alto, precisión baja). "
        "TabFM sale calibrado a la prevalencia real, así que el mismo 0.5 le queda conservador "
        "(precisión alta, recall bajo). Las filas se comparan de verdad por las métricas de "
        "ordenamiento — AUC, PR-AUC y lift del decil — que son invariantes al umbral; el punto "
        "operativo de cada modelo se elige después.",
        "\n## Costo y tiempo",
        f"\n- Consulta `AI.PREDICT`: **{secs_tabfm:.1f} s**, "
        f"`total_bytes_billed` = {info['bytes_billed']:,} bytes "
        f"→ **~USD {info['usd_on_demand']:.4f}** a precio on-demand (6.25 USD/TiB).",
        "- Durante el Preview TabFM se factura como una consulta BigQuery normal "
        "(bytes procesados u on-demand slots). **Desde el 30/10/2026 pasa a precio por "
        "tokens**: `(filas_train × cols + filas_pred × (cols-1)) × n_ensembles`, sobre el "
        "costo normal de BigQuery. Un GroupKFold de 5 folds multiplicaría eso por 5.",
        "- Sin endpoints ni instancias: costo recurrente **cero**. Las tablas temporales "
        "se borran al final del script.",
        "\n## Limitaciones",
        "\n- **20 features** máximo (el bloqueo real para este dataset).",
        "- Máximo 10 clases en clasificación (irrelevante acá, es binario).",
        "- El label debe ser `BOOL` o `STRING`; con `INT64` hace regresión en silencio.",
        "- Estado **Preview**: sin SLA, la API y el precio pueden cambiar.",
        "- No hay control de semilla ni de `n_ensembles` → las probabilidades no son "
        "bit-a-bit reproducibles entre corridas.",
        "- No devuelve importancias ni nada explicable: para la tesis no reemplaza el "
        "análisis SHAP/permutación que ya tenemos sobre XGBoost.",
        "\n## Lectura",
        "\n" + lectura(res, delta),
    ]
    (REPORTS / "tabfm.md").write_text("\n".join(md) + "\n")


def lectura(res, delta):
    tabfm, xgb_full, xgb_top = res.index[2], res.index[0], res.index[1]
    auc_t = res.loc[tabfm, "oot_AUC"]
    auc_f = res.loc[xgb_full, "oot_AUC"]
    auc_p = res.loc[xgb_top, "oot_AUC"]
    if delta > 0.01:
        veredicto = (f"TabFM **supera** al XGBoost tuneado con las {res.loc[xgb_full, 'n_feats']:.0f} "
                     f"features ({auc_t:.4f} vs {auc_f:.4f}, +{delta:.4f} de AUC) y encima con "
                     "20 features y sin tuning.")
    elif delta < -0.01:
        veredicto = (f"TabFM **pierde** contra el XGBoost tuneado ({auc_t:.4f} vs {auc_f:.4f}, "
                     f"{delta:+.4f} de AUC).")
    else:
        veredicto = (f"TabFM **empata** con el XGBoost tuneado ({auc_t:.4f} vs {auc_f:.4f}, "
                     f"{delta:+.4f} de AUC): la diferencia está dentro del ruido de un bloque "
                     "de 885 filas.")
    return (
        f"{veredicto} Contra el XGBoost con las mismas 20 features ({auc_p:.4f}) la brecha "
        f"es de {auc_t - auc_p:+.4f}, que es la comparación limpia modelo-contra-modelo.\n\n"
        "**Para la tesis**: sirve como punto de referencia externo — 'un modelo fundacional "
        "sin entrenar ni tunear llega hasta acá' — y como evidencia de que el techo de ~0.76 "
        "de AUC es del problema y de los datos, no del algoritmo. No sirve como modelo "
        "principal: el tope de 20 features obliga a tirar 90 variables construidas a mano, "
        "que es justamente el trabajo que la tesis documenta.\n\n"
        "**Para producción**: no. El límite de 20 features, el estado Preview sin SLA, la "
        "falta de explicabilidad y el cambio a precio por tokens del 30/10/2026 lo hacen "
        "peor apuesta que un XGBoost que ya está entrenado, es reproducible y cuesta cero "
        "por inferencia. El cuello de botella del proyecto sigue sin ser el algoritmo: es "
        "que el uplift de la acción de retención no está medido."
    )


if __name__ == "__main__":
    main()
