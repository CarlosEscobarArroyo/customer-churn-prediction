"""Sensibilidad al horizonte de churn k (meses sin compra que definen el abandono).

Reentrena LogReg / RandomForest / XGBoost con los hiperparámetros YA tuneados
(05_modelling/*_best_params.json; sin Optuna) para cada k de K_VALUES, con
churn = 1 si la vendedora no compra en [t+1 .. t+k]. Por cada k se repite el
pipeline 02→04 completo:

  1. dataset desde qry_churn.sql con la ventana de etiqueta y el filtro de censura
     reescritos a k (cache: data/processed/horizonte/churn_dataset_k{k}.csv);
  2. split OOT (test = últimos 4 meses etiquetados, gap = k) y GroupKFold(5);
  3. preprocessing fit-en-train + features derivadas (idéntico a 03 y 04);
  4. selección por importancia de permutación > 0 en train (idéntico a 04);
  5. métricas bajo los dos protocolos por modelo.

Ojo al comparar OOT entre k: el bloque de test son los últimos 4 meses CON
etiqueta, que se corren hacia atrás k meses → períodos distintos por k. El
GroupKFold sobre todas las filas es la comparación más homogénea.

Uso:    uv run python 05_modelling/experimentos/05_horizonte_churn.py
Salida: 05_modelling/experimentos/reports/horizonte_churn_modelos.{csv,md}
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (MODELS, REPORTS, SQL_PATH, TARGET, bq_client, evaluate,  # noqa: E402
                      oot_split, prepare, select_features)

SQL = SQL_PATH.read_text()
CACHE = SQL_PATH.parents[1] / "data" / "processed" / "horizonte"
OUT = REPORTS
K_VALUES = [3, 4, 5, 6, 7, 8, 9, 12]


# --- 1. dataset por k: misma query, ventana de etiqueta reescrita -------------
def sql_for(k):
    subs = {"1 FOLLOWING AND 6 FOLLOWING": f"1 FOLLOWING AND {k} FOLLOWING",
            "g.max_rank - 6": f"g.max_rank - {k}"}
    sql = SQL
    for old, new in subs.items():
        assert sql.count(old) == 1, f"qry_churn.sql cambió: no encuentro {old!r}"
        sql = sql.replace(old, new)
    return sql


def load_k(k, client=None):
    path = CACHE / f"churn_dataset_k{k}.csv"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        client = client or bq_client()
        client.query(sql_for(k)).to_dataframe().to_csv(path, index=False)
        print(f"  k={k}: dataset extraído de BigQuery -> {path.name}")
    return pd.read_csv(path, parse_dates=["mes_obs"])


def run_k(k, client=None):
    df = load_k(k, client)
    train_mask, test_mask = oot_split(df["mes_rank"], v=k)
    d, feats_all = prepare(df, train_mask)
    y = d[TARGET].values
    feats = select_features(d[feats_all], y, train_mask)
    X = d[feats]
    mes = d["mes_obs"].dt.strftime("%Y-%m").values
    info = {"k": k, "n_rows": len(d), "n_vend": d["id_vendedor"].nunique(),
            "prevalencia": y.mean(), "n_feat": len(feats),
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "prev_test": y[test_mask].mean(),
            "test_periodo": f"{mes[test_mask].min()}..{mes[test_mask].max()}"}
    return [{**info, "modelo": name,
             **evaluate(name, X, y, d["id_vendedor"].values, mes, train_mask, test_mask)}
            for name in MODELS]


def write_report(res):
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "horizonte_churn_modelos.csv", index=False)
    ds = res.drop_duplicates("k").set_index("k")[
        ["n_rows", "n_vend", "prevalencia", "n_feat", "n_train", "n_test", "prev_test", "test_periodo"]]
    piv = lambda m: res.pivot(index="k", columns="modelo", values=m)[MODELS].round(4)  # noqa: E731
    fin = max(pd.Period(p, "M") + k for p, k in zip(res["test_periodo"].str[-7:], res["k"]))
    md = [
        "# Sensibilidad al horizonte de churn k — modelos tuneados (sin Optuna)",
        f"\n> Generado por `05_modelling/05_horizonte_churn.py` el {time.strftime('%Y-%m-%d %H:%M')}.",
        f"> Último mes con pedidos en la fuente: {fin} (puede estar incompleto); "
        f"último mes observado con etiqueta = {fin} − k. k ∈ {K_VALUES}; "
        "hiperparámetros de `05_modelling/*_best_params.json` (tuneados con k=6).",
        "\n## Dataset por k\n", ds.round(4).to_markdown(),
        "\n## AUC GroupKFold(5) por vendedora — comparación principal\n", piv("gkf_AUC").to_markdown(),
        "\n## Lift PR-AUC / prevalencia (GroupKFold)\n", piv("gkf_liftPR").round(3).to_markdown(),
        "\n## AUC out-of-period (test = últimos 4 meses etiquetados, gap = k)\n", piv("oot_AUC").to_markdown(),
        "\n## Recall @0.5 (OOT)\n", piv("oot_rec").to_markdown(),
        "\n## Precision @0.5 (OOT)\n", piv("oot_prec").to_markdown(),
        "\n## Tabla completa\n",
        res.drop(columns=ds.columns).set_index(["k", "modelo"]).round(4).to_markdown(),
        "\n## Notas\n",
        "- El bloque OOT cambia de período con k (ver `test_periodo`): las AUC OOT entre k "
        "no son estrictamente comparables; GroupKFold usa todas las filas y sí lo es.",
        "- La prevalencia cae con k, por eso se reporta PR-AUC/prevalencia (lift) y no PR-AUC cruda.",
        "- Selección de features por k (importancia de permutación > 0 en train), como en 04.",
    ]
    (OUT / "horizonte_churn_modelos.md").write_text("\n".join(md) + "\n")
    print(f"\nreporte -> {OUT / 'horizonte_churn_modelos.md'}")


def main(client=None):
    assert sql_for(6) == SQL, "k=6 debe reproducir qry_churn.sql exactamente"
    for k in K_VALUES:           # extraer todo primero (BigQuery), luego modelar
        load_k(k, client)
    rows = []
    for k in K_VALUES:
        t0 = time.time()
        rows += run_k(k)
        print(pd.DataFrame(rows[-3:]).set_index("modelo")[
            ["n_rows", "prevalencia", "n_feat", "gkf_AUC", "gkf_liftPR", "oot_AUC", "oot_rec"]]
            .round(4).to_string(), f"\n  k={k} listo en {time.time() - t0:.0f}s\n", flush=True)
    write_report(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
