"""Comparación ampliada de modelos (LogReg, RF, XGBoost, LightGBM, CatBoost, TabFM) y
ensemble de los 3 mejores bajo los dos protocolos del repo.

## Qué compara

Sobre el dataset vigente (`data/processed/churn_dataset.csv`, 30.821 filas,
91 features tras `pipeline.prepare`), con el split OOT de `pipeline.oot_split`
(train 28.735 filas, test 885 = últimos 4 meses etiquetados, gap 6) y el
GroupKFold(5) por vendedora de `pipeline.evaluate_fn`:

- **LogReg / RandomForest / XGBoost**: `pipeline.make_model` con los
  hiperparámetros tuneados de `05_modelling/*_best_params.json` (referencia).
- **LightGBM**: `LGBMClassifier(n_estimators=800, learning_rate=0.02, num_leaves=15,
  min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
  reg_lambda=1.0, scale_pos_weight=neg/pos del train del fold)`.
- **CatBoost**: `CatBoostClassifier(iterations=1000, depth=4, learning_rate=0.03,
  l2_leaf_reg=3, auto_class_weights="Balanced")`.
  LightGBM y CatBoost **no se tunean**: son un régimen regularizado análogo al
  XGBoost tuneado (depth 3, lr 0.011, 950 árboles), defaults razonables para
  saber si la familia aporta antes de invertir en Optuna.
- **TabFM** (BigQuery `AI.PREDICT`, Preview, zero-shot): mismo código que
  `10_tabfm.py` (se importa), sobre las 20 features de mayor ganancia de un
  XGBoost ajustado solo en el train del OOT (tope duro de 20 columnas). Corre
  bajo los dos protocolos: 5 llamadas GroupKFold + 1 OOT (~90 s cada una).
  Las predicciones se cachean en `reports/tabfm_predicciones.csv`
  (`id_vendedor, mes_rank, split, p`): si el archivo existe no se vuelve a
  llamar a BigQuery. **TabFM no es bit-reproducible entre corridas** (sin
  semilla ni control de `n_ensembles`); las 20 features se eligen con el
  train del OOT, que solapa con los folds GKF (sesgo mínimo, solo selección).

## Ensemble

Los 6 candidatos se ordenan por `gkf_AUC` y se toman los 3 mejores. Dos
ensembles sin ajuste (no hay fit, así que no hay leakage): **promedio de
rangos** (rango normalizado dentro de cada fold GKF / dentro del bloque OOT) y
**promedio de probabilidades**. Además: rank-average de los 6 y un **stacking**
(regresión logística sobre las OOF de los 3 mejores, ajustada solo en las filas
del train del OOT y aplicada al OOT). Del stacking solo se reporta OOT: su
métrica GKF no sería honesta (el meta-modelo vería las OOF de todos los folds).

## Cómo re-ejecutar

    uv run python 05_modelling/experimentos/11_ensemble.py

TabFM necesita el token de la cuenta personal (`gcloud auth print-access-token
--account=...`, ver `ACCOUNT` en `10_tabfm.py`). Las tablas temporales se borran
en cada llamada; no queda nada con costo recurrente. Si BigQuery falla, el
script sigue sin TabFM y lo marca en el reporte.
Salida: `05_modelling/experimentos/reports/ensemble.{md,csv}`.
"""
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (MODELS, PROC, REPORTS, RS, TARGET, evaluate_fn, lift10,  # noqa: E402
                      make_model, oot_split, prepare)

tabfm = importlib.import_module("10_tabfm")
CACHE = REPORTS / "tabfm_predicciones.csv"
N_TOP = 3
COLS = ["gkf_AUC", "gkf_liftPR", "gkf_lift10", "oot_AUC", "oot_AUCstd", "oot_PRAUC",
        "oot_lift10", "segundos"]


def make_extra(name, y_tr):
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=800, learning_rate=0.02, num_leaves=15,
                              min_child_samples=30, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=spw,
                              random_state=RS, n_jobs=-1, verbose=-1)
    if name == "CatBoost":
        return CatBoostClassifier(iterations=1000, depth=4, learning_rate=0.03, l2_leaf_reg=3,
                                  auto_class_weights="Balanced", random_seed=RS, verbose=0,
                                  thread_count=-1)
    return make_model(name, y_tr)


def capturar(fit_predict, n):
    """Envuelve fit_predict para quedarse con OOF (5 folds GKF) y predicción OOT."""
    oof, calls = np.full(n, np.nan), []

    def fp(tr, te):
        p = fit_predict(tr, te)
        calls.append(p)
        if len(calls) <= 5:
            oof[te] = p
        return p
    return fp, oof, calls


def tabfm_fit_predict(d, feats_top, y, cache):
    """fit_predict de TabFM con caché por fila: llama a AI.PREDICT solo si faltan filas."""
    keys = list(zip(d["id_vendedor"], d["mes_rank"]))
    info = {"bytes": 0, "usd": 0.0, "llamadas": 0, "n": 0}

    def fp(tr, te):
        split = "oof" if info["n"] < 5 else "oot"  # evaluate_fn: 5 folds GKF y luego OOT
        info["n"] += 1
        hit = [cache.get((split, *keys[i])) for i in te]
        if all(h is not None for h in hit):
            print(f"  TabFM {split}: {len(te)} filas desde caché", flush=True)
            return np.array(hit, dtype=float)
        bigquery, client = tabfm.bq_personal()  # token nuevo por llamada
        p, secs, meta = tabfm.run_tabfm(bigquery, client, feats_top, d.iloc[tr], y[tr], d.iloc[te])
        assert len(p) == len(te)
        info["bytes"] += meta["bytes_billed"]
        info["usd"] += meta["usd_on_demand"]
        info["llamadas"] += 1
        print(f"  TabFM {split}: {secs:.0f}s, ~USD {meta['usd_on_demand']:.4f}", flush=True)
        for i, pi in zip(te, p):
            cache[(split, *keys[i])] = pi
        pd.DataFrame([{"split": s, "id_vendedor": v, "mes_rank": r, "p": pv}
                      for (s, v, r), pv in cache.items()]
                     ).to_csv(CACHE, index=False)
        return p
    return fp, info


def metricas(oof, p, y, yt, mt):
    """Mismas fórmulas que pipeline.evaluate_fn, a partir de OOF y OOT ya combinadas."""
    aucs = [roc_auc_score(yt[mt == mm], p[mt == mm]) for mm in np.unique(mt)
            if 0 < yt[mt == mm].mean() < 1]
    out = {"oot_AUC": roc_auc_score(yt, p), "oot_AUCstd": float(np.std(aucs)),
           "oot_PRAUC": average_precision_score(yt, p), "oot_lift10": lift10(yt, p)}
    if oof is not None:
        out.update({"gkf_AUC": roc_auc_score(y, oof),
                    "gkf_liftPR": average_precision_score(y, oof) / y.mean(),
                    "gkf_lift10": lift10(y, oof)})
    return out


def rank_norm(p, fold):
    """Rango normalizado en [0, 1] dentro de cada fold/bloque."""
    out = np.zeros(len(p))
    for f in np.unique(fold):
        m = fold == f
        out[m] = rankdata(p[m]) / m.sum()
    return out


def main():
    t_all = time.time()
    df = pd.read_csv(PROC / "churn_dataset.csv", parse_dates=["mes_obs"])
    train_mask, test_mask = oot_split(df["mes_rank"])
    d, feats = prepare(df, train_mask)
    X, y, groups = d[feats], d[TARGET].values, d["id_vendedor"].values
    mes = d["mes_obs"].dt.strftime("%Y-%m").values
    n = len(d)
    assert test_mask.sum() == 885 and len(feats) == 91
    print(f"{n:,} filas | {len(feats)} features | train {train_mask.sum():,} "
          f"| test {test_mask.sum()} (prev {y[test_mask].mean():.3f})", flush=True)

    preds, rows = {}, []
    for name in MODELS + ["LightGBM", "CatBoost"]:
        def fit_predict(tr, te):
            m = make_extra(name, y[tr]).fit(X.iloc[tr], y[tr])
            return m.predict_proba(X.iloc[te])[:, 1]
        fp, oof, calls = capturar(fit_predict, n)
        t0 = time.time()
        ev = evaluate_fn(fp, y, groups, mes, train_mask, test_mask)
        preds[name] = (oof, calls[5])
        rows.append({"modelo": name, "segundos": time.time() - t0, **ev})
        print(f"{name:12s} gkf_AUC {ev['gkf_AUC']:.4f} | oot_AUC {ev['oot_AUC']:.4f} "
              f"± {ev['oot_AUCstd']:.3f} | lift10 {ev['oot_lift10']:.2f} | "
              f"{rows[-1]['segundos']:.0f}s", flush=True)

    # --- TabFM: 20 features top por ganancia del XGBoost ajustado solo en train OOT
    xgb = make_model("XGBoost", y[train_mask]).fit(X[train_mask], y[train_mask])
    gain = pd.Series(xgb.get_booster().get_score(importance_type="gain"))
    top = list(gain.reindex(feats).fillna(0).sort_values(ascending=False).head(20).index)
    cache = {}
    if CACHE.exists():
        c = pd.read_csv(CACHE)
        cache = {(s, v, r): p for s, v, r, p in zip(c["split"], c["id_vendedor"], c["mes_rank"], c["p"])}
        print(f"caché TabFM: {len(cache):,} predicciones en {CACHE.name}", flush=True)
    tabfm_ok, tabfm_info = False, {}
    try:
        fp_t, tabfm_info = tabfm_fit_predict(d, top, y, cache)
        fp, oof, calls = capturar(fp_t, n)
        t0 = time.time()
        ev = evaluate_fn(fp, y, groups, mes, train_mask, test_mask)
        preds["TabFM"] = (oof, calls[5])
        rows.append({"modelo": "TabFM", "segundos": time.time() - t0, **ev})
        tabfm_ok = True
        print(f"{'TabFM':12s} gkf_AUC {ev['gkf_AUC']:.4f} | oot_AUC {ev['oot_AUC']:.4f} "
              f"± {ev['oot_AUCstd']:.3f} | lift10 {ev['oot_lift10']:.2f} | "
              f"{rows[-1]['segundos']:.0f}s | {tabfm_info}", flush=True)
    except Exception as exc:  # se sigue sin TabFM y se reporta el error
        tabfm_info["error"] = f"{type(exc).__name__}: {exc}"
        print(f"TabFM FALLÓ: {tabfm_info['error']}", flush=True)

    # --- ensembles
    res = pd.DataFrame(rows).set_index("modelo")
    orden = res["gkf_AUC"].sort_values(ascending=False).index.tolist()
    top3 = orden[:N_TOP]
    print(f"orden por gkf_AUC: {orden} -> top {N_TOP}: {top3}", flush=True)
    fold = np.zeros(n, dtype=int)
    for k, (_, va) in enumerate(GroupKFold(5).split(np.zeros(n), y, groups)):
        fold[va] = k
    yt, mt = y[test_mask], mes[test_mask]
    tr_rows = train_mask  # filas de train del OOT (para el stacking)

    def combinar(names, modo):
        oofs = np.array([preds[m][0] for m in names])
        oots = np.array([preds[m][1] for m in names])
        if modo == "rank":
            oofs = np.array([rank_norm(o, fold) for o in oofs])
            oots = np.array([rank_norm(o, np.zeros(len(o))) for o in oots])
        return oofs.mean(0), oots.mean(0)

    ens = {}
    for modo, tag in (("rank", "rank-avg"), ("prob", "prob-avg")):
        oof, p = combinar(top3, modo)
        ens[f"Ensemble top{N_TOP} {tag}"] = metricas(oof, p, y, yt, mt)
    oof, p = combinar(orden, "rank")
    ens[f"Ensemble {len(orden)} rank-avg"] = metricas(oof, p, y, yt, mt)
    z_tr = np.column_stack([preds[m][0][tr_rows] for m in top3])
    z_te = np.column_stack([preds[m][1] for m in top3])
    meta = LogisticRegression(class_weight="balanced", max_iter=1000).fit(z_tr, y[tr_rows])
    ens[f"Stacking top{N_TOP} (solo OOT)"] = metricas(None, meta.predict_proba(z_te)[:, 1], y, yt, mt)
    ens_df = pd.DataFrame(ens).T
    ens_df["coef_stacking"] = pd.Series(dtype="object")
    ens_df.loc[f"Stacking top{N_TOP} (solo OOT)", "coef_stacking"] = str(
        dict(zip(top3, meta.coef_[0].round(3))))

    full = pd.concat([res, ens_df])
    full["top3"] = [m in top3 for m in full.index]
    print(full[COLS].round(4).to_string(), flush=True)
    REPORTS.mkdir(exist_ok=True)
    full.round(4).to_csv(REPORTS / "ensemble.csv")
    write_report(full, top3, orden, top, tabfm_ok, tabfm_info)
    print(f"total {time.time() - t_all:.0f}s | reporte -> {REPORTS / 'ensemble.md'}")


def lectura(full, top3, tabfm_ok):
    ref = full.loc["XGBoost"]
    ind = full.loc[[m for m in full.index if not m.startswith(("Ensemble", "Stacking"))]]
    nuevos = ind.drop(index=["LogReg", "RandomForest", "XGBoost"])
    mejor = nuevos["gkf_AUC"].idxmax()
    d_gkf, d_oot = nuevos.loc[mejor, "gkf_AUC"] - ref["gkf_AUC"], nuevos.loc[mejor, "oot_AUC"] - ref["oot_AUC"]
    e3 = full.loc[f"Ensemble top{N_TOP} rank-avg"]
    mejor_ind = ind["gkf_AUC"].idxmax()
    de_gkf, de_oot = e3["gkf_AUC"] - ind["gkf_AUC"].max(), e3["oot_AUC"] - ind["oot_AUC"].max()
    rango_g = ind["gkf_AUC"].max() - ind["gkf_AUC"].min()
    rango_o = ind["oot_AUC"].max() - ind["oot_AUC"].min()
    std = ref["oot_AUCstd"]

    def veredicto(dg, do):
        if dg > 0.005 and do > std:
            return "**supera fuera del ruido** en ambos protocolos"
        if dg > 0.005 or do > std:
            return "supera en un protocolo pero no en el otro: no es concluyente"
        return "**empata** (dentro del ruido: < 0.005 en GKF, < std por mes en OOT)"

    p = [
        f"- **Modelos nuevos vs XGBoost tuneado.** El mejor nuevo por GKF es **{mejor}** "
        f"({nuevos.loc[mejor, 'gkf_AUC']:.4f} GKF / {nuevos.loc[mejor, 'oot_AUC']:.4f} OOT) "
        f"vs XGBoost {ref['gkf_AUC']:.4f} / {ref['oot_AUC']:.4f}: Δ {d_gkf:+.4f} GKF, "
        f"{d_oot:+.4f} OOT → {veredicto(d_gkf, d_oot)}. Los {len(ind)} modelos individuales "
        f"caben en un rango de {rango_g:.4f} de AUC GKF y {rango_o:.4f} de AUC OOT "
        f"(std por mes del OOT ≈ {std:.3f}).",
        f"- **Ensemble.** El rank-average de los {N_TOP} mejores ({', '.join(top3)}) da "
        f"{e3['gkf_AUC']:.4f} GKF / {e3['oot_AUC']:.4f} OOT: Δ {de_gkf:+.4f} / {de_oot:+.4f} "
        f"contra el mejor individual ({mejor_ind}) → {veredicto(de_gkf, de_oot)}. "
        f"Prob-avg: {full.loc[f'Ensemble top{N_TOP} prob-avg', 'gkf_AUC']:.4f} / "
        f"{full.loc[f'Ensemble top{N_TOP} prob-avg', 'oot_AUC']:.4f}. Rank-avg de los "
        f"{len(ind)}: {full.loc[f'Ensemble {len(ind)} rank-avg', 'gkf_AUC']:.4f} / "
        f"{full.loc[f'Ensemble {len(ind)} rank-avg', 'oot_AUC']:.4f}. Stacking (solo OOT): "
        f"{full.loc[f'Stacking top{N_TOP} (solo OOT)', 'oot_AUC']:.4f}.",
    ]
    if not tabfm_ok:
        p.append("- **TabFM no corrió** (ver error arriba); la comparación es entre 5 modelos.")
    if max(d_gkf, de_gkf) <= 0.005 and max(d_oot, de_oot) <= std:
        p.append(
            "- **Recomendación: no cambia.** XGBoost tuneado sigue como modelo final. Que seis "
            "familias distintas (lineal, bagging, tres boostings y un modelo fundacional "
            "zero-shot) y sus combinaciones queden en la misma franja es la firma de un "
            "**techo de señal**: las features capturan lo que hay en el historial de compra y "
            "el resto del churn es idiosincrático (no está en los datos). Un ensemble solo suma "
            "cuando los errores de los miembros están descorrelacionados; acá todos aprenden la "
            "misma función de la recencia/frecuencia y sus errores coinciden. Agregar complejidad "
            "de modelo no mueve la aguja; lo que la movería es información nueva (contacto de la "
            "coordinadora, motivo de baja, uplift medido de la acción de retención).")
    else:
        p.append("- **Recomendación:** hay una mejora fuera del ruido; vale revisarla antes de "
                 "cambiar el modelo final (¿se sostiene con otra semilla / otro bloque OOT?).")
    return "\n".join(p)


def write_report(full, top3, orden, feats_top, tabfm_ok, tabfm_info):
    tabla = full[COLS].round(4)
    tabla.index = [f"{m} ★" if m in top3 else m for m in tabla.index]
    md = [
        "# Comparación ampliada de modelos y ensemble (LogReg, RF, XGB, LightGBM, CatBoost, TabFM)",
        f"\n> Generado por `05_modelling/experimentos/11_ensemble.py` el {time.strftime('%Y-%m-%d %H:%M')}.",
        "> 91 features, hiperparámetros del repo para LogReg/RF/XGB; LightGBM y CatBoost con "
        "defaults regularizados (sin Optuna); TabFM zero-shot con 20 features (tope de `AI.PREDICT`). "
        "GroupKFold(5) por vendedora + OOT (885 filas, últimos 4 meses, gap 6). ★ = top 3 por `gkf_AUC`.",
        "\n## Resultados\n", tabla.to_markdown(),
        "\n`segundos` = 5 folds + OOT (fit+predict); para TabFM, latencia acumulada de las 6 consultas "
        "`AI.PREDICT` (0 s si vino de caché). `gkf_liftPR` = PR-AUC / prevalencia; `lift10` = lift del "
        "decil top; `oot_AUCstd` = std del AUC por mes del bloque OOT.",
        f"\nOrden por `gkf_AUC`: {' > '.join(orden)}. Ensembles sin ajuste (rank-avg = rango normalizado "
        "por fold/bloque; prob-avg = promedio de probabilidades). El stacking (regresión logística sobre "
        "las OOF del top 3, ajustada solo en las filas de train del OOT) se reporta **solo en OOT**: su "
        "métrica GKF no sería honesta porque el meta-modelo vería las OOF de todos los folds. "
        f"Coeficientes: {full.loc[f'Stacking top{N_TOP} (solo OOT)', 'coef_stacking']}.",
        "\n## TabFM",
        ("\n" + (f"6 llamadas a `AI.PREDICT` (5 folds + OOT) con las 20 features top por ganancia del "
                 f"XGBoost ajustado en train OOT: `{', '.join(feats_top)}`. "
                 f"Esta corrida: {tabfm_info.get('llamadas', 0)} llamadas nuevas, "
                 f"{tabfm_info.get('bytes', 0):,} bytes facturados (~USD {tabfm_info.get('usd', 0):.4f}); "
                 "el resto salió de `reports/tabfm_predicciones.csv`. **No es bit-reproducible** entre "
                 "corridas (sin semilla ni control de `n_ensembles`): por eso se cachean las predicciones. "
                 "Las 20 features se eligen con el train del OOT, que solapa con los folds GKF (sesgo "
                 "mínimo, solo selección de columnas). Tablas temporales borradas en cada llamada; "
                 "costo recurrente cero.")
         if tabfm_ok else f"\n**TabFM no corrió**: `{tabfm_info.get('error')}`. Tabla sin TabFM."),
        "\n## Lectura\n", lectura(full, top3, tabfm_ok),
    ]
    (REPORTS / "ensemble.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
