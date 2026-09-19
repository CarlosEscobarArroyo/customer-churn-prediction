"""Ablación de los 4 grupos de features nuevas de qry_churn.sql (2026-09):
PAGO, CAMPAÑAS, RED y MIX. NOTA: la ablación original (reporte en reports/) corrió
sobre la revisión del SQL que incluía las 4 familias; tras ella solo CAMPAÑAS quedó
en qry_churn.sql, por lo que hoy GRUPOS se filtra a las columnas presentes en el
dataset (re-ejecutar evalúa solo los grupos disponibles). Entrena LogReg / RF / XGBoost (hiperparámetros ya
tuneados, sin Optuna) sobre: base (schema anterior), base + cada grupo por
separado, y todas. Sin selección por permutación (aísla el efecto de las
features). Reporta los dos protocolos y, para XGBoost con todas, la importancia
por permutación (AUC, en el test OOT) de cada feature nueva.

Uso:    uv run python 05_modelling/experimentos/06_features_nuevas.py
Lee:    data/processed/churn_dataset.csv  (00_dataset_construction/build_churn_dataset.py)
Salida: 05_modelling/experimentos/reports/features_nuevas_ablation.{csv,md}
"""
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (MODELS, PROC, REPORTS, TARGET, evaluate, make_model,  # noqa: E402
                      oot_split, prepare)

GRUPOS = {
    "pago": ["ratio_pago_u3", "ratio_pago_u6", "ratio_pago_u12", "ratio_pago_acum"],
    "campanas": ["camp_saltadas", "camp_part_u12", "tasa_camp_u3", "tasa_camp_u6",
                 "tasa_camp_u12", "pct_directo_u12", "es_nueva_u12"],
    "red": ["tiene_lider", "red_size", "red_tasa_act_u3", "red_tasa_act_u12", "lider_act_u3",
            "lider_act_u12", "lider_recencia", "equipo_size", "equipo_tasa_act_u3"],
    "mix": ["pct_no_ropa_u12", "pct_accesorios_u12", "pct_belleza_u12", "unidades_u3",
            "unidades_u12", "precio_unit_u12"],
}


def main():
    global NUEVAS
    df = pd.read_csv(PROC / "churn_dataset.csv", parse_dates=["mes_obs"])
    # solo los grupos cuyas columnas existen en la revisión vigente del SQL
    for g in list(GRUPOS):
        GRUPOS[g] = [c for c in GRUPOS[g] if c in df.columns]
        if not GRUPOS[g]:
            del GRUPOS[g]
    NUEVAS = [c for g in GRUPOS.values() for c in g]
    train_mask, test_mask = oot_split(df["mes_rank"])
    d, feats_all = prepare(df, train_mask)
    y, groups = d[TARGET].values, d["id_vendedor"].values
    mes = d["mes_obs"].dt.strftime("%Y-%m").values
    base = [c for c in feats_all if c not in NUEVAS]
    configs = {"base": base, **{f"base+{g}": base + cols for g, cols in GRUPOS.items()},
               "todas": base + NUEVAS}
    print(f"{len(d):,} filas | base {len(base)} features | nuevas {len(NUEVAS)}", flush=True)

    rows = []
    for cfg, feats in configs.items():
        t0 = time.time()
        for name in MODELS:
            rows.append({"config": cfg, "n_feat": len(feats), "modelo": name,
                         **evaluate(name, d[feats], y, groups, mes, train_mask, test_mask)})
        print(pd.DataFrame(rows[-3:]).set_index("modelo")[["gkf_AUC", "gkf_liftPR", "oot_AUC", "oot_AUCstd"]]
              .round(4).to_string(), f"\n  {cfg} listo en {time.time() - t0:.0f}s\n", flush=True)
    res = pd.DataFrame(rows)

    # importancia por permutación de las nuevas (XGB todas, AUC en test OOT)
    X = d[configs["todas"]]
    m = make_model("XGBoost", y[train_mask]).fit(X[train_mask], y[train_mask])
    imp = permutation_importance(m, X[test_mask], y[test_mask], scoring="roc_auc",
                                 n_repeats=10, random_state=42, n_jobs=-1)
    rank = (pd.DataFrame({"feature": X.columns, "dAUC": imp.importances_mean, "std": imp.importances_std})
            .sort_values("dAUC", ascending=False).reset_index(drop=True))
    rank["rank"] = rank.index + 1
    rank["grupo"] = rank["feature"].map({c: g for g, cols in GRUPOS.items() for c in cols}).fillna("base")
    write_report(res, rank, len(base))


def write_report(res, rank, n_base):
    REPORTS.mkdir(exist_ok=True)
    res.to_csv(REPORTS / "features_nuevas_ablation.csv", index=False)
    piv = lambda m: res.pivot(index="config", columns="modelo", values=m)[MODELS]  # noqa: E731
    order = list(dict.fromkeys(res["config"]))
    delta = lambda m: (piv(m) - piv(m).loc["base"]).loc[order].round(4)  # noqa: E731
    nuevas = rank[rank["grupo"] != "base"]
    md = [
        "# Features nuevas (PAGO / CAMPAÑAS / RED / MIX) — ablación por grupo",
        f"\n> Generado por `05_modelling/06_features_nuevas.py` el {time.strftime('%Y-%m-%d %H:%M')}.",
        f"> Base = {n_base} features del schema anterior (sin selección por permutación); "
        "modelos con hiperparámetros de `05_modelling/*_best_params.json`.",
        "\n## AUC GroupKFold(5) por vendedora\n", piv("gkf_AUC").loc[order].round(4).to_markdown(),
        "\n### Δ vs base\n", delta("gkf_AUC").to_markdown(),
        "\n## AUC out-of-period (± std por mes)\n",
        (piv("oot_AUC").loc[order].round(4).astype(str) + " ± "
         + piv("oot_AUCstd").loc[order].round(3).astype(str)).to_markdown(),
        "\n### Δ vs base\n", delta("oot_AUC").to_markdown(),
        "\n## Lift PR-AUC / prevalencia (GroupKFold)\n", piv("gkf_liftPR").loc[order].round(3).to_markdown(),
        "\n## Tabla completa\n", res.set_index(["config", "modelo"]).round(4).to_markdown(),
        "\n## Importancia por permutación de las features nuevas (XGBoost todas, AUC en test OOT)\n",
        f"Rank sobre {len(rank)} features. dAUC = caída de AUC al permutar la variable "
        "(10 repeticiones); negativo o ~0 = no aporta fuera de muestra.\n",
        nuevas[["rank", "feature", "grupo", "dAUC", "std"]].round(4).to_markdown(index=False),
        "\n### Top 15 global (para contexto)\n",
        rank.head(15)[["rank", "feature", "grupo", "dAUC", "std"]].round(4).to_markdown(index=False),
    ]
    (REPORTS / "features_nuevas_ablation.md").write_text("\n".join(md) + "\n")
    print(f"\nreporte -> {REPORTS / 'features_nuevas_ablation.md'}")


if __name__ == "__main__":
    main()
