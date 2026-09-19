"""Ablación de time slices (Gattermann-Itschert & Thonemann 2021, §5.5 y §6.3-6.4).

Con el MISMO test out-of-period de siempre (últimos 4 meses etiquetados, gap 6),
entrena XGBoost (hiperparámetros tuneados) variando cuántos meses de historia
entran al train:

  * multi-slicing:  los últimos K meses de observación del train (K = 1 .. todos);
  * downsized:      igual cantidad de filas que el caso K, pero muestreadas al azar
                    de TODOS los meses del train, con una sola fila por vendedora
                    cuando el tamaño lo permite (como en el paper). Separa el efecto
                    "más filas" del efecto "más diversidad temporal".

Uso:    uv run python 05_modelling/experimentos/07_ablacion_slices.py
Lee:    data/processed/churn_dataset.csv
Salida: 05_modelling/experimentos/reports/ablacion_slices.{csv,md}
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import PROC, REPORTS, TARGET, lift10, make_model, oot_split, prepare  # noqa: E402

K_VALUES = [1, 2, 3, 6, 12, 24, 36, 60, None]   # None = todo el histórico
SEEDS = range(5)


def main():
    df = pd.read_csv(PROC / "churn_dataset.csv", parse_dates=["mes_obs"])
    train_mask, test_mask = oot_split(df["mes_rank"])
    d, feats = prepare(df, train_mask)
    X, y, g, rank = d[feats], d[TARGET].values, d["id_vendedor"].values, d["mes_rank"].values
    te = np.where(test_mask)[0]
    tr_all = np.where(train_mask)[0]
    meses_train = np.sort(np.unique(rank[tr_all]))[::-1]          # del más reciente al más viejo
    print(f"train {len(tr_all):,} filas en {len(meses_train)} meses | test {len(te):,} filas", flush=True)

    def auc_de(idx):
        p = make_model("XGBoost", y[idx]).fit(X.iloc[idx], y[idx]).predict_proba(X.iloc[te])[:, 1]
        return roc_auc_score(y[te], p), lift10(y[te], p)

    rows = []
    for K in K_VALUES:
        t0 = time.time()
        k = len(meses_train) if K is None else K
        idx_k = tr_all[np.isin(rank[tr_all], meses_train[:k])]
        n = len(idx_k)
        auc, l10 = auc_de(idx_k)
        rows.append({"K_meses": k, "variante": "multi-slicing (últimos K meses)", "n_train": n,
                     "n_vend": len(np.unique(g[idx_k])), "AUC": auc, "AUC_std": 0.0, "lift10": l10})
        if K is not None:                                           # downsized: mismo n, todos los meses
            aucs, lifts, una_por_vend = [], [], n <= len(np.unique(g[tr_all]))
            for s in SEEDS:
                rng = np.random.default_rng(s)
                if una_por_vend:
                    perm = rng.permutation(tr_all)
                    _, first = np.unique(g[perm], return_index=True)   # una fila al azar por vendedora
                    idx_d = rng.choice(perm[first], n, replace=False)
                else:
                    idx_d = rng.choice(tr_all, n, replace=False)
                a, lf = auc_de(idx_d)
                aucs.append(a)
                lifts.append(lf)
            rows.append({"K_meses": k, "variante": "downsized (mismo n, todos los meses"
                         + (", 1 fila/vendedora)" if una_por_vend else ")"),
                         "n_train": n, "n_vend": int(np.mean([len(np.unique(g[idx_d]))])),
                         "AUC": np.mean(aucs), "AUC_std": np.std(aucs), "lift10": np.mean(lifts)})
        print(pd.DataFrame(rows[-2:] if K is not None else rows[-1:]).round(4).to_string(index=False),
              f"\n  K={k} listo en {time.time() - t0:.0f}s\n", flush=True)
    write_report(pd.DataFrame(rows), len(te), y[te].mean())


def write_report(res, n_test, prev_test):
    REPORTS.mkdir(exist_ok=True)
    res.to_csv(REPORTS / "ablacion_slices.csv", index=False)
    ms = res[res["variante"].str.startswith("multi")].set_index("K_meses")
    ds = res[res["variante"].str.startswith("down")].set_index("K_meses")
    base = ms["AUC"].iloc[0]
    tabla = pd.DataFrame({
        "n_train": ms["n_train"], "AUC multi-slicing": ms["AUC"].round(4),
        "Δ vs K=1": (ms["AUC"] - base).round(4), "lift10": ms["lift10"].round(2),
        "AUC downsized": ds["AUC"].round(4), "± seeds": ds["AUC_std"].round(4),
        "Δ downsized vs K=1": (ds["AUC"] - base).round(4)})
    md = [
        "# Ablación de time slices — XGBoost tuneado, test out-of-period fijo",
        f"\n> Generado por `05_modelling/07_ablacion_slices.py` el {time.strftime('%Y-%m-%d %H:%M')}.",
        f"> Test = últimos 4 meses etiquetados ({n_test} filas, prevalencia {prev_test:.3f}), gap 6 meses. "
        "Train = últimos K meses de observación. Downsized = mismo n de filas muestreadas de todos los "
        "meses (1 fila por vendedora mientras n ≤ nº de vendedoras), promedio de 5 semillas.",
        "\n## Resultados\n", tabla.to_markdown(),
        "\n## Lectura\n",
        f"- De K=1 a todo el histórico el AUC cambia {ms['AUC'].iloc[-1] - base:+.4f} "
        f"(lift decil {ms['lift10'].iloc[0]:.2f} → {ms['lift10'].iloc[-1]:.2f}).",
        f"- Mejor K: {ms['AUC'].idxmax()} meses (AUC {ms['AUC'].max():.4f}).",
        "- Si la columna *downsized* sube con K a igual n, la ganancia es diversidad temporal, "
        "no volumen (Gattermann-Itschert & Thonemann 2021, Fig. 9).",
        "\n## Tabla completa\n", res.round(4).to_markdown(index=False),
    ]
    (REPORTS / "ablacion_slices.md").write_text("\n".join(md) + "\n")
    print(f"\nreporte -> {REPORTS / 'ablacion_slices.md'}")


if __name__ == "__main__":
    main()
