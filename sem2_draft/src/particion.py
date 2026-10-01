"""Partición temporal única del proyecto (metodología §4.3).

  • OOT: últimos `oot_months` meses observados. Se evalúa UNA sola vez con el
    modelo final; no interviene en ninguna decisión.
  • Desarrollo: todo lo anterior al OOT menos una brecha de `gap` meses
    (= horizonte de churn), para que ninguna etiqueta de desarrollo se solape
    con el OOT.
  • Validación temporal: `n_folds` bloques consecutivos de `months` meses al
    final del desarrollo. Cada bloque se entrena con toda la historia previa
    a él, menos la misma brecha de `gap` meses (ventana expansiva).

Todo se expresa en `mes_rank` (entero consecutivo del mes de observación).
"""

import numpy as np

GAP = 6          # horizonte de la etiqueta
OOT_MONTHS = 4
N_FOLDS = 4
FOLD_MONTHS = 4


def oot_split(rank, gap=GAP, oot_months=OOT_MONTHS):
    """Máscaras booleanas (desarrollo, oot) sobre un array/Series de mes_rank."""
    rank = np.asarray(rank)
    oot_start = rank.max() - oot_months + 1
    return rank <= oot_start - 1 - gap, rank >= oot_start


def temporal_folds(rank, n_folds=N_FOLDS, months=FOLD_MONTHS, gap=GAP):
    """Lista de (idx_train, idx_val) sobre el pool de desarrollo, en orden cronológico."""
    rank = np.asarray(rank)
    last = int(rank.max())
    folds = []
    for i in range(n_folds):
        end = last - months * (n_folds - 1 - i)
        start = end - months + 1
        tr = np.flatnonzero(rank <= start - gap - 1)
        va = np.flatnonzero((rank >= start) & (rank <= end))
        assert rank[tr].max() + gap < start, "solape de etiquetas entre train y validación"
        folds.append((tr, va))
    return folds


def describe(df):
    """Tabla resumen de la partición con fechas reales (para el notebook y el JSON)."""
    import pandas as pd

    dev, oot = oot_split(df.mes_rank)
    rows = []

    def add(name, tr, va):
        rows.append({
            "bloque": name,
            "train_desde": tr.mes_obs.min().strftime("%Y-%m"),
            "train_hasta": tr.mes_obs.max().strftime("%Y-%m"),
            "val_desde": va.mes_obs.min().strftime("%Y-%m"),
            "val_hasta": va.mes_obs.max().strftime("%Y-%m"),
            "n_train": len(tr), "n_val": len(va),
            "churn_val": round(va.churn.mean(), 4),
        })

    d = df[dev].reset_index(drop=True)
    for i, (tr, va) in enumerate(temporal_folds(d.mes_rank), 1):
        add(f"val_{i}", d.iloc[tr], d.iloc[va])
    add("oot", df[dev], df[oot])
    return pd.DataFrame(rows)
