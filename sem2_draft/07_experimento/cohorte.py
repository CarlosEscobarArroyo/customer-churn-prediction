"""Conforma la cohorte mensual del experimento de campo (metodología §4.5).

  1. Re-entrena el modelo final (`04_modelado/modelo_final.json`) con todas las filas etiquetadas.
  2. Extrae del DW las vendedoras elegibles del último mes completo (compra en el mes y >= 1 compra
     previa; misma query de `00_datos`, sin exigir 6 meses de futuro) y excluye las de cohortes previas.
  3. Calcula su probabilidad de abandono y las asigna al azar 50/50 a tratamiento y control.
  4. Selecciona el 30 %: en tratamiento, las de mayor probabilidad; en control, al azar.

Salidas: `data/experimento/cohorte_<AAAA-MM>.csv` (con id_vendedor; no versionado),
`models/modelo_experimento_<AAAA-MM>.joblib` y `reports/experimento.md` (solo conteos).

Uso:  uv run python sem2_draft/07_experimento/cohorte.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import datos, modelos  # noqa: E402

SEED = 42
PCT = 0.30
OUT = datos.DATA / "experimento"

# --- 1. modelo final con toda la historia etiquetada ---------------------------------------
spec = json.loads((datos.ROOT / "04_modelado" / "modelo_final.json").read_text())
train = datos.load()
modelo = modelos.desde_spec(spec).fit(train, train[datos.TARGET])

# --- 2. elegibles del último mes completo ---------------------------------------------------
FILTRO = "AND f.mes_rank <= g.max_rank - 6"
sql = (datos.ROOT / "00_datos" / "qry_churn.sql").read_text()
assert FILTRO in sql, "cambió el filtro de etiqueta en qry_churn.sql"
df = datos.bq_client().query(sql.replace(FILTRO, "AND f.mes_rank = g.max_rank")).to_dataframe()
df = datos.preprocess(df.drop(columns=datos.TARGET))  # la etiqueta aún no es observable
df["mes_obs"] = pd.to_datetime(df.mes_obs)
assert df.mes_obs.nunique() == 1 and not df.id_vendedor.duplicated().any()
mes = df.mes_obs.iloc[0].strftime("%Y-%m")
assert df.mes_rank.iloc[0] == train.mes_rank.max() + 6, "el dataset de entrenamiento y el DW no están al mismo corte"

destino = OUT / f"cohorte_{mes}.csv"
assert not destino.exists(), f"{destino.name} ya existe: la asignación de una cohorte no se regenera"
previas = [pd.read_csv(f).id_vendedor for f in OUT.glob("cohorte_*.csv")]
n_previas = 0
if previas:
    ya = df.id_vendedor.isin(pd.concat(previas))
    n_previas, df = int(ya.sum()), df[~ya]

# --- 3. puntaje y asignación aleatoria ------------------------------------------------------
df = df.sort_values("id_vendedor").reset_index(drop=True)
df["p_churn"] = modelo.predict_proba(df)[:, 1]
rng = np.random.default_rng(SEED + df.mes_rank.iloc[0])
orden = rng.permutation(len(df))
df["grupo"] = "control"
df.loc[orden[: (len(df) + 1) // 2], "grupo"] = "tratamiento"

# --- 4. selección del 30 % ------------------------------------------------------------------
df["seleccionada"] = False
t, c = df[df.grupo == "tratamiento"], df[df.grupo == "control"]
df.loc[t.p_churn.nlargest(round(PCT * len(t))).index, "seleccionada"] = True
df.loc[rng.choice(c.index, round(PCT * len(c)), replace=False), "seleccionada"] = True

assert df.groupby("grupo").seleccionada.mean().sub(PCT).abs().max() < 0.01
tt = df[df.grupo == "tratamiento"]
assert tt[tt.seleccionada].p_churn.min() >= tt[~tt.seleccionada].p_churn.max()

OUT.mkdir(exist_ok=True)
df[["id_vendedor", "mes_obs", "p_churn", "grupo", "seleccionada"]].to_csv(destino, index=False)
(datos.ROOT / "models").mkdir(exist_ok=True)
joblib.dump(modelo, datos.ROOT / "models" / f"modelo_experimento_{mes}.joblib")

# --- reporte (sin id_vendedor) --------------------------------------------------------------
res = df.groupby(["grupo", "seleccionada"]).p_churn.agg(n="size", p_media="mean", p_min="min", p_max="max")
lineas = [
    f"## Cohorte {mes}",
    "",
    f"- Entrenamiento: {len(train):,} filas, observaciones {train.mes_obs.min():%Y-%m} → {train.mes_obs.max():%Y-%m} "
    f"(etiquetas con pedidos hasta {mes}); modelo `{spec['final']['config']}`.",
    f"- Elegibles: **{len(df)}** (excluidas por cohortes previas: {n_previas}). "
    f"Tratamiento **{len(t)}**, control **{len(c)}**; semilla {SEED} + mes_rank.",
    f"- Seleccionadas ({PCT:.0%}): tratamiento **{int(tt.seleccionada.sum())}** (mayor probabilidad), "
    f"control **{int(df[df.grupo == 'control'].seleccionada.sum())}** (al azar).",
    f"- Probabilidad media de abandono: {df.p_churn.mean():.3f} (elegibles).",
    "",
    "| grupo | seleccionada | n | p media | p mín | p máx |",
    "|---|---|---:|---:|---:|---:|",
    *[f"| {g} | {'sí' if s else 'no'} | {r.n:.0f} | {r.p_media:.3f} | {r.p_min:.3f} | {r.p_max:.3f} |"
      for (g, s), r in res.iterrows()],
    "",
]
rep = datos.ROOT / "reports" / "experimento.md"
cab = "# Experimento de campo — cohortes (§4.5)\n\nGenerado por `07_experimento/cohorte.py`; una sección por cohorte.\n\n"
rep.write_text((rep.read_text() if rep.exists() else cab) + "\n".join(lineas) + "\n")
print("\n".join(lineas))
