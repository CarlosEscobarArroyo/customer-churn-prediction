"""Corre qry_churn.sql en BigQuery y deja dos CSV en sem2/data/ (no versionados).

  churn_dataset.csv            salida cruda de la query (36 variables)
  churn_dataset_processed.csv  imputación a cero + 6 derivadas (42 variables)

Uso:  uv run python sem2/00_datos/build_dataset.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import datos  # noqa: E402

sql = (Path(__file__).parent / "qry_churn.sql").read_text()
df = datos.bq_client().query(sql).to_dataframe()
datos.DATA.mkdir(exist_ok=True)
df.to_csv(datos.RAW, index=False)

proc = datos.preprocess(df)
assert proc[datos.features(proc)].notna().all().all(), "quedaron nulos tras preprocesar"
assert len(datos.features(proc)) == 42, len(datos.features(proc))
proc.to_csv(datos.PROCESSED, index=False)

print(f"{len(df):,} filas | {df.id_vendedor.nunique():,} vendedoras | "
      f"{df.mes_obs.min()} → {df.mes_obs.max()} | churn {df.churn.mean():.4f}")
print(f"-> {datos.RAW.name}, {datos.PROCESSED.name} ({proc.shape[1]} cols)")
