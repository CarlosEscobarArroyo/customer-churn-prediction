"""Extrae las 8 tablas crudas de BigQuery a CSV.

Destino: clean/data/raw/fact_tables_&_dimensions/
Idempotente: re-correr sobrescribe los CSV. Los CSV no se versionan
(.gitignore excluye *.csv). Fuente: glamour-peru-dw.glamour_dw.
"""
from pathlib import Path

from google.cloud import bigquery

PROJECT = "glamour-peru-dw"
DATASET = "glamour_dw"
TABLES = [
    "dim_campana", "dim_coordinadora", "dim_fecha", "dim_producto",
    "dim_ubicacion", "dim_vendedor", "fact_pedidos", "fact_pedidos_detalle",
]
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "fact_tables_&_dimensions"
OUT.mkdir(parents=True, exist_ok=True)

client = bigquery.Client(project=PROJECT)
for t in TABLES:
    df = client.query(f"SELECT * FROM `{PROJECT}.{DATASET}.{t}`").to_dataframe()
    path = OUT / f"{t}.csv"
    df.to_csv(path, index=False)
    print(f"{t:24s} {len(df):>7,} filas  {df.shape[1]:>2} cols -> {path.name}")
