"""Corre qry_churn.sql en BigQuery y guarda el resultado en CSV local.

Destino: clean/data/processed/churn_dataset.csv  (no versionado; .gitignore)
Idempotente: re-correr regenera el CSV desde la query autoritativa.
"""
from pathlib import Path

from google.cloud import bigquery

BASE = Path(__file__).resolve().parents[1]
SQL = (BASE / "00_dataset_construction" / "qry_churn.sql").read_text()
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

df = bigquery.Client(project="glamour-peru-dw").query(SQL).to_dataframe()
path = OUT / "churn_dataset.csv"
df.to_csv(path, index=False)
print(f"{len(df):,} filas x {df.shape[1]} cols -> {path}")
print(f"churn rate: {df['churn'].mean():.4f} | vendedoras: {df['id_vendedor'].nunique():,}")
