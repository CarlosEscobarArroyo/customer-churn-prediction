"""Carga y preprocesamiento del dataset de churn (metodología §4.1).

El preprocesamiento es sin estado (imputación a cero + ratios), así que puede
aplicarse a todo el dataset antes de partir. La estandarización se ajusta
dentro de cada entrenamiento, en el pipeline del modelo.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # sem2/
DATA = ROOT / "data"
RAW = DATA / "churn_dataset.csv"
PROCESSED = DATA / "churn_dataset_processed.csv"

ID = ["id_vendedor", "mes_obs", "mes_rank"]
TARGET = "churn"

# NULL en SQL cuando no hay historia suficiente (SAFE_DIVIDE / LAG) -> 0
ZERO = [
    "monto_cv_u12", "monto_ult_vs_media", "tend_monto_u3_vs_prev3",
    "tend_nped_u3_vs_prev3", "monto_por_prod_acum",
    *[f"d_{m}_m{n}" for m in ("monto", "nped") for n in (1, 3, 6, 9, 12)],
]


def bq_client():
    """Cliente BigQuery; usa el token de la cuenta gmail si ADC no tiene acceso."""
    import os
    import subprocess

    from google.cloud import bigquery

    account = os.environ.get("BQ_ACCOUNT", "carlos.escobar.arroyo@gmail.com")
    token = subprocess.run(
        ["gcloud", "auth", "print-access-token", f"--account={account}"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    from google.oauth2.credentials import Credentials

    return bigquery.Client(project="glamour-peru-dw", credentials=Credentials(token))


def _ratio(a, b):
    return (a / b.replace(0, np.nan)).fillna(0)


def preprocess(df):
    """Imputa a cero y agrega las 6 variables derivadas (36 SQL + 6 = 42)."""
    d = df.copy()
    d[ZERO] = d[ZERO].fillna(0)
    d["ticket_prom_u12"] = _ratio(d.monto_u12, d.n_ped_u12)
    d["ticket_prom_u3"] = _ratio(d.monto_u3, d.n_ped_u3)
    d["intensidad_u3"] = _ratio(d.n_ped_u3, d.meses_activos_u3)
    d["basket_size_u12"] = _ratio(d.n_prod_u12, d.n_ped_u12)
    d["recencia_norm"] = d.meses_desde_compra_previa * d.meses_activos_u12 / 12
    d["tasa_act_reciente_vs_hist"] = _ratio(d.meses_activos_u3 / 3, d.meses_activos_u12 / 12)
    return d


def load(path=PROCESSED):
    df = pd.read_csv(path, parse_dates=["mes_obs"])
    assert not df.duplicated(ID[:2]).any(), "claves (id_vendedor, mes_obs) duplicadas"
    return df


def features(df):
    """Nombres de las variables predictoras (todo lo que no es id ni target)."""
    return [c for c in df.columns if c not in ID + [TARGET]]
