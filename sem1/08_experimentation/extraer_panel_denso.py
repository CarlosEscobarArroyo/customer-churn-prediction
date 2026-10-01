"""Extrae de BigQuery los datos auxiliares de 08_experimentation (una sola vez):

  * data/processed/panel_denso_mensual.csv — CTE `panel` de qry_churn.sql tal cual
    (vendedora × mes desde su primer mes con compra; meses activos e inactivos). Es la
    base de las secuencias (Opción A) y de las features de ritmo (Opción B).
  * data/processed/dim_vendedor.csv — atributos del star schema (tipo, ccodrelacion = líder,
    sexo, fechas, ubicación) para las segmentaciones de negocio (Opción D).

Reutiliza el SQL autoritativo cortándolo antes de la CTE `feat`, así el panel es idéntico al
que alimenta el dataset de churn. Usa el token de la cuenta personal (ver
05_modelling/experimentos/10_tabfm.py) sin tocar las ADC.
Uso: uv run python 08_experimentation/extraer_panel_denso.py
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_utils import DIMV, PANEL, SQL_PATH  # noqa: E402

ACCOUNT = "carlos.escobar.arroyo@gmail.com"
PROJECT = "glamour-peru-dw"


def bq_personal():
    from google.cloud import bigquery
    from google.oauth2.credentials import Credentials
    tok = subprocess.check_output(["gcloud", "auth", "print-access-token", f"--account={ACCOUNT}"], text=True).strip()
    return bigquery.Client(project=PROJECT, credentials=Credentials(tok))


def main():
    c = bq_personal()
    sql = SQL_PATH.read_text()
    head = sql.split("-- Features de ventana + etiqueta")[0].rstrip()
    assert head.endswith("),"), "el SQL cambió: no se encontró el cierre de la CTE panel"
    q = head[:-1] + "\nSELECT * FROM panel ORDER BY id_vendedor, mes_rank"
    panel = c.query(q).result().to_dataframe()
    panel.to_csv(PANEL, index=False)
    print(f"panel denso: {panel.shape} → {PANEL}")
    q2 = f"""SELECT dv.id_vendedor, dv.tipo_vendedor, dv.ccodrelacion, dv.csexpersona AS sexo,
      dv.fecha_ingreso, dv.fecha_nacimiento, du.departamento, du.provincia, du.distrito
    FROM `{PROJECT}.glamour_dw.dim_vendedor` dv
    LEFT JOIN `{PROJECT}.glamour_dw.dim_ubicacion` du ON du.ccodubigeo = dv.ccodubigeo"""
    dv = c.query(q2).result().to_dataframe()
    dv.to_csv(DIMV, index=False)
    print(f"dim_vendedor: {dv.shape} → {DIMV}")


if __name__ == "__main__":
    main()
