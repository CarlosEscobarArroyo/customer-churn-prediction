"""Construye notebooks/06_horizonte_v4.ipynb desde cero.

Replica el Bloque 2 de NB 01 (validación de la ventana k de churn) pero en
escala mensual, contra la misma blacklist de campañas que `qry_churn_v4.sql`.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "06_horizonte_v4.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS: list[dict] = []

CELLS.append(md("""\
# Validación del horizonte `HORIZON_CHURN` en escala mensual (v4)

Replica el Bloque 2 del NB 01 (`elección de la ventana k`) en la granularidad
mensual de v4, para responder:

> **¿`HORIZON_CHURN = 4` meses es la mejor ventana, o conviene otro valor?**

Decisión heredada de `qry_churn_v4.sql` (header):
> "4 meses sin compra es señal robusta de pérdida; 6 meses sería tarde para
> retención. La elección está abierta a revisión."

Análisis (espejo del NB 01 pero en meses):

1. Panel de eventos `(id_vendedor, mes)` con la misma limpieza que v4
   (blacklist 20102, 20201, 23105).
2. Distribución de gaps entre compras consecutivas (CDF y modas).
3. Curva silent vs k=1..12 sobre sub-muestra fully-observable + hazard.
4. Sensibilidad: para k ∈ [2..8], churn rate, n y Cohen's kappa vs k=4.
5. Codo automático.
6. Cohortes pre-2025 vs post-2025 (régimen 1 campaña/mes vs 2-3) para
   detectar si el horizonte óptimo cambió.
7. Síntesis y recomendación.

Filtro de población: `cum_purchases >= 3`, consistente con
`compras_historicas >= 3` en v4.
"""))

CELLS.append(md("## 1. Setup y carga del panel de eventos\n"))

CELLS.append(code("""\
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import bigquery
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 220)

PROJECT, DATASET = 'glamour-peru-dw', 'glamour_dw'
BLACKLIST = (20102, 20201, 23105)

bq = bigquery.Client(project=PROJECT)
"""))

CELLS.append(code("""\
QUERY_EVENTS = f\"\"\"
WITH
  rango_fechas AS (
    SELECT
      DATE_TRUNC(MIN(d.date), MONTH) AS primer_mes,
      DATE_TRUNC(MAX(d.date), MONTH) AS ultimo_mes
    FROM `{PROJECT}.{DATASET}.fact_pedidos` p
    JOIN `{PROJECT}.{DATASET}.dim_fecha` d ON p.id_fecha = d.id_fecha
  ),
  meses_ordenados AS (
    SELECT
      mes,
      ROW_NUMBER() OVER (ORDER BY mes) AS mes_rank
    FROM rango_fechas,
    UNNEST(GENERATE_DATE_ARRAY(primer_mes, ultimo_mes, INTERVAL 1 MONTH)) AS mes
  ),
  pedidos_limpios AS (
    SELECT
      p.id_vendedor,
      DATE_TRUNC(d.date, MONTH) AS mes
    FROM `{PROJECT}.{DATASET}.fact_pedidos` p
    JOIN `{PROJECT}.{DATASET}.dim_fecha` d ON p.id_fecha = d.id_fecha
    WHERE p.id_campana NOT IN {BLACKLIST}
  )
SELECT DISTINCT
  pl.id_vendedor,
  pl.mes,
  m.mes_rank
FROM pedidos_limpios pl
JOIN meses_ordenados m ON pl.mes = m.mes
ORDER BY pl.id_vendedor, m.mes_rank
\"\"\"

events = bq.query(QUERY_EVENTS).to_dataframe()
events['mes'] = pd.to_datetime(events['mes'])

events['cum_purchases'] = events.groupby('id_vendedor').cumcount() + 1
events['next_rank']     = events.groupby('id_vendedor')['mes_rank'].shift(-1)
events['gap_to_next']   = events['next_rank'] - events['mes_rank']

LAST_RANK = int(events['mes_rank'].max())
events['observable_horizon'] = LAST_RANK - events['mes_rank']

print(f'Eventos (vendedora, mes)   : {len(events):,}')
print(f'Vendedoras únicas          : {events[\"id_vendedor\"].nunique():,}')
print(f'Rango de meses             : {events[\"mes\"].min().date()} → {events[\"mes\"].max().date()}')
print(f'Último mes_rank observado  : {LAST_RANK}')
print(f'Compras promedio por vend. : {len(events) / events[\"id_vendedor\"].nunique():.2f}')
events.head(10)
"""))

CELLS.append(md("""\
**Insight — Panel de eventos mensuales**

Equivalente al panel de NB 01 pero a granularidad mensual: una fila por mes
calendario en el que la vendedora tuvo al menos un pedido (excluyendo la
blacklist de campañas). `gap_to_next` mide la distancia (en meses) hasta la
siguiente compra; es el insumo central para definir churn. Las observaciones
con `observable_horizon < k` no son evaluables para esa k.
"""))

CELLS.append(md("## 2. Distribución de gaps cerrados\n"))

CELLS.append(code("""\
events_h3 = events[events['cum_purchases'] >= 3]
gaps = events_h3['gap_to_next'].dropna().astype(int)

print(f'Gaps cerrados (cum_purchases >= 3): {len(gaps):,}')
print('\\nPercentiles del gap (meses):')
print(gaps.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

gap_freq = gaps.value_counts().sort_index()
gap_pct  = (gap_freq / len(gaps) * 100).round(2)
gap_cdf  = (gap_freq.cumsum() / len(gaps) * 100).round(2)
tabla_gaps = pd.DataFrame({'freq': gap_freq, 'pct': gap_pct, 'cdf': gap_cdf}).head(15)
print('\\nDistribución (k=1..15):')
print(tabla_gaps)

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].bar(gap_freq.head(15).index, gap_freq.head(15).values, color='steelblue')
ax[0].set_xlabel('gap (meses entre compras consecutivas)')
ax[0].set_ylabel('frecuencia')
ax[0].set_title('Distribución de gaps cerrados (k=1..15)')

ax[1].plot(gap_cdf.head(15).index, gap_cdf.head(15).values, marker='o', color='crimson')
ax[1].axhline(80, ls='--', color='gray', lw=0.7)
ax[1].axhline(95, ls='--', color='gray', lw=0.7)
ax[1].axvline(4,  ls='--', color='red',  lw=0.7, label='k=4 (v4)')
ax[1].set_xlabel('k')
ax[1].set_ylabel('% de gaps ≤ k')
ax[1].set_title('CDF acumulada')
ax[1].legend()
plt.tight_layout()
plt.show()

GAP_STATS = {
    'n_gaps':       len(gaps),
    'p50':          float(gaps.quantile(0.5)),
    'p75':          float(gaps.quantile(0.75)),
    'p90':          float(gaps.quantile(0.9)),
    'pct_ge_3':     float((gaps >= 3).mean()),
    'pct_ge_4':     float((gaps >= 4).mean()),
    'pct_ge_5':     float((gaps >= 5).mean()),
    'pct_ge_6':     float((gaps >= 6).mean()),
}
GAP_STATS
"""))

CELLS.append(md("""\
**Insight — Distribución de gaps**

- Mediana, P75, P90 del gap mensual.
- `% gaps ≥ k` es la *cota inferior* de la tasa de falsos churn si se elige
  ese k (gaps que terminan en compra pero serían marcados como churn).
- Un k corto (3 meses) etiqueta como churn una porción significativa de
  vendedoras esporádicas-pero-vuelven; un k largo (≥6) reduce ese ruido pero
  retrasa la señal de retención.
"""))

CELLS.append(md("## 3. Curva silent vs k + hazard\n"))

CELLS.append(code("""\
MAX_K = 12
subset = events_h3[events_h3['observable_horizon'] >= MAX_K].copy()
print(f'Sub-muestra fully-observable (>= {MAX_K} meses futuros): {len(subset):,} '
      f'({len(subset)/len(events_h3):.1%} de events_h3)')

rows = []
for k in range(1, MAX_K + 1):
    silent = (subset['gap_to_next'].isna() | (subset['gap_to_next'] > k)).mean()
    rows.append({'k': k, 'pct_silent': silent})
horizon_curve = pd.DataFrame(rows)

horizon_curve['hazard'] = (
    -horizon_curve['pct_silent'].diff().fillna(1 - horizon_curve['pct_silent'].iloc[0])
    / horizon_curve['pct_silent'].shift(1).fillna(1.0)
)
print(horizon_curve.round(4))

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].plot(horizon_curve['k'], horizon_curve['pct_silent'] * 100, marker='o', color='navy')
ax[0].axvline(4, ls='--', color='red', lw=0.7, label='k=4 (v4)')
ax[0].axvline(6, ls=':',  color='gray', lw=0.7, label='k=6 (v3)')
ax[0].set_xlabel('horizonte k (meses)')
ax[0].set_ylabel('% silent (proxy churn)')
ax[0].set_title('Tasa de churn vs horizonte k')
ax[0].legend()

ax[1].plot(horizon_curve['k'], horizon_curve['hazard'] * 100, marker='o', color='darkgreen')
ax[1].set_xlabel('k')
ax[1].set_ylabel('hazard de retorno en t+k (%)')
ax[1].set_title('Probabilidad de comprar EN k dado silencio previo')
plt.tight_layout()
plt.show()

HORIZON_CURVE = horizon_curve
"""))

CELLS.append(md("""\
**Insight — Curva silent y hazard**

- La curva `pct_silent` debe caer rápido en las primeras k y estabilizarse:
  donde aplane es donde la mayoría de las vendedoras ya volvió.
- El hazard mide la probabilidad condicional de retorno *exactamente* en t+k
  dado silencio previo. Picos al inicio = vendedoras "regulares"; cola larga
  = "esporádicas-pero-vuelven".
- Comparamos la posición de k=4 (v4) y k=6 (v3) sobre la curva.
"""))

CELLS.append(md("## 4. Sensibilidad y kappa vs k=4\n"))

CELLS.append(code("""\
K_VALUES = [2, 3, 4, 5, 6, 7, 8]
K_REF = 4  # horizonte vigente en v4

rows_per_k = {}
rows = []
for k in K_VALUES:
    elig = events_h3[events_h3['observable_horizon'] >= k].copy()
    elig['churn_k'] = (elig['gap_to_next'].isna() | (elig['gap_to_next'] > k)).astype(int)
    rows_per_k[k] = elig
    rows.append({
        'k': k,
        'n_obs':         len(elig),
        'n_vendedoras':  elig['id_vendedor'].nunique(),
        'churn_rate':    elig['churn_k'].mean(),
        'positives':     int(elig['churn_k'].sum()),
    })
sensitivity = pd.DataFrame(rows)
print('Sensibilidad por k:')
print(sensitivity.assign(churn_rate=lambda d: (d['churn_rate']*100).round(2)))

ref = rows_per_k[K_REF][['id_vendedor', 'mes_rank', 'churn_k']].rename(columns={'churn_k': f'churn_{K_REF}'})
rows_kappa = []
for k in K_VALUES:
    if k == K_REF:
        rows_kappa.append({'k': k, 'kappa_vs_ref': 1.0, 'agree_pct': 1.0, 'n_overlap': len(ref)})
        continue
    cmp = (
        rows_per_k[k][['id_vendedor', 'mes_rank', 'churn_k']]
        .rename(columns={'churn_k': f'churn_{k}'})
        .merge(ref, on=['id_vendedor', 'mes_rank'], how='inner')
    )
    if len(cmp) == 0:
        rows_kappa.append({'k': k, 'kappa_vs_ref': np.nan, 'agree_pct': np.nan, 'n_overlap': 0})
        continue
    rows_kappa.append({
        'k': k,
        'kappa_vs_ref': cohen_kappa_score(cmp[f'churn_{k}'], cmp[f'churn_{K_REF}']),
        'agree_pct':    (cmp[f'churn_{k}'] == cmp[f'churn_{K_REF}']).mean(),
        'n_overlap':    len(cmp),
    })
kappa = pd.DataFrame(rows_kappa)
print(f'\\nConcordancia con churn (k={K_REF}):')
print(kappa.round(4))

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].bar(sensitivity['k'].astype(str), sensitivity['churn_rate'] * 100, color='steelblue')
for i, (k_, r_, n_) in enumerate(zip(sensitivity['k'], sensitivity['churn_rate'], sensitivity['n_obs'])):
    ax[0].text(i, r_ * 100 + 0.5, f'{r_:.1%}\\n(n={n_:,})', ha='center', fontsize=8)
ax[0].axvline(K_VALUES.index(K_REF), ls='--', color='red', lw=0.7)
ax[0].set_xlabel('k (meses)')
ax[0].set_ylabel('tasa de churn (%)')
ax[0].set_title('Tasa de churn y tamaño por k')

ax[1].plot(kappa['k'], kappa['kappa_vs_ref'], marker='o')
ax[1].set_xlabel('k')
ax[1].set_ylabel(f'Cohen\\'s kappa vs k={K_REF}')
ax[1].set_title(f'Acuerdo del target k vs k={K_REF}')
ax[1].axhline(0.8, ls='--', color='gray', lw=0.7, label='acuerdo fuerte (0.8)')
ax[1].axvline(K_REF, ls='--', color='red', lw=0.7)
ax[1].legend()
plt.tight_layout()
plt.show()

SENSITIVITY = sensitivity
KAPPA = kappa
"""))

CELLS.append(md("""\
**Insight — Sensibilidad y kappa**

- `churn_rate` por k muestra cuánto cambia la prevalencia con la elección
  del horizonte.
- Cohen's kappa contra k=4 mide cuánto se reetiquetaría el dataset si
  cambiamos. κ ≥ 0.9 = casi indistinguible; κ < 0.8 = cambio material.
- El target es robusto si los k vecinos a la referencia tienen κ alto.
"""))

CELLS.append(md("## 5. Codo automático\n"))

CELLS.append(code("""\
rates = sensitivity.set_index('k')['churn_rate']
deltas = rates.diff(-1).abs()
print('Delta absoluta entre tasas consecutivas:')
print(deltas.round(4))

THRESHOLD_PP = 0.02
k_codo_candidates = deltas[deltas < THRESHOLD_PP].index.tolist()
k_codo = k_codo_candidates[0] if k_codo_candidates else int(rates.idxmin())
print(f'\\nk recomendado por codo (Δtasa < {THRESHOLD_PP*100:.0f}pp): {k_codo}')
print(f'k vigente en v4: 4')

K_CODO = k_codo
"""))

CELLS.append(md("## 6. Cohortes pre-2025 vs post-2025\n"))

CELLS.append(md("""\
La motivación de v4 es el cambio de régimen 2025 (de 1 campaña/mes a 2-3).
Verificamos si el horizonte óptimo cambia entre cohortes — si difiere, la
elección debe favorecer al régimen actual.
"""))

CELLS.append(code("""\
CUTOFF = pd.Timestamp('2025-01-01')

events_h3 = events_h3.assign(cohorte=np.where(events_h3['mes'] < CUTOFF, 'pre_2025', 'post_2025'))

cohorte_summary = (
    events_h3.groupby('cohorte')
    .agg(n_eventos=('mes_rank', 'size'),
         n_vendedoras=('id_vendedor', 'nunique'),
         mes_min=('mes', 'min'),
         mes_max=('mes', 'max'))
)
print(cohorte_summary)

rows = []
for cohorte, g in events_h3.groupby('cohorte'):
    sub = g[g['observable_horizon'] >= MAX_K]
    for k in range(1, MAX_K + 1):
        silent = (sub['gap_to_next'].isna() | (sub['gap_to_next'] > k)).mean()
        rows.append({'cohorte': cohorte, 'k': k, 'pct_silent': silent, 'n': len(sub)})
curve_cohort = pd.DataFrame(rows)
print('\\nCurva silent por cohorte:')
print(curve_cohort.pivot(index='k', columns='cohorte', values='pct_silent').round(4))

fig, ax = plt.subplots(figsize=(8, 5))
for cohorte, sub in curve_cohort.groupby('cohorte'):
    n = sub['n'].iloc[0]
    ax.plot(sub['k'], sub['pct_silent'] * 100, marker='o', label=f'{cohorte} (n={n:,})')
ax.axvline(4, ls='--', color='red', lw=0.7, label='k=4 (v4)')
ax.axvline(6, ls=':',  color='gray', lw=0.7, label='k=6 (v3)')
ax.set_xlabel('horizonte k (meses)')
ax.set_ylabel('% silent (proxy churn)')
ax.set_title('Curva silent por cohorte temporal')
ax.legend()
plt.tight_layout()
plt.show()

# Sensibilidad por cohorte
rows = []
for cohorte, g in events_h3.groupby('cohorte'):
    for k in K_VALUES:
        elig = g[g['observable_horizon'] >= k]
        churn_k = (elig['gap_to_next'].isna() | (elig['gap_to_next'] > k)).astype(int)
        rows.append({'cohorte': cohorte, 'k': k,
                     'n_obs': len(elig),
                     'churn_rate': churn_k.mean()})
sens_cohort = pd.DataFrame(rows)
print('\\nSensibilidad por cohorte:')
print(sens_cohort.pivot(index='k', columns='cohorte', values='churn_rate').round(4))

CURVE_COHORT = curve_cohort
SENS_COHORT  = sens_cohort
"""))

CELLS.append(md("""\
**Insight — Cohortes**

- Si la curva post_2025 cae más rápido que la pre_2025, las vendedoras del
  régimen actual vuelven antes → un k más corto sigue siendo apropiado.
- Si la post_2025 es más plana, aumenta la fracción de "silent legítimos" y
  un k corto inflaría falsos churn.
- La cohorte post_2025 tiene `observable_horizon` muy limitado (pocos meses
  desde 2025-01); leer la curva con cautela en k altos.
"""))

CELLS.append(md("## 7. Síntesis y recomendación\n"))

CELLS.append(code("""\
print('=' * 70)
print('VALIDACIÓN DEL HORIZONTE — v4 mensual')
print('=' * 70)
print(f'\\nDataset de eventos: {len(events):,} filas, '
      f'{events[\"id_vendedor\"].nunique():,} vendedoras')
print(f'Filtro población   : cum_purchases >= 3')
print(f'Blacklist campañas : {BLACKLIST}')
print(f'k vigente en v4    : 4 meses')
print(f'k recomendado codo : {K_CODO} meses (Δtasa < 2pp)')

print('\\n--- Stats de gaps ---')
for k_, v_ in GAP_STATS.items():
    print(f'  {k_:12s} = {v_:.4f}' if isinstance(v_, float) else f'  {k_:12s} = {v_}')

print('\\n--- Sensibilidad ---')
print(SENSITIVITY[['k', 'n_obs', 'churn_rate']].assign(
    churn_rate=lambda d: (d['churn_rate']*100).round(2).astype(str) + '%'
).to_string(index=False))

print('\\n--- Concordancia (kappa vs k=4) ---')
print(KAPPA.round(4).to_string(index=False))

print('\\n--- Curva silent (global) ---')
print(HORIZON_CURVE.round(4).to_string(index=False))

print('\\n--- Curva silent por cohorte ---')
print(CURVE_COHORT.pivot(index='k', columns='cohorte', values='pct_silent').round(4))

print('\\n' + '=' * 70)
print('Decisión queda escrita en la celda markdown siguiente.')
"""))

CELLS.append(md("""\
### Decisión

*Esta celda se completa después de inspeccionar los outputs.*

Bullets a llenar a partir de los resultados:

- **k recomendado por codo**: …
- **Tasa de churn @ k=4 vs alternativas**: …
- **Kappa vs k=4** para los k vecinos: …
- **Diferencia entre cohortes** (pre/post-2025): …
- **Decisión final**: mantener k=4 / migrar a k=…

Si se decide migrar, el cambio en `qry_churn_v4.sql` es puntual:
- Constante `HORIZON_CHURN` (header + sección 7).
- CTE `target`: ajustar el número de `LEAD(compro, n)` y la suma final.
- CTE `panel`: `m.mes_rank <= vv.ultima_compra_rank + HORIZON_CHURN`.
"""))


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "customer-churn-prediction (3.12.3)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    main()
