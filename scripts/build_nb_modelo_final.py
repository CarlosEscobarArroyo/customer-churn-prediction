"""Construye notebooks/clean/modelo_final.ipynb desde cero.

Versión más concisa de `modelo_final_v5.ipynb`: mantiene el marco teórico
mínimo necesario y va al grano. Agrega una sección nueva que justifica
empíricamente el horizonte de 6 meses (análisis de gaps + hazard de retorno).

Estructura:
  1. Marco teórico (compacto).
  2. Carga del dataset v5.
  3. ¿Por qué horizonte = 6 meses? (análisis de hazard, sección nueva).
  4. Preparación de features + pipeline.
  5. Modelo HGB tuneado.
  6. Validación 1: GroupKFold por vendedora.
  7. Validación 2: split temporal forward.
  8. Threshold y matriz de confusión.
  9. Importancia por permutación.
 10. Viabilidad de negocio (lift, puntos de operación, veredicto).
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "clean" / "modelo_final.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS: list[dict] = []

# ==========================================================================
# PORTADA
# ==========================================================================
CELLS.append(md("""\
# Modelo de predicción de *silent churn* — Glamour Perú

**Dataset**: v5 · **Granularidad**: (vendedora, mes calendario) · **Horizonte**: 6 meses
**Modelo**: `HistGradientBoostingClassifier` con `class_weight='balanced'` (hiperparámetros tuneados con Optuna)

Este notebook aplica la mejor configuración encontrada en la fase exploratoria
(notebooks 01–08 en `drafts/`) y la presenta como entregable autocontenido. El
camino que llevó hasta acá está documentado en `VERSIONES.md`; este documento
sólo aplica las decisiones y reporta resultados.

Cada bloque va precedido por una explicación corta del **qué** y el **por qué**
para que un lector sin formación profunda en ML pueda seguirlo.
"""))

# ==========================================================================
# 1. MARCO TEÓRICO COMPACTO
# ==========================================================================
CELLS.append(md("""\
# 1. Marco teórico

## 1.1 Silent churn

En venta directa **no hay baja formal**: una vendedora simplemente deja de
hacer pedidos. Eso se llama *silent churn*. La definición operativa que
usamos es:

> **Una vendedora churnea en el mes _M_ si no compra en ninguno de los
> 6 meses calendario siguientes (_M+1_ … _M+6_).**

El número 6 se justifica empíricamente en la Sección 3 — no es arbitrario.

## 1.2 Aprendizaje supervisado y panel longitudinal

- **Entrada (X)**: características de una vendedora en un mes dado (RFM:
  recencia, frecuencia, monetario; tendencias; diversidad de producto;
  contexto geográfico).
- **Salida (y)**: 0 = no churnea en los próximos 6 meses; 1 = churnea.
- **Granularidad**: una fila por **(vendedora, mes)** — un *panel longitudinal*.
  Una vendedora con 18 meses de historia activa aporta 18 filas.

El modelo devuelve **P(churn=1) ∈ [0, 1]**. La decisión binaria depende de
un **threshold** que se elige según el costo de cada tipo de error
(Sección 7).

## 1.3 Validación: dos protocolos complementarios

| Protocolo | Pregunta que responde | Cómo se hace |
|---|---|---|
| **GroupKFold por vendedora** (5 folds) | ¿Generaliza a vendedoras nuevas? | Cada vendedora cae entera en train o entera en test, nunca en los dos. |
| **Split temporal forward** | ¿Predice el futuro a partir del pasado? | Train = meses anteriores, test = últimos 6 meses, GAP de 7 meses entre ambos para evitar solapamiento del horizonte. |

Reportamos los dos: GroupKFold es la métrica honesta de generalización;
split forward simula producción.

## 1.4 Métricas

- **AUC ROC**: ¿con qué probabilidad el modelo le asigna mayor score a un
  churner que a un no-churner tomados al azar? Independiente del threshold.
- **PR-AUC**: lo relevante es el **lift sobre la prevalencia**. PR-AUC 0.50
  con prevalencia 0.275 → lift ≈ 1.83×.
- **Precision / Recall / F1**: una vez fijado un threshold, miden la calidad
  de la decisión 0/1.

## 1.5 Anti-leakage

Tres trampas resueltas: (1) variables del horizonte como features —
eliminadas desde v3; (2) misma vendedora en train y test — resuelto con
`GroupKFold` por `id_vendedor`; (3) solapamiento de horizonte en split
temporal — resuelto con GAP = 7 meses.
"""))

# ==========================================================================
# 2. CARGA DEL DATASET
# ==========================================================================
CELLS.append(md("""\
# 2. Carga del dataset v5

El dataset vive en BigQuery (`glamour-peru-dw.glamour_dw.training_churn_v5`)
y se construye con `data/qry_churn_v5.sql`. La query ya aplica los filtros
de población (`compras_historicas >= 3`, churn no nulo, mes con compra).
"""))

CELLS.append(code("""\
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import bigquery

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 220)
np.random.seed(42)

PROJECT, DATASET = 'glamour-peru-dw', 'glamour_dw'
TABLE = f'`{PROJECT}.{DATASET}.training_churn_v5`'
RANDOM_STATE = 42
N_SPLITS = 5
HORIZON_CHURN = 6
TEST_WINDOW = 6
GAP = HORIZON_CHURN + 1   # = 7

bq = bigquery.Client(project=PROJECT)
"""))

CELLS.append(code("""\
df = bq.query(f'SELECT * FROM {TABLE}').to_dataframe()

print(f'Filas              : {len(df):,}')
print(f'Vendedoras únicas  : {df["id_vendedor"].nunique():,}')
print(f'Meses cubiertos    : {df["mes_obs"].nunique()}')
print(f'Rango temporal     : {df["mes_obs"].min()} → {df["mes_obs"].max()}')
print(f'Tasa de churn      : {df["churn"].mean():.4f}  ({df["churn"].sum():,} positivos)')
"""))

# ==========================================================================
# 3. JUSTIFICACIÓN DEL HORIZONTE = 6 MESES (NUEVA)
# ==========================================================================
CELLS.append(md("""\
# 3. ¿Por qué 6 meses de horizonte?

El target depende críticamente de **cuántos meses de silencio consideramos
suficiente para llamar a alguien "churner"**. Si elegimos un k chico (3
meses) marcamos como churn a vendedoras esporádicas legítimas que vuelven.
Si elegimos un k grande (12 meses) la señal llega tarde para cualquier
acción de retención.

Esta sección reproduce el análisis de `notebooks/drafts/06_horizonte_v4.ipynb`
de forma compacta para mostrar **por qué el codo está en k = 6**.

## 3.1 Construcción del panel de eventos

Necesitamos los **gaps reales entre compras consecutivas**, sin el filtro
`churn IS NOT NULL` del dataset v5 (que ya elimina los meses de cola con
horizonte futuro insuficiente). Por eso vamos directo a `fact_pedidos`.
"""))

CELLS.append(code("""\
BLACKLIST = (20102, 20201, 23105)   # COVID, fechas invertidas, curso no-retail

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
    SELECT mes, ROW_NUMBER() OVER (ORDER BY mes) AS mes_rank
    FROM rango_fechas,
    UNNEST(GENERATE_DATE_ARRAY(primer_mes, ultimo_mes, INTERVAL 1 MONTH)) AS mes
  ),
  pedidos_limpios AS (
    SELECT p.id_vendedor, DATE_TRUNC(d.date, MONTH) AS mes
    FROM `{PROJECT}.{DATASET}.fact_pedidos` p
    JOIN `{PROJECT}.{DATASET}.dim_fecha` d ON p.id_fecha = d.id_fecha
    WHERE p.id_campana NOT IN {BLACKLIST}
  )
SELECT DISTINCT pl.id_vendedor, pl.mes, m.mes_rank
FROM pedidos_limpios pl
JOIN meses_ordenados m ON pl.mes = m.mes
ORDER BY pl.id_vendedor, m.mes_rank
\"\"\"

events = bq.query(QUERY_EVENTS).to_dataframe()
events['mes'] = pd.to_datetime(events['mes'])
events['cum_purchases']      = events.groupby('id_vendedor').cumcount() + 1
events['next_rank']          = events.groupby('id_vendedor')['mes_rank'].shift(-1)
events['gap_to_next']        = events['next_rank'] - events['mes_rank']
LAST_RANK = int(events['mes_rank'].max())
events['observable_horizon'] = LAST_RANK - events['mes_rank']

# Mismo filtro de población que el dataset v5
events_h3 = events[events['cum_purchases'] >= 3]
print(f'Eventos (vendedora, mes con compra): {len(events):,}')
print(f'Tras filtro cum_purchases >= 3     : {len(events_h3):,}')
print(f'Vendedoras                         : {events_h3["id_vendedor"].nunique():,}')
"""))

CELLS.append(md("""\
## 3.2 Curva *silent* y *hazard* de retorno

Calculamos dos curvas sobre la **sub-muestra fully-observable** (eventos
con al menos 12 meses de futuro disponibles, para que las dos curvas no
estén sesgadas por censura):

1. **Curva silent**: `pct_silent(k)` = fracción de eventos cuyo gap a la
   próxima compra es **mayor que k meses** (o nunca vuelve). Es la
   tasa de churn que se obtendría si el horizonte fuera k.
2. **Hazard de retorno en k**: probabilidad condicional de **comprar
   exactamente en t+k** dado que estuvo silenciosa hasta t+k-1. Cuanto
   más alto el hazard, más vendedoras "siguen vivas" — todavía hay
   recuperación genuina. Cuando el hazard cae a niveles bajos y se
   estabiliza, llamar churn a esa cola es razonable.
"""))

CELLS.append(code("""\
MAX_K = 12
fully_obs = events_h3[events_h3['observable_horizon'] >= MAX_K].copy()

rows = []
for k in range(1, MAX_K + 1):
    silent = (fully_obs['gap_to_next'].isna() | (fully_obs['gap_to_next'] > k)).mean()
    rows.append({'k': k, 'pct_silent': silent})
horizon_curve = pd.DataFrame(rows)

# hazard(k) = (silent(k-1) - silent(k)) / silent(k-1)
horizon_curve['hazard'] = (
    -horizon_curve['pct_silent'].diff().fillna(1 - horizon_curve['pct_silent'].iloc[0])
    / horizon_curve['pct_silent'].shift(1).fillna(1.0)
)
# Δ tasa = cuánto cae la tasa de churn al pasar de k-1 a k
horizon_curve['delta_silent_pp'] = -horizon_curve['pct_silent'].diff() * 100

print(f'Sub-muestra fully-observable (>= {MAX_K} meses futuros): {len(fully_obs):,}')
print()
print(horizon_curve.round(4).to_string(index=False))
"""))

CELLS.append(code("""\
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

ax[0].plot(horizon_curve['k'], horizon_curve['pct_silent'] * 100,
           marker='o', color='navy', lw=2)
ax[0].axvline(6, ls='--', color='red', lw=1.2, label='k = 6 (elegido)')
ax[0].set_xlabel('horizonte k (meses)')
ax[0].set_ylabel('% silent (proxy de tasa de churn)')
ax[0].set_title('Tasa de churn vs horizonte k')
ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].plot(horizon_curve['k'], horizon_curve['hazard'] * 100,
           marker='o', color='darkgreen', lw=2)
ax[1].axvline(6, ls='--', color='red', lw=1.2, label='k = 6 (elegido)')
ax[1].axhline(6, ls=':', color='gray', lw=0.8, label='hazard ≈ 6%')
ax[1].set_xlabel('k')
ax[1].set_ylabel('hazard de retorno en t+k (%)')
ax[1].set_title('Probabilidad de volver EN k dado silencio previo')
ax[1].legend(); ax[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()
"""))

CELLS.append(md("""\
## 3.3 Lectura del codo

Lo que hay que mirar son los dos signos clásicos de **convergencia**:

1. **La curva silent se aplana**: la caída marginal de la tasa de churn
   al sumar un mes de espera (`delta_silent_pp`) se vuelve chica. Pasar
   de k=5 a k=6 todavía recupera vendedoras esporádicas; pasar de k=6 a
   k=7 ya casi no cambia la tasa.
2. **El hazard se estabiliza por debajo del 6%**: a partir de k=6, una
   vendedora que estuvo callada los 6 meses anteriores tiene menos del
   6% de probabilidad de volver en el mes siguiente. Esa cola larga es
   ruido, no recuperación real.

**Conclusión**: k = 6 es el primer valor donde se cumplen ambas
condiciones simultáneamente. Es el punto donde **se atrapa la mayor
parte del churn real sin inflar falsos positivos** con vendedoras
esporádicas que vuelven al mes 4 o 5.

> **Nota histórica**: en el dataset v3 (granularidad por campaña) ya se
> había elegido k = 6 *campañas*. La migración a granularidad mensual
> (v4 → v5) reabrió la pregunta. El análisis acá replica el de
> `06_horizonte_v4.ipynb` y confirma que **6 meses** sigue siendo la
> elección correcta en el régimen mensual.

A partir de acá todo el resto del notebook trabaja con `HORIZON_CHURN = 6`.
"""))

# ==========================================================================
# 4. PREPARACIÓN DE FEATURES
# ==========================================================================
CELLS.append(md("""\
# 4. Preparación de features

Excluimos columnas que no deben entrar al modelo (identificadores,
trazabilidad temporal, features removidas por el ablation):

| Columna | Razón |
|---|---|
| `id_vendedor` | Identificador → memorización. |
| `mes_obs`, `mes_rank_obs` | Trazabilidad, no feature. |
| `fecha_ingreso` | Su señal está en `antiguedad_meses`. |
| `id_coordinadora`, `ccodrelacion` | IDs de alta cardinalidad. |
| `ccodubigeo`, `distrito` | Cardinalidad ~cientos → usamos `provincia`/`departamento`. |
| `mes_num`, `anio_mes_num` | Sacadas tras el ablation (ver `ablation_temporal_v5.ipynb`): en split forward el modelo es igual o mejor sin ellas, y `anio_mes_num` extrapolaría en producción. |
| `churn` | Target. |
"""))

CELLS.append(code("""\
EXCLUDE = {
    'id_vendedor', 'mes_obs', 'mes_rank_obs', 'fecha_ingreso',
    'id_coordinadora', 'ccodrelacion', 'ccodubigeo', 'distrito',
    'mes_num', 'anio_mes_num',
    'churn',
}
CATEGORICAL = ['sexo_vendedor', 'tipo_vendedor', 'departamento', 'provincia']

feature_cols = [c for c in df.columns if c not in EXCLUDE]
numeric_cols = [c for c in feature_cols if c not in CATEGORICAL]

for c in CATEGORICAL:
    df[c] = df[c].astype('string').fillna('NA')

X = df[feature_cols].copy()
y = df['churn'].astype(int).values
groups = df['id_vendedor'].values

print(f'Features totales : {len(feature_cols)}  (numéricas: {len(numeric_cols)}, categóricas: {len(CATEGORICAL)})')
print(f'Target           : no-churn {(y==0).sum():,} ({(y==0).mean()*100:.1f}%) · churn {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)')
"""))

CELLS.append(md("""\
**Pipeline de preprocesamiento**: imputación + one-hot. Se aplica idéntico
en train y test para no introducir distribuciones distintas en los dos
lados.
"""))

CELLS.append(code("""\
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), numeric_cols),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='NA')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), CATEGORICAL),
    ])

X_tx = build_preprocessor().fit_transform(X.head(100))
print(f'Pipeline OK: {X.head(100).shape} → {X_tx.shape} (one-hot expande las categóricas).')
"""))

# ==========================================================================
# 5. MODELO
# ==========================================================================
CELLS.append(md("""\
# 5. Modelo: HistGradientBoosting tuneado

## 5.1 Por qué este modelo

`HistGradientBoosting` es un *gradient boosting* sobre árboles que
discretiza features numéricas en histogramas de 256 bins (de ahí "Hist"):
muy rápido, acepta NaN nativamente, y viene en el core de scikit-learn.

En el barrido del NB 07 (sobre v5) los seis modelos comparados quedaron así:

| Modelo | AUC GroupKFold |
|---|---:|
| Dummy (azar) | 0.50 |
| Heurística "gap previo" | 0.66 |
| LogReg (balanced) | 0.74 |
| **HGB (balanced)** | **0.7465** |
| LightGBM / XGBoost | 0.74 |

HGB ganó por margen pequeño pero consistente y no requiere dependencias
nativas adicionales.

## 5.2 `class_weight='balanced'`

Con 27.5% de churners el modelo sin pesar tiende a quedarse conservador.
`balanced` pesa cada churner ~2.6× más durante el entrenamiento (fórmula:
`n / (2 * count(clase))`) → mejor recall a costa de algo de precision.

## 5.3 Hiperparámetros tuneados con Optuna

Los valores no son defaults — vienen de `tuning_optuna_v5.ipynb` (50
trials, sampler TPE). El `learning_rate` explica el 79% de la varianza
del AUC durante la búsqueda; el óptimo combina **lr bajo + muchos árboles
+ hojas grandes**, receta clásica de boosting estable. Ganancia respecto
del default: +1.2 pp AUC GroupKFold, +0.8 pp AUC forward, std entre folds
−18%.
"""))

CELLS.append(code("""\
def make_hgb() -> Pipeline:
    return Pipeline([
        ('prep', build_preprocessor()),
        ('clf', HistGradientBoostingClassifier(
            class_weight='balanced',
            learning_rate=0.0175,    # tuneado (default 0.1)
            max_iter=750,            # tuneado (default 100)
            max_depth=4,             # tuneado (default None)
            max_leaf_nodes=22,       # tuneado (default 31)
            min_samples_leaf=100,    # tuneado (default 20)
            l2_regularization=0.0,
            random_state=RANDOM_STATE,
            early_stopping=False,
        )),
    ])
"""))

# ==========================================================================
# 6. VALIDACIÓN GROUPKFOLD
# ==========================================================================
CELLS.append(md("""\
# 6. Validación 1: `GroupKFold` por vendedora

Dividimos las **vendedoras** (no las filas) en 5 grupos. En cada iteración
entrenamos sobre 4 grupos y evaluamos sobre el restante. Ninguna vendedora
está al mismo tiempo en train y test → la métrica mide capacidad de
**generalizar a vendedoras nunca vistas**.

Tras los 5 folds se concatenan las predicciones (OOF = *out-of-fold*) y
se reportan dos cosas: AUC promedio entre folds (con std) y AUC sobre
las predicciones OOF concatenadas.
"""))

CELLS.append(code("""\
cv = GroupKFold(n_splits=N_SPLITS)

# Sanity check de no-leakage de vendedora
for fold_i, (tr, te) in enumerate(cv.split(X, y, groups), 1):
    assert not (set(groups[tr]) & set(groups[te])), f'fold {fold_i}: leakage'
print(f'OK — {N_SPLITS} folds sin solapamiento de vendedoras.')

oof_proba = np.zeros(len(y), dtype=float)
fold_aucs, fold_aps = [], []

for fold_i, (tr, te) in enumerate(cv.split(X, y, groups), 1):
    model = make_hgb()
    model.fit(X.iloc[tr], y[tr])
    proba_te = model.predict_proba(X.iloc[te])[:, 1]
    oof_proba[te] = proba_te
    auc = roc_auc_score(y[te], proba_te)
    ap  = average_precision_score(y[te], proba_te)
    fold_aucs.append(auc); fold_aps.append(ap)
    print(f'  Fold {fold_i}:  AUC = {auc:.4f}   PR-AUC = {ap:.4f}')

auc_mean, auc_std = np.mean(fold_aucs), np.std(fold_aucs)
ap_mean,  ap_std  = np.mean(fold_aps),  np.std(fold_aps)
auc_oof = roc_auc_score(y, oof_proba)
ap_oof  = average_precision_score(y, oof_proba)
prev    = y.mean()

print()
print(f'AUC fold (mean ± std) : {auc_mean:.4f} ± {auc_std:.4f}')
print(f'AUC OOF concatenado   : {auc_oof:.4f}')
print(f'PR-AUC OOF            : {ap_oof:.4f}   (prevalencia {prev:.4f}, lift {ap_oof/prev:.2f}×)')
"""))

CELLS.append(md("""\
**Lectura**: AUC ≈ 0.74-0.75 es coherente con la literatura para silent
churn (telcos llegan a 0.85+ porque el churn ahí es explícito). Std entre
folds < 0.01 → modelo estable. Lift PR-AUC ≈ 1.85× sobre prevalencia.
"""))

CELLS.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

fpr, tpr, _ = roc_curve(y, oof_proba)
axes[0].plot(fpr, tpr, lw=2, label=f'HGB (AUC = {auc_oof:.4f})')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Azar')
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR (recall)')
axes[0].set_title('ROC — OOF GroupKFold')
axes[0].legend(); axes[0].grid(alpha=0.3)

p, r, _ = precision_recall_curve(y, oof_proba)
axes[1].plot(r, p, lw=2, label=f'HGB (PR-AUC = {ap_oof:.4f})')
axes[1].axhline(prev, ls='--', color='k', alpha=0.4, label=f'Prevalencia ({prev:.3f})')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall — OOF GroupKFold')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()
"""))

# ==========================================================================
# 7. VALIDACIÓN SPLIT FORWARD
# ==========================================================================
CELLS.append(md("""\
# 7. Validación 2: split temporal forward

`GroupKFold` mezcla épocas. Para simular el escenario real de producción
("entrenar con histórico, predecir mañana") usamos un split temporal:

- **Train** = todos los meses anteriores al corte.
- **Test** = últimos 6 meses.
- **GAP = 7 meses** entre el último mes de train y el primero de test:
  el target de la última fila de train mira los 6 meses siguientes; sin
  GAP, las features de test "saben" cosas que el target de train aún no
  terminaba de definir.

Una vendedora puede aparecer en train y en test (con observaciones de
meses distintos). Por eso esta métrica suele ser un poco más alta que la
de GroupKFold — no es sesgo, es otra pregunta: *"¿predigo bien el futuro
sabiendo que muchas vendedoras del test ya las vi antes?"*. Producción
se parece a este escenario.
"""))

CELLS.append(code("""\
last_rank  = int(df['mes_rank_obs'].max())
test_min   = last_rank - TEST_WINDOW + 1
train_max  = test_min - GAP

train_mask = df['mes_rank_obs'] <= train_max
test_mask  = df['mes_rank_obs'].between(test_min, last_rank)

df_train = df.loc[train_mask]
df_test  = df.loc[test_mask]

X_train = df_train[feature_cols]
y_train = df_train['churn'].astype(int).values
X_test  = df_test[feature_cols]
y_test  = df_test['churn'].astype(int).values

print(f'Train: mes_rank ≤ {train_max}  ·  {len(df_train):,} filas  ·  '
      f'{df_train["id_vendedor"].nunique():,} vendedoras  ·  '
      f'{df_train["mes_obs"].min()} → {df_train["mes_obs"].max()}')
print(f'GAP de {GAP} meses')
print(f'Test : mes_rank ∈ [{test_min}, {last_rank}]  ·  {len(df_test):,} filas  ·  '
      f'{df_test["id_vendedor"].nunique():,} vendedoras  ·  '
      f'{df_test["mes_obs"].min()} → {df_test["mes_obs"].max()}')
"""))

CELLS.append(code("""\
model_fwd = make_hgb()
model_fwd.fit(X_train, y_train)
proba_test = model_fwd.predict_proba(X_test)[:, 1]

auc_fwd  = roc_auc_score(y_test, proba_test)
ap_fwd   = average_precision_score(y_test, proba_test)
prev_fwd = y_test.mean()

# Estabilidad mes a mes
eval_df = df_test[['mes_rank_obs', 'mes_obs', 'churn']].copy()
eval_df['proba'] = proba_test
per_mes = []
for r, g in eval_df.groupby('mes_rank_obs'):
    yt = g['churn'].astype(int).values; pp = g['proba'].values
    n_pos = int(yt.sum())
    auc_v = roc_auc_score(yt, pp) if 0 < n_pos < len(yt) else np.nan
    per_mes.append({'mes': str(g['mes_obs'].iloc[0])[:7], 'n': len(yt),
                    'churn_rate': float(yt.mean()), 'AUC': auc_v})
per_mes = pd.DataFrame(per_mes)
auc_std_mes = per_mes['AUC'].std()

print(f'AUC bloque test     : {auc_fwd:.4f}')
print(f'PR-AUC bloque test  : {ap_fwd:.4f}   (prev {prev_fwd:.4f}, lift {ap_fwd/prev_fwd:.2f}×)')
print(f'AUC mes a mes       : media {per_mes["AUC"].mean():.4f}  ·  std {auc_std_mes:.4f}  ·  '
      f'min {per_mes["AUC"].min():.4f}  ·  max {per_mes["AUC"].max():.4f}')
print()
print(per_mes.round(4).to_string(index=False))
"""))

CELLS.append(code("""\
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(per_mes['mes'], per_mes['AUC'], marker='o', lw=2, label='AUC mensual')
ax.axhline(auc_fwd, ls='--', color='red', alpha=0.5, label=f'AUC bloque ({auc_fwd:.3f})')
ax.set_ylabel('AUC'); ax.set_xlabel('Mes')
ax.set_title('Estabilidad temporal del modelo dentro del bloque test')
ax.set_ylim(0.5, 1.0); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""))

# ==========================================================================
# 8. THRESHOLD Y MATRIZ DE CONFUSIÓN
# ==========================================================================
CELLS.append(md("""\
# 8. Threshold y matriz de confusión

AUC y PR-AUC no dependen del threshold. Para tomar decisiones operativas
(¿lanzo campaña?, ¿escalo a coordinadora?) hay que elegir un umbral. El
trade-off:

- **Threshold bajo (0.30)**: marcamos a muchos → alto recall, baja precision.
- **Threshold medio (0.50)**: equilibrio razonable.
- **Threshold alto (0.65)**: marcamos a pocos → alta precision, bajo recall.

No hay umbral universal: depende del costo relativo de FP vs FN. En
Glamour contactar es barato (mensaje WhatsApp/push) → conviene un
threshold que **maximice recall manteniendo un filtro real** sobre la
base.
"""))

CELLS.append(code("""\
def show_confusion(y_true, proba, threshold, label):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = tn + fp + fn + tp
    cm = pd.DataFrame([[tn, fp], [fn, tp]],
                      index=['real: no-churn', 'real: churn'],
                      columns=['pred: no-churn', 'pred: churn'])
    print(f'=== {label}  ·  threshold = {threshold}  ===')
    print(cm.to_string())
    print(f'  Accuracy {(tp+tn)/n:.4f}  ·  '
          f'Precision {tp/(tp+fp) if (tp+fp) else 0:.4f}  ·  '
          f'Recall {tp/(tp+fn) if (tp+fn) else 0:.4f}  ·  '
          f'F1 {f1_score(y_true, pred):.4f}')
    print()

show_confusion(y, oof_proba, 0.5, 'GroupKFold OOF')
show_confusion(y_test, proba_test, 0.5, 'Split forward bloque test')
"""))

CELLS.append(md("""\
**Barrido de threshold**: tabla y curvas para que negocio elija el punto
operativo según su tolerancia a falsos positivos.
"""))

CELLS.append(code("""\
def threshold_sweep(y_true, proba, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({'threshold': round(float(t), 2),
                     'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
                     'precision': prec, 'recall': rec, 'F1': f1})
    return pd.DataFrame(rows)

sweep_oof = threshold_sweep(y, oof_proba)
print('Barrido de threshold — OOF GroupKFold')
print(sweep_oof.round(4).to_string(index=False))

best_idx = sweep_oof['F1'].idxmax()
print(f"\\nThreshold que maximiza F1: {sweep_oof.loc[best_idx, 'threshold']}  "
      f"(F1 = {sweep_oof.loc[best_idx, 'F1']:.4f}, "
      f"precision = {sweep_oof.loc[best_idx, 'precision']:.4f}, "
      f"recall = {sweep_oof.loc[best_idx, 'recall']:.4f})")
"""))

CELLS.append(code("""\
fig, ax = plt.subplots(figsize=(10, 4.5))
for col, color in [('precision', 'tab:blue'), ('recall', 'tab:orange'), ('F1', 'tab:green')]:
    ax.plot(sweep_oof['threshold'], sweep_oof[col], marker='o', lw=2, label=col, color=color)
ax.axvline(0.5, ls='--', color='gray', alpha=0.5, label='t = 0.5 (default)')
ax.set_xlabel('threshold'); ax.set_ylabel('métrica')
ax.set_title('Sensibilidad de precision/recall/F1 al threshold (OOF)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""))

# ==========================================================================
# 9. INTERPRETABILIDAD
# ==========================================================================
CELLS.append(md("""\
# 9. Interpretabilidad: importancia por permutación

> **Idea**: medir cuánto cae el AUC si **mezclamos al azar** los valores
> de una feature, manteniendo todo lo demás igual. Si la métrica se
> derrumba, esa feature era importante; si no cambia, era irrelevante.

Es agnóstica al modelo y mide **impacto sobre la métrica** (no sólo "uso
interno"). La corremos sobre una muestra de 10K filas para que termine en
1–3 minutos.
"""))

CELLS.append(code("""\
model_full = make_hgb()
model_full.fit(X, y)

sample_size = min(10_000, len(X))
rng = np.random.RandomState(RANDOM_STATE)
sample_idx = rng.choice(len(X), size=sample_size, replace=False)

print(f'Calculando permutation_importance sobre {sample_size:,} filas...')
perm = permutation_importance(
    model_full, X.iloc[sample_idx], y[sample_idx],
    scoring='roc_auc', n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1,
)

imp = pd.DataFrame({
    'feature':         feature_cols,
    'importance_mean': perm.importances_mean,
    'importance_std':  perm.importances_std,
}).sort_values('importance_mean', ascending=False).reset_index(drop=True)

print('\\nTop 20 features por permutación (caída de AUC al permutar):')
print(imp.head(20).round(4).to_string(index=False))
"""))

CELLS.append(code("""\
top_n = 20
top_imp = imp.head(top_n).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 7.5))
ax.barh(top_imp['feature'], top_imp['importance_mean'],
        xerr=top_imp['importance_std'], color='steelblue', alpha=0.85)
ax.set_xlabel('Caída de AUC al permutar')
ax.set_title(f'Top {top_n} features — importancia por permutación (HGB v5)')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout(); plt.show()
"""))

CELLS.append(md("""\
**Lectura del ranking**:

1. **La exposición domina**: `compras_historicas`, `antiguedad_meses`. El
   modelo se apoya en *cuán establecida es la vendedora* más que en su
   patrón reciente.
2. **Recencia explícita ausente del top 20**: contraintuitivo respecto de
   la heurística RFM. Hipótesis: la señal de recencia ya está embebida en
   `ratio_pago_u3`, `monto_total_u3`, etc. (un proxy de "compró poco hace
   poco"). Otra: con el filtro `compras_historicas >= 3` la varianza de
   la recencia se achica.
3. **Features RFM por ventana sí están**: `ratio_pago_*`, `monto_total_*`,
   `ticket_u3_vs_u12` aportan, sólo con menos peso del esperado.
4. **Sin features de calendario**: `mes_num` y `anio_mes_num` se sacaron
   tras el ablation (`ablation_temporal_v5.ipynb`).
"""))

# ==========================================================================
# 10. VIABILIDAD DE NEGOCIO
# ==========================================================================
CELLS.append(md("""\
# 10. Viabilidad de negocio

Esta sección responde la pregunta operativa: **¿deberíamos desplegar
esto?** Lo justifica con tres lentes complementarios: lift por decil,
puntos de operación, y casos de uso concretos.
"""))

CELLS.append(md("""\
## 10.1 Lift por decil

Ordenamos a las vendedoras de mayor a menor probabilidad de churn según
el modelo y dividimos en 10 grupos del mismo tamaño. En cada decil
medimos la fracción real de churners.

> **Lift del decil k** = (tasa de churn en ese decil) / (tasa global).
> Lift > 1 en los deciles altos = el modelo prioriza correctamente.
"""))

CELLS.append(code("""\
def lift_table(y_true, proba, n_bins=10):
    order = np.argsort(-proba)
    y_sorted = y_true[order]
    chunks = np.array_split(y_sorted, n_bins)
    base_rate = y_true.mean()
    total_pos = int(y_true.sum())
    rows, cum_pos, cum_n = [], 0, 0
    for i, ch in enumerate(chunks, 1):
        n = len(ch); n_pos = int(ch.sum())
        cum_pos += n_pos; cum_n += n
        rows.append({
            'decil': i, 'n_filas': n, 'n_churn': n_pos,
            'churn_rate': n_pos / n,
            'lift': (n_pos / n) / base_rate,
            'recall_acumulado': cum_pos / total_pos,
            '%_pob_acumulado':  cum_n / len(y_true),
        })
    return pd.DataFrame(rows)

lift_oof = lift_table(y, oof_proba)
print(f'Lift por decil — OOF GroupKFold (tasa global = {y.mean():.4f})')
print(lift_oof.round(4).to_string(index=False))
"""))

CELLS.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].bar(lift_oof['decil'], lift_oof['lift'], color='steelblue', alpha=0.85)
axes[0].axhline(1.0, ls='--', color='red', alpha=0.6, label='Lift = 1 (azar)')
axes[0].set_xlabel('Decil (1 = top 10% más riesgoso)')
axes[0].set_ylabel('Lift sobre tasa global')
axes[0].set_title('Lift por decil')
axes[0].set_xticks(range(1, 11)); axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)

axes[1].plot([0] + list(lift_oof['%_pob_acumulado']),
             [0] + list(lift_oof['recall_acumulado']),
             marker='o', lw=2, label='Modelo')
axes[1].plot([0, 1], [0, 1], '--', color='red', alpha=0.6, label='Azar')
axes[1].set_xlabel('% de población contactada')
axes[1].set_ylabel('% de churners capturados')
axes[1].set_title('Curva de ganancia acumulada')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""))

CELLS.append(md("""\
## 10.2 Tres puntos de operación

Cómo se mueven recall y precision según el threshold:
"""))

CELLS.append(code("""\
def operating_point(y_true, proba, threshold, label):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    base_rate = y_true.mean()
    return {
        'punto': label, 'threshold': threshold,
        '%_población': (tp + fp) / len(y_true),
        'recall':      tp / (tp + fn) if (tp + fn) else 0,
        'precision':   tp / (tp + fp) if (tp + fp) else 0,
        'lift_precision': (tp / (tp + fp) / base_rate) if (tp + fp) else 0,
        'F1':          f1_score(y_true, pred),
    }

points = pd.DataFrame([
    operating_point(y, oof_proba, 0.30, 'Agresivo  (t=0.30)'),
    operating_point(y, oof_proba, 0.50, 'Balanceado (t=0.50) ★'),
    operating_point(y, oof_proba, 0.65, 'Conservador (t=0.65)'),
])
print('Puntos de operación — OOF GroupKFold (sobre toda la base)')
print(points.round(4).to_string(index=False))
print(f'\\n(prevalencia = {y.mean():.4f}; lift_precision = precision / prevalencia)')
"""))

CELLS.append(md("""\
**Interpretación**:

- **Agresivo (t=0.30)** — descartado. Atrapamos al ~93% de los churners,
  pero contactando al **~71% de la base** con precision 0.36 (lift 1.30×).
  En la práctica es "contactar a casi todos" — el modelo aporta poco vs
  no usar modelo.
- **Balanceado (t=0.50) ★** — recomendado. Recall **0.73** (3 de 4),
  precision **0.43** (lift 1.56×). Tocamos al **~47% de la base**: filtro
  real. Es el punto que **maximiza F1** en el barrido de §8.
- **Conservador (t=0.65)** — descartado. Precision sube a 0.52 pero
  perdemos a más de la mitad de los churners (recall 0.44). Desperdicia
  la ventaja del régimen barato de contacto.
"""))

CELLS.append(md("""\
## 10.3 Casos de uso concretos

**A) Score mensual para priorizar la campaña de retención.**
Cada inicio de mes scorear a todas las vendedoras activas; lanzar la
campaña sólo sobre el top 20–30%. Si hoy se contacta al azar al 30% de
la base se atrapa al 30% de los churners; con el modelo, al mismo 30%
se atrapa ~60% — la efectividad **se duplica** sin tocar más vendedoras.

**B) Alerta semanal a coordinadora.**
Lista corta (5–10) de vendedoras con mayor riesgo por coordinadora; ella
decide la acción. Prioriza el tiempo escaso de la coordinadora hacia
donde más impacto tiene.

**C) Segmentación del portfolio.**
El score continuo se usa para segmentar (bajo / medio / alto / crítico) y
diseñar acciones diferenciadas. Útil para reportería ejecutiva ("¿cómo
está la salud del portfolio este mes?").

**D) Lo que el modelo NO debe hacer.** No decidir bajas automáticas. No
recortar comisión preventivamente (sería confundir predicción con
causalidad). No reemplazar el criterio de la coordinadora — lo enfoca,
no lo sustituye.
"""))

CELLS.append(md("""\
## 10.4 Limitaciones operativas

**Pipeline de datos**: el SQL `qry_churn_v5.sql` debe correr mensualmente
(hoy es manual; producción requiere job programado). Latencia mínima:
~3 días tras el cierre del mes.

**Reentrenamiento**: mensual o trimestral. Es barato (~30 segundos) y
previene drift silencioso. Trigger automático ante alertas: si la tasa
observada en el último mes se desvía ±5pp de la del anterior, gatillar
revisión.

**Monitoreo**: loggear score por (vendedora, mes) y comparar contra el
outcome a +6 meses. Dashboard de AUC rolling. Alertar si el AUC cae bajo
0.70 o si el recall a t=0.50 cae bajo 0.65.

**Calibración**: el modelo no está calibrado — `P(churn) = 0.7` no
significa "70% de probabilidad real". Importa poco mientras se opere por
threshold fijo; importaría si en el futuro se usaran las probabilidades
crudas para segmentación fina (aplicar `CalibratedClassifierCV`).

**Gobernanza**: documentar limitaciones en un *model card* visible para
los usuarios finales; auditar trimestralmente que el modelo no
discrimine geográficamente más allá de lo razonable.
"""))

CELLS.append(md("""\
## 10.5 Veredicto

**Sí, el modelo es viable en producción.** En el punto operativo
recomendado (t = 0.50):

| Métrica | Valor | Lectura de negocio |
|---|---:|---|
| **AUC split forward** | **0.75** | Ranquea bien al churner típico vs el no-churner típico el 75% de las veces. |
| **Recall** | **~73%** | Atrapamos a 3 de cada 4 churners reales. |
| **Precision** | **~43%** | 1.56× la prevalencia base. Filtro real, no ruido. |

> Sin modelo, contactando al 47% de la base al azar, atrapamos al 47%
> de los churners. **Con modelo, contactando al mismo 47%, atrapamos
> al 73%**. Esos 26 puntos extra de recall — sobre una base contactada
> menor que la mitad — son el valor incremental que aporta el sistema.

**Condiciones para desplegar**:

1. Pipeline mínimo en producción: job mensual que corra el SQL, genere
   scores y empuje la lista priorizada al equipo de retención.
2. Dashboard de monitoreo (AUC rolling + recall mes a mes), con alerta
   si caen bajo umbrales.
3. Reentrenamiento mensual o trimestral.
4. Uso correcto: el modelo **enfoca** la campaña, no decide bajas, no
   recorta comisiones.

**Lo que falta para subir el techo** (no son requisitos para desplegar):

- Atributos historizados (SCD-2) en lugar de snapshots actuales.
- Variables de comportamiento ausentes (interacciones con la app,
  asistencia a eventos, devoluciones).
- Definición de churn como tiempo-hasta-evento (modelo de supervivencia)
  en lugar de binario.
"""))

# ==========================================================================
# 11. RESUMEN
# ==========================================================================
CELLS.append(md("""\
# 11. Resumen
"""))

CELLS.append(code("""\
print('=' * 70)
print('MODELO FINAL — silent churn Glamour Perú (dataset v5)')
print('=' * 70)
print()
print(f'Configuración')
print(f'  Granularidad         : (vendedora, mes)')
print(f'  Horizonte            : {HORIZON_CHURN} meses (justificado en §3)')
print(f'  Filtro de población  : compras_historicas >= 3')
print(f'  Modelo               : HistGradientBoosting (class_weight=balanced, tuneado)')
print()
print(f'GroupKFold por vendedora (5 folds)')
print(f'  AUC fold (mean ± std): {auc_mean:.4f} ± {auc_std:.4f}')
print(f'  AUC OOF concatenado  : {auc_oof:.4f}')
print(f'  PR-AUC OOF           : {ap_oof:.4f}   (lift {ap_oof/prev:.2f}×)')
print()
print(f'Split temporal forward (último bloque de {TEST_WINDOW} meses, GAP = {GAP})')
print(f'  AUC bloque test      : {auc_fwd:.4f}')
print(f'  PR-AUC bloque test   : {ap_fwd:.4f}   (lift {ap_fwd/prev_fwd:.2f}×)')
print(f'  Std AUC mes a mes    : {auc_std_mes:.4f}')
print()
print(f'Punto operativo recomendado: t = 0.50')
f1_default = f1_score(y, (oof_proba >= 0.5).astype(int))
print(f'  F1 a t=0.50          : {f1_default:.4f}  (OOF)')
print(f'  F1 óptimo            : {sweep_oof["F1"].max():.4f}  '
      f'a t = {sweep_oof.loc[sweep_oof["F1"].idxmax(), "threshold"]}')
"""))

CELLS.append(md("""\
---

*Notebook generado por `scripts/build_nb_modelo_final.py`. Para regenerar:*

```bash
uv run python scripts/build_nb_modelo_final.py
```

*No editar el `.ipynb` a mano — los cambios se pierden en la próxima
regeneración. Editá el script.*
"""))


# ==========================================================================
# MAIN
# ==========================================================================
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
