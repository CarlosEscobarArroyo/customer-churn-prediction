"""Construye notebooks/07_baselines_v5.ipynb desde cero.

Replica NB 05 sobre `training_churn_v5` (HORIZON_CHURN = 6 meses) y compara
contra v4 (HORIZON_CHURN = 4 meses) bajo GroupKFold y split temporal forward.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "07_baselines_v5.ipynb"


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

CELLS.append(md("""\
# Baselines sobre `training_churn_v5` (HORIZON_CHURN = 6 meses)

Replica NB 05 sobre v5 para responder:

1. **¿v5 mantiene el AUC bajo GroupKFold?** Cota mínima: AUC ≥ 0.74
   (cota observada en v4).
2. **¿v5 reduce o mantiene el drift temporal?** Cotas a verificar contra v4:
   - AUC del split temporal forward ≥ 0.729 (cota v4).
   - Std del AUC por mes dentro del bloque test ≤ 0.030 (cota v4).
   - Diferencia churn rate train vs test ≤ |0.36pp| (cota v4).
3. **¿La definición más conservadora gana en PR-AUC?** Esperado: cae churn
   rate (32.6% → 27.5%) pero el target tiene menos ruido (16% vs 24% de
   falsos churn) → PR-AUC podría subir o quedar plano.

Modelo principal: HGB balanceado (ganador del NB 03 y NB 05).
"""))

CELLS.append(md("## 1. Setup y carga\n"))

CELLS.append(code("""\
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import bigquery

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve, roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import lightgbm as lgb
    _ = lgb.LGBMClassifier()
    LGBM_OK = True
except Exception as e:
    print(f'[!] LightGBM no disponible — se saltea. ({type(e).__name__})')
    LGBM_OK = False

try:
    import xgboost as xgb
    _ = xgb.XGBClassifier()
    XGB_OK = True
except Exception as e:
    print(f'[!] XGBoost no disponible — se saltea. ({type(e).__name__})')
    XGB_OK = False

if not (LGBM_OK and XGB_OK):
    print('[!] Probable causa: falta `libomp` en macOS. Solución: `brew install libomp` y re-correr.')
    print('    HGB sigue funcionando y es el modelo primario; la comparación v4 vs v5 no se bloquea.')

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 220)
np.random.seed(42)

PROJECT, DATASET = 'glamour-peru-dw', 'glamour_dw'
TABLE = f'`{PROJECT}.{DATASET}.training_churn_v5`'
RANDOM_STATE = 42
N_SPLITS = 5
HORIZON_CHURN = 6  # meses

bq = bigquery.Client(project=PROJECT)
df = bq.query(f'SELECT * FROM {TABLE}').to_dataframe()

print(f'shape           : {df.shape}')
print(f'vendedoras      : {df["id_vendedor"].nunique():,}')
print(f'meses           : {df["mes_obs"].nunique()}')
print(f'churn rate      : {df["churn"].mean():.4f}')
print(f'rango mes_obs   : {df["mes_obs"].min()} → {df["mes_obs"].max()}')
print(f'rango mes_rank  : {df["mes_rank_obs"].min()} → {df["mes_rank_obs"].max()}')
df.head(3)
"""))

CELLS.append(md("## 2. Features y pipeline (idéntico a NB 05)\n"))

CELLS.append(code("""\
EXCLUDE = {
    'id_vendedor', 'mes_obs', 'mes_rank_obs',
    'fecha_ingreso',
    'id_coordinadora', 'ccodubigeo', 'distrito',
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

print(f'#features   : {len(feature_cols)}')
print(f'#numéricas  : {len(numeric_cols)}')
print(f'#categóricas: {len(CATEGORICAL)}  (cardinalidades: {[X[c].nunique() for c in CATEGORICAL]})')
print(f'positivos   : {y.sum():,} ({y.mean()*100:.2f}%)')

def build_preprocessor(scale=False):
    num_steps = [('impute', SimpleImputer(strategy='median'))]
    if scale: num_steps.append(('scale', StandardScaler()))
    return ColumnTransformer([
        ('num', Pipeline(num_steps), numeric_cols),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='NA')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), CATEGORICAL),
    ])
"""))

CELLS.append(md("## 3. Función de evaluación con GroupKFold\n"))

CELLS.append(code("""\
cv = GroupKFold(n_splits=N_SPLITS)

def evaluate(model_factory, X, y, groups, name, predict_proba=True):
    oof = np.zeros(len(y), dtype=float)
    fold_aucs, fold_aps = [], []
    for tr, te in cv.split(X, y, groups):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        if predict_proba:
            m = model_factory(); m.fit(Xtr, ytr)
            proba = m.predict_proba(Xte)[:, 1]
        else:
            proba = model_factory()(Xte)
        oof[te] = proba
        fold_aucs.append(roc_auc_score(yte, proba))
        fold_aps.append(average_precision_score(yte, proba))

    pred05 = (oof >= 0.5).astype(int)
    p, r, t = precision_recall_curve(y, oof)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    bi = int(np.argmax(f1[:-1]))
    bt = float(t[bi])
    pb = (oof >= bt).astype(int)

    return {
        'name': name, 'oof': oof,
        'auc_mean': np.mean(fold_aucs), 'auc_std': np.std(fold_aucs),
        'ap_mean': np.mean(fold_aps), 'ap_std': np.std(fold_aps),
        'auc_oof': roc_auc_score(y, oof), 'ap_oof': average_precision_score(y, oof),
        'f1_05': f1_score(y, pred05),
        'precision_05': precision_score(y, pred05, zero_division=0),
        'recall_05': recall_score(y, pred05),
        'best_thr': bt, 'f1_best': f1_score(y, pb),
        'precision_best': precision_score(y, pb, zero_division=0),
        'recall_best': recall_score(y, pb),
    }

def summarize(results):
    return pd.DataFrame([{
        'modelo': r['name'],
        'AUC (mean ± std)': f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}",
        'PR-AUC (mean ± std)': f"{r['ap_mean']:.4f} ± {r['ap_std']:.4f}",
        'F1@0.5': f"{r['f1_05']:.3f}",
        'Recall@0.5': f"{r['recall_05']:.3f}",
        'Precision@0.5': f"{r['precision_05']:.3f}",
        'best thr': f"{r['best_thr']:.3f}",
        'F1@best': f"{r['f1_best']:.3f}",
        'Recall@best': f"{r['recall_best']:.3f}",
        'Precision@best': f"{r['precision_best']:.3f}",
    } for r in results])

for tr, te in cv.split(X, y, groups):
    assert not (set(groups[tr]) & set(groups[te]))
print(f'OK — GroupKFold {N_SPLITS} folds, sin solapamiento de vendedoras.')
"""))

CELLS.append(md("""\
## 4. Baselines bajo GroupKFold

Mismos 6 modelos que NB 05. La heurística usa
`1 - exp(-gap / HORIZON_CHURN)` con `HORIZON_CHURN = 6` (escala alineada
al nuevo horizonte; en NB 05 era 4).
"""))

CELLS.append(code("""\
def make_dummy():
    return Pipeline([('prep', build_preprocessor(False)),
                     ('clf', DummyClassifier(strategy='stratified', random_state=RANDOM_STATE))])

def make_heuristic():
    def scorer(Xte):
        gap = Xte['meses_desde_compra_previa'].fillna(0).astype(float).values
        return 1.0 - np.exp(-gap / float(HORIZON_CHURN))
    return scorer

def make_logreg():
    return Pipeline([('prep', build_preprocessor(True)),
                     ('clf', LogisticRegression(max_iter=2000, class_weight='balanced',
                                                random_state=RANDOM_STATE, n_jobs=-1))])

def make_hgb():
    return Pipeline([('prep', build_preprocessor(False)),
                     ('clf', HistGradientBoostingClassifier(class_weight='balanced',
                                                            max_iter=400, learning_rate=0.05,
                                                            random_state=RANDOM_STATE,
                                                            early_stopping=False))])

def make_lgbm():
    return Pipeline([('prep', build_preprocessor(False)),
                     ('clf', lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05,
                                                num_leaves=63, min_child_samples=40,
                                                subsample=0.9, colsample_bytree=0.9,
                                                is_unbalance=True, random_state=RANDOM_STATE,
                                                n_jobs=-1, verbose=-1))]) if LGBM_OK else None

p_pos = float(y.mean())
def make_xgb():
    if not XGB_OK:
        return None
    return Pipeline([('prep', build_preprocessor(False)),
                     ('clf', xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6,
                                               min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
                                               scale_pos_weight=(1 - p_pos) / p_pos,
                                               objective='binary:logistic', eval_metric='auc',
                                               tree_method='hist', random_state=RANDOM_STATE,
                                               n_jobs=-1))])

model_specs = [
    ('Dummy (stratified)',          make_dummy,     True,  True),
    ('Heurística (gap previo)',     make_heuristic, False, True),
    ('LogReg (balanced)',           make_logreg,    True,  True),
    ('HGB (balanced)',              make_hgb,       True,  True),
    ('LightGBM (unbalanced)',       make_lgbm,      True,  LGBM_OK),
    ('XGBoost (scale_pos_weight)',  make_xgb,       True,  XGB_OK),
]

results = []
for name, fac, pp, available in model_specs:
    if not available:
        print(f'→ saltando {name} (dependencia nativa no disponible)')
        continue
    print(f'→ entrenando {name} ...', flush=True)
    r = evaluate(fac, X, y, groups, name=name, predict_proba=pp)
    results.append(r)
    print(f'   AUC OOF {r["auc_oof"]:.4f}  PR-AUC {r["ap_oof"]:.4f}')

summarize(results)
"""))

CELLS.append(md("## 5. Curvas ROC y PR (todos los modelos)\n"))

CELLS.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
prev = y.mean()
for r in results:
    fpr, tpr, _ = roc_curve(y, r['oof'])
    axes[0].plot(fpr, tpr, label=f"{r['name']}  (AUC {r['auc_oof']:.3f})")
    p, rc, _ = precision_recall_curve(y, r['oof'])
    axes[1].plot(rc, p, label=f"{r['name']}  (AP {r['ap_oof']:.3f})")
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); axes[0].set_title('ROC (OOF)'); axes[0].legend(fontsize=9)
axes[1].axhline(prev, ls='--', color='k', alpha=0.4, label=f'prev ({prev:.3f})')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].set_title('PR (OOF)'); axes[1].legend(fontsize=9)
plt.tight_layout(); plt.show()
"""))

CELLS.append(md("""\
## 6. Comparación lado a lado: v4 vs v5 (GroupKFold)

Los números de v4 vienen del NB 05 (mismo hardware, mismas seeds).
"""))

CELLS.append(code("""\
v4_bench = {
    'Dummy (stratified)':         {'auc': 0.5014, 'ap': 0.3269},
    'Heurística (gap previo)':    {'auc': 0.6551, 'ap': 0.4663},
    'LogReg (balanced)':          {'auc': 0.7403, 'ap': 0.5487},
    'HGB (balanced)':             {'auc': 0.7413, 'ap': 0.5531},
    'LightGBM (unbalanced)':      {'auc': 0.7288, 'ap': 0.5325},
    'XGBoost (scale_pos_weight)': {'auc': 0.7373, 'ap': 0.5445},
}
rows = []
for r in results:
    v4 = v4_bench.get(r['name'], {})
    auc_v4 = v4.get('auc', np.nan); ap_v4 = v4.get('ap', np.nan)
    rows.append({
        'modelo': r['name'],
        'AUC v4': f'{auc_v4:.4f}' if auc_v4 == auc_v4 else '—',
        'AUC v5': f"{r['auc_oof']:.4f}",
        'Δ AUC':  f"{(r['auc_oof'] - auc_v4):+.4f}" if auc_v4 == auc_v4 else '—',
        'PR-AUC v4': f'{ap_v4:.4f}' if ap_v4 == ap_v4 else '—',
        'PR-AUC v5': f"{r['ap_oof']:.4f}",
        'Δ PR-AUC': f"{(r['ap_oof'] - ap_v4):+.4f}" if ap_v4 == ap_v4 else '—',
    })
pd.DataFrame(rows)
"""))

CELLS.append(md("""\
**Aviso sobre los números v4 hardcodeados**: si `v4_bench` no coincide con
los outputs reales del NB 05 (puede pasar si NB 05 fue re-ejecutado con
seeds distintas), re-leer NB 05 y actualizar este diccionario. Los valores
están tomados de la última ejecución conocida del NB 05.
"""))

CELLS.append(md("""\
## 7. Split temporal forward sobre v5

- **Test**: bloque de los **últimos 6 meses** del dataset.
- **Train**: `mes_rank_obs ≤ (test_min_rank − GAP)`.
- **GAP = HORIZON_CHURN + 1 = 7 meses**. Asegura que el target del último
  train (que mira 6 meses adelante) termine antes del primer mes del test.
  En NB 05 era 5 (HORIZON_CHURN=4 + 1).

Modelo: HGB balanceado.

Comparación clave contra v4 (NB 05):
- AUC bloque test: v4 = 0.729 → v5 ¿se mantiene?
- Std AUC por mes test: v4 = 0.030 → v5 ¿se mantiene?
- |Δ churn rate train vs test|: v4 = 0.36pp → v5 ¿se mantiene?
"""))

CELLS.append(code("""\
TEST_WINDOW = 6
GAP = HORIZON_CHURN + 1   # = 7

last_rank = int(df['mes_rank_obs'].max())
test_min = last_rank - TEST_WINDOW + 1
train_max = test_min - GAP

train_mask_t = df['mes_rank_obs'] <= train_max
test_mask_t  = df['mes_rank_obs'].between(test_min, last_rank)

df_train_t = df.loc[train_mask_t]
df_test_t  = df.loc[test_mask_t]

X_train_t, y_train_t = df_train_t[feature_cols], df_train_t['churn'].astype(int).values
X_test_t,  y_test_t  = df_test_t[feature_cols],  df_test_t['churn'].astype(int).values

model_t = make_hgb()
model_t.fit(X_train_t, y_train_t)
proba_t = model_t.predict_proba(X_test_t)[:, 1]

print(f'Train: mes_rank_obs ≤ {train_max}  →  {len(df_train_t):,} filas · '
      f'{df_train_t["id_vendedor"].nunique():,} vendedoras · '
      f'churn rate {df_train_t["churn"].mean():.4f}')
print(f'Test : mes_rank_obs ∈ [{test_min}, {last_rank}]  →  {len(df_test_t):,} filas · '
      f'{df_test_t["id_vendedor"].nunique():,} vendedoras · '
      f'churn rate {df_test_t["churn"].mean():.4f}')
overlap = len(set(df_train_t['id_vendedor']) & set(df_test_t['id_vendedor']))
n_test_v = df_test_t['id_vendedor'].nunique()
print(f'Vendedoras presentes en train Y test: {overlap:,} '
      f'({overlap / n_test_v * 100:.1f}% del test)' if n_test_v else
      f'Vendedoras presentes en train Y test: {overlap:,}')

if len(y_test_t) and y_test_t.sum() and y_test_t.sum() < len(y_test_t):
    auc_t = roc_auc_score(y_test_t, proba_t)
    ap_t  = average_precision_score(y_test_t, proba_t)
    pred_t = (proba_t >= 0.5).astype(int)

    print()
    print(f'ROC-AUC      : {auc_t:.4f}  (NB 05 v4 = 0.7286)')
    print(f'PR-AUC       : {ap_t:.4f}  (prevalencia test {y_test_t.mean():.4f})')
    print(f'F1@0.5       : {f1_score(y_test_t, pred_t):.3f}')
    print(f'Recall@0.5   : {recall_score(y_test_t, pred_t):.3f}')
    print(f'Precision@0.5: {precision_score(y_test_t, pred_t, zero_division=0):.3f}')
    print()
    cm = confusion_matrix(y_test_t, pred_t)
    print('Matriz de confusión @ thr=0.5')
    print(pd.DataFrame(cm, index=['real: no-churn', 'real: churn'],
                       columns=['pred: no-churn', 'pred: churn']))
else:
    auc_t = ap_t = float('nan')
    print('\\n[!] Test set degenerado (0 positivos o 0 negativos). '
          'Probable causa: pérdida de horizonte observable al final del panel.')
"""))

CELLS.append(md("## 8. Estabilidad por mes dentro del test\n"))

CELLS.append(code("""\
eval_t = df_test_t[['mes_rank_obs', 'mes_obs', 'churn']].copy()
eval_t['proba'] = proba_t

per_mes = []
for r, g in eval_t.groupby('mes_rank_obs'):
    yt = g['churn'].astype(int).values
    pp = g['proba'].values
    n_pos = int(yt.sum())
    auc_v = roc_auc_score(yt, pp) if 0 < n_pos < len(yt) else np.nan
    ap_v  = average_precision_score(yt, pp) if n_pos else np.nan
    per_mes.append({
        'rank': int(r),
        'mes': str(g['mes_obs'].iloc[0])[:10],
        'n': len(yt), 'n_churn': n_pos,
        'churn_rate': float(yt.mean()),
        'AUC': auc_v, 'AP': ap_v,
    })
per_mes = pd.DataFrame(per_mes).sort_values('rank').reset_index(drop=True)
print(per_mes.to_string(index=False))
print()
auc_std = per_mes['AUC'].std()
print(f"AUC por mes — media {per_mes['AUC'].mean():.3f}, "
      f"std {auc_std:.3f}, "
      f"min {per_mes['AUC'].min():.3f}, max {per_mes['AUC'].max():.3f}")
print(f"AUC bloque agregado: {auc_t:.3f}")
print()
print(f"Comparación con NB 05 (v4):  std AUC v4 = 0.030, v5 = {auc_std:.3f}  "
      f"→ {'MEJOR (más estable)' if auc_std < 0.030 else ('similar' if abs(auc_std - 0.030) < 0.01 else 'peor')}")
"""))

CELLS.append(md("## 9. Veredicto\n"))

CELLS.append(code("""\
hgb_result = next(r for r in results if r['name'] == 'HGB (balanced)')

print('='*70)
print('VEREDICTO v5 (HORIZON_CHURN = 6) vs v4 (HORIZON_CHURN = 4)')
print('='*70)
print()
print(f"{'Métrica':<40} {'v4':>10} {'v5':>10} {'Δ':>10}  Pasa")
print('-'*70)

# AUC GroupKFold (HGB)
auc_v4_hgb = v4_bench['HGB (balanced)']['auc']
auc_v5_hgb = hgb_result['auc_oof']
ok = auc_v5_hgb >= 0.74
print(f"{'AUC GroupKFold (HGB)':<40} {auc_v4_hgb:>10.4f} {auc_v5_hgb:>10.4f} {auc_v5_hgb-auc_v4_hgb:>+10.4f}  {'✓' if ok else '✗'}")

# PR-AUC GroupKFold (HGB)
ap_v4_hgb = v4_bench['HGB (balanced)']['ap']
ap_v5_hgb = hgb_result['ap_oof']
print(f"{'PR-AUC GroupKFold (HGB)':<40} {ap_v4_hgb:>10.4f} {ap_v5_hgb:>10.4f} {ap_v5_hgb-ap_v4_hgb:>+10.4f}")

# AUC split forward
auc_v4_fwd = 0.7286
ok = auc_t >= auc_v4_fwd - 0.01
print(f"{'AUC split forward':<40} {auc_v4_fwd:>10.4f} {auc_t:>10.4f} {auc_t-auc_v4_fwd:>+10.4f}  {'✓' if ok else '✗'}")

# Std AUC por mes
std_v4 = 0.030
ok = auc_std <= std_v4 + 0.01
print(f"{'Std AUC por mes (test)':<40} {std_v4:>10.4f} {auc_std:>10.4f} {auc_std-std_v4:>+10.4f}  {'✓' if ok else '✗'}")

# Drift churn rate
drift_v4 = abs(0.3258 - 0.3222)
drift_v5 = abs(df_train_t['churn'].mean() - df_test_t['churn'].mean())
ok = drift_v5 <= drift_v4 + 0.02
print(f"{'|Δ churn rate train vs test|':<40} {drift_v4:>10.4f} {drift_v5:>10.4f} {drift_v5-drift_v4:>+10.4f}  {'✓' if ok else '✗'}")

print()
print('Decisión queda escrita en la siguiente celda markdown.')
"""))

CELLS.append(md("""\
### Decisión

*Esta celda se completa después de inspeccionar los outputs.*

Llenar:

| Criterio | Cota (v4) | Resultado v5 | Pasa |
|---|---|---|---|
| AUC GroupKFold (HGB) | ≥ 0.74 | … | ☐ |
| PR-AUC GroupKFold (HGB) | ≥ 0.55 | … | ☐ |
| AUC split forward | ≥ 0.72 | … | ☐ |
| Std AUC por mes (test) | ≤ 0.04 | … | ☐ |
| |Δ churn rate train vs test| | ≤ 0.02 | … | ☐ |

**Decisión:**

- **3+ criterios pasan** → migrar a v5 como dataset productivo. Actualizar
  `STATUS.md` para que `training_churn_v5` sea la referencia y v4 quede
  como histórica.
- **AUC sube y PR-AUC sube** → ganancia neta: target menos ruidoso
  (16% vs 24% de falsos churn) compensa la menor prevalencia.
- **AUC plano y PR-AUC sube** → equivalente en ranking pero con menos
  ruido en el target → migrar igual.
- **AUC cae > 1pp** → revisar si la ventana de target perdió señal útil
  en los últimos meses observables; considerar k=5 como compromiso.
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
