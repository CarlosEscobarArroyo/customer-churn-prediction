"""Construye notebooks/08_baselines_v6.ipynb desde cero.

Comparación 3-way:
- v5         : full data (2017-2025) + filtro hist >= 3.        — vigente
- v6         : post-pandemia (2022+) + filtro hist >= 3.
- v6_all     : post-pandemia (2022+) sin filtro de población.

Modelo principal: HGB balanceado (mismo hiperparámetros que NB 05/07).
Protocolos: GroupKFold (5 folds, por vendedora) + split temporal forward
(últimos 6 meses test, GAP = 7).
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "08_baselines_v6.ipynb"


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
# Comparación 3-way: v5 vs v6 (post-pandemia) vs v6_all (post-pandemia + todos)

Hipótesis a contrastar:

1. **Filtrar pre-pandemia mejora la generalización**: si v6 ≥ v5 en split
   forward, los datos pre-2022 no aportan al régimen actual.
2. **El filtro `compras_historicas >= 3` sigue siendo necesario**: si v6_all
   cae > 1pp respecto a v6, vendedores con 1-2 meses de historia saturan
   el target (regla "1 compra → churn" domina).
3. **Volumen vs limpieza**: v6 pierde más de la mitad de la muestra de v5
   (9.841 vs 23.684). Es un trade-off entre tamaño y especificidad temporal.

Modelo: HGB balanceado.
Protocolos: GroupKFold + split temporal forward (últimos 6 meses test).
"""))

CELLS.append(md("## 1. Setup\n"))

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

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 220)
np.random.seed(42)

PROJECT, DATASET = 'glamour-peru-dw', 'glamour_dw'
RANDOM_STATE = 42
N_SPLITS = 5
HORIZON_CHURN = 6   # meses

VARIANTS = ['v5', 'v6', 'v6_all']

bq = bigquery.Client(project=PROJECT)
dfs = {}
for v in VARIANTS:
    table = f'`{PROJECT}.{DATASET}.training_churn_{v}`'
    dfs[v] = bq.query(f'SELECT * FROM {table}').to_dataframe()
    df_ = dfs[v]
    print(f'{v:<8} → {len(df_):>6,} filas · {df_["id_vendedor"].nunique():>5,} vendedoras · '
          f'{df_["mes_obs"].nunique():>3} meses · churn {df_["churn"].mean():.4f} · '
          f'mes_obs [{df_["mes_obs"].min()} → {df_["mes_obs"].max()}]')
"""))

CELLS.append(md("## 2. Pipeline (idéntico a NB 05/07)\n"))

CELLS.append(code("""\
EXCLUDE = {
    'id_vendedor', 'mes_obs', 'mes_rank_obs',
    'fecha_ingreso',
    'id_coordinadora', 'ccodubigeo', 'distrito',
    'churn',
}
CATEGORICAL = ['sexo_vendedor', 'tipo_vendedor', 'departamento', 'provincia']

def prepare(df_: pd.DataFrame):
    feature_cols = [c for c in df_.columns if c not in EXCLUDE]
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL]
    df_ = df_.copy()
    for c in CATEGORICAL:
        df_[c] = df_[c].astype('string').fillna('NA')
    return df_, feature_cols, numeric_cols

def build_preprocessor(numeric_cols, scale=False):
    num_steps = [('impute', SimpleImputer(strategy='median'))]
    if scale: num_steps.append(('scale', StandardScaler()))
    return ColumnTransformer([
        ('num', Pipeline(num_steps), numeric_cols),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='NA')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), CATEGORICAL),
    ])

def make_hgb(numeric_cols):
    return Pipeline([('prep', build_preprocessor(numeric_cols, False)),
                     ('clf', HistGradientBoostingClassifier(class_weight='balanced',
                                                            max_iter=400, learning_rate=0.05,
                                                            random_state=RANDOM_STATE,
                                                            early_stopping=False))])

def make_logreg(numeric_cols):
    return Pipeline([('prep', build_preprocessor(numeric_cols, True)),
                     ('clf', LogisticRegression(max_iter=2000, class_weight='balanced',
                                                random_state=RANDOM_STATE, n_jobs=-1))])

def make_dummy(numeric_cols):
    return Pipeline([('prep', build_preprocessor(numeric_cols, False)),
                     ('clf', DummyClassifier(strategy='stratified', random_state=RANDOM_STATE))])

def make_heuristic_scorer():
    def scorer(Xte):
        gap = Xte['meses_desde_compra_previa'].fillna(0).astype(float).values
        return 1.0 - np.exp(-gap / float(HORIZON_CHURN))
    return scorer
"""))

CELLS.append(md("## 3. GroupKFold sobre cada variante\n"))

CELLS.append(code("""\
cv = GroupKFold(n_splits=N_SPLITS)

def evaluate_gkf(df_, name):
    df_, feature_cols, numeric_cols = prepare(df_)
    X = df_[feature_cols].copy()
    y = df_['churn'].astype(int).values
    groups = df_['id_vendedor'].values

    out = {'variant': name, 'n': len(df_), 'n_vend': df_['id_vendedor'].nunique(),
           'churn_rate': float(y.mean()), 'models': {}}

    # heurística (no requiere fit)
    scorer = make_heuristic_scorer()
    proba_h = scorer(X)
    out['models']['Heurística'] = {
        'auc': roc_auc_score(y, proba_h),
        'ap':  average_precision_score(y, proba_h),
    }

    for model_name, factory in [('LogReg', make_logreg), ('HGB', make_hgb)]:
        oof = np.zeros(len(y), dtype=float)
        fold_aucs, fold_aps = [], []
        for tr, te in cv.split(X, y, groups):
            m = factory(numeric_cols); m.fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[te])[:, 1]
            oof[te] = p
            fold_aucs.append(roc_auc_score(y[te], p))
            fold_aps.append(average_precision_score(y[te], p))
        out['models'][model_name] = {
            'auc_mean': float(np.mean(fold_aucs)),
            'auc_std':  float(np.std(fold_aucs)),
            'ap_mean':  float(np.mean(fold_aps)),
            'ap_std':   float(np.std(fold_aps)),
            'auc':      float(roc_auc_score(y, oof)),
            'ap':       float(average_precision_score(y, oof)),
        }
        print(f'{name:<8} {model_name:<8} AUC {out["models"][model_name]["auc"]:.4f}  '
              f'AP {out["models"][model_name]["ap"]:.4f}')
    return out

gkf_results = {}
for v in VARIANTS:
    print(f'\\n→ {v}')
    gkf_results[v] = evaluate_gkf(dfs[v], v)

print('\\nDone.')
"""))

CELLS.append(md("## 4. Comparación GroupKFold lado a lado\n"))

CELLS.append(code("""\
rows = []
for v in VARIANTS:
    r = gkf_results[v]
    for m in ['Heurística', 'LogReg', 'HGB']:
        d = r['models'][m]
        rows.append({
            'variant': v,
            'modelo':  m,
            'n_obs':   r['n'],
            'n_vend':  r['n_vend'],
            'churn_rate': r['churn_rate'],
            'AUC':     d['auc'],
            'PR-AUC':  d['ap'],
            'AUC mean ± std': f"{d.get('auc_mean', d['auc']):.4f} ± {d.get('auc_std', 0):.4f}",
        })
gkf_summary = pd.DataFrame(rows)
print(gkf_summary.to_string(index=False))
print()

# pivot: AUC HGB por variante para comparar rápido
pivot = gkf_summary[gkf_summary['modelo'] == 'HGB'].set_index('variant')[['AUC', 'PR-AUC', 'churn_rate', 'n_obs']]
print('HGB por variante:')
print(pivot.round(4))
"""))

CELLS.append(md("## 5. Split temporal forward sobre cada variante\n"))

CELLS.append(code("""\
TEST_WINDOW = 6
GAP = HORIZON_CHURN + 1   # 7

def evaluate_forward(df_, name):
    df_, feature_cols, numeric_cols = prepare(df_)

    last_rank = int(df_['mes_rank_obs'].max())
    test_min  = last_rank - TEST_WINDOW + 1
    train_max = test_min - GAP

    df_tr = df_[df_['mes_rank_obs'] <= train_max]
    df_te = df_[df_['mes_rank_obs'].between(test_min, last_rank)]

    if df_tr.empty or df_te.empty:
        return None

    X_tr, y_tr = df_tr[feature_cols], df_tr['churn'].astype(int).values
    X_te, y_te = df_te[feature_cols], df_te['churn'].astype(int).values

    if y_te.sum() == 0 or y_te.sum() == len(y_te):
        return None

    m = make_hgb(numeric_cols); m.fit(X_tr, y_tr)
    proba = m.predict_proba(X_te)[:, 1]

    auc_blk = roc_auc_score(y_te, proba)
    ap_blk  = average_precision_score(y_te, proba)

    # AUC por mes test
    eval_t = df_te[['mes_rank_obs']].copy()
    eval_t['y'] = y_te
    eval_t['p'] = proba
    per_mes = []
    for r, g in eval_t.groupby('mes_rank_obs'):
        yt, pp = g['y'].values, g['p'].values
        if 0 < yt.sum() < len(yt):
            per_mes.append(roc_auc_score(yt, pp))
    auc_std = float(np.std(per_mes)) if per_mes else float('nan')
    auc_mean_per_mes = float(np.mean(per_mes)) if per_mes else float('nan')

    return {
        'variant': name,
        'n_train': len(df_tr), 'n_test': len(df_te),
        'churn_train': float(y_tr.mean()),
        'churn_test':  float(y_te.mean()),
        'overlap_pct': len(set(df_tr['id_vendedor']) & set(df_te['id_vendedor'])) /
                       max(df_te['id_vendedor'].nunique(), 1) * 100,
        'auc_block': auc_blk, 'ap_block': ap_blk,
        'auc_mean_per_mes': auc_mean_per_mes, 'auc_std_per_mes': auc_std,
        'train_max_rank': train_max, 'test_min_rank': test_min, 'last_rank': last_rank,
    }

fwd_results = {}
for v in VARIANTS:
    fwd_results[v] = evaluate_forward(dfs[v], v)
    r = fwd_results[v]
    if r is None:
        print(f'{v}: insuficiente data para split forward.')
        continue
    print(f"{v:<8} train [{r['n_train']:>5,} filas, churn {r['churn_train']:.3f}] "
          f"→ test [{r['n_test']:>5,} filas, churn {r['churn_test']:.3f}] "
          f"AUC bloque {r['auc_block']:.4f}  std/mes {r['auc_std_per_mes']:.4f}  "
          f"overlap {r['overlap_pct']:.1f}%")
"""))

CELLS.append(md("## 6. Comparación split forward lado a lado\n"))

CELLS.append(code("""\
rows = []
for v in VARIANTS:
    r = fwd_results[v]
    if r is None:
        continue
    rows.append({
        'variant':       v,
        'n_train':       r['n_train'],
        'n_test':        r['n_test'],
        'churn_train':   r['churn_train'],
        'churn_test':    r['churn_test'],
        '|drift|':       abs(r['churn_train'] - r['churn_test']),
        'overlap_pct':   r['overlap_pct'],
        'AUC bloque':    r['auc_block'],
        'PR-AUC bloque': r['ap_block'],
        'AUC mean/mes':  r['auc_mean_per_mes'],
        'std AUC/mes':   r['auc_std_per_mes'],
    })
fwd_summary = pd.DataFrame(rows)
print(fwd_summary.round(4).to_string(index=False))
"""))

CELLS.append(md("## 7. Veredicto\n"))

CELLS.append(code("""\
def fmt(r, key, fmt_str='.4f'):
    return ('—' if r is None else format(r[key], fmt_str))

print('=' * 90)
print('VEREDICTO 3-WAY: v5 vs v6 (post-pandemia) vs v6_all (post-pandemia + todos)')
print('=' * 90)
print()
print(f"{'Métrica':<35} {'v5':>15} {'v6':>15} {'v6_all':>15}")
print('-' * 90)

# AUC GroupKFold (HGB)
v5_a = gkf_results['v5']['models']['HGB']['auc']
v6_a = gkf_results['v6']['models']['HGB']['auc']
v6a_a = gkf_results['v6_all']['models']['HGB']['auc']
print(f"{'AUC GroupKFold (HGB)':<35} {v5_a:>15.4f} {v6_a:>15.4f} {v6a_a:>15.4f}")

# PR-AUC GroupKFold (HGB)
v5_p = gkf_results['v5']['models']['HGB']['ap']
v6_p = gkf_results['v6']['models']['HGB']['ap']
v6a_p = gkf_results['v6_all']['models']['HGB']['ap']
print(f"{'PR-AUC GroupKFold (HGB)':<35} {v5_p:>15.4f} {v6_p:>15.4f} {v6a_p:>15.4f}")

# churn rate
print(f"{'Churn rate (global)':<35} "
      f"{gkf_results['v5']['churn_rate']:>15.4f} "
      f"{gkf_results['v6']['churn_rate']:>15.4f} "
      f"{gkf_results['v6_all']['churn_rate']:>15.4f}")

# n filas / vendedoras
print(f"{'n filas':<35} "
      f"{gkf_results['v5']['n']:>15,} "
      f"{gkf_results['v6']['n']:>15,} "
      f"{gkf_results['v6_all']['n']:>15,}")
print(f"{'n vendedoras':<35} "
      f"{gkf_results['v5']['n_vend']:>15,} "
      f"{gkf_results['v6']['n_vend']:>15,} "
      f"{gkf_results['v6_all']['n_vend']:>15,}")

# split forward
print()
print(f"{'AUC split forward':<35} "
      f"{fmt(fwd_results['v5'], 'auc_block')!s:>15} "
      f"{fmt(fwd_results['v6'], 'auc_block')!s:>15} "
      f"{fmt(fwd_results['v6_all'], 'auc_block')!s:>15}")
print(f"{'Std AUC por mes (test)':<35} "
      f"{fmt(fwd_results['v5'], 'auc_std_per_mes')!s:>15} "
      f"{fmt(fwd_results['v6'], 'auc_std_per_mes')!s:>15} "
      f"{fmt(fwd_results['v6_all'], 'auc_std_per_mes')!s:>15}")
print(f"{'|Δ churn rate train↔test|':<35} "
      f"{fmt(fwd_results['v5'], '|drift|') if False else format(abs(fwd_results['v5']['churn_train']-fwd_results['v5']['churn_test']), '.4f'):>15} "
      f"{format(abs(fwd_results['v6']['churn_train']-fwd_results['v6']['churn_test']), '.4f'):>15} "
      f"{format(abs(fwd_results['v6_all']['churn_train']-fwd_results['v6_all']['churn_test']), '.4f'):>15}")
print()
print('Decisión queda escrita en la celda markdown siguiente.')
"""))

CELLS.append(md("""\
### Decisión

*Esta celda se completa después de inspeccionar los outputs.*

Llenar:

| Hipótesis | Predicción | Resultado | Veredicto |
|---|---|---|---|
| Filtrar pre-pandemia mejora generalización | v6 ≥ v5 en split forward | … | ☐ |
| Filtro hist≥3 sigue necesario post-pandemia | v6 > v6_all en AUC | … | ☐ |
| Volumen de v5 compensa el ruido pre-pandemia | v5 ≥ v6 en GroupKFold | … | ☐ |

**Decisión final**: …

Opciones según resultado:
- **v6 gana** (AUC sube en forward) → migrar a v6 productivo.
- **v6_all gana** (AUC sube quitando filtro) → considerar relajar el filtro
  de población; documentar que en el régimen post-pandemia los vendedores
  con poca historia no saturan el target.
- **v5 gana** (volumen importa más que limpieza temporal) → mantener v5.
- **Empate técnico (Δ < 0.5pp)** → quedarse con v5 (más datos, menos
  riesgo de overfitting a régimen actual; más fácil mantener).
"""))


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "customer-churn-prediction (3.12.3)",
                "language": "python", "name": "python3",
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
