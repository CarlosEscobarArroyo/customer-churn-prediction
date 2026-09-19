# %% [markdown]
# # Opción C — Google Trends como señal externa
#
# **Pregunta.** ¿El interés de búsqueda en Perú (venta por catálogo, ropa, competidores,
# alternativas comerciales, empleo/crédito) aporta capacidad predictiva sobre el historial
# de compras? Y si aporta, ¿mejora el **nivel de riesgo de cada mes** (todas las vendedoras
# suben o bajan juntas) o la **selección individual** dentro del mes?
#
# **Diseño.**
# - Series mensuales `geo=PE`, 2016-01 → 2026-02 (cubren el panel: ranks 1–117), una
#   consulta por término (cada serie en su propia escala 0–100). Cache en
#   `data/external/google_trends_pe.csv`; si Google bloquea (429) se puede exportar a mano
#   desde trends.google.com al mismo CSV (columnas `date`, `<término>`).
# - Cobertura geográfica: `interest_by_region` (departamentos) para dos términos, contra la
#   distribución de departamentos del panel.
# - **Información conocida en cada fecha**: para la fila de mes t se usan valores ≤ t (el mes t
#   está completo al cerrar el mes, que es cuando se predice). La normalización 0–100 de Trends
#   usa el máximo de toda la ventana pedida (mira el futuro solo en la *escala*, no en la forma):
#   por eso además se usan transformaciones libres de escala (variación interanual, z-score
#   expansivo con datos ≤ t). Cambios de metodología de Google: 2016-01 y 2022-01 (marcados
#   en la UI de Trends); el panel arranca en 2016-11, así que solo importa el segundo.
# - Evaluación en dos niveles: (1) **mensual**: correlación con la tasa de churn del mes
#   (solo meses de train, con y sin estacionalidad) y pronóstico expansivo de la tasa
#   mensual (Ridge: estacionalidad vs estacionalidad + Trends, entrenando solo con meses cuya
#   etiqueta ya se conoce, t−7); (2) **por fila**: XGBoost base vs +Trends en OOT y en
#   *rolling-origin* sobre los últimos 12 meses etiquetados (train ≤ t−7, test = mes t), con el
#   **oráculo mensual** (tasa real del mes como feature) como techo de cualquier variable
#   constante por mes. AUC agrupado vs AUC medio dentro del mes separa "nivel" de "selección".
# - GroupKFold no se usa como criterio: con features constantes por mes, mezcla los mismos
#   meses en train y validación y es optimista (se reporta solo como referencia).

# %%
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(next(p for p in [Path.cwd(), Path.cwd() / "08_experimentation"] if (p / "exp_utils.py").exists())))
from exp_utils import (EXTERNAL, TARGET, fmt, load_base, paired_delta, run_protocols, save_report, xgb_fp)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

# %% [markdown]
# ## 1. Descarga (con cache) y calidad de las series

# %%
TERMS = {
    "catalogo": ["venta por catalogo", "catalogo", "catalogo de ropa", "vender por catalogo", "glamour"],
    "competencia": ["yanbal", "natura", "unique", "esika", "belcorp"],
    "alternativas": ["shein", "temu", "gamarra", "ropa por mayor", "trabajo desde casa"],
    "macro": ["trabajo", "prestamo", "ofertas", "ropa", "cts"],
}
ALL_TERMS = [t for ts in TERMS.values() for t in ts]
CACHE = EXTERNAL / "google_trends_pe.csv"
CACHE_REG = EXTERNAL / "google_trends_pe_regiones.csv"
TIMEFRAME = "2016-01-01 2026-02-28"


def fetch_trends():
    from pytrends.request import TrendReq
    pt = TrendReq(hl="es-PE", tz=300, timeout=(10, 30))
    series, fails = {}, []
    for term in ALL_TERMS:
        for intento in range(3):
            try:
                pt.build_payload([term], geo="PE", timeframe=TIMEFRAME)
                s = pt.interest_over_time()
                if len(s):
                    s = s[~s["isPartial"].astype(bool)][term]
                series[term] = s
                break
            except Exception as e:  # 429 u otros: esperar y reintentar
                print(f"  {term}: {type(e).__name__} (intento {intento + 1})", flush=True)
                time.sleep(60 * (intento + 1))
        else:
            fails.append(term)
        time.sleep(2)
    out = pd.DataFrame(series)
    out.index.name = "date"
    return out, fails


EXTERNAL.mkdir(exist_ok=True)
if CACHE.exists():
    gt = pd.read_csv(CACHE, parse_dates=["date"]).set_index("date")
    print("cache:", CACHE, gt.shape)
else:
    gt, fails = fetch_trends()
    gt.to_csv(CACHE)
    print("descargado:", gt.shape, "| fallidos:", fails)
gt = gt.reindex(columns=[t for t in ALL_TERMS if t in gt.columns]).astype(float)

# calidad: cobertura, ceros, primer mes con señal
qual = pd.DataFrame({
    "familia": {t: f for f, ts in TERMS.items() for t in ts},
    "n_meses": gt.notna().sum(), "pct_ceros": (gt == 0).mean().round(3),
    "media": gt.mean().round(1), "max": gt.max(), "std": gt.std().round(1),
    "primer_mes_>0": gt.apply(lambda s: s[s > 0].index.min().strftime("%Y-%m") if (s > 0).any() else None),
    "corr_2016-21_vs_2022+": gt.apply(lambda s: np.nan),
}).reindex(gt.columns)
# salto de nivel en el cambio de metodología 2022-01: media 12 m antes vs 12 m después
pre = gt.loc["2021-01":"2021-12"].mean()
post = gt.loc["2022-01":"2022-12"].mean()
qual["salto_2022_ratio"] = (post / pre.replace(0, np.nan)).round(2)
qual = qual.drop(columns=["corr_2016-21_vs_2022+"])
print(qual.to_markdown())

# %%
if CACHE_REG.exists():
    reg = pd.read_csv(CACHE_REG, index_col=0)
else:
    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="es-PE", tz=300, timeout=(10, 30))
        reg = {}
        for term in ["venta por catalogo", "ropa"]:
            pt.build_payload([term], geo="PE", timeframe=TIMEFRAME)
            reg[term] = pt.interest_by_region(resolution="REGION")[term]
            time.sleep(2)
        reg = pd.DataFrame(reg)
        reg.to_csv(CACHE_REG)
    except Exception as e:
        print("regiones no disponibles:", type(e).__name__, e)
        reg = pd.DataFrame()
d, feats, tr, te = load_base()
y, g, mes = d[TARGET].values, d["id_vendedor"].values, d["mes_rank"].values
dep_cols = [c for c in feats if c.startswith("departamento_")]
dep_share = d[dep_cols].mean().rename(lambda c: c.replace("departamento_", "")).sort_values(ascending=False)
if len(reg):
    reg.index = reg.index.str.lower().str.replace("region", "").str.strip()
    cov = pd.DataFrame({"share_panel": dep_share.round(3)}).join(reg, how="left")
    print("Cobertura regional (interés por departamento, 0–100 relativo; NaN = sin dato en Trends):")
    print(cov.head(12).to_markdown())
    print(f"departamentos del panel con dato regional de Trends: {cov['venta por catalogo'].notna().mean():.0%}")

# %% [markdown]
# ## 2. Alineación temporal y features conocidas en t
#
# `mes_rank` = (año − 2016)·12 + mes − 10 (rank 2 = 2016-12, rank 111 = 2026-01).
# Por término: `t0` (valor del mes t), `yoy` (t − (t−12), libre de escala) y `zexp`
# (z-score con media y desvío de los valores ≤ t; requiere ≥ 12 meses).

# %%
gt = gt.copy()
gt["mes_rank"] = (gt.index.year - 2016) * 12 + gt.index.month - 10
assert gt.loc["2016-12-01", "mes_rank"] == 2 and gt.loc["2026-01-01", "mes_rank"] == 111
gtr = gt.set_index("mes_rank").sort_index()
F = {}
for t in ALL_TERMS:
    if t not in gtr.columns:
        continue
    s = gtr[t]
    F[f"gt_{t}_t0"] = s
    F[f"gt_{t}_yoy"] = s - s.shift(12)
    mu, sd = s.expanding(12).mean(), s.expanding(12).std()
    F[f"gt_{t}_zexp"] = (s - mu) / sd.replace(0, np.nan)
F = pd.DataFrame(F)
gt_cols = list(F.columns)
core_cols = [c for c in gt_cols if any(c.startswith(f"gt_{t}_") for t in TERMS["catalogo"])]
X_gt = F.reindex(mes).reset_index(drop=True)
print(f"{len(gt_cols)} features de Trends ({len(core_cols)} de la familia catálogo); NaN medio por fila: {X_gt.isna().mean().mean():.3f}")

# %% [markdown]
# ## 3. Nivel mensual: ¿Trends anticipa la tasa de churn del mes?

# %%
mensual = d.groupby("mes_rank").agg(churn_rate=(TARGET, "mean"), n=(TARGET, "size"))
mensual["mes_obs"] = pd.to_datetime(d.groupby("mes_rank")["mes_obs"].first())
mensual["moy"] = mensual["mes_obs"].dt.month
train_months = mensual.index[mensual.index <= mes[tr].max()]
# estacionalidad estimada solo con meses de train
season = mensual.loc[train_months].groupby("moy")["churn_rate"].mean()
mensual["churn_deseason"] = mensual["churn_rate"] - mensual["moy"].map(season)

corr_rows = []
for t in gtr.columns:
    for lag in range(0, 7):
        s = gtr[t].shift(lag).reindex(train_months)
        ok = s.notna()
        if ok.sum() < 24:
            continue
        r_raw = spearmanr(s[ok], mensual.loc[train_months, "churn_rate"][ok]).correlation
        r_des = spearmanr(s[ok], mensual.loc[train_months, "churn_deseason"][ok]).correlation
        corr_rows.append({"termino": t, "lag": lag, "rho_bruto": r_raw, "rho_desestac": r_des, "n": int(ok.sum())})
corr = pd.DataFrame(corr_rows)
best = corr.loc[corr.groupby("termino")["rho_desestac"].apply(lambda s: s.abs().idxmax())]
best = best.sort_values("rho_desestac", key=np.abs, ascending=False)
print(f"Correlación (Spearman) con la tasa mensual de churn en los {len(train_months)} meses de train; mejor lag por término:")
print(best.round(3).to_markdown(index=False))
# referencia: con 20 términos × 7 lags, |rho| ≈ 0.25 aparece por azar (n≈90 → SE ≈ 0.105)
print(f"SE aproximado de rho con n={len(train_months)}: {1 / np.sqrt(len(train_months) - 3):.3f}; "
      f"{(best['rho_desestac'].abs() > 2 / np.sqrt(len(train_months) - 3)).sum()} términos superan 2 SE (de {len(best)} probados, ~1 esperado por azar)")

# %%
# Pronóstico expansivo de la tasa mensual: meses objetivo = últimos 22 etiquetados; train = meses ≤ t-7
TOP_K = 5
top_terms = best.head(TOP_K)["termino"].tolist()


def month_design(cols_gt):
    X = pd.get_dummies(mensual["moy"], prefix="m", dtype=float)
    for t in cols_gt:
        X[f"{t}_l0"] = gtr[t].reindex(mensual.index).values
        X[f"{t}_l1"] = gtr[t].shift(1).reindex(mensual.index).values
    return X.fillna(0)


targets = mensual.index[mensual.index >= mensual.index.max() - 21]
fc = []
for t in targets:
    tr_m = mensual.index[mensual.index <= t - 7]
    row = {"mes_rank": t, "real": mensual.loc[t, "churn_rate"]}
    for name, cols in (("estacional", []), ("estacional+trends", top_terms)):
        X = month_design(cols)
        m = Ridge(alpha=1.0).fit(X.loc[tr_m], mensual.loc[tr_m, "churn_rate"])
        row[name] = float(m.predict(X.loc[[t]])[0])
    row["media_train"] = mensual.loc[tr_m, "churn_rate"].mean()
    fc.append(row)
fc = pd.DataFrame(fc)
rmse = {k: float(np.sqrt(((fc[k] - fc["real"]) ** 2).mean())) for k in ("media_train", "estacional", "estacional+trends")}
print("RMSE del pronóstico de la tasa mensual (22 meses):", {k: round(v, 4) for k, v in rmse.items()})
print(f"desvío de la tasa mensual real en esos meses: {fc['real'].std():.4f}")

# %% [markdown]
# ## 4. Nivel de fila: XGBoost base vs +Trends (OOT, rolling-origin y oráculo mensual)

# %%
X_tab = d[feats].reset_index(drop=True)
oracle = d.groupby("mes_rank")[TARGET].transform("mean").reset_index(drop=True)  # techo in-sample
VARIANTES = {
    "base": X_tab,
    "base+trends_catalogo": pd.concat([X_tab, X_gt[core_cols]], axis=1),
    "base+trends_todos": pd.concat([X_tab, X_gt], axis=1),
    "base+oraculo_mes": pd.concat([X_tab, oracle.rename("tasa_real_mes")], axis=1),
}
res, poots, oofs = [], {}, {}
for name, X in VARIANTES.items():
    m, oof, p = run_protocols(xgb_fp(X, y), y, g, mes, tr, te, gkf=True)
    poots[name], oofs[name] = p, oof
    res.append({"variante": name, "n_feats": X.shape[1], **m})
    print(f"{name:22s} GKF(optimista) {m['gkf_AUC']:.4f}  OOT {m['oot_AUC']:.4f}", flush=True)
res = pd.DataFrame(res)
print(fmt(res))
deltas = pd.DataFrame([{"variante": n, **{f"oot_{k}": v for k, v in paired_delta(y[te], poots["base"], poots[n]).items()}}
                       for n in VARIANTES if n != "base"])
print(deltas.round(4).to_markdown(index=False))

# %%
# Rolling-origin: 12 meses de test (ranks 100..111), train ≤ t-7. AUC por mes + AUC agrupado.
ROLL = list(range(mes.max() - 11, mes.max() + 1))
roll_rows, roll_pred = [], {n: np.full(len(d), np.nan) for n in VARIANTES}
for t in ROLL:
    tr_i, te_i = np.where(mes <= t - 7)[0], np.where(mes == t)[0]
    row = {"mes_rank": t, "n_test": len(te_i), "prev": y[te_i].mean()}
    for name, X in VARIANTES.items():
        p = xgb_fp(X, y)(tr_i, te_i)
        roll_pred[name][te_i] = p
        row[name] = roc_auc_score(y[te_i], p)
    roll_rows.append(row)
roll = pd.DataFrame(roll_rows)
print(roll.round(4).to_markdown(index=False))
mask_roll = np.isin(mes, ROLL)
resumen = pd.DataFrame({
    "AUC_medio_dentro_del_mes": roll[list(VARIANTES)].mean(),
    "AUC_agrupado_12m": {n: roc_auc_score(y[mask_roll], roll_pred[n][mask_roll]) for n in VARIANTES},
    "meses_gana_a_base": {n: int((roll[n] > roll["base"]).sum()) for n in VARIANTES},
})
resumen["Δ_agrupado_vs_base"] = resumen["AUC_agrupado_12m"] - resumen.loc["base", "AUC_agrupado_12m"]
resumen["Δ_dentro_mes_vs_base"] = resumen["AUC_medio_dentro_del_mes"] - resumen.loc["base", "AUC_medio_dentro_del_mes"]
print(resumen.round(4).to_markdown())

# %% [markdown]
# ## 5. Reporte

# %%
rs = res.set_index("variante")
txt = f"""# Opción C — Google Trends

> Generado por `08_experimentation/03_opcion_C_google_trends.ipynb` el {pd.Timestamp.now():%Y-%m-%d %H:%M}.
> {gt.shape[1] - 1} términos (geo=PE, mensual, {gt.index.min():%Y-%m} → {gt.index.max():%Y-%m}), {len(gt_cols)} features. OOT = {int(te.sum())} filas.

## Calidad de las series

{qual.to_markdown()}

`salto_2022_ratio` = media 2022 / media 2021 (cambio de metodología de Google en 2022-01).

## Nivel mensual

Mejor lag por término (Spearman con la tasa mensual de churn, meses de train, desestacionalizada):

{best.round(3).to_markdown(index=False)}

SE(rho) ≈ {1 / np.sqrt(len(train_months) - 3):.3f} con n = {len(train_months)} meses.

Pronóstico expansivo de la tasa mensual (22 meses, train ≤ t−7), RMSE:
{pd.Series(rmse).round(4).to_markdown()}
Desvío de la tasa real: {fc['real'].std():.4f}. Términos usados: {top_terms}.

## Nivel de fila

{fmt(res)}

ΔAUC OOT pareado vs base (IC 95 %):

{deltas.round(4).to_markdown(index=False)}

Rolling-origin (12 meses, train ≤ t−7):

{resumen.round(4).to_markdown()}

Detalle por mes:

{roll.round(4).to_markdown(index=False)}

## Lectura

- El oráculo mensual (`base+oraculo_mes`) marca el techo de cualquier variable constante por mes:
  OOT {rs.loc['base+oraculo_mes', 'oot_AUC']:.4f} vs base {rs.loc['base', 'oot_AUC']:.4f};
  agrupado 12 m {resumen.loc['base+oraculo_mes', 'AUC_agrupado_12m']:.4f} vs {resumen.loc['base', 'AUC_agrupado_12m']:.4f}
  y dentro del mes {resumen.loc['base+oraculo_mes', 'AUC_medio_dentro_del_mes']:.4f} vs {resumen.loc['base', 'AUC_medio_dentro_del_mes']:.4f}.
- Trends: OOT {rs.loc['base+trends_todos', 'oot_AUC']:.4f} (todos) / {rs.loc['base+trends_catalogo', 'oot_AUC']:.4f} (catálogo);
  rolling agrupado {resumen.loc['base+trends_todos', 'AUC_agrupado_12m']:.4f}, dentro del mes {resumen.loc['base+trends_todos', 'AUC_medio_dentro_del_mes']:.4f}.
- Una variable constante por mes solo puede mover el AUC agrupado (nivel del mes); si el AUC dentro
  del mes no cambia, no mejora la selección individual de vendedoras.
- GroupKFold con features mensuales es optimista (mismos meses en train y validación); se reporta pero no decide.
"""
save_report("opcion_C_google_trends", txt, res.merge(deltas, on="variante", how="left"))
