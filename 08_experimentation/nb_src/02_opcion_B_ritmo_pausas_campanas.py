# %% [markdown]
# # Opción B — Ritmo individual, pausas, rachas y dinámica de campañas
#
# **Pregunta.** Las features vigentes resumen la actividad en ventanas fijas (u3/u6/u12,
# acumuladas, deltas). ¿Aporta algo describir el **ritmo propio** de cada vendedora
# (cada cuánto compra, cuánto se desvía hoy de ese ritmo), su **historial de pausas y
# retornos**, sus **rachas** y su **secuencia de participación en campañas**?
#
# **Diseño.**
# - Todas las features se calculan sobre el panel denso con la historia `[primer mes .. t]`
#   de cada fila (nada posterior a t). Sin ajuste en train: son estadísticos individuales
#   (los dos estimadores "encogidos" usan constantes fijas, no aprendidas).
# - Cuatro familias (prefijo `r_`): **ritmo** (gaps entre compras), **pausas/reactivaciones**,
#   **rachas**, **campañas**; más dos estimadores del **tiempo hasta la próxima compra**
#   basados en la distribución individual de gaps (`r_p_gap_le6`, `r_esp_gap`).
# - El modelado explícito del tiempo hasta la próxima compra ya se hizo en
#   `05_modelling/experimentos/09_supervivencia.py` (XGBoost Cox AUC@6 OOT 0.7616 vs
#   clasificador 0.7638: empate); aquí no se repite, se usa como referencia.
# - Coordinación con la Opción A: las variantes combinadas usan las salidas guardadas por
#   el notebook A (`data/processed/exp_A_lstm_outputs.npz`) en vez de re-entrenar redes.
#   `flat24` (24 meses aplanados) es dato crudo y se evalúa en ambos protocolos; las salidas
#   aprendidas (p(LSTM), embedding auxiliar) se ajustaron con el train OOT, así que solo se
#   combinan en el protocolo OOT.

# %%
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(next(p for p in [Path.cwd(), Path.cwd() / "08_experimentation"] if (p / "exp_utils.py").exists())))
from exp_utils import (PROC, TARGET, PanelCube, fmt, load_base, load_panel, make_model,  # noqa: E402
                       oot_block, paired_delta, run_protocols, save_report, xgb_fp)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

# %%
d, feats, tr, te = load_base()
y, g, mes = d[TARGET].values, d["id_vendedor"].values, d["mes_rank"].values
panel = load_panel()
cube = PanelCube(panel)
# canal extra: campañas saltadas al iniciar una campaña nueva ese mes (NaN si no inició ninguna)
M_salt = np.full_like(cube.present, np.nan)
M_salt[cube.pos[panel["id_vendedor"]].values, panel["mes_rank"].values] = panel["camp_saltadas"].values

# %% [markdown]
# ## 1. Features de ritmo, pausas, rachas y campañas (historia ≤ t)

# %%
PRIOR_N = 3        # peso del prior en los estimadores encogidos (constante, no ajustada)
PRIOR_P6 = 0.7     # prior de "el siguiente gap es ≤ 6 meses"
PRIOR_GAP = 2.5    # prior del gap medio (meses)
PAUSA_LARGA = 4    # gap ≥ 4 meses = pausa larga (volver tras ella = reactivación)


def runs(a):
    """Longitudes de rachas de 1s en un vector binario."""
    if a.sum() == 0:
        return np.array([0])
    edges = np.diff(np.r_[0, a, 0])
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def ritmo_feats(vid, t):
    pos, p0 = cube.pos[vid], int(cube.primer[vid])
    act = cube.M["activo"][pos, p0:t + 1]
    cp, cd = cube.M["n_camp_part"][pos, p0:t + 1], cube.M["n_camp_disp"][pos, p0:t + 1]
    salt = M_salt[pos, p0:t + 1]
    ranks = np.flatnonzero(act) + p0                 # meses con compra (incluye t)
    gaps = np.diff(ranks)                            # ≥ 1; len = compras_hist
    hab = gaps[:-1]                                  # ritmo habitual = gaps ANTES del que entra a t
    last = gaps[-1]
    f = {}
    # --- ritmo
    f["r_gap_mean"], f["r_gap_median"], f["r_gap_max"] = gaps.mean(), np.median(gaps), gaps.max()
    f["r_gap_std"] = gaps.std() if len(gaps) > 1 else np.nan
    f["r_gap_cv"] = f["r_gap_std"] / f["r_gap_mean"] if len(gaps) > 1 else np.nan
    f["r_gap_hab_mean"] = hab.mean() if len(hab) else np.nan
    f["r_gap_last_vs_hab"] = last / hab.mean() if len(hab) else np.nan
    f["r_gap_last_z"] = (last - hab.mean()) / (hab.std() + 0.5) if len(hab) > 1 else np.nan
    f["r_gap_trend"] = gaps[-3:].mean() - gaps[:-3].mean() if len(gaps) > 3 else np.nan
    # --- tiempo hasta la próxima compra (estimadores individuales encogidos)
    n = len(gaps)
    f["r_p_gap_le6"] = ((gaps <= 6).sum() + PRIOR_N * PRIOR_P6) / (n + PRIOR_N)
    f["r_p_gap_le3"] = ((gaps <= 3).sum() + PRIOR_N * 0.5) / (n + PRIOR_N)
    f["r_esp_gap"] = (gaps.sum() + PRIOR_N * PRIOR_GAP) / (n + PRIOR_N)
    # --- pausas y reactivaciones
    f["r_n_pausas"] = int((gaps >= 2).sum())
    f["r_n_pausas_largas"] = int((gaps >= PAUSA_LARGA).sum())
    f["r_frac_pausas"] = f["r_n_pausas"] / n
    f["r_es_reactivacion"] = int(last >= PAUSA_LARGA)
    react = ranks[1:][gaps >= PAUSA_LARGA]
    f["r_meses_desde_reactivacion"] = t - react[-1] if len(react) else np.nan
    f["r_densidad_vida"] = act.mean()
    f["r_densidad_u12_vs_vida"] = act[-12:].mean() - act.mean()
    # --- rachas
    r = runs(act)
    f["r_racha_actual"] = int(r[-1])
    f["r_racha_max"], f["r_n_rachas"] = int(r.max()), int(len(r))
    f["r_racha_actual_vs_max"] = r[-1] / r.max()
    # --- campañas
    f["r_camp_tasa_vida"] = cp.sum() / cd.sum() if cd.sum() > 0 else np.nan
    disp = cd > 0
    cpd = cp[disp]
    f["r_camp_racha"] = int(runs((cpd > 0).astype(float))[-1]) if len(cpd) else 0
    f["r_camp_tasa_u6_vs_vida"] = (cp[-6:].sum() / cd[-6:].sum() if cd[-6:].sum() > 0 else np.nan) - f["r_camp_tasa_vida"]
    s_hist = salt[:-1][~np.isnan(salt[:-1])]
    f["r_camp_saltadas_media"] = s_hist.mean() if len(s_hist) else np.nan
    f["r_camp_saltadas_max"] = s_hist.max() if len(s_hist) else np.nan
    f["r_camp_saltadas_delta"] = (salt[-1] if not np.isnan(salt[-1]) else 0) - (s_hist.mean() if len(s_hist) else 0)
    return f


R = pd.DataFrame([ritmo_feats(v, t) for v, t in zip(d["id_vendedor"].values, d["mes_rank"].values)])
rcols = list(R.columns)
# sanity: gap que entra a t = meses_desde_compra_previa; nº de gaps = compras_hist
assert np.allclose(R["r_gap_last_vs_hab"].fillna(0) * R["r_gap_hab_mean"].fillna(0),
                   d["meses_desde_compra_previa"].values * R["r_gap_hab_mean"].notna())
print(f"{len(rcols)} features de ritmo; NaN por columna (sin historia suficiente):")
print(R.isna().mean().round(3).sort_values(ascending=False).head(8).to_string())
print(R.describe().T[["mean", "50%", "min", "max"]].round(2).to_string())

# %% [markdown]
# ## 2. ¿Qué información nueva traen? Redundancia con las features vigentes

# %%
X_tab = d[feats].reset_index(drop=True)
red = []
for c in rcols:
    ok = R[c].notna()
    cors = {f: abs(spearmanr(R.loc[ok, c], X_tab.loc[ok, f]).correlation) for f in feats
            if X_tab[f].nunique() > 2}
    fbest = max(cors, key=cors.get)
    red.append({"feature": c, "max_|rho|_vigente": cors[fbest], "con": fbest,
                "AUC_univariado": max(roc_auc_score(y[ok], R.loc[ok, c]), 1 - roc_auc_score(y[ok], R.loc[ok, c]))})
red = pd.DataFrame(red).sort_values("max_|rho|_vigente")
print(red.round(3).to_markdown(index=False))

# %% [markdown]
# ## 3. Variantes y protocolos
#
# | id | entrada de XGBoost |
# |---|---|
# | `base` | 91 features vigentes |
# | `ritmo_solo` | solo las features `r_*` |
# | `base+ritmo` | 91 + `r_*` |
# | `base+flat24` | 91 + 24 meses aplanados (referencia de la Opción A) |
# | `base+ritmo+flat24` | ¿el ritmo agrega sobre el detalle mensual crudo? |
# | `base+ritmo+lstm` (solo OOT) | 91 + `r_*` + p(LSTM-churn) + embedding auxiliar de la Opción A |

# %%
A = np.load(PROC / "exp_A_lstm_outputs.npz") if (PROC / "exp_A_lstm_outputs.npz").exists() else None
if A is not None:
    assert (A["id_vendedor"] == g).all() and (A["mes_rank"] == mes).all()
    X_flat = pd.DataFrame(A["flat24"], columns=A["flat_cols"])
    X_lstm = pd.DataFrame(np.c_[A["emb_aux_oot"]], columns=[f"emb{i}" for i in range(A["emb_aux_oot"].shape[1])])
    # p_lstm: en train OOT usamos la OOF del GroupKFold de A (redes que no vieron esas filas);
    # en test OOT, la p del stacking OOT de A
    p_te_full = np.full(len(d), np.nan)
    p_te_full[te] = A["p_oot_stack"]
    X_lstm["p_lstm"] = np.where(te, p_te_full, A["oof_stack"])
else:
    print("Sin salidas de la Opción A: correr antes 01_opcion_A_lstm_xgboost.ipynb")

VARIANTES = {
    "base": (X_tab, True),
    "ritmo_solo": (R, True),
    "base+ritmo": (pd.concat([X_tab, R], axis=1), True),
}
if A is not None:
    VARIANTES["base+flat24"] = (pd.concat([X_tab, X_flat], axis=1), True)
    VARIANTES["base+ritmo+flat24"] = (pd.concat([X_tab, R, X_flat], axis=1), True)
    VARIANTES["base+ritmo+lstm"] = (pd.concat([X_tab, R, X_lstm], axis=1), False)

res, oofs, poots = [], {}, {}
for name, (X, gkf) in VARIANTES.items():
    m, oof, p = run_protocols(xgb_fp(X, y), y, g, mes, tr, te, gkf=gkf)
    oofs[name], poots[name] = oof, p
    res.append({"variante": name, "n_feats": X.shape[1], **m})
    print(f"{name:20s} GKF {m.get('gkf_AUC', float('nan')):.4f}  OOT {m['oot_AUC']:.4f}", flush=True)
res = pd.DataFrame(res)
print(fmt(res))

# %%
deltas = []
for name in res["variante"]:
    if name == "base":
        continue
    row = {"variante": name}
    if not np.isnan(oofs[name]).all():
        dg = paired_delta(y, oofs["base"], oofs[name])
        row.update({"ΔAUC_gkf": dg["delta"], "gkf_lo": dg["ci_lo"], "gkf_hi": dg["ci_hi"]})
    do = paired_delta(y[te], poots["base"], poots[name])
    row.update({"ΔAUC_oot": do["delta"], "oot_lo": do["ci_lo"], "oot_hi": do["ci_hi"]})
    deltas.append(row)
deltas = pd.DataFrame(deltas)
print(deltas.round(4).to_markdown(index=False))

# %% [markdown]
# ## 4. Importancia por permutación de las features de ritmo (modelo `base+ritmo`, bloque OOT)

# %%
Xbr = VARIANTES["base+ritmo"][0]
mdl = make_model("XGBoost", y[tr]).fit(Xbr[tr], y[tr])
pi = permutation_importance(mdl, Xbr[te], y[te], scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=1)
imp = pd.DataFrame({"feature": Xbr.columns, "ΔAUC_perm": pi.importances_mean, "std": pi.importances_std})
imp["familia"] = np.where(imp["feature"].str.startswith("r_"), "ritmo (nueva)", "vigente")
print(imp.sort_values("ΔAUC_perm", ascending=False).head(15).round(4).to_markdown(index=False))
gain = pd.Series(mdl.feature_importances_, index=Xbr.columns)
print(f"\ngain acumulado de las features r_*: {gain[rcols].sum():.1%} ({len(rcols)} de {len(Xbr.columns)} columnas)")
print("top ritmo por gain:", gain[rcols].sort_values(ascending=False).head(6).round(3).to_dict())

# %% [markdown]
# ## 5. Tiempo hasta la próxima compra: ¿cuánto predice el ritmo individual por sí solo?

# %%
from scipy.stats import rankdata  # noqa: E402


def _rk(a):
    """Escala a (0, 1] por rango: oot_block exige un score tipo probabilidad (Brier no aplica aquí)."""
    return rankdata(a) / len(a)


ttnp = pd.DataFrame([
    {"score": "r_p_gap_le6 (share hist. de gaps ≤ 6, encogido)", "AUC_gkf(all)": 1 - roc_auc_score(y, R["r_p_gap_le6"]),
     **{k: v for k, v in oot_block(y[te], _rk(-R.loc[te, "r_p_gap_le6"].values), mes[te]).items() if k in ("oot_AUC", "oot_prec10", "oot_rec30")}},
    {"score": "r_esp_gap (gap medio encogido)", "AUC_gkf(all)": roc_auc_score(y, R["r_esp_gap"]),
     **{k: v for k, v in oot_block(y[te], _rk(R.loc[te, "r_esp_gap"].values), mes[te]).items() if k in ("oot_AUC", "oot_prec10", "oot_rec30")}},
    {"score": "meses_activos_u12 (vigente, referencia)", "AUC_gkf(all)": 1 - roc_auc_score(y, d["meses_activos_u12"]),
     **{k: v for k, v in oot_block(y[te], _rk(-d.loc[te, "meses_activos_u12"].values.astype(float)), mes[te]).items() if k in ("oot_AUC", "oot_prec10", "oot_rec30")}},
    {"score": "XGBoost Cox (09_supervivencia, referencia)", "AUC_gkf(all)": 0.7511, "oot_AUC": 0.7616, "oot_prec10": np.nan, "oot_rec30": np.nan},
])
print(ttnp.round(4).to_markdown(index=False))

# %% [markdown]
# ## 6. Reporte

# %%
rs = res.set_index("variante")
txt = f"""# Opción B — Ritmo individual, pausas, rachas y campañas

> Generado por `08_experimentation/02_opcion_B_ritmo_pausas_campanas.ipynb` el {pd.Timestamp.now():%Y-%m-%d %H:%M}.
> {len(rcols)} features `r_*` calculadas sobre el panel denso con historia ≤ t. OOT = {int(te.sum())} filas.

## Resultados

{fmt(res)}

## ΔAUC pareado vs base (IC 95 % bootstrap)

{deltas.round(4).to_markdown(index=False)}

## Redundancia con las features vigentes (|rho| de Spearman máxima) y AUC univariado

{red.round(3).to_markdown(index=False)}

## Importancia por permutación (OOT, modelo base+ritmo): top 15

{imp.sort_values("ΔAUC_perm", ascending=False).head(15).round(4).to_markdown(index=False)}

Gain acumulado de `r_*`: {gain[rcols].sum():.1%}.

## Tiempo hasta la próxima compra con el ritmo individual

{ttnp.round(4).to_markdown(index=False)}

## Lectura

- `base+ritmo`: GKF {rs.loc['base+ritmo', 'gkf_AUC']:.4f} vs base {rs.loc['base', 'gkf_AUC']:.4f};
  OOT {rs.loc['base+ritmo', 'oot_AUC']:.4f} vs {rs.loc['base', 'oot_AUC']:.4f}.
- `ritmo_solo` ({len(rcols)} columnas) llega a {rs.loc['ritmo_solo', 'gkf_AUC']:.4f} GKF: el ritmo resume
  la mayor parte de la señal que ya tienen las 91 features (ver redundancia).
- Las variantes con salidas de la Opción A (`flat24`, `lstm`) muestran si el ritmo a mano agrega algo
  sobre el detalle mensual crudo o aprendido: ver deltas.
"""
save_report("opcion_B_ritmo", txt, res.merge(deltas, on="variante", how="left"))
R.assign(id_vendedor=g, mes_rank=mes).to_csv(PROC / "exp_B_features_ritmo.csv", index=False)
print("→", PROC / "exp_B_features_ritmo.csv")
