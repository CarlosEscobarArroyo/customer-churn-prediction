# %% [markdown]
# # Opción A — LSTM como entrada para XGBoost
#
# **Pregunta.** ¿Una representación aprendida del historial mensual de compras y campañas
# (LSTM) aporta información que las 91 features tabulares del modelo final no capturan?
#
# **Diseño.**
# - Secuencia por fila `(id_vendedor, t)`: los últimos **24 meses** del panel denso
#   (`data/processed/panel_denso_mensual.csv`, CTE `panel` de `qry_churn.sql`), 6 canales
#   (`activo`, `log monto`, `n_ped`, `log n_prod`, `campañas participadas`, `campañas disponibles`)
#   + máscara de relleno. Solo información ≤ t.
# - Tres formas de usar la LSTM, comparadas contra el XGBoost vigente y contra una
#   alternativa sencilla (los 24 meses "aplanados" como columnas):
#   1. **LSTM-churn** entrenada con la etiqueta a 6 meses → su probabilidad entra a XGBoost
#      como *stacking* honesto (OOF interno por vendedora dentro de train).
#   2. **LSTM-aux** auto-supervisada sobre el panel denso (tarea auxiliar: ¿compra el mes
#      siguiente? ¿compra en los próximos 3?) → su estado oculto (32 dims) entra a XGBoost.
#      No ve la etiqueta de churn y se entrena con muchas más filas (meses activos e inactivos).
#   3. **LSTM-churn embedding** (estado oculto de la red supervisada, ajustado in-sample):
#      se reporta como sensibilidad porque el árbol puede sobre-confiar en un embedding que
#      ya vio la etiqueta de las mismas filas.
# - Protocolos idénticos al resto del repo: GroupKFold(5) por vendedora y OOT (train ≤ rank 101,
#   gap 6, test = ranks 108–111). Toda red, escala y árbol se ajusta solo con el train del fold.
# - Horizonte final: 6 meses (sin cambios).
#
# **Lectura de resultados.** ΔAUC con IC bootstrap pareado; el OOT tiene 885 filas
# (SE ≈ 0.02), así que solo el GroupKFold (30 821 filas) discrimina diferencias < 0.01.

# %%
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(next(p for p in [Path.cwd(), Path.cwd() / "08_experimentation"] if (p / "exp_utils.py").exists())))
from exp_utils import (L_SEQ, PROC, RS, TARGET, PanelCube, fmt, load_base, load_panel,  # noqa: E402
                       lstm_embed, lstm_prob, paired_delta, run_protocols, save_report,
                       train_lstm, xgb_fp)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

# %% [markdown]
# ## 1. Datos: tabla vigente + secuencias del panel denso

# %%
d, feats, tr, te = load_base()
y, g, mes = d[TARGET].values, d["id_vendedor"].values, d["mes_rank"].values
panel = load_panel()
cube = PanelCube(panel)
X_seq = cube.window(d["id_vendedor"].values, d["mes_rank"].values, L=L_SEQ)
n_ch = X_seq.shape[-1]
print(f"filas {len(d):,} | features tabulares {len(feats)} | secuencias {X_seq.shape} (L={L_SEQ}, canales={n_ch} incl. máscara)")
print(f"panel denso: {len(panel):,} filas vendedora×mes, {panel['id_vendedor'].nunique():,} vendedoras, ranks 1..{cube.R}")
# sanity: la última posición es t (compró en t) y los 12 últimos activos coinciden con meses_activos_u12
assert (X_seq[:, -1, 0] == 1).all()
assert np.allclose(X_seq[:, -12:, 0].sum(1), d["meses_activos_u12"].values)
cob = X_seq[..., -1].mean(1)
print(f"cobertura media de la ventana de 24 meses: {cob.mean():.2f} (filas con historia completa: {(cob == 1).mean():.1%})")

# %% [markdown]
# ## 2. Variantes
#
# | id | modelo | entrada |
# |---|---|---|
# | `base` | XGBoost tuneado | 91 features |
# | `flat24` | XGBoost | 91 + 24×2 columnas (activo, log monto de cada uno de los 24 meses) |
# | `lstm_solo` | LSTM-churn (end-to-end) | secuencia |
# | `stack_lstm` | XGBoost | 91 + p(LSTM-churn) (OOF interno 3-fold por vendedora) |
# | `emb_aux` | XGBoost | 91 + 32 dims de la LSTM auxiliar (panel denso, sin etiqueta) |
# | `emb_churn` | XGBoost | 91 + 32 dims de la LSTM-churn (in-sample, sensibilidad) |
# | `todo` | XGBoost | 91 + flat24 + p(LSTM-churn) + emb_aux |
# | `rank_avg` | promedio de rangos | base y lstm_solo (¿complementarios?) |

# %%
X_tab = d[feats].reset_index(drop=True)
flat_cols = [f"m{L_SEQ - i}_{c}" for i in range(L_SEQ) for c in ("activo", "log_monto")]
X_flat = pd.DataFrame(X_seq[:, :, :2].reshape(len(d), -1), columns=flat_cols)
X_tabflat = pd.concat([X_tab, X_flat], axis=1)

AUX_N = 120_000      # filas del panel denso muestreadas para la tarea auxiliar (por fold)
HID = 32


def aux_dataset(train_idx, seed=RS):
    """Filas (vendedora, mes) del panel denso de las vendedoras de train, con mes ≤ max rank
    de train + 3 (los targets auxiliares t+1..t+3 no pasan del gap del protocolo).
    Incluye meses SIN compra: la LSTM auxiliar ve la dinámica completa, no solo los meses activos."""
    ids_tr = np.unique(g[train_idx])
    max_r = int(mes[train_idx].max()) + 3
    sub = panel[panel["id_vendedor"].isin(ids_tr) & (panel["mes_rank"] <= max_r)]
    sub = sub[sub["mes_rank"] > sub.groupby("id_vendedor")["mes_rank"].transform("min")]  # ≥1 mes de historia
    sub = sub.sample(min(AUX_N, len(sub)), random_state=seed)
    vi = cube.pos[sub["id_vendedor"]].values
    r = sub["mes_rank"].values
    act = cube.M["activo"]
    y1 = act[vi, r + 1]
    y3 = np.maximum.reduce([act[vi, r + k] for k in (1, 2, 3)])
    Xa = cube.window(sub["id_vendedor"].values, r, L=L_SEQ)
    return Xa, np.c_[y1, y3], sub["id_vendedor"].values


def stack_prob(train_idx, test_idx, seed=RS):
    """p(LSTM-churn) OOF en train (GroupKFold interno de 3) y promedio de las 3 redes en test."""
    p_tr, p_te = np.zeros(len(train_idx)), np.zeros(len(test_idx))
    for k, (a, b) in enumerate(GroupKFold(3).split(train_idx, groups=g[train_idx])):
        net = train_lstm(X_seq[train_idx[a]], y[train_idx[a]], g[train_idx[a]], hidden=HID, seed=seed + k)
        p_tr[b] = lstm_prob(net, X_seq[train_idx[b]])
        p_te += lstm_prob(net, X_seq[test_idx]) / 3
    return p_tr, p_te


def fp_xgb_plus(extra_fn, base=X_tab):
    """XGBoost sobre base + columnas extra calculadas por `extra_fn(tr, te) -> (E_tr, E_te)`."""
    def fp(tr_idx, te_idx):
        E_tr, E_te = extra_fn(tr_idx, te_idx)
        cols = [f"x{i}" for i in range(E_tr.shape[1])]
        Xtr = pd.concat([base.iloc[tr_idx].reset_index(drop=True), pd.DataFrame(E_tr, columns=cols)], axis=1)
        Xte = pd.concat([base.iloc[te_idx].reset_index(drop=True), pd.DataFrame(E_te, columns=cols)], axis=1)
        Xall = pd.concat([Xtr, Xte]).reset_index(drop=True)
        yy = np.r_[y[tr_idx], y[te_idx]]
        return xgb_fp(Xall, yy)(np.arange(len(tr_idx)), np.arange(len(tr_idx), len(yy)))
    return fp


def extra_stack(tr_idx, te_idx):
    p_tr, p_te = stack_prob(tr_idx, te_idx)
    return p_tr[:, None], p_te[:, None]


def extra_aux(tr_idx, te_idx):
    Xa, Ya, ga = aux_dataset(tr_idx)
    net = train_lstm(Xa, Ya, ga, hidden=HID, epochs=12)
    return lstm_embed(net, X_seq[tr_idx]), lstm_embed(net, X_seq[te_idx])


def extra_emb_churn(tr_idx, te_idx):
    net = train_lstm(X_seq[tr_idx], y[tr_idx], g[tr_idx], hidden=HID)
    return lstm_embed(net, X_seq[tr_idx]), lstm_embed(net, X_seq[te_idx])


def extra_todo(tr_idx, te_idx):
    s_tr, s_te = extra_stack(tr_idx, te_idx)
    a_tr, a_te = extra_aux(tr_idx, te_idx)
    return np.c_[s_tr, a_tr], np.c_[s_te, a_te]


def fp_lstm_solo(tr_idx, te_idx):
    return lstm_prob(train_lstm(X_seq[tr_idx], y[tr_idx], g[tr_idx], hidden=HID), X_seq[te_idx])


VARIANTES = {
    "base": xgb_fp(X_tab, y),
    "flat24": xgb_fp(X_tabflat, y),
    "lstm_solo": fp_lstm_solo,
    "stack_lstm": fp_xgb_plus(extra_stack),
    "emb_aux": fp_xgb_plus(extra_aux),
    "emb_churn": fp_xgb_plus(extra_emb_churn),
    "todo": fp_xgb_plus(extra_todo, base=X_tabflat),
}

# %% [markdown]
# ## 3. Ejecución (GroupKFold + OOT por variante)

# %%
res, oofs, poots = [], {}, {}
for name, fp in VARIANTES.items():
    t0 = time.time()
    m, oof, p = run_protocols(fp, y, g, mes, tr, te)
    oofs[name], poots[name] = oof, p
    res.append({"variante": name, **m, "seg": round(time.time() - t0)})
    print(f"{name:12s} GKF {m['gkf_AUC']:.4f}  OOT {m['oot_AUC']:.4f}  ({res[-1]['seg']} s)", flush=True)

# promedio de rangos base + LSTM (complementariedad sin re-entrenar)
from scipy.stats import rankdata  # noqa: E402
rk = lambda a: rankdata(a) / len(a)  # noqa: E731
oof_ra = (rk(oofs["base"]) + rk(oofs["lstm_solo"])) / 2
p_ra = (rk(poots["base"]) + rk(poots["lstm_solo"])) / 2
from exp_utils import lift10, oot_block  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
res.append({"variante": "rank_avg", "gkf_AUC": roc_auc_score(y, oof_ra), "gkf_AUC_std": np.nan,
            "gkf_PRAUC": average_precision_score(y, oof_ra), "gkf_lift10": lift10(y, oof_ra),
            **oot_block(y[te], p_ra, mes[te]), "seg": 0})
oofs["rank_avg"], poots["rank_avg"] = oof_ra, p_ra
res = pd.DataFrame(res)
print(fmt(res))

# %% [markdown]
# ## 4. ¿Las diferencias son reales? ΔAUC pareado vs `base` (IC 95 % bootstrap)

# %%
deltas = []
for name in res["variante"]:
    if name == "base":
        continue
    dg = paired_delta(y, oofs["base"], oofs[name])
    do = paired_delta(y[te], poots["base"], poots[name])
    deltas.append({"variante": name, "ΔAUC_gkf": dg["delta"], "gkf_lo": dg["ci_lo"], "gkf_hi": dg["ci_hi"],
                   "ΔAUC_oot": do["delta"], "oot_lo": do["ci_lo"], "oot_hi": do["ci_hi"]})
deltas = pd.DataFrame(deltas)
print(deltas.round(4).to_markdown(index=False))

# %% [markdown]
# ## 5. ¿Información complementaria o redundante?
#
# - Correlación de Spearman entre p(base) y p(LSTM) en OOF y OOT.
# - Solapamiento del top 10 % de riesgo entre ambos modelos.
# - AUC in-sample del XGBoost con `emb_churn` (si sube mucho vs. base, el embedding
#   sobre-ajusta y el árbol lo sobre-confía: justifica preferir `stack_lstm`/`emb_aux`).

# %%
from scipy.stats import spearmanr  # noqa: E402

comp = {}
for proto, P in (("gkf", oofs), ("oot", poots)):
    a, b = P["base"], P["lstm_solo"]
    n = max(len(a) // 10, 1)
    top_a, top_b = set(np.argsort(-a)[:n]), set(np.argsort(-b)[:n])
    comp[proto] = {"spearman_base_lstm": spearmanr(a, b).correlation,
                   "overlap_top10": len(top_a & top_b) / n}
comp = pd.DataFrame(comp).T
print(comp.round(3).to_markdown())

# in-sample: base vs emb_churn sobre el bloque de train OOT
tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
E_tr, E_te = extra_emb_churn(tr_idx, te_idx)
from exp_utils import make_model  # noqa: E402
Xtr_e = np.c_[X_tab.iloc[tr_idx].values, E_tr]
m_e = make_model("XGBoost", y[tr_idx]).fit(Xtr_e, y[tr_idx])
m_b = make_model("XGBoost", y[tr_idx]).fit(X_tab.iloc[tr_idx], y[tr_idx])
insample = {"base_train_AUC": roc_auc_score(y[tr_idx], m_b.predict_proba(X_tab.iloc[tr_idx])[:, 1]),
            "emb_churn_train_AUC": roc_auc_score(y[tr_idx], m_e.predict_proba(Xtr_e)[:, 1]),
            "emb_churn_gain_share": float(m_e.feature_importances_[len(feats):].sum())}
print(insample)

# %% [markdown]
# ## 6. Salidas para la Opción B (coordinación: no repetir el aprendizaje de secuencias)
#
# Se guardan, en el orden de filas de `churn_dataset.csv`, la p(LSTM-churn) OOF/OOT y el
# embedding auxiliar del bloque OOT, para que el notebook de ritmo/pausas pueda comparar
# "features de ritmo hechas a mano" vs "representación aprendida" sin re-entrenar redes.

# %%
Xa, Ya, ga = aux_dataset(tr_idx)
net_aux = train_lstm(Xa, Ya, ga, hidden=HID, epochs=12)
emb_aux_all = lstm_embed(net_aux, X_seq)  # red ajustada con train OOT; válida para filas de test OOT
np.savez(PROC / "exp_A_lstm_outputs.npz", oof_lstm=oofs["lstm_solo"], p_oot_lstm=poots["lstm_solo"],
         oof_stack=oofs["stack_lstm"], p_oot_stack=poots["stack_lstm"], emb_aux_oot=emb_aux_all,
         flat24=X_flat.values, flat_cols=np.array(flat_cols), id_vendedor=g, mes_rank=mes)
print("→", PROC / "exp_A_lstm_outputs.npz")

# %% [markdown]
# ## 7. Reporte

# %%
best = res.sort_values("gkf_AUC", ascending=False).iloc[0]
base_row = res.set_index("variante").loc["base"]
txt = f"""# Opción A — LSTM como entrada para XGBoost

> Generado por `08_experimentation/01_opcion_A_lstm_xgboost.ipynb` el {pd.Timestamp.now():%Y-%m-%d %H:%M}.
> {len(d):,} filas, {len(feats)} features tabulares, secuencias de {L_SEQ} meses × {n_ch - 1} canales.
> OOT = {int(te.sum())} filas (ranks {mes[te].min()}–{mes[te].max()}), prevalencia {y[te].mean():.3f}.

## Resultados

{fmt(res)}

## ΔAUC pareado vs base (IC 95 % bootstrap, 1000 réplicas)

{deltas.round(4).to_markdown(index=False)}

## Complementariedad base ↔ LSTM

{comp.round(3).to_markdown()}

In-sample (bloque train OOT): AUC train base {insample['base_train_AUC']:.4f} vs con `emb_churn`
{insample['emb_churn_train_AUC']:.4f}; el embedding concentra {insample['emb_churn_gain_share']:.1%} del gain.

## Lectura

- Mejor GroupKFold: `{best['variante']}` {best['gkf_AUC']:.4f} vs base {base_row['gkf_AUC']:.4f}
  (OOT {best['oot_AUC']:.4f} vs {base_row['oot_AUC']:.4f}).
- La LSTM sola alcanza {res.set_index('variante').loc['lstm_solo', 'gkf_AUC']:.4f} GKF /
  {res.set_index('variante').loc['lstm_solo', 'oot_AUC']:.4f} OOT con solo 6 canales mensuales:
  la secuencia contiene casi la misma información que las 91 features (lo mismo se ve en `flat24`).
- Criterio de decisión: una variante "aporta" si su ΔAUC GKF tiene IC que excluye 0 **y** el ΔAUC OOT
  tiene el mismo signo. Ver tabla de deltas.
"""
save_report("opcion_A_lstm", txt, res.merge(deltas, on="variante", how="left"))
