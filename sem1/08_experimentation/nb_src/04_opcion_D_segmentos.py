# %% [markdown]
# # Opción D — ¿Distintos tipos de vendedoras necesitan modelos distintos?
#
# **Pregunta.** El XGBoost global ya recibe las variables que definen los segmentos (tipo,
# antigüedad, frecuencia, monto, recencia). ¿Gana algo (a) hacer explícito el segmento,
# (b) entrenar un modelo especializado por segmento, o (c) partir del global y afinarlo por
# segmento (información compartida)? ¿Y en qué segmentos el modelo es débil?
#
# **Diseño.**
# - Segmentaciones con información disponible en t. De **negocio** (star schema
#   `dim_vendedor`/`dim_ubicacion` + panel): tipo de vendedora, si reporta a una Líder
#   (`ccodrelacion ≠ 0`), región, antigüedad de la relación (`compras_hist`), frecuencia
#   (`meses_activos_u12`), nivel de compra (`monto_u12`, terciles fijados en train), regularidad
#   (racha actual), reactivación (`meses_desde_compra_previa`). **Aprendidas**: KMeans (k=4)
#   sobre features de comportamiento estandarizadas, ajustado solo con filas de train y
#   asignado al resto por distancia (nunca se re-ajusta con test).
# - Cuatro enfoques por segmentación, con los mismos hiperparámetros tuneados:
#   `global` (modelo vigente, evaluado por segmento), `global+seg` (segmento como one-hot),
#   `especializado` (un XGBoost por segmento; si el segmento tiene < 800 filas de train se usa
#   el global), `fine-tune` (el booster global sigue entrenando 150 árboles con lr 0.01 solo
#   con las filas del segmento: comparte lo aprendido y especializa el margen).
# - Protocolos: GroupKFold(5) y OOT. Se reporta AUC agrupado (todas las filas) y por segmento.
#   Tamaño, prevalencia y estabilidad de cada segmento (transiciones mes a mes de la misma
#   vendedora y participación por año).

# %%
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(next(p for p in [Path.cwd(), Path.cwd() / "08_experimentation"] if (p / "exp_utils.py").exists())))
from exp_utils import (DIMV, PROC, RS, TARGET, load_base, make_model, oot_block,  # noqa: E402
                       paired_delta, save_report)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

# %%
d, feats, tr, te = load_base()
y, g, mes = d[TARGET].values, d["id_vendedor"].values, d["mes_rank"].values
X_tab = d[feats].reset_index(drop=True)
dv = pd.read_csv(DIMV).drop_duplicates("id_vendedor").set_index("id_vendedor")
raw = pd.read_csv(PROC / "churn_dataset.csv", usecols=["id_vendedor", "mes_rank", "departamento", "tipo_vendedor"])
raw[["departamento", "tipo_vendedor"]] = raw[["departamento", "tipo_vendedor"]].fillna("DESCONOCIDO")
assert (raw["id_vendedor"].values == g).all() and (raw["mes_rank"].values == mes).all()
ritmo_path = PROC / "exp_B_features_ritmo.csv"
ritmo = pd.read_csv(ritmo_path) if ritmo_path.exists() else None

# %% [markdown]
# ## 1. Segmentaciones

# %%
def cortes_train(s, q=(1 / 3, 2 / 3)):
    """Terciles fijados con filas de train; etiquetas bajo/medio/alto."""
    c = s[tr].quantile(q).values
    return pd.Series(np.select([s <= c[0], s <= c[1]], ["bajo", "medio"], "alto"), index=s.index)


SEG = {}
SEG["tipo_vendedor"] = raw["tipo_vendedor"].values
tiene_lider = (dv["ccodrelacion"].reindex(g).fillna(0).values != 0)
SEG["red"] = np.where(raw["tipo_vendedor"] == "Líder", "Líder",
                      np.where(tiene_lider, "Asesora con líder", "Asesora sin líder"))
SEG["region"] = np.where(raw["departamento"].str.lower().isin(["lima", "callao"]), "Lima-Callao", "Provincias")
SEG["antiguedad_relacion"] = pd.cut(d["compras_hist"], [0, 2, 9, np.inf], labels=["1-2 compras", "3-9", "10+"]).astype(str).values
SEG["frecuencia_u12"] = pd.cut(d["meses_activos_u12"], [0, 3, 8, 12], labels=["1-3 meses", "4-8", "9-12"]).astype(str).values
SEG["nivel_compra_u12"] = cortes_train(d["monto_u12"]).values
SEG["reactivacion"] = pd.cut(d["meses_desde_compra_previa"], [0, 1, 3, np.inf], labels=["continua (gap 1)", "pausa corta (2-3)", "reactivada (4+)"]).astype(str).values
if ritmo is not None:
    SEG["regularidad_racha"] = pd.cut(ritmo["r_racha_actual"], [0, 1, 3, np.inf], labels=["racha 1", "racha 2-3", "racha 4+"]).astype(str).values

# aprendida: KMeans sobre comportamiento (fit en train, asignación por distancia)
beh = ["meses_activos_u12", "n_ped_u12", "monto_u12", "compras_hist", "meses_desde_compra_previa",
       "camp_saltadas", "antiguedad_meses", "monto_acum", "tasa_camp_u12"]
Zb = np.log1p(d[beh].clip(lower=0).values)
sc = StandardScaler().fit(Zb[tr])
km = KMeans(n_clusters=4, n_init=10, random_state=RS).fit(sc.transform(Zb[tr]))
lab = km.predict(sc.transform(Zb))
# nombres legibles por el perfil de cada cluster (media de meses activos y monto)
prof = pd.DataFrame(Zb, columns=beh).assign(k=lab).groupby("k").mean()
orden = prof.sort_values(["meses_activos_u12", "monto_u12"]).index.tolist()
nombres = {k: f"C{i}" for i, k in enumerate(orden)}
SEG["kmeans4"] = np.array([nombres[k] for k in lab], dtype=object)
print("Perfil de los clusters KMeans (medias en log1p):")
print(pd.DataFrame(Zb, columns=beh).assign(k=SEG["kmeans4"]).groupby("k").mean().round(2).to_markdown())

tam = []
for s, v in SEG.items():
    vc = pd.Series(v).astype(str)
    for val in sorted(vc.unique()):
        m = vc.values == val
        tam.append({"segmentacion": s, "segmento": val, "n_rows": int(m.sum()), "n_train": int((m & tr).sum()),
                    "n_oot": int((m & te).sum()), "prev": y[m].mean(), "n_vend": len(np.unique(g[m]))})
tam = pd.DataFrame(tam)
print(tam.round(3).to_markdown(index=False))

# %% [markdown]
# ## 2. Estabilidad: ¿una vendedora cambia de segmento entre meses consecutivos? ¿cambia el mix por año?

# %%
orden_idx = np.lexsort((mes, g))
stab = []
for s, v in SEG.items():
    vs, gs, ms = v[orden_idx], g[orden_idx], mes[orden_idx]
    consec = (gs[1:] == gs[:-1]) & (ms[1:] - ms[:-1] <= 3)   # misma vendedora, observaciones a ≤ 3 meses
    permanece = (vs[1:] == vs[:-1])[consec].mean()
    anio = pd.to_datetime(d["mes_obs"]).dt.year.values
    mix = pd.crosstab(anio, v, normalize="index")
    drift = float(mix.diff().abs().sum(axis=1).mean())
    stab.append({"segmentacion": s, "n_segmentos": len(np.unique(v)), "permanece_≤3m": permanece,
                 "cambio_mix_anual_medio": drift, "min_share_oot": pd.Series(v[te]).value_counts(normalize=True).min()})
stab = pd.DataFrame(stab)
print(stab.round(3).to_markdown(index=False))

# %% [markdown]
# ## 3. Enfoques: global, global+segmento, especializado, fine-tune

# %%
MIN_TRAIN = 800
FT_TREES, FT_LR = 150, 0.01


_GLOBAL_CACHE = {}


def fit_global(Xtr, ytr, key=None):
    """XGBoost tuneado; con `key` (índices de train) cachea el global, que es el mismo para
    todas las segmentaciones y enfoques del mismo fold."""
    if key is None:
        return make_model("XGBoost", ytr).fit(Xtr, ytr)
    k = (len(key), int(key[0]), int(key[-1]), int(key.sum()))
    if k not in _GLOBAL_CACHE:
        _GLOBAL_CACHE[k] = make_model("XGBoost", ytr).fit(Xtr, ytr)
    return _GLOBAL_CACHE[k]


def predict_approach(approach, seg, tr_i, te_i, Xb):
    """Devuelve p_te para el enfoque. `seg` vector de etiquetas de segmento (todas las filas)."""
    str_, ste = seg[tr_i], seg[te_i]
    if approach == "global+seg":
        Xs = pd.concat([Xb, pd.get_dummies(pd.Series(seg), prefix="seg", dtype=int)], axis=1)
        return fit_global(Xs.iloc[tr_i], y[tr_i]).predict_proba(Xs.iloc[te_i])[:, 1]
    glob = fit_global(Xb.iloc[tr_i], y[tr_i], key=tr_i)
    p = glob.predict_proba(Xb.iloc[te_i])[:, 1]
    if approach == "global":
        return p
    for val in np.unique(ste):
        mtr, mte = str_ == val, ste == val
        if mtr.sum() < MIN_TRAIN or len(np.unique(y[tr_i][mtr])) < 2:
            continue  # segmento chico: se queda con el global
        if approach == "especializado":
            m = fit_global(Xb.iloc[tr_i[mtr]], y[tr_i][mtr])
        else:  # fine-tune: continúa el booster global sobre el segmento
            m = make_model("XGBoost", y[tr_i][mtr]).set_params(n_estimators=FT_TREES, learning_rate=FT_LR)
            m.fit(Xb.iloc[tr_i[mtr]], y[tr_i][mtr], xgb_model=glob.get_booster())
        p[mte] = m.predict_proba(Xb.iloc[te_i[mte]])[:, 1]
    return p


APPROACHES = ["global", "global+seg", "especializado", "fine-tune"]
folds = list(GroupKFold(5).split(X_tab, y, g))
tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]

res, per_seg, preds = [], [], {}
for sname, seg in SEG.items():
    for ap in APPROACHES:
        oof = np.zeros(len(y))
        for a, b in folds:
            oof[b] = predict_approach(ap, seg, a, b, X_tab)
        p = predict_approach(ap, seg, tr_idx, te_idx, X_tab)
        preds[(sname, ap)] = (oof, p)
        m = {"gkf_AUC": roc_auc_score(y, oof), **oot_block(y[te], p, mes[te])}
        res.append({"segmentacion": sname, "enfoque": ap, **m})
        for val in np.unique(seg):
            mk, mo = seg == val, seg[te] == val
            per_seg.append({"segmentacion": sname, "segmento": str(val), "enfoque": ap,
                            "AUC_gkf": roc_auc_score(y[mk], oof[mk]) if 0 < y[mk].mean() < 1 else np.nan,
                            "AUC_oot": roc_auc_score(y[te][mo], p[mo]) if mo.sum() > 20 and 0 < y[te][mo].mean() < 1 else np.nan})
        print(f"{sname:20s} {ap:14s} GKF {m['gkf_AUC']:.4f}  OOT {m['oot_AUC']:.4f}", flush=True)
res = pd.DataFrame(res)
per_seg = pd.DataFrame(per_seg)

# %%
cols = ["segmentacion", "enfoque", "gkf_AUC", "oot_AUC", "oot_AUCstd", "oot_PRAUC", "oot_prec10", "oot_rec30", "oot_brier"]
print(res[cols].round(4).to_markdown(index=False))

# ΔAUC pareado de cada enfoque vs global (misma segmentación)
deltas = []
for sname in SEG:
    oof_g, p_g = preds[(sname, "global")]
    for ap in APPROACHES[1:]:
        oof_a, p_a = preds[(sname, ap)]
        dg, do = paired_delta(y, oof_g, oof_a), paired_delta(y[te], p_g, p_a)
        deltas.append({"segmentacion": sname, "enfoque": ap, "ΔAUC_gkf": dg["delta"], "gkf_lo": dg["ci_lo"], "gkf_hi": dg["ci_hi"],
                       "ΔAUC_oot": do["delta"], "oot_lo": do["ci_lo"], "oot_hi": do["ci_hi"]})
deltas = pd.DataFrame(deltas)
print(deltas.round(4).to_markdown(index=False))

# %% [markdown]
# ## 4. ¿Dónde es débil el modelo global? AUC por segmento (global vs mejor enfoque)

# %%
piv = per_seg.pivot_table(index=["segmentacion", "segmento"], columns="enfoque", values=["AUC_gkf", "AUC_oot"])
piv.columns = [f"{a}_{b}" for a, b in piv.columns]
piv = piv.join(tam.set_index(["segmentacion", "segmento"])[["n_rows", "n_oot", "prev"]])
piv["mejor_gkf"] = piv[[f"AUC_gkf_{a}" for a in APPROACHES]].idxmax(axis=1).str.replace("AUC_gkf_", "")
piv["Δ_mejor_vs_global_gkf"] = piv[[f"AUC_gkf_{a}" for a in APPROACHES]].max(axis=1) - piv["AUC_gkf_global"]
print(piv.round(3).to_markdown())

# %% [markdown]
# ## 5. Reporte

# %%
best_rows = res.sort_values("gkf_AUC", ascending=False).head(5)
glob_ref = res[res["enfoque"] == "global"].iloc[0]
sig = deltas[(deltas["gkf_lo"] > 0) & (deltas["ΔAUC_oot"] > 0)]
txt = f"""# Opción D — Modelos por segmento de vendedoras

> Generado por `08_experimentation/04_opcion_D_segmentos.ipynb` el {pd.Timestamp.now():%Y-%m-%d %H:%M}.
> {len(SEG)} segmentaciones × {len(APPROACHES)} enfoques. Global de referencia: GKF {glob_ref['gkf_AUC']:.4f} / OOT {glob_ref['oot_AUC']:.4f}.
> Segmento con < {MIN_TRAIN} filas de train usa el global en `especializado` y `fine-tune`.

## Tamaño y prevalencia por segmento

{tam.round(3).to_markdown(index=False)}

## Estabilidad

{stab.round(3).to_markdown(index=False)}

`permanece_≤3m`: fracción de pares de observaciones de la misma vendedora (≤ 3 meses de distancia) que quedan en el mismo segmento.
`cambio_mix_anual_medio`: suma de |Δ share| entre años consecutivos (0 = mix constante).

## AUC agrupado por segmentación y enfoque

{res[cols].round(4).to_markdown(index=False)}

## ΔAUC pareado vs global (IC 95 % bootstrap)

{deltas.round(4).to_markdown(index=False)}

Enfoques con IC GKF > 0 **y** ΔOOT > 0: {len(sig)} de {len(deltas)}{': ' + ', '.join(sig['segmentacion'] + '/' + sig['enfoque']) if len(sig) else ''}.

## AUC por segmento (global vs enfoques)

{piv.round(3).to_markdown()}

## Lectura

- Top 5 por AUC GKF: {', '.join(f"{r.segmentacion}/{r.enfoque} {r.gkf_AUC:.4f}" for r in best_rows.itertuples())}.
- Los segmentos con menor AUC del global señalan dónde falta información, no necesariamente dónde
  un modelo aparte ayuda: comparar `Δ_mejor_vs_global_gkf` con el IC de la tabla de deltas.
- El segmento debe ser conocido en t: todas las segmentaciones usan solo columnas ≤ t y los cortes /
  KMeans se fijan con train.
"""
save_report("opcion_D_segmentos", txt, res.merge(deltas, on=["segmentacion", "enfoque"], how="left"))
per_seg.to_csv(PROC.parent.parent / "08_experimentation" / "reports" / "opcion_D_segmentos_por_segmento.csv", index=False)
