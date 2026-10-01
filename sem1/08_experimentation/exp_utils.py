"""Utilidades compartidas de 08_experimentation.

Reutiliza 05_modelling/experimentos/pipeline.py (split OOT, preprocessing fit-en-train,
XGBoost tuneado sin Optuna) y agrega lo que necesitan las cuatro líneas:
  * `load_base()`     dataset vigente ya preparado (mismas 91 features del modelo final tras preprocessing).
  * `PanelCube`       panel denso mensual (vendedora × mes) como matrices por canal;
                      `window()` devuelve la secuencia de los últimos L meses de cada fila.
  * `run_protocols()` GroupKFold(5) + OOT con métricas extendidas (AUC, PR-AUC, precisión
                      y cobertura en el top 10 % / 30 %, std mensual, Brier, ECE) y las
                      predicciones (para bootstrap pareado entre variantes).
  * `paired_delta()`  IC bootstrap del ΔAUC entre dos modelos sobre las mismas filas.
  * LSTM (torch)      `train_lstm()`, `lstm_embed()`, `lstm_prob()`.

Convención temporal: todo lo que se ajusta (escalas, KMeans, LSTM, XGBoost) se ajusta
SOLO con las filas de train del protocolo en curso (fold o bloque OOT).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "05_modelling" / "experimentos"))
from pipeline import (ID, PARAMS, PROC, RS, SQL_PATH, TARGET, lift10, make_model,  # noqa: E402,F401
                      oot_split, prepare)

REPORTS = HERE / "reports"
PANEL = PROC / "panel_denso_mensual.csv"
DIMV = PROC / "dim_vendedor.csv"
EXTERNAL = BASE / "data" / "external"
L_SEQ = 24
SEQ_CH = ["activo", "log_monto", "n_ped", "log_n_prod", "n_camp_part", "n_camp_disp"]
METRIC_COLS = ["gkf_AUC", "gkf_AUC_std", "gkf_PRAUC", "gkf_lift10", "oot_AUC", "oot_AUCstd",
               "oot_PRAUC", "oot_prec10", "oot_rec10", "oot_prec30", "oot_rec30",
               "oot_brier", "oot_ece"]


# --- datos --------------------------------------------------------------------
def load_base():
    """Dataset vigente preparado como en 02→04. Devuelve (d, feats, train_mask, test_mask)."""
    df = pd.read_csv(PROC / "churn_dataset.csv", parse_dates=["mes_obs"])
    tr, te = oot_split(df["mes_rank"])
    d, feats = prepare(df, tr)
    return d, feats, tr, te


def load_panel():
    """Panel denso (CTE `panel` de qry_churn.sql): una fila por vendedora × mes desde su
    primer mes con compra. Extraído a data/processed/panel_denso_mensual.csv."""
    p = pd.read_csv(PANEL, parse_dates=["mes_obs"])
    p["camp_saltadas"] = p["camp_saltadas"].astype("float")
    return p.sort_values(["id_vendedor", "mes_rank"]).reset_index(drop=True)


class PanelCube:
    """Matrices (n_vendedoras × max_rank + 1) por canal; columna 0 = relleno (todo 0)."""

    def __init__(self, panel):
        self.ids = np.sort(panel["id_vendedor"].unique())
        self.pos = pd.Series(np.arange(len(self.ids)), index=self.ids)
        self.R = int(panel["mes_rank"].max())
        vi = self.pos[panel["id_vendedor"]].values
        ri = panel["mes_rank"].values
        shape = (len(self.ids), self.R + 1)
        src = {"activo": panel["activo"], "log_monto": np.log1p(panel["monto"].clip(lower=0)),
               "n_ped": panel["n_ped"].clip(upper=10), "log_n_prod": np.log1p(panel["n_prod"]),
               "n_camp_part": panel["n_camp_part"].clip(upper=5),
               "n_camp_disp": panel["n_camp_disp"].clip(upper=5),
               "monto": panel["monto"].clip(lower=0)}
        self.M = {}
        for k, v in src.items():
            m = np.zeros(shape, dtype=np.float32)
            m[vi, ri] = v.values
            self.M[k] = m
        self.present = np.zeros(shape, dtype=np.float32)
        self.present[vi, ri] = 1.0
        self.primer = pd.Series(panel.groupby("id_vendedor")["mes_rank"].min().reindex(self.ids).values,
                                index=self.ids)

    def window(self, ids, ranks, L=L_SEQ, channels=SEQ_CH):
        """Secuencia [t-L+1 .. t] de cada (id, t). Devuelve X (n, L, C+1): último canal = mask
        (1 si el mes existe en el panel, 0 si es relleno anterior al primer mes)."""
        vi = self.pos[np.asarray(ids)].values[:, None]
        idx = np.asarray(ranks)[:, None] - np.arange(L - 1, -1, -1)[None, :]
        idx = np.clip(idx, 0, self.R)
        X = np.stack([self.M[c][vi, idx] for c in channels] + [self.present[vi, idx]], axis=-1)
        X[..., :-1] *= X[..., -1:]  # rellenos a 0 (idx clip a 0 ya es 0, pero por si acaso)
        return X.astype(np.float32)

    def series(self, id_, upto, channel="activo"):
        """Vector histórico [primer_mes .. upto] de un canal para una vendedora."""
        return self.M[channel][self.pos[id_], self.primer[id_]: upto + 1]


# --- métricas -----------------------------------------------------------------
def topk(yt, p, frac):
    n = max(int(round(len(yt) * frac)), 1)
    idx = np.argsort(-p)[:n]
    return float(yt[idx].mean()), float(yt[idx].sum() / max(yt.sum(), 1))


def ece(y, p, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def oot_block(yt, p, mt):
    aucs = [roc_auc_score(yt[mt == mm], p[mt == mm]) for mm in np.unique(mt)
            if 0 < yt[mt == mm].mean() < 1]
    p10, r10 = topk(yt, p, 0.10)
    p30, r30 = topk(yt, p, 0.30)
    return {"oot_AUC": roc_auc_score(yt, p), "oot_AUCstd": float(np.std(aucs)),
            "oot_PRAUC": average_precision_score(yt, p),
            "oot_prec10": p10, "oot_rec10": r10, "oot_prec30": p30, "oot_rec30": r30,
            "oot_brier": brier_score_loss(yt, p), "oot_ece": ece(yt, p),
            "oot_lift10": lift10(yt, p)}


def run_protocols(fit_predict, y, groups, mes, train_mask, test_mask, gkf=True):
    """`fit_predict(tr_idx, te_idx) -> p_te`. Devuelve (métricas, oof, p_oot).
    Con gkf=False solo corre el bloque OOT (para features constantes por mes, donde el
    GroupKFold mezcla meses entre train y validación y es optimista)."""
    y = np.asarray(y)
    out, oof = {}, np.full(len(y), np.nan)
    if gkf:
        fold_auc = []
        for tr, va in GroupKFold(5).split(np.zeros(len(y)), y, groups):
            oof[va] = fit_predict(tr, va)
            fold_auc.append(roc_auc_score(y[va], oof[va]))
        out.update({"gkf_AUC": roc_auc_score(y, oof), "gkf_AUC_std": float(np.std(fold_auc)),
                    "gkf_PRAUC": average_precision_score(y, oof), "gkf_lift10": lift10(y, oof)})
    p = fit_predict(np.where(train_mask)[0], np.where(test_mask)[0])
    out.update(oot_block(y[test_mask], p, np.asarray(mes)[test_mask]))
    return out, oof, p


def xgb_fp(X, y, params=None):
    """fit_predict de XGBoost tuneado sobre la matriz X (DataFrame)."""
    def fp(tr, te):
        m = make_model("XGBoost", y[tr])
        if params:
            m.set_params(**params)
        m.fit(X.iloc[tr], y[tr])
        return m.predict_proba(X.iloc[te])[:, 1]
    return fp


def paired_delta(y, p_a, p_b, n_boot=1000, seed=RS, metric=roc_auc_score):
    """ΔAUC (b − a) con IC 95 % bootstrap pareado sobre las mismas filas."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    ok = ~(np.isnan(p_a) | np.isnan(p_b))
    y, p_a, p_b = y[ok], p_a[ok], p_b[ok]
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].mean() < 1:
            d.append(metric(y[i], p_b[i]) - metric(y[i], p_a[i]))
    d = np.array(d)
    return {"delta": float(metric(y, p_b) - metric(y, p_a)),
            "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5))}


def fmt(df, cols=None, nd=4):
    cols = cols or [c for c in METRIC_COLS if c in df.columns]
    return df[[c for c in df.columns if c not in cols] + cols].round(nd).to_markdown(index=False)


def save_report(name, text, df=None):
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{name}.md").write_text(text)
    if df is not None:
        df.to_csv(REPORTS / f"{name}.csv", index=False)
    print(f"→ {REPORTS / name}.md")


# --- LSTM ---------------------------------------------------------------------
def _torch():
    # ponytail: en macOS torch y xgboost traen cada uno su libomp y se bloquean si ambos
    # paralelizan. xgboost ya está importado (pipeline); torch queda en 1 hilo. Si se corre
    # en Linux/GPU se puede subir con torch.set_num_threads(n) después de importar.
    import xgboost  # noqa: F401  (debe cargarse ANTES que torch)
    import torch
    torch.set_num_threads(1)
    return torch


class _SeqNet:
    """Wrapper mínimo: LSTM sobre secuencia (n, L, C) → último estado oculto → cabezal lineal."""

    def __init__(self, n_in, n_out, hidden=32, seed=RS):
        torch = _torch()
        torch.manual_seed(seed)
        nn = torch.nn
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)
        self.mu = np.zeros(n_in, dtype=np.float32)
        self.sd = np.ones(n_in, dtype=np.float32)
        self.hidden = hidden

    def params(self):
        return list(self.lstm.parameters()) + list(self.head.parameters())

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1], self.head(h[-1])


def _scale(net, X):
    torch = _torch()
    Xs = (X - net.mu) / net.sd
    Xs[..., -1] = X[..., -1]  # la máscara no se escala
    return torch.from_numpy(Xs.astype(np.float32))


def train_lstm(X, Y, groups, hidden=32, epochs=15, lr=2e-3, batch=512, seed=RS,
               pos_weight=True, val_frac=0.1, verbose=False):
    """Entrena una LSTM con salidas sigmoide (BCE) sobre X (n, L, C). `Y` (n, k) binaria.
    Early stopping por pérdida en un 10 % de vendedoras apartadas (GroupShuffle por grupo).
    Escala por canal ajustada SOLO con X (train). Devuelve la red."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=np.float32).reshape(len(X), -1)
    net = _SeqNet(X.shape[-1], Y.shape[1], hidden, seed)
    flat = X.reshape(-1, X.shape[-1])[X[..., -1].reshape(-1) > 0]
    net.mu, net.sd = flat.mean(0), flat.std(0) + 1e-6
    ug = np.unique(groups)
    val_g = set(rng.choice(ug, int(len(ug) * val_frac), replace=False))
    is_val = np.array([g in val_g for g in groups])
    Xt, Yt = _scale(net, X), torch.from_numpy(Y)
    pw = torch.tensor(((1 - Y[~is_val]).sum(0) / np.maximum(Y[~is_val].sum(0), 1)) if pos_weight
                      else np.ones(Y.shape[1]), dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(net.params(), lr=lr, weight_decay=1e-5)
    tr_idx, va_idx = np.where(~is_val)[0], np.where(is_val)[0]
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.lstm.train()
        rng.shuffle(tr_idx)
        for i in range(0, len(tr_idx), batch):
            b = tr_idx[i:i + batch]
            opt.zero_grad()
            _, logit = net.forward(Xt[b])
            loss = loss_fn(logit, Yt[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.params(), 1.0)
            opt.step()
        net.lstm.eval()
        with torch.no_grad():
            vl = float(loss_fn(net.forward(Xt[va_idx])[1], Yt[va_idx]))
        if verbose:
            print(f"  ep {ep + 1:2d} val_loss {vl:.4f}")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = [p.detach().clone() for p in net.params()]
        else:
            bad += 1
            if bad >= 3:
                break
    with torch.no_grad():
        for p, s in zip(net.params(), best_state):
            p.copy_(s)
    net.lstm.eval()
    return net


def _infer(net, X, batch=4096):
    torch = _torch()
    Xt = _scale(net, X)
    H, Lg = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            h, lg = net.forward(Xt[i:i + batch])
            H.append(h.numpy())
            Lg.append(lg.numpy())
    return np.vstack(H), np.vstack(Lg)


def lstm_embed(net, X):
    return _infer(net, X)[0]


def lstm_prob(net, X, col=0):
    return 1 / (1 + np.exp(-_infer(net, X)[1][:, col]))
