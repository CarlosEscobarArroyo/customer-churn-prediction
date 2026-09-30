"""Temporal ablation of lagged department sales activity, excluding the focal seller.

Uses the complete local monthly purchase panel, never churn labels, to construct
context. Model selection remains exploratory temporal validation, not final OOT.
"""
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from scripts.climate_trends_ablation import summarize
from scripts.external_features_ablation import BASE_REPORT, PARAMS, ExternalFeatures
from scripts.master_features_ablation import development_data, normalize_category
from scripts.recent_windows import recent_training_indices
from scripts.temporal_optuna import META, ROOT, build_model, digest, write_json

PANEL = ROOT / "data/processed/panel_denso_mensual.csv"
MASTER = ROOT / "data/processed/dim_vendedor.csv"
MEASURES = {"monto": "ventas", "n_ped": "pedidos", "activo": "vendedoras_activas"}
HORIZONS = (0, 6, 12)


def department_key(s):
    return s.map(normalize_category).replace({"__sin_dato__": None, "sin departamento": None})


def prepare_activity(panel, master, end):
    """Complete calendar rolling means, retaining genuine zero-activity months."""
    p = panel[["id_vendedor", "mes_obs", *MEASURES]].copy()
    p["mes_obs"] = pd.to_datetime(p.mes_obs)
    if p.duplicated(["id_vendedor", "mes_obs"]).any():
        raise ValueError("Duplicate seller/month source keys")
    if master.id_vendedor.duplicated().any():
        raise ValueError("Duplicate master seller keys")
    if p.mes_obs.isna().any() or not p.mes_obs.dt.is_month_start.all():
        raise ValueError("Source months must be first-of-month dates")
    if not np.isfinite(p[list(MEASURES)]).all().all() or (p[list(MEASURES)] < 0).any().any():
        raise ValueError("Activity must be finite and nonnegative")
    if not p.activo.isin([0, 1]).all() or not p.activo.eq(p.n_ped.gt(0)).all():
        raise ValueError("Activity indicator does not agree with orders")
    p = p.loc[p.mes_obs <= pd.Timestamp(end)].sort_values(["id_vendedor", "mes_obs"]).reset_index(drop=True)
    if p.empty or p.mes_obs.max() != pd.Timestamp(end):
        raise ValueError("Purchase panel does not cover required reference month")
    calendar = pd.date_range(p.mes_obs.min(), end, freq="MS")
    p["month_number"] = p.mes_obs.dt.year * 12 + p.mes_obs.dt.month
    bounds = p.groupby("id_vendedor").month_number.agg(["min", "max", "size"])
    if not bounds["max"].eq(p.month_number.max()).all() or not bounds["size"].eq(bounds["max"] - bounds["min"] + 1).all():
        raise ValueError("Purchase panel must be dense from first purchase through reference end")
    m = master[["id_vendedor", "departamento"]].copy()
    m["department_key"] = department_key(m.departamento)
    p = p.merge(m[["id_vendedor", "department_key"]], on="id_vendedor", how="left", validate="many_to_one")
    quality = {"source_rows_through_reference": len(p), "source_sellers": int(p.id_vendedor.nunique()),
               "source_first_month": str(calendar.min().date()), "source_last_month": str(calendar.max().date()),
               "rows_without_department": int(p.department_key.isna().sum()),
               "active_seller_months_without_department": int(p.loc[p.department_key.isna(), "activo"].sum()),
               "sales_without_department": float(p.loc[p.department_key.isna(), "monto"].sum()),
               "total_sales": float(p.monto.sum())}
    regions = sorted(m.department_key.dropna().unique())
    index = pd.MultiIndex.from_product([regions, calendar], names=["department_key", "mes_obs"])
    regional = p.groupby(["department_key", "mes_obs"])[list(MEASURES)].sum().reindex(index, fill_value=0).reset_index()
    for col in MEASURES:
        regional[f"mean3_{col}"] = regional.groupby("department_key")[col].transform(
            lambda x: x.rolling(3, min_periods=3).mean())
    # Before a seller's first observed purchase, their contribution is zero.
    # Division by 3 is deliberate even for a newly observed seller's first month.
    own = p[["id_vendedor", "mes_obs"]].copy()
    rolled = p.groupby("id_vendedor")[list(MEASURES)].rolling(3, min_periods=1).sum().droplevel(0) / 3
    for col in MEASURES:
        own[f"own3_{col}"] = rolled[col]
    return regional, own, quality


def attach_activity(pool, regional, own):
    d = pool.reset_index(drop=True).copy()
    d["department_key"] = department_key(d.departamento)
    for h in HORIZONS:
        ref = f"regional_ref_{h}"
        d[ref] = pd.to_datetime(d.mes_obs) - pd.DateOffset(months=h + 1)
        r = regional[["department_key", "mes_obs"] + [f"mean3_{c}" for c in MEASURES]].rename(columns={"mes_obs": ref})
        o = own.rename(columns={"mes_obs": ref})
        d = d.merge(r, on=["department_key", ref], how="left", sort=False, validate="many_to_one")
        d = d.merge(o, on=["id_vendedor", ref], how="left", sort=False, validate="many_to_one")
        peers = []
        for col, label in MEASURES.items():
            name = f"regional_{label}_mean3_h{h}"
            d[name] = d[f"mean3_{col}"] - d[f"own3_{col}"].fillna(0)
            if d[name].min() < -1e-7:
                raise ValueError("Own activity exceeds department total")
            d[name] = d[name].clip(lower=0)
            d[f"regional_log_{label}_h{h}"] = np.log1p(d[name])
            peers.append(name)
        d[f"regional_sin_historia_h{h}"] = d[peers].isna().any(axis=1).astype(float)
        d = d.drop(columns=[f"{prefix}_{col}" for prefix in ("mean3", "own3") for col in MEASURES])
    for h in (6, 12):
        for label in MEASURES.values():
            now, then = d[f"regional_{label}_mean3_h0"], d[f"regional_{label}_mean3_h{h}"]
            denominator = now + then
            change = (now - then).div(denominator.where(denominator.ne(0)))
            d[f"regional_cambio_{label}_{h}m"] = change.mask(denominator.eq(0), 0)
    pd.testing.assert_frame_equal(d[pool.columns], pool.reset_index(drop=True))
    return d


def variants():
    def levels(h):
        return tuple(f"regional_log_{label}_h{h}" for label in MEASURES.values())
    def changes(h):
        return tuple(f"regional_cambio_{label}_{h}m" for label in MEASURES.values())
    def flags(*hs):
        return tuple(f"regional_sin_historia_h{h}" for h in hs)
    return {"transaccional": (), "mas_departamento": (),
            "actividad_reciente": levels(0) + flags(0),
            "actividad_6m": levels(0) + levels(6) + changes(6) + flags(0, 6),
            "actividad_12m": levels(0) + levels(12) + changes(12) + flags(0, 12),
            "actividad_6m_12m": levels(0) + levels(6) + levels(12) + changes(6) + changes(12) + flags(0, 6, 12)}


def run(output=ROOT / "reports/regional_activity_v1", threads=4):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    pool, folds = development_data()
    base_protocol = json.loads((BASE_REPORT / "protocol.json").read_text())
    versions = {p: importlib.metadata.version(p) for p in base_protocol["versions"]}
    parameters = {n: digest(PARAMS / f"{n}_best.json") for n in ("logreg", "xgboost")}
    assert base_protocol["raw_sha256"] == digest(ROOT / "data/processed/churn_dataset.csv")
    assert base_protocol["parameter_files"] == parameters and base_protocol["versions"] == versions
    for key, file in (("code_sha256", "master_features_ablation"), ("temporal_code_sha256", "temporal_optuna"), ("ensemble_code_sha256", "recent_windows")):
        assert base_protocol[key] == digest(ROOT / f"scripts/{file}.py")
    config = variants()
    protocol = {"code_sha256": digest(__file__), "raw_sha256": base_protocol["raw_sha256"],
                "sources": {str(p.relative_to(ROOT)): digest(p) for p in (PANEL, MASTER)},
                "dependencies": {f: digest(ROOT / f) for f in (
                    "scripts/master_features_ablation.py", "scripts/temporal_optuna.py",
                    "scripts/recent_windows.py", "scripts/external_features_ablation.py", "scripts/climate_trends_ablation.py")},
                "parameters": parameters, "versions": versions, "threads": threads,
                "training_end": "2025-01-01", "gap_months": 6,
                "variants": {k: list(v) for k, v in config.items()},
                "cached_references": {f"{v}/{i}": digest(BASE_REPORT / f"predictions_{v}_fold{i}.parquet") for v in ("transaccional", "mas_departamento") for i in range(4)},
                "window": "mean of 3 complete months; endpoints t-1, t-7, t-13",
                "leave_one_seller_out": True, "labels_used_in_features": False,
                "location_caveat": "current department snapshot, historical moves unavailable",
                "status": "development temporal validation reused for selection; not final independent OOT"}
    if (out / "protocol.json").exists() and json.loads((out / "protocol.json").read_text()) != protocol:
        raise ValueError("Changed protocol: choose a new output directory")
    panel = pd.read_csv(PANEL, parse_dates=["mes_obs"], usecols=["id_vendedor", "mes_obs", *MEASURES])
    master = pd.read_csv(MASTER, usecols=["id_vendedor", "departamento"])
    # Verify this panel and geography describe the same development snapshot.
    aligned = pool.merge(panel, on=["id_vendedor", "mes_obs"], how="left", validate="one_to_one")
    assert aligned.activo.eq(1).all()
    # SQL SAFE_DIVIDE returns NULL when the historical mean is zero; the
    # nonnegative monthly amounts must then also be zero.
    reconstructed_amount = (aligned.monto_mean_u12 * aligned.monto_ult_vs_media).mask(aligned.monto_mean_u12.eq(0), 0)
    np.testing.assert_allclose(aligned.monto, reconstructed_amount, atol=1e-7)
    matched = pool.merge(master, on="id_vendedor", how="left", suffixes=("", "_master"), validate="many_to_one")
    assert matched.departamento.map(normalize_category).eq(matched.departamento_master.map(normalize_category)).all()
    regional, own, quality = prepare_activity(panel, master, pool.mes_obs.max() - pd.DateOffset(months=1))
    joined = attach_activity(pool, regional, own)
    coverage = []
    for partition, idx in (("development", np.arange(len(joined))), ("validation", np.concatenate([v for _, v in folds]))):
        d = joined.iloc[idx]
        coverage.append({"partition": partition, "rows": len(d), "missing_department": int(d.department_key.isna().sum()),
                         **{f"missing_window_h{h}": int(d[f"regional_sin_historia_h{h}"].sum()) for h in HORIZONS}})
    write_json(out / "protocol.json", protocol)
    write_json(out / "source_quality.json", quality)
    write_json(out / "variables.json", {"variants": config, "department_control": "cached original one-hot department variant",
        "levels": "log1p(mean monthly department activity over 3 months, excluding focal seller)",
        "changes": "(recent - historic)/(recent + historic); both zero -> 0; missing -> missing",
        "horizons": {str(h): f"t-{h+3} through t-{h+1}" for h in HORIZONS},
        "imputation": "median fitted within each training component; missing window indicators"})
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    regional.to_parquet(out / "department_month_activity.parquet", index=False)
    joined.to_parquet(out / "joined_development.parquet", index=False)
    params = {n: json.loads((PARAMS / f"{n}_best.json").read_text())["params"] for n in ("logreg", "xgboost")}
    write_json(out / "parameters.json", params)
    summaries, blocks_all, months_all, components = [], [], [], []
    for variant, columns in config.items():
        chunks = []
        for i, (tr, va) in enumerate(folds):
            expected = joined.iloc[va][META].reset_index(drop=True).assign(fold=i)
            file = out / f"predictions_{variant}_fold{i}.parquet"
            source = BASE_REPORT / file.name if not columns else file
            if source.exists():
                predicted = pd.read_parquet(source)
                pd.testing.assert_frame_equal(predicted[expected.columns], expected)
            else:
                scores = []
                for name, months in (("logreg", 36), ("xgboost", None)):
                    a = joined.iloc[recent_training_indices(joined, tr, months)]
                    assert a.mes_rank.max() + 6 < joined.iloc[va].mes_rank.min()
                    model = make_pipeline(ExternalFeatures(columns), SimpleImputer(strategy="median", keep_empty_features=True),
                                          build_model(name, params[name], a.churn.to_numpy(), threads))
                    print(f"Training {variant}/{name}/fold{i}", flush=True)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(a, a.churn)
                    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                        raise RuntimeError(f"Convergence failure: {variant}/{name}/fold{i}")
                    scores.append(model.predict_proba(joined.iloc[va])[:, 1])
                predicted = expected.assign(score_logreg=scores[0], score_xgboost=scores[1], score=(scores[0]+scores[1])/2)
            np.testing.assert_allclose(predicted.score, (predicted.score_logreg+predicted.score_xgboost)/2)
            assert predicted.score.between(0, 1).all() and np.isfinite(predicted.score).all()
            if not file.exists():
                predicted.to_parquet(file, index=False)
            chunks.append(predicted)
        prediction = pd.concat(chunks, ignore_index=True)
        summary, blocks, monthly = summarize(prediction)
        summaries.append({"variant": variant, "n_added_numeric_columns": len(columns), **summary})
        blocks_all.append(blocks.assign(variant=variant))
        months_all.append(monthly.assign(variant=variant))
        for name in ("logreg", "xgboost"):
            s, _, _ = summarize(prediction.assign(score=prediction[f"score_{name}"]))
            components.append({"variant": variant, "model": name, **s})
        print(f"RESULT {variant}: AUC={summary['auc_mean']:.6f}, monthly={summary['auc_monthly_mean']:.6f}, TP10={summary['tp_top10']}", flush=True)
    comparison = pd.DataFrame(summaries)
    for metric in ("auc_mean", "auc_monthly_mean", "tp_top10", "tp_top20"):
        comparison[f"delta_{metric}"] = comparison[metric] - comparison.loc[0, metric]
    comparison.to_csv(out / "comparison.csv", index=False)
    pd.concat(blocks_all).to_csv(out / "block_metrics.csv", index=False)
    pd.concat(months_all).to_csv(out / "monthly_metrics.csv", index=False)
    pd.DataFrame(components).to_csv(out / "component_metrics.csv", index=False)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/regional_activity_v1")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Positive thread count required")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(run(args.output, args.threads).to_string(index=False))


if __name__ == "__main__":
    main()
