"""Fixed-parameter temporal ablation of local NASA POWER and Google Trends data.

Offline exploratory comparison: downloaded snapshots are not historical vintages.
No regional Trends values without monthly dates enter the backtest.
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
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from scripts.external_features_ablation import BASE_REPORT, PARAMS, ExternalFeatures
from scripts.master_features_ablation import development_data, score_metrics
from scripts.nasa_power import AGGREGATIONS, attach_climate
from scripts.recent_windows import recent_training_indices
from scripts.temporal_optuna import META, ROOT, build_model, digest, write_json

NASA = ROOT / "data/external/nasa_power"
TRENDS = ROOT / "data/external/google_trends"
TERMS = (
    "venta por catalogo", "catalogo", "catalogo de ropa", "vender por catalogo",
    "glamour", "yanbal", "natura", "unique", "esika", "belcorp", "shein",
    "temu", "gamarra", "ropa por mayor", "trabajo desde casa", "trabajo",
    "prestamo", "ofertas", "ropa", "cts",
)
CLIMATE = tuple(f"nasa_{c}" for c in AGGREGATIONS) + (
    "nasa_temperatura_delta12_c", "nasa_precipitacion_delta12_mm", "nasa_sin_dato",
)


def trends_column(term, transform):
    return f"trends_{term.replace(' ', '_')}_{transform}"


def prepare_trends(frame):
    """Expanding normalization only uses values up to each reference month.

    It cancels a common multiplicative scale, but cannot undo integer rounding,
    historical revisions, sampling changes or missing publication vintages.
    """
    source = frame.copy()
    source["date"] = pd.to_datetime(source.date, errors="raise")
    if source.date.duplicated().any() or not source.date.dt.is_month_start.all():
        raise ValueError("Expected unique first-of-month Trends keys")
    source = source.sort_values("date").set_index("date")
    if not source.index.equals(pd.date_range(source.index.min(), source.index.max(), freq="MS", name="date")):
        raise ValueError("Missing calendar month in Trends")
    values = source[list(TERMS)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values).all().all() or not values.ge(0).all().all() or not values.le(100).all().all():
        raise ValueError("Trends indices must be finite values in [0, 100]")
    result = pd.DataFrame(index=source.index)
    for term in TERMS:
        s = values[term]
        result[trends_column(term, "indice")] = s
        mean, std = s.expanding(min_periods=12).mean(), s.expanding(min_periods=12).std()
        result[trends_column(term, "zexp")] = (s - mean) / std.replace(0, np.nan)
    return result.rename_axis("trends_mes_ref").reset_index()


def attach_trends(pool, source, lag):
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("Positive integer lag required")
    d = pool.copy()
    d["trends_mes_ref"] = d.mes_obs - pd.DateOffset(months=lag)
    joined = d.merge(source, on="trends_mes_ref", how="left", validate="many_to_one", sort=False)
    pd.testing.assert_frame_equal(joined[pool.columns], pool.reset_index(drop=True))
    raw = [trends_column(t, "indice") for t in TERMS]
    if joined[raw].isna().any().any():
        raise ValueError("Trends source does not cover all lagged observation months")
    return joined


def variants():
    z = tuple(trends_column(t, "zexp") for t in TERMS)
    raw = tuple(trends_column(t, "indice") for t in TERMS)
    return {"transaccional": (), "mas_nasa": CLIMATE,
            "mas_trends_catalogo": z[:5], "mas_trends": z,
            "mas_trends_indices": raw, "mas_nasa_trends": CLIMATE + z}


def summarize(prediction):
    summary, blocks, monthly = score_metrics(prediction)
    auc = prediction.groupby("mes_obs").apply(
        lambda g: roc_auc_score(g.churn, g.score), include_groups=False)
    summary["auc_monthly_mean"] = auc.mean()
    for fraction in (10, 20, 30):
        m = monthly.loc[monthly.top_fraction.eq(fraction / 100)]
        summary[f"tp_top{fraction}"] = int(m.tp.sum())
        summary[f"n_top{fraction}"] = int(m.n_top.sum())
    monthly["auc"] = pd.to_datetime(monthly.month).map(auc)
    return summary, blocks, monthly


def run(output=ROOT / "reports/climate_trends_v1", lag=3, threads=4):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    pool, folds = development_data()
    sources = [NASA / "clima_mensual_provincias.parquet", NASA / "protocol.json",
               NASA / "diccionario.json", TRENDS / "google_trends_pe.csv",
               TRENDS / "google_trends_pe_regiones.csv"]
    base_protocol = json.loads((BASE_REPORT / "protocol.json").read_text())
    parameter_hashes = {n: digest(PARAMS / f"{n}_best.json") for n in ("logreg", "xgboost")}
    versions = {p: importlib.metadata.version(p) for p in base_protocol["versions"]}
    assert base_protocol["raw_sha256"] == digest(ROOT / "data/processed/churn_dataset.csv")
    assert base_protocol["parameter_files"] == parameter_hashes
    assert base_protocol["versions"] == versions
    assert base_protocol["code_sha256"] == digest(ROOT / "scripts/master_features_ablation.py")
    assert base_protocol["temporal_code_sha256"] == digest(ROOT / "scripts/temporal_optuna.py")
    assert base_protocol["ensemble_code_sha256"] == digest(ROOT / "scripts/recent_windows.py")
    nasa_protocol = json.loads((NASA / "protocol.json").read_text())
    assert nasa_protocol["dataset_sha256"] == base_protocol["raw_sha256"]
    assert nasa_protocol["code_sha256"] == digest(ROOT / "scripts/nasa_power.py")
    config = variants()
    protocol = {
        "code_sha256": digest(__file__), "raw_sha256": base_protocol["raw_sha256"],
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "dependencies": {f: digest(ROOT / f) for f in (
            "scripts/master_features_ablation.py", "scripts/temporal_optuna.py",
            "scripts/recent_windows.py", "scripts/nasa_power.py", "scripts/external_features_ablation.py")},
        "parameters": parameter_hashes, "versions": versions,
        "lag_months_assumed": lag, "threads": threads, "training_end": "2025-01-01",
        "gap_months": 6, "variants": {k: list(v) for k, v in config.items()},
        "baseline_predictions": {str(i): digest(BASE_REPORT / f"predictions_transaccional_fold{i}.parquet") for i in range(len(folds))},
        "status": "exploratory historical snapshots; no historical publication vintages",
        "regional_trends": "excluded: undated aggregate over extraction window, not department-month history",
    }
    if (out / "protocol.json").exists() and json.loads((out / "protocol.json").read_text()) != protocol:
        raise ValueError("Changed protocol: choose a new output directory")
    provincial = pd.read_parquet(sources[0])
    trends = prepare_trends(pd.read_csv(TRENDS / "google_trends_pe.csv"))
    joined = attach_trends(attach_climate(pool, provincial, lag), trends, lag)
    pd.testing.assert_frame_equal(joined[META], pool[META])
    assert (joined.nasa_mes_ref == joined.trends_mes_ref).all()
    assert (joined.nasa_mes_ref < joined.mes_obs).all()
    coverage = []
    for partition, idx in (("development", np.arange(len(joined))),
                           ("validation", np.concatenate([va for _, va in folds]))):
        d = joined.iloc[idx]
        coverage.append({"partition": partition, "rows": len(d),
                         "nasa_missing_location_or_month": int(d.nasa_sin_dato.sum()),
                         "nasa_missing_yearly_change": int(d[list(CLIMATE[-3:-1])].isna().any(axis=1).sum()),
                         "trends_missing_index": int(d[[trends_column(t, "indice") for t in TERMS]].isna().any(axis=1).sum()),
                         "trends_missing_zexp": int(d[[trends_column(t, "zexp") for t in TERMS]].isna().any(axis=1).sum())})
    write_json(out / "protocol.json", protocol)
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    joined.to_parquet(out / "joined_development.parquet", index=False)
    trends.to_csv(out / "source_trends_transformed.csv", index=False)
    write_json(out / "variables.json", {"variants": config, "terms": TERMS,
        "nasa_definitions": json.loads((NASA / "diccionario.json").read_text()),
        "zexp": "(x_ref - mean(x_2016_01..ref)) / sample_std(x_2016_01..ref), minimum 12 months; zero std -> missing",
        "imputation": "median fitted only on each component training set; empty features retained"})
    params = {n: json.loads((PARAMS / f"{n}_best.json").read_text())["params"] for n in ("logreg", "xgboost")}
    write_json(out / "parameters.json", params)
    summaries, block_tables, month_tables, components, training = [], [], [], [], []
    for variant, columns in config.items():
        chunks = []
        for i, (tr, va) in enumerate(folds):
            expected = joined.iloc[va][META].reset_index(drop=True).assign(fold=i)
            file = out / f"predictions_{variant}_fold{i}.parquet"
            source = file if columns else BASE_REPORT / f"predictions_transaccional_fold{i}.parquet"
            for name, months in (("logreg", 36), ("xgboost", None)):
                a = joined.iloc[recent_training_indices(joined, tr, months)]
                assert a.mes_rank.max() + 6 < joined.iloc[va].mes_rank.min()
                training.append({"variant": variant, "fold": i, "model": name,
                                 "train_start": str(a.mes_obs.min().date()), "train_end": str(a.mes_obs.max().date()),
                                 "train_rows": len(a), "valid_start": str(expected.mes_obs.min().date()),
                                 "valid_end": str(expected.mes_obs.max().date()), "valid_rows": len(expected)})
            if source.exists():
                predicted = pd.read_parquet(source)
                pd.testing.assert_frame_equal(predicted[expected.columns], expected)
            else:
                scores = []
                for name, months in (("logreg", 36), ("xgboost", None)):
                    a = joined.iloc[recent_training_indices(joined, tr, months)]
                    model = make_pipeline(ExternalFeatures(columns),
                        SimpleImputer(strategy="median", keep_empty_features=True),
                        build_model(name, params[name], a.churn.to_numpy(), threads))
                    print(f"Training {variant}/{name}/fold{i} ({len(a)} rows)", flush=True)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(a, a.churn)
                    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                        raise RuntimeError(f"Convergence failure: {variant}/{name}/fold{i}")
                    scores.append(model.predict_proba(joined.iloc[va])[:, 1])
                predicted = expected.assign(score_logreg=scores[0], score_xgboost=scores[1], score=(scores[0]+scores[1])/2)
            np.testing.assert_allclose(predicted.score, (predicted.score_logreg + predicted.score_xgboost)/2)
            assert np.isfinite(predicted.score).all() and predicted.score.between(0, 1).all()
            if not file.exists():
                predicted.to_parquet(file, index=False)
            chunks.append(predicted)
        prediction = pd.concat(chunks, ignore_index=True)
        summary, blocks, monthly = summarize(prediction)
        summaries.append({"variant": variant, "n_added_columns": len(columns), **summary})
        block_tables.append(blocks.assign(variant=variant))
        month_tables.append(monthly.assign(variant=variant))
        for name in ("logreg", "xgboost"):
            c, _, _ = summarize(prediction.assign(score=prediction[f"score_{name}"]))
            components.append({"variant": variant, "model": name, **c})
        print(f"RESULT {variant}: AUC={summary['auc_mean']:.6f}, monthly={summary['auc_monthly_mean']:.6f}, TP10={summary['tp_top10']}", flush=True)
    comparison = pd.DataFrame(summaries)
    for metric in ("auc_mean", "auc_monthly_mean", "tp_top10", "tp_top20"):
        comparison[f"delta_{metric}"] = comparison[metric] - comparison.loc[0, metric]
    comparison.to_csv(out / "comparison.csv", index=False)
    pd.concat(block_tables).to_csv(out / "block_metrics.csv", index=False)
    pd.concat(month_tables).to_csv(out / "monthly_metrics.csv", index=False)
    pd.DataFrame(components).to_csv(out / "component_metrics.csv", index=False)
    pd.DataFrame(training).to_csv(out / "training_windows.csv", index=False)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/climate_trends_v1")
    parser.add_argument("--lag", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.lag < 1 or args.threads < 1:
        parser.error("Positive lag and threads required")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(run(args.output, args.lag, args.threads).to_string(index=False))


if __name__ == "__main__":
    main()
