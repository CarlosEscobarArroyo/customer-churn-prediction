"""Controlled temporal comparison of the 42-feature ensemble and master data.

Run from the notebook or with python -m scripts.master_features_ablation.
The default compares fixed exported parameters. Use --params-dir for another
export, or explicitly --reconstruct to rebuild a temporal hyperparameter search.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.metadata
import json
from pathlib import Path
import unicodedata
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from scripts.recent_windows import ProbabilityEnsemble, recent_training_indices
from scripts.temporal_optuna import (
    META, ZERO, build_model, check_numeric, digest, engineer, load_raw,
    objective, prepare, suggest, temporal_folds, write_json,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_COLUMNS = (
    "meses_activos_u3 meses_activos_u6 meses_activos_u12 n_ped_u3 n_ped_u6 "
    "n_ped_u12 monto_u3 monto_u6 monto_u12 monto_mean_u12 monto_std_u12 "
    "monto_cv_u12 monto_ult_vs_media meses_desde_compra_previa compras_hist "
    "n_prod_u12 n_cat_max_u12 tend_monto_u3_vs_prev3 tend_nped_u3_vs_prev3 "
    "monto_acum n_ped_acum n_prod_acum n_cat_max_acum ticket_acum "
    "monto_por_prod_acum monto_mensual_acum d_monto_m1 d_monto_m3 d_monto_m6 "
    "d_monto_m9 d_monto_m12 d_nped_m1 d_nped_m3 d_nped_m6 d_nped_m9 d_nped_m12"
).split()
VARIANTS = {
    "transaccional": (),
    "mas_sexo": ("sexo",),
    "mas_edad": ("edad",),
    "mas_antiguedad": ("antiguedad_meses",),
    "mas_departamento": ("departamento",),
    "mas_provincia": ("provincia",),
    "mas_ubicacion": ("departamento", "provincia"),
    "mas_demografia_antiguedad": ("sexo", "edad", "antiguedad_meses"),
    "mas_todas": ("sexo", "edad", "antiguedad_meses", "departamento", "provincia"),
}
NUMERIC_MASTER = ("edad", "antiguedad_meses")
CATEGORICAL_MASTER = ("sexo", "departamento", "provincia")


def normalize_category(value):
    if pd.isna(value) or not str(value).strip():
        return "__sin_dato__"
    value = unicodedata.normalize("NFKD", str(value).strip().casefold())
    return " ".join("".join(c for c in value if not unicodedata.combining(c)).split())


class MasterFeatures(TransformerMixin, BaseEstimator):
    """Explicit schema: never ingest new columns or labels accidentally."""

    def __init__(self, extras=()):
        self.extras = extras

    def fit(self, X, y=None):
        self.columns_ = self.transform(X).columns.tolist()
        return self

    def transform(self, X):
        if not set(self.extras).issubset(NUMERIC_MASTER + CATEGORICAL_MASTER):
            raise ValueError("Unsupported master feature")
        d = X[BASE_COLUMNS].copy()
        d[ZERO] = d[ZERO].fillna(0)
        d = engineer(d)
        check_numeric(d)
        for col in self.extras:
            if col in NUMERIC_MASTER:
                d[col] = pd.to_numeric(X[col], errors="raise")
                d[f"{col}_sin_dato"] = d[col].isna().astype(float)
            else:
                d[col] = X[col].map(normalize_category)
        return d


def preprocessor(frame, extras):
    categorical = [c for c in extras if c in CATEGORICAL_MASTER]
    numeric = [c for c in frame.columns if c not in categorical]
    transformers = [("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric)]
    if categorical:
        transformers.append(("categorical", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float64), categorical))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def make_component(train, name, extras, params, threads):
    features = MasterFeatures(extras)
    frame = features.fit_transform(train)
    return make_pipeline(features, preprocessor(frame, extras),
                         build_model(name, params, train.churn.to_numpy(), threads))


def development_data():
    raw = load_raw()
    # Fixed calendar cutoff: never slide folds when a newer CSV arrives.
    pool = raw.loc[raw.mes_obs <= "2025-01-01"].reset_index(drop=True)
    folds = temporal_folds(pool)
    starts = [str(pool.iloc[va].mes_obs.min().date()) for _, va in folds]
    if starts != ["2023-10-01", "2024-02-01", "2024-06-01", "2024-10-01"]:
        raise ValueError("Original validation calendar could not be reproduced")
    return pool, folds


def parallel_logistic_objective(trial, folds, workers):
    """Same objective/report order; independent SAGA fits release the GIL."""
    params = suggest(trial, "logreg")

    def fit_fold(fold):
        cols = fold["info"]["selected"] if trial.params["features"] == "selected" else fold["X_train"].columns
        model = build_model("logreg", params, fold["y_train"], 1)
        model.fit(fold["X_train"][cols], fold["y_train"])
        p = model.predict_proba(fold["X_valid"][cols])[:, 1]
        y = fold["y_valid"]
        monthly = {}
        for month in np.unique(fold["valid_months"]):
            mask = fold["valid_months"] == month
            if len(np.unique(y[mask])) == 2:
                monthly[month] = float(roc_auc_score(y[mask], p[mask]))
        return {"auc": float(roc_auc_score(y, p)),
                "average_precision": float(average_precision_score(y, p)),
                "lift10": float(y[np.argsort(-p)[:max(1, len(y) // 10)]].mean() / y.mean()),
                "recall_at_05": float(recall_score(y, p >= 0.5)),
                "monthly_auc": monthly, "n_features": len(cols)}

    metrics = []
    with ThreadPoolExecutor(max_workers=min(workers, len(folds))) as executor:
        futures = [executor.submit(fit_fold, fold) for fold in folds]
        for i, future in enumerate(futures):
            metrics.append(future.result())
            trial.set_user_attr("fold_metrics", list(metrics))
            trial.report(float(np.mean([m["auc"] for m in metrics])), i)
            if trial.should_prune():
                for pending in futures[i + 1:]:
                    pending.cancel()
                raise optuna.TrialPruned()
    trial.set_user_attr("auc_std_between_blocks", float(np.std([m["auc"] for m in metrics])))
    return float(np.mean([m["auc"] for m in metrics]))


def obtain_params(pool, args, out):
    if args.params_dir:
        params = {name: json.loads((Path(args.params_dir) / f"{name}_best.json").read_text())
                  for name in ("logreg", "xgboost")}
        if any(p["params"]["features"] != "all" for p in params.values()):
            raise ValueError("Expected original winners using all 42 transaction features")
        write_json(out / "parameter_source.json", {
            "kind": "supplied", "path": str(Path(args.params_dir).resolve()),
            "search_history": {name: p.get("search_history") for name, p in params.items()},
            "parameters": {name: p["params"] for name, p in params.items()}})
        return {name: p["params"] for name, p in params.items()}

    tune = out / "reconstructed_tuning"
    tune.mkdir(exist_ok=True)
    cache = tune / "folds.joblib"
    if cache.exists():
        prepared = joblib.load(cache)
    else:
        # Match original 46-column schema. Original selector explicitly drops master data.
        from scripts.temporal_optuna import MASTER
        prepared = prepare(pool[META + BASE_COLUMNS + MASTER], args.threads)
        joblib.dump(prepared, cache)
    params = {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for name in ("logreg", "xgboost"):
        study = optuna.create_study(
            storage=f"sqlite:///{tune / 'studies.sqlite3'}", study_name=name,
            direction="maximize", load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
        for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        while len(study.trials) < args.trials:
            study.sampler = optuna.samplers.TPESampler(seed=42 + len(study.trials))
            if name == "logreg":
                study.optimize(lambda t: parallel_logistic_objective(t, prepared, args.threads), n_trials=1)
            else:
                study.optimize(lambda t: objective(t, name, prepared, args.threads), n_trials=1)
            print(f"Tuning {name}: {len(study.trials)}/{args.trials}; AUC={study.best_value:.6f}", flush=True)
        eligible = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
                    and t.params["features"] == "all"]
        if not eligible:
            raise ValueError("No completed trial with all 42 features")
        best = max(eligible, key=lambda t: t.value)
        write_json(tune / f"{name}_best.json", {
            "trial": best.number, "params": best.params, "auc": best.value,
            "metrics": best.user_attrs})
        study.trials_dataframe().to_csv(tune / f"{name}_trials.csv", index=False)
        params[name] = best.params
    write_json(out / "parameter_source.json", {
        "kind": "reconstructed_search", "trials_per_model": args.trials,
        "parameters": params,
        "note": "New local search, not proof of bitwise reproduction of historical tuning. "
                "42 features and original ensemble windows are fixed for all variants."})
    return params


def score_metrics(predictions):
    blocks, monthly = [], []
    for fold, group in predictions.groupby("fold"):
        blocks.append({"fold": int(fold), "auc": roc_auc_score(group.churn, group.score),
                       "average_precision": average_precision_score(group.churn, group.score)})
    for month, group in predictions.groupby("mes_obs"):
        ranked = group.sort_values(["score", "id_vendedor"], ascending=[False, True])
        for proportion in (0.1, 0.2, 0.3):
            top = ranked.head(max(1, int(len(group) * proportion)))
            tp = int(top.churn.sum())
            monthly.append({"month": str(pd.Timestamp(month).date()), "top_fraction": proportion,
                            "n": len(group), "n_top": len(top), "positives": int(group.churn.sum()),
                            "tp": tp, "precision": tp / len(top),
                            "recall": tp / group.churn.sum(),
                            "lift": top.churn.mean() / group.churn.mean()})
    blocks, monthly = pd.DataFrame(blocks), pd.DataFrame(monthly)
    summary = {"auc_mean": blocks.auc.mean(), "auc_std": blocks.auc.std(ddof=0),
               "ap_mean": blocks.average_precision.mean()}
    for proportion in (0.1, 0.2, 0.3):
        m = monthly[monthly.top_fraction == proportion]
        label = str(int(proportion * 100))
        summary[f"precision_top{label}"] = m.tp.sum() / m.n_top.sum()
        summary[f"recall_top{label}"] = m.tp.sum() / m.positives.sum()
        summary[f"lift_top{label}"] = m.lift.mean()
    return summary, blocks, monthly


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pool, folds = development_data()
    parameter_files = ({name: digest(Path(args.params_dir) / f"{name}_best.json")
                        for name in ("logreg", "xgboost")} if args.params_dir else None)
    protocol = {"raw_sha256": digest(ROOT / "data/processed/churn_dataset.csv"),
                "code_sha256": digest(__file__),
                "temporal_code_sha256": digest(ROOT / "scripts/temporal_optuna.py"),
                "ensemble_code_sha256": digest(ROOT / "scripts/recent_windows.py"),
                "parameter_files": parameter_files, "trials": args.trials, "threads": args.threads,
                "pool_rows": len(pool), "training_end": "2025-01-01", "gap_months": 6,
                "base_columns": BASE_COLUMNS, "variants": {k: list(v) for k, v in VARIANTS.items()},
                "versions": {p: importlib.metadata.version(p) for p in
                             ("numpy", "pandas", "scikit-learn", "xgboost", "optuna")}}
    if (out / "protocol.json").exists() and json.loads((out / "protocol.json").read_text()) != protocol:
        raise ValueError("Protocol changed: use another output directory")
    write_json(out / "protocol.json", protocol)
    profile = []
    for col in NUMERIC_MASTER + CATEGORICAL_MASTER:
        profile.append({"variable": col, "missing": int(pool[col].isna().sum()),
                        "missing_fraction": float(pool[col].isna().mean()),
                        "distinct_raw": int(pool[col].nunique())})
    pd.DataFrame(profile).to_csv(out / "data_profile.csv", index=False)
    dates = []
    for i, (tr, va) in enumerate(folds):
        for name, months in (("logreg", 36), ("xgboost", None)):
            selected = recent_training_indices(pool, tr, months)
            a, b = pool.iloc[selected], pool.iloc[va]
            assert a.mes_rank.max() + 6 < b.mes_rank.min()
            dates.append({"fold": i, "model": name, "train_start": str(a.mes_obs.min().date()),
                          "train_end": str(a.mes_obs.max().date()), "n_train": len(a),
                          "valid_start": str(b.mes_obs.min().date()),
                          "valid_end": str(b.mes_obs.max().date()), "n_valid": len(b)})
    pd.DataFrame(dates).to_csv(out / "folds.csv", index=False)
    params = obtain_params(pool, args, out)
    summaries, block_tables, monthly_tables, dimensions = [], [], [], []
    for variant, extras in VARIANTS.items():
        chunks = []
        for i, (tr, va) in enumerate(folds):
            path = out / f"predictions_{variant}_fold{i}.parquet"
            expected = pool.iloc[va][META].reset_index(drop=True).assign(fold=i)
            if path.exists():
                predicted = pd.read_parquet(path)
                pd.testing.assert_frame_equal(predicted[expected.columns], expected)
            else:
                scores = []
                for name, months in (("logreg", 36), ("xgboost", None)):
                    a = pool.iloc[recent_training_indices(pool, tr, months)]
                    model = make_component(a, name, extras, params[name], args.threads)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(a, a.churn)
                    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
                    if convergence:
                        raise RuntimeError(f"{variant}/{name}/fold{i} did not converge: {convergence}")
                    scores.append(model.predict_proba(pool.iloc[va])[:, 1])
                    dimensions.append({"variant": variant, "fold": i, "model": name,
                                       "n_encoded": len(model.steps[1][1].get_feature_names_out())})
                predicted = expected.assign(score_logreg=scores[0], score_xgboost=scores[1],
                                             score=(scores[0] + scores[1]) / 2)
                predicted.to_parquet(path, index=False)
            assert np.isfinite(predicted.score).all() and predicted.score.between(0, 1).all()
            chunks.append(predicted)
        predictions = pd.concat(chunks, ignore_index=True)
        summary, blocks, monthly = score_metrics(predictions)
        summaries.append({"variant": variant, "extras": ", ".join(extras), **summary})
        block_tables.append(blocks.assign(variant=variant))
        monthly_tables.append(monthly.assign(variant=variant))
        print(f"{variant}: AUC={summary['auc_mean']:.6f}; lift10={summary['lift_top10']:.4f}", flush=True)
    comparison = pd.DataFrame(summaries)
    baseline = comparison.loc[comparison.variant == "transaccional", "auc_mean"].iloc[0]
    comparison["delta_auc"] = comparison.auc_mean - baseline
    comparison = comparison.sort_values("auc_mean", ascending=False)
    comparison.to_csv(out / "comparison.csv", index=False)
    pd.concat(block_tables).to_csv(out / "block_metrics.csv", index=False)
    pd.concat(monthly_tables).to_csv(out / "monthly_metrics.csv", index=False)
    if dimensions:
        pd.DataFrame(dimensions).to_csv(out / "encoded_dimensions.csv", index=False)
    # Export the winner as an experimental candidate, preserving the original artifact.
    winner = comparison.iloc[0].variant
    trained, columns = [], {}
    for name, months in (("logreg", 36), ("xgboost", None)):
        a = pool.iloc[recent_training_indices(pool, np.arange(len(pool)), months)]
        model = make_component(a, name, VARIANTS[winner], params[name], args.threads)
        model.fit(a, a.churn)
        trained.append(model)
        columns[name] = model.steps[1][1].get_feature_names_out().tolist()
    candidate = ProbabilityEnsemble(trained)
    joblib.dump(candidate, out / "candidate.joblib")
    reloaded = joblib.load(out / "candidate.joblib")
    np.testing.assert_allclose(candidate.predict_proba(pool.tail(10)), reloaded.predict_proba(pool.tail(10)))
    write_json(out / "candidate.json", {"variant": winner, "extras": VARIANTS[winner],
               "encoded_features": columns, "sha256": digest(out / "candidate.joblib"),
               "status": "experimental, selected on reused validation; master data are current snapshots"})
    write_report(out, comparison, pd.concat(block_tables), sum(len(v) for _, v in folds))
    return comparison


def write_report(out, comparison, blocks, n_valid):
    source = json.loads((out / "parameter_source.json").read_text())
    winner = comparison.iloc[0]
    display = comparison[["variant", "auc_mean", "delta_auc", "ap_mean", "precision_top10", "recall_top10", "lift_top10"]]
    text = f"""# Aporte de sexo, edad, antigüedad y ubicación al ensemble

Mejor variante por AUC temporal medio: **{winner.variant}**, AUC **{winner.auc_mean:.5f}**,
cambio frente a la base de esta corrida: **{winner.delta_auc:+.5f}**.

## Comparación controlada

{display.to_markdown(index=False, floatfmt='.5f')}

Todas las variantes usan las mismas 42 variables transaccionales, logística con 36 meses,
XGBoost con toda la historia permitida y promedio 50/50. Los parámetros se fijan antes
de la ablación; no se retunean por variante. Las columnas de campañas no se incorporan.
Origen de parámetros: `{source['kind']}`; consultar `parameter_source.json`.
Historial de búsqueda de los parámetros suministrados: `{source.get('search_history', {})}`.
No es necesario ejecutar 100 trials para esta comparación: basta con parámetros fijos
y una base evaluada en los mismos períodos. La búsqueda previa se conserva como antecedente.

Cuatro bloques de validación de cuatro meses entre octubre de 2023 y enero de 2025,
gap de seis meses y {n_valid} observaciones vendedora-mes. El corte se fija por fecha;
no se desplaza con los meses adicionales del CSV actual. No se evalúa el OOT.

## AUC por bloque

{blocks.pivot(index='variant', columns='fold', values='auc').to_markdown(floatfmt='.5f')}

## Tratamiento y límites

- Edad y antigüedad: mediana aprendida exclusivamente en cada entrenamiento e indicador
  explícito de dato ausente. Edad se refiere al mes observado, a partir de fecha de nacimiento;
  antigüedad es meses desde la fecha de ingreso registrada.
- Sexo, departamento y provincia: normalización de espacios, mayúsculas y tildes;
  one-hot aprendido en entrenamiento, categorías nuevas ignoradas. Faltantes tienen categoría propia.
- La imputación, vocabulario y escalado se ajustan por separado para cada modelo y fold.
- Ubicación es departamento/provincia del snapshot actual. Una mejora retrospectiva no
  acredita que esa ubicación estuviera disponible en la fecha histórica. Lo mismo exige
  verificar la calidad de sexo, nacimiento e ingreso antes de una decisión operativa.
- Son métricas de selección sobre validaciones reutilizadas. Las pequeñas diferencias no
  demuestran superioridad estadística ni equivalen a una evaluación final independiente.
- La referencia histórica de AUC 0,78961 procede de otra corrida. La comparación del aporte
  de añadir variables se realiza contra `transaccional` de esta misma ejecución.

## Artefactos

En `{out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}`: protocolo con hashes/versiones,
parámetros, perfil de faltantes, fechas de folds, predicciones por variante/fold,
`comparison.csv`, `block_metrics.csv`, `monthly_metrics.csv`, `candidate.json` y
`candidate.joblib`. El candidato se guarda por separado del ensemble histórico y su recarga
se comprueba. Las métricas de precisión y recall son agregadas; el lift es promedio mensual.

Ejecución y análisis: [notebook 08](../05_modelling/08_ablacion_datos_maestros.ipynb).
"""
    import os
    local_notebook = os.path.relpath(ROOT / "05_modelling/08_ablacion_datos_maestros.ipynb", out)
    (out / "report.md").write_text(text.replace("../05_modelling/08_ablacion_datos_maestros.ipynb", local_notebook))
    (ROOT / "07_results/ablacion_datos_maestros.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/master_features_comparison_v1")
    parser.add_argument("--params-dir", default="reports/master_features_v1/reconstructed_tuning")
    parser.add_argument("--reconstruct", action="store_true", help="Explicitly run a new temporal search")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.reconstruct:
        args.params_dir = None
    if args.trials < 1 or args.threads < 1:
        parser.error("Positive trial/thread counts required")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    from scripts.master_features_ablation import main as entrypoint
    entrypoint()
