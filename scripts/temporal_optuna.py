"""Temporal churn experiments. Run from repository root; see README.md."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
ID = ["id_vendedor", "mes_obs", "mes_rank"]
META = ID + ["churn"]
MASTER = ["sexo", "edad", "antiguedad_meses", "tipo_vendedor", "departamento", "provincia"]
ZERO = ["tend_monto_u3_vs_prev3", "tend_nped_u3_vs_prev3", "monto_cv_u12",
        "monto_ult_vs_media", "monto_por_prod_acum"] + [
    f"d_{m}_m{n}" for m in ("monto", "nped") for n in (1, 3, 6, 9, 12)]
MODELS = ["logreg", "rf", "xgboost", "catboost"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def check_numeric(frame):
    if any(not pd.api.types.is_numeric_dtype(t) for t in frame.dtypes):
        raise ValueError("Non-numeric feature detected; refusing silent coercion")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Missing or infinite feature detected")


def engineer(df):
    d = df.copy()

    def ratio(a, b):
        return (a / b.replace(0, np.nan)).fillna(0)

    d["ticket_prom_u12"] = ratio(d.monto_u12, d.n_ped_u12)
    d["ticket_prom_u3"] = ratio(d.monto_u3, d.n_ped_u3)
    d["intensidad_u3"] = ratio(d.n_ped_u3, d.meses_activos_u3)
    d["basket_size_u12"] = ratio(d.n_prod_u12, d.n_ped_u12)
    d["recencia_norm"] = d.meses_desde_compra_previa * d.meses_activos_u12 / 12
    d["tasa_act_reciente_vs_hist"] = ratio(d.meses_activos_u3 / 3, d.meses_activos_u12 / 12)
    return d


def load_raw():
    df = pd.read_csv(PROC / "churn_dataset.csv", parse_dates=["mes_obs"])
    if df.duplicated(["id_vendedor", "mes_rank"]).any() or df[META].isna().any().any():
        raise ValueError("Invalid dataset keys")
    if set(df.churn.unique()) != {0, 1}:
        raise ValueError("Invalid target")
    ordinal = df.mes_obs.dt.year * 12 + df.mes_obs.dt.month
    if (ordinal - df.mes_rank).nunique() != 1:
        raise ValueError("mes_rank is not aligned to calendar months")
    # Only structurally missing transactional features may be filled with zero.
    transactional = df.drop(columns=META + MASTER).copy()
    transactional[ZERO] = transactional[ZERO].fillna(0)
    check_numeric(transactional)
    return df


def repair():
    """Restore legacy 68-feature CSV; preserve selection and existing model compatibility."""
    raw = load_raw()
    pre = pd.read_csv(PROC / "churn_dataset_processed.csv", parse_dates=["mes_obs"])
    split = pd.read_csv(PROC / "oot_split.csv", parse_dates=["mes_obs"])
    for other in [pre, split]:
        if not raw[META].equals(other[META]):
            raise ValueError("Source/split/preprocessed rows are not aligned")
    start = raw.mes_rank.max() - 3
    expected = np.where(raw.mes_rank <= start - 7, "train",
                        np.where(raw.mes_rank >= start, "test", "gap"))
    if not np.array_equal(expected, split.partition):
        raise ValueError("Persisted split differs from expected legacy split")
    # Independently verify the upstream preprocessor before using its artifact.
    enc = raw.drop(columns=["edad", "provincia"]).copy()
    enc[ZERO] = enc[ZERO].fillna(0)
    enc["antiguedad_meses"] = enc.antiguedad_meses.fillna(
        raw.loc[expected == "train", "antiguedad_meses"].median())
    cats = ["sexo", "tipo_vendedor", "departamento"]
    for col in cats:
        levels = sorted(raw.loc[expected == "train", col].fillna("DESCONOCIDO").unique())
        enc[col] = pd.Categorical(enc[col].fillna("DESCONOCIDO"), categories=levels)
    enc = pd.get_dummies(enc, columns=cats, dtype=int)
    pd.testing.assert_frame_equal(enc, pre, check_dtype=False, rtol=1e-9, atol=1e-9)
    selected = json.loads((PROC / "selected_features.json").read_text())
    rebuilt = engineer(pre)[META + selected]
    check_numeric(rebuilt[selected])
    destination = PROC / "churn_dataset_features.csv"
    old = pd.read_csv(destination, low_memory=False, parse_dates=["mes_obs"])
    if not rebuilt[META].equals(old[META]) or rebuilt.columns.tolist() != old.columns.tolist():
        raise ValueError("Legacy artifact schema/keys differ")
    changes = {}
    for col in selected:
        diff = ~np.isclose(pd.to_numeric(old[col], errors="coerce"), rebuilt[col],
                           rtol=1e-9, atol=1e-9, equal_nan=True)
        if diff.any():
            changes[col] = np.flatnonzero(diff).tolist()
    before = digest(destination)
    backup = PROC / f"churn_dataset_features.before_repair_{before[:12]}.csv"
    if changes:
        if not backup.exists():
            shutil.copy2(destination, backup)
        temp = destination.with_suffix(".tmp.csv")
        rebuilt.to_csv(temp, index=False)
        check_numeric(pd.read_csv(temp)[selected])
        temp.replace(destination)
    manifest = {"changed_positions_zero_based": changes, "before_sha256": before,
                "after_sha256": digest(destination), "raw_sha256": digest(PROC / "churn_dataset.csv"),
                "preprocessed_sha256": digest(PROC / "churn_dataset_processed.csv"),
                "backup": str(backup) if changes else None}
    report = ROOT / "reports/data_repair"
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / f"repair_{time.time_ns()}.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


class TransactionFeatures(TransformerMixin, BaseEstimator):
    """Deterministic transaction-only features; no snapshot master data."""

    def fit(self, X, y=None):
        self.columns_ = self._convert(X).columns.tolist()
        return self

    def _convert(self, X):
        d = X.drop(columns=META + MASTER, errors="ignore").copy()
        d[ZERO] = d[ZERO].fillna(0)
        check_numeric(d)
        d = engineer(d)
        check_numeric(d)
        return d

    def transform(self, X):
        d = self._convert(X)
        if d.columns.tolist() != self.columns_:
            raise ValueError("Feature schema changed")
        return d


class KeepColumns(TransformerMixin, BaseEstimator):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


def temporal_folds(df, n_folds=4, months=4, gap=6):
    last = int(df.mes_rank.max())
    folds = []
    for i in range(n_folds):
        end = last - months * (n_folds - 1 - i)
        start = end - months + 1
        tr = np.flatnonzero(df.mes_rank.to_numpy() <= start - gap - 1)
        va = np.flatnonzero(df.mes_rank.between(start, end).to_numpy())
        if len(tr) < 100 or len(va) < 40:
            raise ValueError("Insufficient rows for temporal split")
        if df.iloc[tr].churn.nunique() != 2 or df.iloc[va].churn.nunique() != 2:
            raise ValueError("Both target classes required in each partition")
        if int(df.iloc[tr].mes_rank.max()) + gap >= int(df.iloc[va].mes_rank.min()):
            raise ValueError("Temporal label overlap")
        folds.append((tr, va))
    return folds


def select_inside_training(train, threads):
    """Permutation AUC on an inner temporal split, never on the objective validation."""
    tr, va = temporal_folds(train, n_folds=1)[0]
    transform = TransactionFeatures().fit(train.iloc[tr])
    a, b = transform.transform(train.iloc[tr]), transform.transform(train.iloc[va])
    model = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=20,
                                   class_weight="balanced", n_jobs=threads, random_state=42)
    model.fit(a, train.iloc[tr].churn)
    imp = permutation_importance(model, b, train.iloc[va].churn,
                                 scoring="roc_auc", n_repeats=3, n_jobs=1, random_state=42)
    selected = a.columns[imp.importances_mean > 0].tolist()
    if not selected:
        selected = a.columns.tolist()
    return selected, dict(zip(a.columns, imp.importances_mean.tolist()))


def prepare(df, threads):
    result = []
    for i, (tr, va) in enumerate(temporal_folds(df)):
        train, valid = df.iloc[tr], df.iloc[va]
        selector, importance = select_inside_training(train, threads)
        transformer = TransactionFeatures().fit(train)
        info = {"fold": i, "train_rows": len(tr), "valid_rows": len(va),
                "train_end": str(train.mes_obs.max().date()),
                "valid_start": str(valid.mes_obs.min().date()),
                "valid_end": str(valid.mes_obs.max().date()),
                "selected": selector, "inner_permutation_auc": importance}
        result.append({"X_train": transformer.transform(train), "y_train": train.churn.to_numpy(),
                       "X_valid": transformer.transform(valid), "y_valid": valid.churn.to_numpy(),
                       "valid_months": valid.mes_obs.astype(str).to_numpy(), "info": info})
        print(f"Prepared fold {i}: {info['train_end']} -> {info['valid_start']}.."
              f"{info['valid_end']}; {len(selector)} selected features", flush=True)
    return result


def suggest(trial, name):
    trial.suggest_categorical("features", ["all", "selected"])
    p = {"balance": trial.suggest_categorical("balance", [False, True])}
    if name == "logreg":
        p.update(C=trial.suggest_float("C", 1e-4, 100, log=True),
                 l1_ratio=trial.suggest_float("l1_ratio", 0, 1))
    elif name == "rf":
        p.update(n_estimators=trial.suggest_int("n_estimators", 200, 700, step=100),
                 max_depth=trial.suggest_int("max_depth", 3, 18),
                 min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 100),
                 max_features=trial.suggest_float("max_features", 0.2, 1.0),
                 max_samples=trial.suggest_float("max_samples", 0.5, 1.0))
    elif name == "xgboost":
        p.update(n_estimators=trial.suggest_int("n_estimators", 300, 1500, step=100),
                 max_depth=trial.suggest_int("max_depth", 2, 7),
                 learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                 min_child_weight=trial.suggest_float("min_child_weight", 3, 80, log=True),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                 reg_lambda=trial.suggest_float("reg_lambda", 0.1, 30, log=True),
                 gamma=trial.suggest_float("gamma", 0, 5))
    else:
        p.update(iterations=trial.suggest_int("iterations", 300, 1500, step=100),
                 depth=trial.suggest_int("depth", 3, 7),
                 learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                 l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.1, 30, log=True),
                 random_strength=trial.suggest_float("random_strength", 0.01, 10, log=True),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0),
                 rsm=trial.suggest_float("rsm", 0.5, 1.0))
    return p


def build_model(name, params, y, threads):
    p = {k: v for k, v in params.items() if k != "features"}
    balance = p.pop("balance")
    weight = "balanced" if balance else None
    spw = float((y == 0).sum() / (y == 1).sum()) if balance else 1.0
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(
            **p, solver="saga", penalty="elasticnet", class_weight=weight,
            max_iter=5000, tol=1e-4, random_state=42))
    if name == "rf":
        return RandomForestClassifier(**p, class_weight=weight, n_jobs=threads, random_state=42)
    if name == "xgboost":
        return XGBClassifier(**p, scale_pos_weight=spw, tree_method="hist",
                             eval_metric="logloss", n_jobs=threads, random_state=42)
    return CatBoostClassifier(**p, scale_pos_weight=spw, bootstrap_type="Bernoulli",
                              thread_count=threads, random_seed=42, verbose=False,
                              allow_writing_files=False)


def objective(trial, name, folds, threads):
    params = suggest(trial, name)
    metrics = []
    for i, fold in enumerate(folds):
        cols = fold["info"]["selected"] if trial.params["features"] == "selected" else fold["X_train"].columns
        model = build_model(name, params, fold["y_train"], threads)
        model.fit(fold["X_train"][cols], fold["y_train"])
        p = model.predict_proba(fold["X_valid"][cols])[:, 1]
        y = fold["y_valid"]
        monthly = {}
        for month in np.unique(fold["valid_months"]):
            mask = fold["valid_months"] == month
            if len(np.unique(y[mask])) == 2:
                monthly[month] = float(roc_auc_score(y[mask], p[mask]))
        metrics.append({"auc": float(roc_auc_score(y, p)),
                        "average_precision": float(average_precision_score(y, p)),
                        "lift10": float(y[np.argsort(-p)[:max(1, len(y) // 10)]].mean() / y.mean()),
                        "recall_at_05": float(recall_score(y, p >= 0.5)),
                        "monthly_auc": monthly, "n_features": len(cols)})
        trial.set_user_attr("fold_metrics", metrics)
        trial.report(float(np.mean([m["auc"] for m in metrics])), i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("auc_std_between_blocks", float(np.std([m["auc"] for m in metrics])))
    return float(np.mean([m["auc"] for m in metrics]))


def export_best(study, name, train, threads, out):
    best = study.best_trial
    transformer = TransactionFeatures().fit(train)
    if best.params["features"] == "selected":
        cols, _ = select_inside_training(train, threads)
    else:
        cols = transformer.columns_
    model = build_model(name, best.params, train.churn.to_numpy(), threads)
    pipeline = make_pipeline(transformer, KeepColumns(cols), model)
    pipeline.fit(train, train.churn)
    path = out / f"{name}_pipeline.joblib"
    temporary = path.with_suffix(".tmp")
    joblib.dump(pipeline, temporary)
    temporary.replace(path)
    write_json(out / f"{name}_best.json", {"trial": best.number, "cv_auc": best.value,
               "params": best.params, "metrics": best.user_attrs,
               "training_rows": len(train), "training_end": str(train.mes_obs.max().date()),
               "features": cols, "model_sha256": digest(path),
               "note": "CV-selected candidate; historical OOT not evaluated; not production-approved"})


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    oot_start = int(raw.mes_rank.max()) - 3
    train = raw[raw.mes_rank <= oot_start - 7].reset_index(drop=True)
    import importlib.metadata
    protocol = {"version": 1, "code_sha256": digest(__file__),
                "raw_sha256": digest(PROC / "churn_dataset.csv"),
                "train_rows": len(train), "train_end": str(train.mes_obs.max().date()),
                "folds": 4, "validation_months": 4, "gap_months": 6,
                "objective": "unweighted mean validation block ROC AUC",
                "excluded_master_columns": MASTER, "seed": 42, "threads": args.threads,
                "packages": {p: importlib.metadata.version(p) for p in
                             ["numpy", "pandas", "scikit-learn", "optuna", "xgboost", "catboost"]}}
    manifest = out / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError("Protocol/data/code changed: use a new output directory")
    write_json(manifest, protocol)
    cache = out / "folds.joblib"
    if cache.exists():
        folds = joblib.load(cache)
    else:
        folds = prepare(train, args.threads)
        joblib.dump(folds, cache)
        write_json(out / "folds.json", [f["info"] for f in folds])
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{out / 'studies.sqlite3'}",
                                       engine_kwargs={"connect_args": {"timeout": 60}})
    studies = {}
    for name in args.models:
        study = optuna.create_study(storage=storage, study_name=f"temporal_v1_{name}",
                                   direction="maximize", load_if_exists=True,
                                   sampler=optuna.samplers.TPESampler(seed=42),
                                   pruner=optuna.pruners.MedianPruner(
                                       n_startup_trials=10, n_warmup_steps=1))
        for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        studies[name] = study
    write_json(out / "process.json", {"pid": os.getpid(), "started": time.ctime(),
                                     "target_trials_per_model": args.trials, "models": args.models})
    while True:
        progressed = False
        for name, study in studies.items():
            if (out / "STOP").exists():
                print("STOP requested; exiting between trials", flush=True)
                return
            finished = [t for t in study.trials if t.state.is_finished()]
            if len(finished) >= args.trials:
                continue
            progressed = True
            # Independent deterministic seed per trial permits reproducible resumption.
            study.sampler = optuna.samplers.TPESampler(seed=42 + len(study.trials))
            previous = study.best_trial.number if any(
                t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) else None
            study.optimize(lambda t: objective(t, name, folds, args.threads), n_trials=1)
            if previous != study.best_trial.number:
                export_best(study, name, train, args.threads, out)
            study.trials_dataframe().to_csv(out / f"{name}_trials.csv", index=False)
            print(f"{name}: {len(study.trials)}/{args.trials}, best mean temporal AUC="
                  f"{study.best_value:.6f}", flush=True)
        if not progressed:
            break
    print("All trial budgets completed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["repair", "run"])
    parser.add_argument("--output", default="reports/optuna_temporal_v1")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    if args.trials < 1 or args.threads < 1:
        parser.error("trials and threads must be positive")
    if args.command == "repair":
        repair()
    else:
        # Fail rather than allow concurrent writers/resume to invalidate live trials.
        import fcntl
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "runner.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)


if __name__ == "__main__":
    # Keep pickle class paths importable from a later process.
    from scripts.temporal_optuna import main as entrypoint
    entrypoint()
