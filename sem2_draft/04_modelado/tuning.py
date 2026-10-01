"""Búsqueda de hiperparámetros con Optuna (metodología §4.3).

Cinco algoritmos con idéntico presupuesto de trials. El objetivo es el AUC medio de los
4 bloques de validación temporal (`src.evaluacion.evaluar`); el peso de la clase positiva
forma parte del espacio de búsqueda (§4.1 / §4.3).

Cada estudio se persiste en `optuna.db` (SQLite, no versionado), de modo que una corrida
interrumpida se reanuda sin repetir trials. `tune()` devuelve el mejor conjunto de
hiperparámetros; `src.modelos.build()` construye el estimador a partir de ese dict.
"""

from pathlib import Path

import numpy as np
import optuna

from src import evaluacion
from src.modelos import SEED, N_JOBS, build  # noqa: F401  (build se reexporta para el notebook)

ALGOS = ["logreg", "rf", "xgboost", "lightgbm", "catboost"]
DB = Path(__file__).parent / "optuna.db"

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Espacios de búsqueda. `w` es el peso de la clase positiva: 1 = sin ponderar,
# `ratio` = balanceado (negativos / positivos del entrenamiento).
# ---------------------------------------------------------------------------
def espacio(algo, trial, ratio, con_peso=True):
    p = {"w": trial.suggest_float("w", 1.0, ratio) if con_peso else 1.0}
    if algo == "logreg":
        p.update(C=trial.suggest_float("C", 1e-3, 1e2, log=True),
                 l1_ratio=trial.suggest_categorical("l1_ratio", [0.0, 1.0]))   # 0 = L2, 1 = L1
    elif algo == "rf":
        p.update(n_estimators=trial.suggest_int("n_estimators", 100, 500, step=100),
                 max_depth=trial.suggest_int("max_depth", 3, 16),
                 min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 100, log=True),
                 max_features=trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.5, 1.0]))
    elif algo == "xgboost":
        p.update(n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 max_depth=trial.suggest_int("max_depth", 2, 8),
                 min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 gamma=trial.suggest_float("gamma", 0.0, 5.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True))
    elif algo == "lightgbm":
        p.update(n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 4, 128, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True))
    elif algo == "catboost":
        p.update(iterations=trial.suggest_int("iterations", 100, 1000, step=50),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 depth=trial.suggest_int("depth", 3, 8),
                 l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
                 random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
                 bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0))
    else:
        raise ValueError(algo)
    return p


def make_model_factory(algo, p, sampler=None):
    """`make_model(y_train) -> estimador` para `evaluacion.evaluar`.

    `sampler` (imblearn) envuelve el estimador en un pipeline que remuestrea solo el
    entrenamiento (§4.1); en ese caso el peso de clase no aplica.
    """
    def make_model(y):
        est = build(algo, p)
        if sampler is None:
            return est
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline([("sampler", sampler()), ("clf", est)])
    return make_model


def tune(algo, dev, feats, folds, n_trials, nombre, sampler=None, seed=SEED):
    """Corre (o reanuda) el estudio `nombre` hasta completar `n_trials`. Devuelve el estudio."""
    ratio = float(np.mean([evaluacion.scale_pos_weight(dev.churn.iloc[tr].to_numpy())
                           for tr, _ in folds]))
    con_peso = sampler is None

    def objective(trial):
        p = espacio(algo, trial, ratio, con_peso)
        r = evaluacion.evaluar(make_model_factory(algo, p, sampler), dev, feats, folds)
        trial.set_user_attr("aucs", r["aucs"])
        trial.set_user_attr("lift10", r["lift10"])
        return r["auc"]

    study = optuna.create_study(study_name=nombre, storage=f"sqlite:///{DB}",
                                direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    hechos = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if hechos < n_trials:
        study.optimize(objective, n_trials=n_trials - hechos, gc_after_trial=True)
    return study


def mejores_params(study, algo, ratio_fallback=1.0):
    """Dict de hiperparámetros del mejor trial, listo para `build`."""
    p = dict(study.best_params)
    p.setdefault("w", ratio_fallback)
    return p
