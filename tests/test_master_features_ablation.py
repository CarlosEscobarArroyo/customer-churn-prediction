import numpy as np
import pandas as pd
import optuna

from scripts.master_features_ablation import (
    BASE_COLUMNS, MasterFeatures, normalize_category, parallel_logistic_objective,
    preprocessor, score_metrics,
)
from scripts.temporal_optuna import objective


def test_explicit_schema_excludes_labels_campaigns_and_unused_master_data():
    raw = pd.DataFrame(1.0, index=range(3), columns=BASE_COLUMNS)
    raw = raw.assign(churn=[0, 1, 0], camp_saltadas=999, edad=40, id_vendedor=987)
    baseline = MasterFeatures().fit_transform(raw)
    assert baseline.shape == (3, 42)
    changed = raw.assign(churn=1, camp_saltadas=-100, edad=90, id_vendedor=123)
    pd.testing.assert_frame_equal(baseline, MasterFeatures().fit_transform(changed))


def test_imputation_and_vocabulary_use_only_training():
    train = pd.DataFrame(1.0, index=range(3), columns=BASE_COLUMNS)
    train = train.assign(edad=[20, 40, np.nan], provincia=[" LÍMA ", "lima", None])
    valid = train.iloc[:2].copy().assign(edad=[np.nan, 90], provincia=["LIMA", "NUEVA"])
    features = MasterFeatures(("edad", "provincia")).fit(train)
    a, b = features.transform(train), features.transform(valid)
    encoder = preprocessor(a, features.extras).fit(a)
    values = encoder.transform(b)
    names = encoder.get_feature_names_out().tolist()
    assert values[0, names.index("edad")] == 30
    assert values[0, names.index("edad_sin_dato")] == 1
    assert not any("nueva" in n for n in names)
    categorical = [i for i, n in enumerate(names) if n.startswith("provincia_")]
    assert values[1, categorical].sum() == 0
    assert normalize_category("  SAN   MARTÍN ") == "san martin"


def test_monthly_prioritization_does_not_rank_months_together():
    predictions = pd.DataFrame({
        "fold": [0] * 20, "mes_obs": ["2024-01-01"] * 10 + ["2024-02-01"] * 10,
        "id_vendedor": list(range(20)), "churn": [1] + [0] * 9 + [1] * 5 + [0] * 5,
        "score": list(np.linspace(.99, .90, 10)) + list(np.linspace(.89, .1, 10)),
    })
    summary, _, monthly = score_metrics(predictions)
    top = monthly[monthly.top_fraction == .1]
    assert top.n_top.tolist() == [1, 1]
    assert summary["precision_top10"] == 1
    assert summary["recall_top10"] == 2 / 6
    assert summary["lift_top10"] == 6  # mean(10x, 2x), not aggregate precision/prevalence


def test_parallel_folds_preserve_original_objective_and_diagnostics():
    rng = np.random.default_rng(31)
    folds = []
    for _ in range(4):
        x = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd"))
        y = (x.a + rng.normal(size=100) > 0).astype(int).to_numpy()
        folds.append({"X_train": x.iloc[:60], "y_train": y[:60],
                      "X_valid": x.iloc[60:], "y_valid": y[60:],
                      "valid_months": np.repeat(["2024-01", "2024-02"], 20),
                      "info": {"selected": ["a", "c"]}})
    for feature_mode in ("all", "selected"):
        params = {"features": feature_mode, "balance": True, "C": .1, "l1_ratio": .3}
        serial, parallel = optuna.trial.FixedTrial(params), optuna.trial.FixedTrial(params)
        assert objective(serial, "logreg", folds, 1) == parallel_logistic_objective(parallel, folds, 4)
        assert serial.user_attrs == parallel.user_attrs
