import numpy as np
import pandas as pd
import pytest

from scripts import temporal_optuna as experiment


def panel():
    ranks = np.repeat(np.arange(1, 101), 20)
    return pd.DataFrame({"mes_rank": ranks, "churn": np.tile([0, 1], len(ranks) // 2)})


def test_temporal_folds_use_calendar_gap_and_disjoint_validation():
    df = panel()
    folds = experiment.temporal_folds(df)
    seen = set()
    for tr, va in folds:
        assert df.iloc[tr].mes_rank.max() + 6 < df.iloc[va].mes_rank.min()
        assert df.iloc[va].mes_rank.nunique() == 4
        assert not set(va) & seen
        assert not set(tr) & set(va)
        seen.update(va)
    assert df.iloc[folds[-1][1]].mes_rank.max() == 100


def test_inner_selection_split_is_inside_outer_training():
    df = panel()
    for tr, va in experiment.temporal_folds(df):
        train = df.iloc[tr].reset_index(drop=True)
        inner_tr, inner_va = experiment.temporal_folds(train, n_folds=1)[0]
        assert train.iloc[inner_tr].mes_rank.max() + 6 < train.iloc[inner_va].mes_rank.min()
        assert train.iloc[inner_va].mes_rank.max() < df.iloc[va].mes_rank.min()


@pytest.mark.parametrize("bad", ["Directo", np.nan, np.inf])
def test_invalid_numeric_data_fails_loudly(bad):
    with pytest.raises(ValueError):
        experiment.check_numeric(pd.DataFrame({"numeric": [1, bad]}))


def test_one_class_validation_rejected():
    df = panel()
    df.loc[df.mes_rank > 96, "churn"] = 0
    with pytest.raises(ValueError, match="Both target classes"):
        experiment.temporal_folds(df)


def test_transaction_features_exclude_identity_target_and_master_data():
    df = pd.DataFrame({c: [1.0, 2.0] for c in experiment.ZERO})
    for c in ["monto_u12", "monto_u3", "n_ped_u12", "n_ped_u3",
              "meses_activos_u3", "meses_activos_u12", "n_prod_u12",
              "meses_desde_compra_previa"]:
        df[c] = [1.0, 2.0]
    for c in experiment.META + experiment.MASTER:
        df[c] = ["never use", "future information"]
    transform = experiment.TransactionFeatures().fit(df)
    result = transform.transform(df)
    assert not set(result.columns) & set(experiment.META + experiment.MASTER)
    assert "ticket_prom_u12" in result
    assert result.ticket_prom_u12.tolist() == [1, 1]
    df["monto_u12"] = df["monto_u12"].astype(object)
    df.loc[0, "monto_u12"] = "Directo"
    with pytest.raises(ValueError):
        transform.transform(df)


def test_class_balance_is_computed_from_training_labels():
    params = {"balance": True, "n_estimators": 10, "max_depth": 2}
    model = experiment.build_model("xgboost", params, np.array([0, 0, 0, 1]), 1)
    assert model.get_params()["scale_pos_weight"] == 3
    assert params["balance"] is True
