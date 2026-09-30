import numpy as np
import pandas as pd
import pytest

from scripts.climate_trends_ablation import TERMS, attach_trends, prepare_trends, trends_column


def sample():
    source = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=30, freq="MS")})
    for i, term in enumerate(TERMS):
        source[term] = np.arange(30) + i
    return source


def test_expanding_features_do_not_use_future_and_cancel_multiplicative_scale():
    source = sample()
    result = prepare_trends(source)
    changed = source.copy()
    changed.loc[20:, list(TERMS)] = 99
    pd.testing.assert_frame_equal(result.iloc[:20], prepare_trends(changed).iloc[:20])
    scaled = source.copy()
    scaled[list(TERMS)] *= 0.5
    z = [trends_column(t, "zexp") for t in TERMS]
    np.testing.assert_allclose(result[z], prepare_trends(scaled)[z], equal_nan=True)
    assert result[z].iloc[:11].isna().all().all()
    source[TERMS[0]] = 0
    assert prepare_trends(source)[z[0]].isna().all()


def test_trends_lag_and_row_order_and_missing_calendar_validation():
    source = sample()
    pool = pd.DataFrame({"id_vendedor": [8, 1], "mes_obs": pd.to_datetime(["2021-05-01", "2021-04-01"]), "churn": [1, 0]})
    joined = attach_trends(pool, prepare_trends(source), 3)
    assert joined.id_vendedor.tolist() == [8, 1]
    assert joined[trends_column(TERMS[0], "indice")].tolist() == [13, 12]
    assert joined.trends_mes_ref.tolist() == list(pd.to_datetime(["2021-02-01", "2021-01-01"]))
    with pytest.raises(ValueError, match="Missing calendar"):
        prepare_trends(source.drop(index=12))
    with pytest.raises(ValueError, match="unique"):
        prepare_trends(pd.concat([source, source.iloc[:1]]))
    with pytest.raises(ValueError, match="Positive"):
        attach_trends(pool, prepare_trends(source), 0)
