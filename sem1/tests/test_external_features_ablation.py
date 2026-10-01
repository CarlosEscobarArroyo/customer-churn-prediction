import numpy as np
import pandas as pd
import pytest

import scripts.external_features_ablation as ext
from scripts.master_features_ablation import BASE_COLUMNS


def test_lagged_calendar_join_preserves_rows_and_keeps_callao_separate():
    pool = pd.DataFrame({"id_vendedor": [2, 1, 3], "mes_rank": [50]*3,
                         "mes_obs": pd.to_datetime(["2024-04-01"]*3), "churn": [1, 0, 1],
                         "departamento": [" LÍMA ", "CALLAO", "desconocido"]})
    macro = pd.DataFrame({"mes_ref": pd.to_datetime(["2024-01-01", "2024-04-01"]),
                          "inflacion": [2., 99.]})
    regional = pd.DataFrame({"mes_ref": pd.to_datetime(["2024-01-01"]*2),
                             "department_key": ["lima", "callao"], ext.REGIONAL: [5., 12.]})
    joined = ext.attach_sources(pool, macro, regional, 3)
    assert joined.id_vendedor.tolist() == [2, 1, 3]
    assert joined.inflacion.tolist() == [2., 2., 2.]
    assert joined[ext.REGIONAL].iloc[:2].tolist() == [5., 12.]
    assert pd.isna(joined[ext.REGIONAL].iloc[2])


def test_regional_growth_uses_exact_same_month_a_year_earlier(monkeypatch):
    macro = pd.DataFrame({"mes_ref": pd.to_datetime(["2020-01-01"]), "inflacion": [2.]})
    credit = pd.DataFrame({"mes_ref": pd.to_datetime(["2020-01-01", "2021-01-01", "2021-03-01"]),
                          "departamento_fuente": ["Lima"]*3,
                          "credito_total_millones_soles": [100., 120., 150.]})
    monkeypatch.setattr(ext, "read_source_table", lambda path, sheet: macro.copy() if sheet == "Macro_mensual" else credit.copy())
    _, growth, _ = ext.prepare_sources("unused")
    assert np.isclose(growth.loc[growth.mes_ref == "2021-01-01", ext.REGIONAL].iloc[0], 20.)
    assert pd.isna(growth.loc[growth.mes_ref == "2021-03-01", ext.REGIONAL].iloc[0])
    credit.loc[len(credit)] = credit.iloc[0]
    with pytest.raises(ValueError, match="Duplicate"):
        ext.prepare_sources("unused")


def test_only_selected_external_features_enter_model():
    raw = pd.DataFrame(1., index=range(2), columns=BASE_COLUMNS)
    raw = raw.assign(churn=[1, 0], department_key=["lima", "callao"],
                     future_value=999, inflacion=[2., 3.])
    features = ext.ExternalFeatures(("inflacion",)).fit_transform(raw)
    assert features.shape == (2, 43)
    assert features.external_inflacion.tolist() == [2., 3.]
    assert not {"churn", "department_key", "future_value"} & set(features.columns)
