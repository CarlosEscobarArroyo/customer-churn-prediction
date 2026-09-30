import numpy as np
import pandas as pd
import pytest

from scripts.regional_activity_ablation import attach_activity, prepare_activity


def source():
    dates = pd.date_range("2020-01-01", "2022-01-01", freq="MS")
    panel = pd.DataFrame([{"id_vendedor": seller, "mes_obs": date, "monto": amount,
                           "n_ped": 1, "activo": 1}
                          for seller, amount in [(1, 10.), (2, 20.), (3, 90.)] for date in dates])
    master = pd.DataFrame({"id_vendedor": [1, 2, 3], "departamento": [" LÍMA ", "lima", "Callao"]})
    pool = pd.DataFrame({"id_vendedor": [1, 3, 2, 999], "mes_obs": pd.to_datetime(["2021-07-01"]*4),
                         "departamento": ["Lima", "Callao", "lima", None], "churn": [1, 0, 0, 1]})
    return panel, master, pool


def test_exact_past_windows_leave_out_own_sales_and_keep_callao_separate():
    panel, master, pool = source()
    reg, own, _ = prepare_activity(panel, master, "2021-06-01")
    d = attach_activity(pool, reg, own)
    assert d.regional_ventas_mean3_h0.iloc[:3].tolist() == [20., 0., 10.]
    assert d.regional_vendedoras_activas_mean3_h0.iloc[:3].tolist() == [1., 0., 1.]
    assert d.regional_ref_6.eq(pd.Timestamp("2020-12-01")).all()
    assert d.regional_ref_12.eq(pd.Timestamp("2020-06-01")).all()
    assert d.regional_cambio_ventas_6m.iloc[:3].eq(0).all()
    assert pd.isna(d.regional_ventas_mean3_h0.iloc[3])
    assert d.regional_sin_historia_h0.iloc[3] == 1
    changed = panel.copy()
    changed.loc[changed.id_vendedor.eq(1), "monto"] *= 100
    r2, o2, _ = prepare_activity(changed, master, "2021-06-01")
    d2 = attach_activity(pool, r2, o2)
    for h in (0, 6, 12):
        assert np.isclose(d2[f"regional_ventas_mean3_h{h}"].iloc[0], d[f"regional_ventas_mean3_h{h}"].iloc[0])


def test_future_and_labels_do_not_change_context_and_missing_history_is_not_zero():
    panel, master, pool = source()
    reg, own, _ = prepare_activity(panel, master, "2021-06-01")
    d = attach_activity(pool, reg, own)
    panel.loc[panel.mes_obs.gt("2021-06-01"), "monto"] = 99999.
    r2, o2, _ = prepare_activity(panel, master, "2021-06-01")
    d2 = attach_activity(pool.assign(churn=1-pool.churn), r2, o2)
    pd.testing.assert_frame_equal(d.filter(regex='^regional_'), d2.filter(regex='^regional_'))
    early = pool.iloc[:1].assign(mes_obs=pd.Timestamp("2020-03-01"))
    e = attach_activity(early, reg, own)
    assert e.regional_sin_historia_h0.iloc[0] == 1  # Only two observed source months.
    assert pd.isna(e.regional_log_ventas_h0.iloc[0])


def test_reject_duplicate_or_incomplete_purchase_panel():
    panel, master, _ = source()
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_activity(pd.concat([panel, panel.iloc[:1]]), master, "2021-06-01")
    with pytest.raises(ValueError, match="dense"):
        prepare_activity(panel.drop(index=4), master, "2021-06-01")
