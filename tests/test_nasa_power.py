import json
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import nasa_power as nasa


def payload(start="2024-02-01", end="2024-02-29"):
    dates = pd.date_range(start, end).strftime("%Y%m%d")
    return {
        "header": {"time_standard": "LST", "fill_value": -999.0},
        "parameters": {p: {"units": unit} for p, (_, unit) in nasa.PARAMETERS.items()},
        "properties": {"parameter": {p: dict.fromkeys(dates, 2.0) for p in nasa.PARAMETERS}},
    }


def test_leap_month_rainfall_units_and_missing_days():
    source = payload()
    daily = nasa.parse_daily(source, "2024-02-01", "2024-02-29").assign(grid_id="test")
    month = nasa.aggregate_monthly(daily).iloc[0]
    assert month.precipitacion_total_mm == 58
    assert month.dias_lluvia_ge1mm == 29
    assert month.temperatura_media_c == 2
    assert month.mes_completo
    source["properties"]["parameter"]["PRECTOTCORR"]["20240201"] = -999
    del source["properties"]["parameter"]["T2M"]["20240202"]
    daily = nasa.parse_daily(source, "2024-02-01", "2024-02-29").assign(grid_id="test")
    month = nasa.aggregate_monthly(daily).iloc[0]
    assert pd.isna(month.precipitacion_total_mm)
    assert pd.isna(month.dias_lluvia_ge1mm)
    assert pd.isna(month.temperatura_media_c)
    assert month.humedad_relativa_pct == 2
    assert month.dias_validos_precipitacion_mm == 28
    assert not month.mes_completo


def test_partial_month_and_duplicate_days():
    daily = nasa.parse_daily(payload(), "2024-02-01", "2024-02-29").assign(grid_id="test")
    month = nasa.aggregate_monthly(daily.iloc[:10]).iloc[0]
    assert not month.mes_completo
    assert pd.isna(month.precipitacion_total_mm)
    with pytest.raises(ValueError, match="Duplicate"):
        nasa.aggregate_monthly(pd.concat([daily, daily.iloc[:1]]))


def test_units_are_checked():
    source = payload()
    source["parameters"]["PRECTOTCORR"]["units"] = "mm/hour"
    with pytest.raises(ValueError, match="unit"):
        nasa.parse_daily(source, "2024-02-01", "2024-02-29")


def test_normalization_alias_grid_reuse_and_unknown_location():
    dataset = pd.DataFrame({"departamento": [" TÚMBES ", "Lima", "Callao", None],
                            "provincia": ["contralmirante villa", "Lima", "Callao", None]})
    catalog = pd.DataFrame({"departamento": ["TUMBES", "LIMA", "CALLAO"],
                            "provincia": ["CONTRALMIRANTE VILLAR", "LIMA", "CALLAO"],
                            "inei": ["240200", "150100", "070100"],
                            "capital": ["Zorritos", "Lima", "Callao"],
                            "latitude": [-3.68, -12.04, -12.06],
                            "longitude": [-80.67, -77.03, -77.15]})
    locations = nasa.locations_for_dataset(dataset, catalog)
    assert locations.grid_id.notna().sum() == 3
    assert locations.iloc[1].grid_id == locations.iloc[2].grid_id
    assert locations.iloc[0].ubigeo_provincia == "240200"
    assert pd.isna(locations.iloc[3].grid_id)


def test_lag_exact_year_join_and_row_preservation():
    months = pd.to_datetime(["2023-01-01", "2024-01-01", "2024-02-01"])
    source = pd.DataFrame({"departamento_key": "lima", "provincia_key": "lima", "mes_ref": months})
    for feature in nasa.AGGREGATIONS:
        source[feature] = [10.0, 13.0, 999.0]
    dataset = pd.DataFrame({"mes_obs": ["2024-04-01", "2024-05-01", "2024-04-01"],
                            "departamento": ["Lima", "Lima", None],
                            "provincia": ["LIMA", "Lima", None], "churn": [0, 1, 0]})
    joined = nasa.attach_climate(dataset, source, lag=3)
    pd.testing.assert_frame_equal(joined[dataset.columns], dataset)
    assert joined.iloc[0].nasa_temperatura_media_c == 13
    assert joined.iloc[0].nasa_temperatura_delta12_c == 3
    assert pd.isna(joined.iloc[1].nasa_temperatura_delta12_c)
    assert joined.iloc[2].nasa_sin_dato == 1
    changed = source.copy()
    changed.loc[2, "temperatura_media_c"] = -500
    assert nasa.attach_climate(dataset, changed, 3).iloc[0].nasa_temperatura_media_c == 13
    with pytest.raises(ValueError, match="positive"):
        nasa.attach_climate(dataset, source, lag=0)
    with pytest.raises(ValueError, match="Duplicate"):
        nasa.attach_climate(dataset, pd.concat([source, source.iloc[:1]]))


def test_cached_download_does_not_hit_network(tmp_path, monkeypatch):
    row = SimpleNamespace(grid_id="test", grid_latitude=-12.0, grid_longitude=-76.875)
    url = nasa.request_url(row.grid_latitude, row.grid_longitude, "2024-02-01", "2024-02-29")
    key = nasa.hashlib.sha256(url.encode()).hexdigest()
    (tmp_path / f"{key}.json").write_text(json.dumps({"url": url, "response": payload(),
                                                    "downloaded_at_utc": "2026-09-20T00:00:00Z"}))
    def forbidden(*args):
        raise AssertionError("Cache should prevent network calls")
    monkeypatch.setattr(nasa, "get_bytes", forbidden)
    frame, record = nasa.download_grid(row, "2024-02-01", "2024-02-29", tmp_path)
    assert len(frame) == 29
    assert record["url"] == url


def test_retry_transient_but_not_bad_request(monkeypatch):
    from urllib.error import HTTPError
    attempts = []
    def fail(request, timeout):
        attempts.append(request)
        raise HTTPError(request.full_url, 503, "Unavailable", {}, None)
    monkeypatch.setattr(nasa, "urlopen", fail)
    monkeypatch.setattr(nasa.time, "sleep", lambda _: None)
    with pytest.raises(HTTPError):
        nasa.get_bytes("https://example.test", attempts=3)
    assert len(attempts) == 3
    def bad_request(request, timeout):
        raise HTTPError(request.full_url, 422, "Invalid", {}, None)
    monkeypatch.setattr(nasa, "urlopen", bad_request)
    with pytest.raises(HTTPError) as error:
        nasa.get_bytes("https://example.test")
    assert error.value.code == 422
