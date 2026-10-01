"""Download NASA POWER daily meteorology and build lagged provincial features.

Only public coordinates leave the computer. Seller records and labels stay local.
Provincial capitals are spatial proxies, not province-wide climate averages.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
API = "https://power.larc.nasa.gov/api/temporal/daily/point"
DOCS = "https://power.larc.nasa.gov/docs/services/api/temporal/daily/"
GEO_URL = ("https://raw.githubusercontent.com/jmcastagnetto/"
           "ubigeo-peru-aumentado/main/ubigeo_provincia.csv")
PARAMETERS = {
    "T2M": ("temperatura_media_c", "C"),
    "T2M_MAX": ("temperatura_maxima_c", "C"),
    "T2M_MIN": ("temperatura_minima_c", "C"),
    "RH2M": ("humedad_relativa_pct", "%"),
    "PRECTOTCORR": ("precipitacion_mm", "mm/day"),
    "WS2M": ("viento_m_s", "m/s"),
}
AGGREGATIONS = {
    "temperatura_media_c": ("temperatura_media_c", "mean"),
    "temperatura_maxima_media_c": ("temperatura_maxima_c", "mean"),
    "temperatura_minima_media_c": ("temperatura_minima_c", "mean"),
    "temperatura_maxima_c": ("temperatura_maxima_c", "max"),
    "temperatura_minima_c": ("temperatura_minima_c", "min"),
    "humedad_relativa_pct": ("humedad_relativa_pct", "mean"),
    "precipitacion_total_mm": ("precipitacion_mm", "sum"),
    "precipitacion_maxima_diaria_mm": ("precipitacion_mm", "max"),
    "dias_lluvia_ge1mm": ("precipitacion_mm", "rain_days"),
    "viento_m_s": ("viento_m_s", "mean"),
}
KEYS = ["departamento_key", "provincia_key"]


def normalize(value):
    if pd.isna(value):
        return "__sin_dato__"
    text = unicodedata.normalize("NFKD", str(value).strip().casefold())
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split()) or "__sin_dato__"


def geo_keys(frame):
    frame = frame.copy()
    for name, key in zip(("departamento", "provincia"), KEYS):
        frame[key] = frame[name].map(normalize)
    # Explicit correction of the truncated spelling in the local source.
    alias = frame.departamento_key.eq("tumbes") & frame.provincia_key.eq("contralmirante villa")
    frame.loc[alias, "provincia_key"] = "contralmirante villar"
    return frame


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def get_bytes(url, attempts=4):
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "GlamourClimateResearch/1.0"})
            with urlopen(request, timeout=120) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            retryable = not isinstance(error, HTTPError) or error.code in (429, 500, 502, 503, 504)
            if not retryable or attempt == attempts - 1:
                raise
            delay = min(2 ** (attempt + 1), 30)
            if isinstance(error, HTTPError):
                retry_after = error.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = max(delay, int(retry_after))
            print(f"Reintento en {delay}s: {error}", flush=True)
            time.sleep(delay)
    raise RuntimeError("Unreachable")


def locations_for_dataset(dataset, catalog):
    source = geo_keys(catalog)
    if source.duplicated(KEYS).any():
        raise ValueError("Duplicate province keys in geographic catalog")
    source = source.rename(columns={"inei": "ubigeo_provincia"})
    locations = geo_keys(dataset)[KEYS].drop_duplicates().merge(
        source[KEYS + ["ubigeo_provincia", "capital", "latitude", "longitude"]],
        on=KEYS, how="left", validate="one_to_one")
    matched = locations.latitude.notna() & locations.longitude.notna()
    if not (locations.loc[matched, "latitude"].between(-90, 90).all()
            and locations.loc[matched, "longitude"].between(-180, 180).all()):
        raise ValueError("Invalid geographic coordinates")
    # MERRA-2 native grid: 0.5 degrees latitude x 0.625 longitude.
    # Query the nearest cell center once, then reuse it across provinces.
    locations["grid_latitude"] = np.floor(locations.latitude / 0.5 + 0.5) * 0.5
    locations["grid_longitude"] = np.floor(locations.longitude / 0.625 + 0.5) * 0.625
    locations["grid_id"] = pd.Series(pd.NA, index=locations.index, dtype="string")
    locations.loc[matched, "grid_id"] = locations.loc[matched].apply(
        lambda r: f"{r.grid_latitude:.3f}_{r.grid_longitude:.3f}", axis=1)
    return locations


def request_url(latitude, longitude, start, end):
    return API + "?" + urlencode({
        "parameters": ",".join(PARAMETERS), "community": "AG",
        "longitude": longitude, "latitude": latitude,
        "start": pd.Timestamp(start).strftime("%Y%m%d"),
        "end": pd.Timestamp(end).strftime("%Y%m%d"),
        "format": "JSON", "time-standard": "LST",
    })


def parse_daily(payload, start, end):
    expected = pd.date_range(start, end, freq="D", name="fecha")
    header = payload["header"]
    if header["time_standard"] != "LST":
        raise ValueError("Unexpected time standard")
    data = payload["properties"]["parameter"]
    frame = pd.DataFrame(index=expected)
    for parameter, (column, unit) in PARAMETERS.items():
        if payload["parameters"][parameter]["units"] != unit:
            raise ValueError(f"Unexpected unit for {parameter}")
        series = pd.Series(data[parameter], dtype=float)
        series.index = pd.to_datetime(series.index, format="%Y%m%d")
        if series.index.duplicated().any() or not series.index.isin(expected).all():
            raise ValueError(f"Invalid dates for {parameter}")
        frame[column] = series.reindex(expected).replace(header["fill_value"], np.nan)
        if np.isinf(frame[column]).any():
            raise ValueError(f"Non-finite values for {parameter}")
    return frame.reset_index()


def download_grid(row, start, end, cache):
    url = request_url(row.grid_latitude, row.grid_longitude, start, end)
    key = hashlib.sha256(url.encode()).hexdigest()
    path = cache / f"{key}.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record["url"] != url:
            raise ValueError("Cache URL mismatch")
    else:
        payload = json.loads(get_bytes(url))
        parse_daily(payload, start, end)  # Validate before caching.
        record = {"url": url, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                  "response": payload}
        save_json(path, record)
        time.sleep(0.5)
    frame = parse_daily(record["response"], start, end)
    frame.insert(0, "grid_id", row.grid_id)
    return frame, {"grid_id": row.grid_id, "url": url, "file": str(path),
                   "sha256": sha256(path), "downloaded_at_utc": record["downloaded_at_utc"]}


def aggregate_monthly(daily):
    if daily.duplicated(["grid_id", "fecha"]).any():
        raise ValueError("Duplicate grid/day keys")
    d = daily.copy()
    d["mes_ref"] = pd.to_datetime(d.fecha).dt.to_period("M").dt.to_timestamp()
    groups = d.groupby(["grid_id", "mes_ref"], sort=True)
    result = groups.size().rename("dias_recibidos").to_frame()
    result["dias_esperados"] = result.index.get_level_values("mes_ref").days_in_month
    for column in (v[0] for v in PARAMETERS.values()):
        result[f"dias_validos_{column}"] = groups[column].count()
    for feature, (column, operation) in AGGREGATIONS.items():
        if operation == "rain_days":
            values = groups[column].agg(lambda x: int(x.ge(1).sum()))
        else:
            values = groups[column].agg(operation)
        # A partial month must never look like a complete total or average.
        complete = result[f"dias_validos_{column}"].eq(result.dias_esperados)
        result[feature] = values.where(complete)
    result["mes_completo"] = result[[f"dias_validos_{v[0]}" for v in PARAMETERS.values()]].eq(
        result.dias_esperados, axis=0).all(axis=1)
    return result.reset_index()


def attach_climate(dataset, provincial, lag=3):
    """Exact calendar join; no future values, interpolation or imputation."""
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("lag must be a positive integer")
    source = provincial.copy()
    source["mes_ref"] = pd.to_datetime(source.mes_ref)
    if source.duplicated(KEYS + ["mes_ref"]).any():
        raise ValueError("Duplicate province/month keys")
    if not source.mes_ref.dt.is_month_start.all():
        raise ValueError("Source months must start on day 1")
    source = source[KEYS + ["mes_ref"] + list(AGGREGATIONS)].rename(
        columns={col: f"nasa_{col}" for col in AGGREGATIONS})
    # Exact prior-year month rather than a row shift across missing months.
    history = source[KEYS + ["mes_ref", "nasa_temperatura_media_c", "nasa_precipitacion_total_mm"]].copy()
    history["mes_ref"] += pd.DateOffset(years=1)
    history = history.rename(columns={"nasa_temperatura_media_c": "temp_prev",
                                      "nasa_precipitacion_total_mm": "rain_prev"})
    source = source.merge(history, on=KEYS + ["mes_ref"], how="left", validate="one_to_one")
    source["nasa_temperatura_delta12_c"] = source.nasa_temperatura_media_c - source.temp_prev
    source["nasa_precipitacion_delta12_mm"] = source.nasa_precipitacion_total_mm - source.rain_prev
    source = source.drop(columns=["temp_prev", "rain_prev"])
    d = geo_keys(dataset)
    months = pd.to_datetime(d.mes_obs)
    if months.isna().any() or not months.dt.is_month_start.all():
        raise ValueError("mes_obs must contain first-of-month dates")
    d["nasa_mes_ref"] = months - pd.DateOffset(months=lag)
    joined = d.merge(source.rename(columns={"mes_ref": "nasa_mes_ref"}),
                     on=KEYS + ["nasa_mes_ref"], how="left", sort=False, validate="many_to_one")
    pd.testing.assert_frame_equal(joined[dataset.columns].reset_index(drop=True),
                                  dataset.reset_index(drop=True))
    joined["nasa_sin_dato"] = joined[[f"nasa_{c}" for c in AGGREGATIONS]].isna().any(axis=1).astype(int)
    return joined.drop(columns=KEYS)


def run(dataset_path=ROOT / "data/processed/churn_dataset.csv",
        output=ROOT / "data/external/nasa_power", start="2016-01-01", end="2025-12-31",
        lag=3, workers=2, catalog_path=None):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end or start < pd.Timestamp("1981-01-01"):
        raise ValueError("Invalid NASA POWER date range")
    if not start.is_month_start or not end.is_month_end:
        raise ValueError("Use complete calendar months for the monthly outputs")
    if lag < 1 or not 1 <= workers <= 3:
        raise ValueError("Use lag >= 1 and between 1 and 3 workers")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "raw"
    cache.mkdir(exist_ok=True)
    geography = out / "ubigeo_provincia_source.csv"
    if not geography.exists():
        geography.write_bytes(Path(catalog_path).read_bytes() if catalog_path else get_bytes(GEO_URL))
    elif catalog_path and sha256(geography) != sha256(catalog_path):
        raise ValueError("Changed geographic catalog; use a new output directory")
    dataset = pd.read_csv(dataset_path)
    locations = locations_for_dataset(dataset, pd.read_csv(geography, dtype={"inei": str}))
    protocol = {"start": str(start.date()), "end": str(end.date()), "lag_months_assumed": lag,
                "parameters": list(PARAMETERS), "time_standard": "LST", "community": "AG",
                "dataset_sha256": sha256(dataset_path), "geography_sha256": sha256(geography),
                "code_sha256": sha256(__file__), "geography_url": GEO_URL, "api_docs": DOCS,
                "spatial_method": "nearest MERRA-2 cell center to provincial capital (0.5 x 0.625 degrees)",
                "availability": "revised historical snapshot, no historical publication vintages",
                "location_limit": "current seller province snapshot; historical residence unverified"}
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Changed extraction protocol; use a new output directory")
    save_json(protocol_path, protocol)
    locations.to_csv(out / "ubicaciones.csv", index=False)
    grids = locations.dropna(subset=["grid_id"]).drop_duplicates("grid_id")
    print(f"{locations.grid_id.notna().sum()} provincias, {len(grids)} celdas NASA; {start.date()} a {end.date()}", flush=True)
    if grids.empty:
        raise ValueError("No provinces matched the geographic catalog")
    frames, manifest = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = [executor.submit(download_grid, row, start, end, cache) for row in grids.itertuples()]
        for i, job in enumerate(as_completed(jobs), 1):
            frame, record = job.result()
            frames.append(frame)
            manifest.append(record)
            print(f"[{i}/{len(jobs)}] {record['grid_id']}", flush=True)
    daily = pd.concat(frames, ignore_index=True).sort_values(["grid_id", "fecha"]).reset_index(drop=True)
    monthly = aggregate_monthly(daily)
    provincial = locations.dropna(subset=["grid_id"]).merge(monthly, on="grid_id", validate="many_to_many")
    if provincial.duplicated(KEYS + ["mes_ref"]).any():
        raise ValueError("Unexpected duplicate provincial months")
    joined = attach_climate(dataset, provincial, lag)
    daily.to_parquet(out / "clima_diario_celdas.parquet", index=False)
    daily.to_csv(out / "clima_diario_celdas.csv", index=False)
    monthly.to_parquet(out / "clima_mensual_celdas.parquet", index=False)
    provincial.to_parquet(out / "clima_mensual_provincias.parquet", index=False)
    provincial.to_csv(out / "clima_mensual_provincias.csv", index=False)
    joined.to_parquet(out / "churn_con_clima.parquet", index=False)
    counts = geo_keys(dataset).groupby(KEYS, dropna=False).size().rename("filas_modelo").reset_index()
    coverage = counts.merge(locations, on=KEYS, validate="one_to_one")
    coverage.to_csv(out / "cobertura_ubicaciones.csv", index=False)
    summary = {"provinces": int(locations.grid_id.notna().sum()), "grid_cells": len(grids),
               "daily_grid_rows": len(daily), "monthly_province_rows": len(provincial),
               "model_rows": len(joined), "model_rows_with_climate": int(joined.nasa_sin_dato.eq(0).sum()),
               "model_rows_missing_geography": int(coverage.loc[coverage.grid_id.isna(), "filas_modelo"].sum()),
               "incomplete_grid_months": int((~monthly.mes_completo).sum()),
               "daily_missing_values": {col: int(daily[col].isna().sum()) for col, _ in PARAMETERS.values()},
               "climate_features": [c for c in joined.columns if c.startswith("nasa_") and c != "nasa_mes_ref"]}
    save_json(out / "manifest.json", sorted(manifest, key=lambda x: x["grid_id"]))
    save_json(out / "resumen.json", summary)
    save_json(out / "diccionario.json", {"daily": PARAMETERS, "monthly": AGGREGATIONS,
              "derived": {"nasa_temperatura_delta12_c": "reference month minus exact same month one year earlier",
                          "nasa_precipitacion_delta12_mm": "reference month minus exact same month one year earlier",
                          "nasa_sin_dato": "1 if any of the ten monthly climate features is missing"},
              "rain_days_threshold_mm": 1, "monthly_missing_policy": "null if any required day is missing",
              "lag_months_assumed": lag, "time_standard": "LST, not civil Peru time"})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/churn_dataset.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/external/nasa_power")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--lag", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--catalog", type=Path, help="Optional local copy of the public geographic catalog")
    args = parser.parse_args()
    run(args.dataset, args.output, args.start, args.end, args.lag, args.workers, args.catalog)


if __name__ == "__main__":
    main()
