"""Exploratory external-data ablation; fixed parameters, no Optuna search.

The supplied workbook is a revised snapshot without historical publication dates.
A configurable calendar lag is an assumption, not proof of point-in-time availability.
"""
import argparse
import fcntl
import importlib.metadata
import json
import posixpath
from pathlib import Path
import warnings
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from scripts.master_features_ablation import (
    MasterFeatures, development_data, normalize_category, score_metrics,
)
from scripts.recent_windows import recent_training_indices
from scripts.temporal_optuna import META, ROOT, build_model, digest, write_json

WORKBOOK = ROOT / "data/fuentes_datos_externos_glamour.xlsx"
BASE_REPORT = ROOT / "reports/master_features_comparison_v1"
PARAMS = ROOT / "reports/master_features_v1/reconstructed_tuning"
REGIONAL = "credito_regional_var12_pct"
PRIORITY = ("inflacion_nacional_var12_pct", "comercio_var12_pct",
            "expect_demanda_3m_indice", REGIONAL)
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_source_table(path, sheet_name):
    """Read literal source cells, preserving blanks; never modify the workbook."""
    with ZipFile(path) as archive:
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        properties = book.find("s:workbookPr", NS)
        if properties is not None and properties.get("date1904") in ("1", "true"):
            raise ValueError("Expected Excel 1900 date system")
        rels = {r.get("Id"): r.get("Target") for r in
                ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(t.text or "" for t in si.iterfind(".//s:t", NS))
                       for si in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        sheet = next(s for s in book.find("s:sheets", NS) if s.get("name") == sheet_name)
        target = rels[sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")]
        member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        records = []
        for row in ET.fromstring(archive.read(member)).findall("s:sheetData/s:row", NS):
            values = {}
            for cell in row.findall("s:c", NS):
                if cell.find("s:f", NS) is not None:
                    raise ValueError("Formula found in source table; verify its cached result first")
                letters = "".join(c for c in cell.get("r") if c.isalpha())
                index = 0
                for letter in letters:
                    index = index * 26 + ord(letter) - ord("A") + 1
                v = cell.find("s:v", NS)
                value = v.text if v is not None else None
                kind = cell.get("t")
                if kind == "inlineStr":
                    value = "".join(t.text or "" for t in cell.iterfind(".//s:t", NS))
                elif kind == "s":
                    value = strings[int(value)]
                elif value is not None and kind not in ("str", "e"):
                    value = float(value)
                elif kind == "e":
                    raise ValueError(f"Source error {value} in {sheet_name}/{cell.get('r')}")
                values[index - 1] = value
            records.append(values)
    width = len(records[0])
    matrix = [[r.get(i) for i in range(width)] for r in records]
    frame = pd.DataFrame(matrix[1:], columns=matrix[0])
    frame["mes_ref"] = pd.to_datetime(frame.mes_ref, unit="D", origin="1899-12-30")
    if not frame.mes_ref.dt.is_month_start.all():
        raise ValueError("Monthly source keys must be first-of-month dates")
    return frame


def prepare_sources(workbook):
    macro = read_source_table(workbook, "Macro_mensual").sort_values("mes_ref")
    regional = read_source_table(workbook, "Credito_regional")
    regional["department_key"] = regional.departamento_fuente.map(normalize_category)
    regional = regional.sort_values(["department_key", "mes_ref"])
    if macro.mes_ref.duplicated().any() or regional.duplicated(["department_key", "mes_ref"]).any():
        raise ValueError("Duplicate source keys would multiply seller-month rows")
    macro_cols = macro.columns.drop("mes_ref").tolist()
    if not np.isfinite(macro[macro_cols].to_numpy(dtype=float)).all():
        raise ValueError("Missing/non-finite macro source value")
    if not np.isfinite(regional.credito_total_millones_soles).all() or (regional.credito_total_millones_soles <= 0).any():
        raise ValueError("Regional credit balances must be positive and finite")
    # Match exact calendar months, not row positions; this cannot bridge missing months.
    prior = regional[["department_key", "mes_ref", "credito_total_millones_soles"]].copy()
    prior["mes_ref"] = prior.mes_ref + pd.DateOffset(years=1)
    prior = prior.rename(columns={"credito_total_millones_soles": "credit_year_ago"})
    regional = regional.merge(prior, on=["department_key", "mes_ref"], how="left", validate="one_to_one")
    regional[REGIONAL] = 100 * (regional.credito_total_millones_soles / regional.credit_year_ago - 1)
    return macro, regional, macro_cols


def attach_sources(pool, macro, regional, lag):
    if lag < 1:
        raise ValueError("A positive lag is required; no publication dates are available")
    d = pool.copy()
    d["external_ref_month"] = d.mes_obs - pd.DateOffset(months=lag)
    d["department_key"] = d.departamento.map(normalize_category)
    d = d.merge(macro.rename(columns={"mes_ref": "external_ref_month"}),
                on="external_ref_month", how="left", validate="many_to_one", sort=False)
    d = d.merge(regional[["department_key", "mes_ref", REGIONAL]].rename(columns={"mes_ref": "external_ref_month"}),
                on=["department_key", "external_ref_month"], how="left", validate="many_to_one", sort=False)
    pd.testing.assert_frame_equal(d[META].reset_index(drop=True), pool[META].reset_index(drop=True))
    assert (d.external_ref_month < d.mes_obs).all()
    macro_cols = macro.columns.drop("mes_ref")
    if d[macro_cols].isna().any().any():
        raise ValueError("Macro source does not cover all required lagged months")
    return d


class ExternalFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        self.columns_ = self.transform(X).columns.tolist()
        return self

    def transform(self, X):
        d = MasterFeatures().transform(X)
        for col in self.columns:
            d[f"external_{col}"] = pd.to_numeric(X[col], errors="raise")
            if col == REGIONAL:
                d["external_credito_regional_sin_dato"] = X[col].isna().astype(float)
        return d


def run(workbook=WORKBOOK, output=ROOT / "reports/external_features_v1", lag=3, threads=4):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    pool, folds = development_data()
    macro, regional, macro_cols = prepare_sources(workbook)
    joined = attach_sources(pool, macro, regional, lag)
    variants = {"transaccional": (), "mas_inflacion": (PRIORITY[0],),
                "mas_comercio": (PRIORITY[1],), "mas_expectativas_demanda": (PRIORITY[2],),
                "mas_credito_regional": (REGIONAL,), "cuatro_fuentes": PRIORITY,
                "todos_externos": tuple(macro_cols) + (REGIONAL,)}
    parameter_hashes = {n: digest(PARAMS / f"{n}_best.json") for n in ("logreg", "xgboost")}
    base_protocol = json.loads((BASE_REPORT / "protocol.json").read_text())
    versions = {p: importlib.metadata.version(p) for p in base_protocol["versions"]}
    assert base_protocol["raw_sha256"] == digest(ROOT / "data/processed/churn_dataset.csv")
    assert base_protocol["parameter_files"] == parameter_hashes
    assert base_protocol["versions"] == versions
    protocol = {"workbook_sha256": digest(workbook), "raw_sha256": base_protocol["raw_sha256"],
                "code_sha256": digest(__file__), "parameters": parameter_hashes,
                "dependencies": {f: digest(ROOT / f) for f in ["scripts/temporal_optuna.py", "scripts/master_features_ablation.py", "scripts/recent_windows.py"]},
                "lag_months_assumed": lag, "threads": threads, "versions": versions,
                "training_end": "2025-01-01", "variants": {k: list(v) for k, v in variants.items()},
                "status": "exploratory revised snapshot; historical publication dates/vintages unavailable"}
    if (out / "protocol.json").exists() and json.loads((out / "protocol.json").read_text()) != protocol:
        raise ValueError("Changed protocol; use a new output directory")
    write_json(out / "protocol.json", protocol)
    write_json(out / "variables.json", {"lag_months": lag, "variants": variants,
               "regional_formula": "100 * (credit_at_ref_month / credit_same_department_12_months_earlier - 1)",
               "macro_columns": macro_cols, "raw_credit_balance_used_as_predictor": False})
    coverage = []
    known_regions = set(regional.department_key)
    for part, indices in [("development", np.arange(len(joined))),
                          ("validation", np.concatenate([v for _, v in folds]))]:
        d = joined.iloc[indices]
        coverage.append({"partition": part, "rows": len(d),
                         "missing_region_key": int((~d.department_key.isin(known_regions)).sum()),
                         "missing_regional_growth": int(d[REGIONAL].isna().sum()),
                         "missing_macro_rows": int(d[macro_cols].isna().any(axis=1).sum())})
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    joined.to_parquet(out / "joined_development.parquet", index=False)
    macro.to_csv(out / "source_macro.csv", index=False)
    regional.to_csv(out / "source_credit_growth.csv", index=False)
    params = {n: json.loads((PARAMS / f"{n}_best.json").read_text())["params"] for n in ("logreg", "xgboost")}
    write_json(out / "parameters.json", params)
    summaries, block_tables, month_tables, component_rows = [], [], [], []
    for variant, columns in variants.items():
        chunks = []
        for i, (tr, va) in enumerate(folds):
            expected = joined.iloc[va][META].reset_index(drop=True).assign(fold=i)
            file = out / f"predictions_{variant}_fold{i}.parquet"
            source = BASE_REPORT / f"predictions_transaccional_fold{i}.parquet" if not columns else file
            if source.exists():
                predicted = pd.read_parquet(source)
                pd.testing.assert_frame_equal(predicted[expected.columns], expected)
            else:
                scores = []
                for name, months in (("logreg", 36), ("xgboost", None)):
                    a = joined.iloc[recent_training_indices(joined, tr, months)]
                    model = make_pipeline(ExternalFeatures(columns),
                        SimpleImputer(strategy="median", keep_empty_features=True),
                        build_model(name, params[name], a.churn.to_numpy(), threads))
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(a, a.churn)
                    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                        raise RuntimeError(f"Convergence failure: {variant}/{name}/fold{i}")
                    scores.append(model.predict_proba(joined.iloc[va])[:, 1])
                predicted = expected.assign(score_logreg=scores[0], score_xgboost=scores[1], score=(scores[0]+scores[1])/2)
            np.testing.assert_allclose(predicted.score, (predicted.score_logreg + predicted.score_xgboost)/2)
            assert np.isfinite(predicted.score).all() and predicted.score.between(0, 1).all()
            if not file.exists():
                predicted.to_parquet(file, index=False)
            chunks.append(predicted)
        prediction = pd.concat(chunks, ignore_index=True)
        summary, blocks, monthly = score_metrics(prediction)
        monthly_auc = prediction.groupby("mes_obs").apply(lambda g: roc_auc_score(g.churn, g.score), include_groups=False)
        summary["auc_monthly_mean"] = monthly_auc.mean()
        summaries.append({"variant": variant, "n_external_variables": len(columns), **summary})
        block_tables.append(blocks.assign(variant=variant))
        monthly["auc"] = pd.to_datetime(monthly.month).map(monthly_auc)
        month_tables.append(monthly.assign(variant=variant))
        for name in ("logreg", "xgboost"):
            component_rows.append({"variant": variant, "model": name,
                "auc_mean": np.mean([roc_auc_score(g.churn, g[f"score_{name}"]) for _, g in prediction.groupby("fold")])})
        print(f"{variant}: AUC bloques={summary['auc_mean']:.6f}; AUC mensual={summary['auc_monthly_mean']:.6f}; lift10={summary['lift_top10']:.4f}", flush=True)
    comparison = pd.DataFrame(summaries)
    comparison["delta_auc"] = comparison.auc_mean - comparison.loc[0, "auc_mean"]
    comparison["delta_auc_monthly"] = comparison.auc_monthly_mean - comparison.loc[0, "auc_monthly_mean"]
    comparison = comparison.sort_values("auc_mean", ascending=False)
    comparison.to_csv(out / "comparison.csv", index=False)
    pd.concat(block_tables).to_csv(out / "block_metrics.csv", index=False)
    pd.concat(month_tables).to_csv(out / "monthly_metrics.csv", index=False)
    pd.DataFrame(component_rows).to_csv(out / "component_metrics.csv", index=False)
    write_report(out, comparison, coverage, lag, variants)
    return comparison


def write_report(out, comparison, coverage, lag, variants):
    cols = ["variant", "auc_mean", "delta_auc", "auc_monthly_mean", "delta_auc_monthly", "precision_top10", "recall_top10", "lift_top10"]
    report = f"""# Evaluación exploratoria de fuentes externas

Se utilizaron las tablas del Excel `data/fuentes_datos_externos_glamour.xlsx`:
120 meses con 11 indicadores nacionales y 3.000 observaciones de crédito regional,
25 ámbitos (24 departamentos y Callao), entre enero de 2016 y diciembre de 2025.

## Resultados

{comparison[cols].to_markdown(index=False, floatfmt='.5f')}

## Comparación y variables

Ensemble 50/50 de logística (36 meses) y XGBoost (historia permitida completa),
42 variables transaccionales y los parámetros ya exportados. **No se ejecutó Optuna.**
Cuatro bloques entre octubre de 2023 y enero de 2025, gap de seis meses,
4.379 observaciones de validación. La base reutiliza predicciones verificadas con los
mismos datos, parámetros y versiones. No se añadieron sexo, edad, antigüedad ni códigos
de ubicación como predictores; departamento solo se utiliza para vincular crédito.

"""
    for name, columns in variants.items():
        report += f"- `{name}`: " + (", ".join(f"`{c}`" for c in columns) or "base sin variables externas") + ".\n"
    report += f"""
Para una observación en el mes `t`, se usa el indicador de `t−{lag}` meses.
Crédito regional es variación porcentual interanual dentro del mismo departamento:
`100 × (saldo(t−{lag}) / saldo(t−{lag}−12) − 1)`; el saldo bruto no entra al modelo.
Callao se conserva separado de Lima. Se normalizan espacios, mayúsculas y tildes.
Los crecimientos ausentes se imputan con la mediana de cada entrenamiento y se
añade un indicador de ausencia. Los porcentajes conservan su escala: 4,35 significa 4,35 %.

## Cobertura

{pd.DataFrame(coverage).to_markdown(index=False)}

La variación anual requiere un año previo; por ello falta en algunos meses iniciales
del entrenamiento. También queda ausente cuando el departamento no se puede vincular.
La imputación nunca usa validación y no se eliminan observaciones para favorecer una variante.

## Alcance de la evidencia

- **Snapshot revisado al 20 de septiembre de 2026.** El Excel no incluye fechas de
  publicación históricas ni versiones de lo conocido en cada fecha. El rezago de {lag}
  meses es una hipótesis explícita, no una verificación de disponibilidad histórica.
  Esta corrida no acredita desempeño sin fuga temporal por revisiones.
- Departamento proviene del maestro actual. Su correspondencia histórica tampoco
  está garantizada; esto afecta la unión con crédito regional.
- Los indicadores nacionales son iguales para todas las vendedoras de un mes.
  Se reportan AUC mensual y priorización dentro de cada mes para distinguir una mejora
  de ranking individual de una mejora en la comparación entre meses. Las filas no
  equivalen a observaciones macroeconómicas independientes.
- Las validaciones ya se utilizaron para seleccionar modelos. Los resultados son
  exploratorios, con parámetros fijos, sin evaluación final independiente ni demostración
  de significancia estadística. No se reemplazó el modelo anterior.
- Las fuentes F08–F13 del catálogo no contienen tablas de datos en el archivo y no se
  incorporaron. Se utilizaron únicamente las dos tablas efectivamente entregadas.

## Reproducibilidad

Artefactos en `{out.relative_to(ROOT)}`: protocolo con hashes, parámetros, variables,
cobertura, tablas fuente, dataset vinculado, predicciones por fold y métricas.
Ejecutar `python -m scripts.external_features_ablation`; un bloqueo evita ejecuciones
simultáneas y el manifiesto impide mezclar protocolos. Las predicciones terminadas se reutilizan.
El Excel original permanece intacto.
"""
    (out / "report.md").write_text(report)
    (ROOT / "07_results/ablacion_fuentes_externas.md").write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/external_features_v1")
    parser.add_argument("--lag", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.lag < 1 or args.threads < 1:
        parser.error("Positive lag and thread count required")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.workbook, args.output, args.lag, args.threads)


if __name__ == "__main__":
    main()
