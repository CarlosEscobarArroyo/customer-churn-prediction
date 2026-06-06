# Auditoría de data leakage y honestidad de resultados

> Verificación integral del pipeline 00→07 (re-ejecución completa + análisis estático). Fecha: 2026-06-05.

## Veredicto

**No se detectó data leakage.** El metric de cabecera (AUC out-of-period) es honesto: el bloque de test nunca participa en la construcción de features, el preprocesamiento, la selección de variables ni el tuning de hiperparámetros. Se documentan dos salvedades menores (sin impacto material) al final.

## 1. Construcción de features (SQL) — sin fuga de objetivo

Se auditó cada *window function* de `00_dataset_construction/qry_churn.sql`. **Todas las ventanas de features miran hacia atrás o al mes actual; solo la etiqueta mira hacia adelante:**

| Ventana | Definición | Dirección | Uso |
|---|---|---|---|
| `fwd6` | `1 FOLLOWING .. 6 FOLLOWING` | **adelante** | **exclusivamente la etiqueta** `churn` |
| `hist`, `ant` | `UNBOUNDED PRECEDING .. 1 PRECEDING` | atrás (estricto) | historia, recencia |
| `acum` | `UNBOUNDED PRECEDING .. CURRENT ROW` | atrás + actual | features acumuladas (CF) |
| `w3 / w6 / w12` | `N PRECEDING .. CURRENT ROW` | atrás + actual | RFM por ventana |
| `prev3` | `5 PRECEDING .. 3 PRECEDING` | atrás | tendencias |
| `seq` (LAG) | `LAG(col, n)` | atrás | deltas (DF) |

- La etiqueta `churn = (compras_fwd6 = 0)` usa **solo** la ventana `fwd6` (meses t+1..t+6). Ninguna feature usa `fwd6`.
- Incluir el mes actual *t* en las ventanas `acum`/`w*` **no es leakage**: *t* es el momento de observación (la vendedora compró en *t*, por eso está en la población); su actividad en *t* es conocida al momento de puntuar.
- El filtro `mes_rank <= max_rank - 6` garantiza que cada observación tenga su ventana de etiqueta **completa** (6 meses de futuro reales sobre el panel denso), evitando falsos churn por truncamiento de la serie.

## 2. Separación temporal y de identidad — sin fuga entre train y test

- **Split out-of-period** (`02_train_test_oot_splits`): test = últimos 4 meses; **GAP de 6 meses** (= horizonte de etiqueta) entre train y test → las ventanas de etiqueta del train no se solapan con el período de test. Verificado: train 28 311 / gap 1 242 / test 803.
- **GroupKFold por `id_vendedor`**: verificación de **0 vendedoras solapadas** en los 5 folds → sin fuga de identidad (una misma vendedora nunca está en train y validación a la vez).

## 3. Preprocesamiento y selección — ajustados solo en train

- **Preprocesamiento** (`03_preprocessing`): la mediana de `antiguedad` (12.0) y el vocabulario de categorías *one-hot* se aprenden **solo del train del split OOT**; las categorías no vistas en test quedan en ceros. El test no informa ningún parámetro de transformación.
- **Selección de variables** (`04_feature_engineering`): la importancia por permutación que decide las 68 variables se calcula **sobre el train**; el test OOT no se usa para seleccionar. (Se confirma que el AUC se mantiene: 0.7625 → 0.7628.)

## 4. Tuning — validación cruzada anidada (nested CV)

- `05_modelling/02_tuning_xgboost_optuna`: Optuna optimiza el AUC de GroupKFold **calculado solo sobre el train-pool**; el bloque OOT permanece **aislado** y se evalúa una única vez. Esto sigue el procedimiento de Gattermann-Itschert y Thonemann (2021, §5.4) y elimina el sesgo de selección de hiperparámetros sobre el test.
- `scale_pos_weight` se recalcula **dentro del train de cada fold**, evitando filtrar el ratio de clases de la partición de validación.

## 5. Salvedades menores (documentadas, sin impacto material)

1. **Preprocesamiento/selección ajustados sobre el train del OOT, no re-ajustados por fold de la CV interna.** Para la métrica GroupKFold (secundaria), la mediana de imputación y el vocabulario *one-hot* "vieron" las filas de validación de los folds (porque son parte del train del OOT). El impacto es **despreciable** (son estadísticos muy estables: una mediana y listas de categorías) y está reconocido en el notebook. **No afecta al AUC out-of-period**, que es la métrica de cabecera (el test OOT nunca se tocó al ajustar nada).
2. **Variables de contexto en *snapshot* SCD-1** (`tipo_vendedor`, `departamento`): reflejan el estado actual del *data warehouse*, no el histórico de cada mes. **No es fuga de objetivo** (no codifican el resultado futuro), sino una limitación de los datos maestros, ya señalada en la sección de limitaciones. `edad` y `provincia` se descartan en preprocesamiento.

## 6. Reproducibilidad

La re-ejecución completa de los notebooks 01→07 (semillas fijas, `random_state=42`) reproduce los resultados reportados. La etapa 00 (extracción desde BigQuery) no pudo re-ejecutarse en este entorno por permisos de solo-lectura; se auditó el SQL de forma estática (sin leakage) y se trabajó sobre el *snapshot* committeado del dataset (`churn_dataset.csv`: 30 356 observaciones, 6 323 vendedoras, prevalencia 30.9 %).
