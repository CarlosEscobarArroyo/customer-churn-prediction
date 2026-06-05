# Diccionario de variables — Feature Engineering & Selection

Describe las variables que intervienen en `01_feature_engineering.ipynb`: las
**base** que llegan desde `data/processed/churn_dataset_processed.csv`, las
**derivadas** que se crean en el notebook, y las **codificadas** (one-hot) de las
categóricas. La fuente autoritativa de las base es
`00_dataset_construction/qry_churn.sql`.

**Granularidad:** una fila por `(id_vendedor, mes_obs)` (mes calendario).
**Ventana de features:** `w = 12` meses hacia atrás `[t-11 .. t]` (incluye el mes t).
**Sufijos de ventana:** `u3` = últimos 3 meses, `u6` = últimos 6, `u12` = últimos 12.

---

## 1 · Identificadores y temporales (NO entran al modelo)

| Variable | Tipo | Descripción |
|---|---|---|
| `id_vendedor` | int | Identificador único de la vendedora. Sirve para `GroupKFold`. |
| `mes_obs` | date | Mes de observación (origen del pronóstico, `t`). Primer día del mes. |
| `mes_rank` | int | Orden temporal global del mes (1 = primer mes con datos). Ordenador del split forward. |

## 2 · Etiqueta (target)

| Variable | Tipo | Descripción |
|---|---|---|
| `churn` | 0/1 | `1` si la vendedora **no compra en ninguno** de los 6 meses siguientes `[t+1 .. t+6]`. Fila excluida si no hay 6 meses de futuro observables (censura a la derecha). |

---

## 3 · Features base (desde el SQL)

### 3.1 Frecuencia (¿cuántas veces compra?)

| Variable | Tipo | Descripción |
|---|---|---|
| `meses_activos_u3` | int | Nº de meses con al menos un pedido en los últimos 3 meses. |
| `meses_activos_u6` | int | Ídem en los últimos 6 meses. |
| `meses_activos_u12` | int | Ídem en los últimos 12 meses. |
| `n_ped_u3` | int | Nº total de pedidos en los últimos 3 meses. |
| `n_ped_u6` | int | Nº total de pedidos en los últimos 6 meses. |
| `n_ped_u12` | int | Nº total de pedidos en los últimos 12 meses. |

### 3.2 Monetario (¿cuánto compra?)

| Variable | Tipo | Descripción |
|---|---|---|
| `monto_u3` | float | Suma del monto comprado en los últimos 3 meses. |
| `monto_u6` | float | Suma del monto comprado en los últimos 6 meses. |
| `monto_u12` | float | Suma del monto comprado en los últimos 12 meses. |
| `monto_mean_u12` | float | Monto mensual promedio en los últimos 12 meses. |
| `monto_std_u12` | float | Desviación estándar del monto mensual en los últimos 12 meses. |
| `monto_cv_u12` | float | Coeficiente de variación = `monto_std_u12 / monto_mean_u12`. Volatilidad relativa del gasto. |
| `monto_ult_vs_media` | float | Monto del último mes `/ monto_mean_u12`. Mide si la última compra está por encima/debajo de su media. |

### 3.3 Recencia y antigüedad (¿hace cuánto / desde cuándo?)

| Variable | Tipo | Descripción |
|---|---|---|
| `meses_desde_compra_previa` | int | Meses transcurridos desde la compra previa a `t` (gap de recencia). |
| `compras_hist` | int | Nº de meses activos en todo el historial **antes** de `t`. |
| `antiguedad_meses` | int | Meses desde `fecha_ingreso` de la vendedora hasta `mes_obs`. |

### 3.4 Diversidad de producto (¿qué tan variado compra?)

| Variable | Tipo | Descripción |
|---|---|---|
| `n_prod_u12` | int | Nº de productos distintos comprados en los últimos 12 meses. |
| `n_cat_max_u12` | int | Máximo de categorías distintas en un mes, dentro de los últimos 12. Productos huérfanos se imputan como `Ropa`. |

### 3.5 Tendencias normalizadas en [-1, 1] (¿está acelerando o frenando?)

Comparan la ventana reciente `u3` `[t-2 .. t]` contra los 3 meses previos a esa
ventana `prev3` `[t-5 .. t-3]`. Fórmula: `(reciente - previo) / (reciente + previo)`.
Valor `> 0` = creciendo, `< 0` = decreciendo.

| Variable | Tipo | Descripción |
|---|---|---|
| `tend_monto_u3_vs_prev3` | float | Tendencia del monto: u3 vs prev3. |
| `tend_nped_u3_vs_prev3` | float | Tendencia del nº de pedidos: u3 vs prev3. |

### 3.6 Cumulative Features — CF (Mirkovic 2022, §3.2)

Agregados sobre **toda la historia disponible** de la vendedora, desde su primer
registro hasta `t` (ventana `UNBOUNDED PRECEDING .. CURRENT ROW`). A diferencia
de `u3/u6/u12` (topadas a 12 meses), capturan el nivel absoluto de la relación de
largo plazo: dos vendedoras con el mismo `u12` pero distinta antigüedad se ven
diferentes aquí.

| Variable | Tipo | Descripción |
|---|---|---|
| `monto_acum` | float | Monto total comprado en toda la historia hasta `t`. |
| `n_ped_acum` | int | Nº total de pedidos en toda la historia. |
| `n_prod_acum` | int | Suma de productos (líneas-mes) en toda la historia. |
| `n_cat_max_acum` | int | Máximo de categorías distintas en un mes, en toda la historia (amplitud de catálogo de por vida). |
| `ticket_acum` | float | `monto_acum / n_ped_acum`. Ticket promedio histórico. **Es la feature más importante del modelo.** |
| `monto_por_prod_acum` | float | `monto_acum / n_prod_acum`. Gasto promedio por producto histórico. |
| `monto_mensual_acum` | float | `monto_acum / meses_activos_acum`. Gasto promedio por mes activo histórico. |

### 3.7 Delta Features — DF (Mirkovic 2022, §3.2)

Diferencia del valor del **mes vigente `mt`** contra el valor del mes `mt-n`,
para `n ∈ {1, 3, 6, 9, 12}` (último mes, trimestre, 2/3 trimestres, año), vía
`LAG` sobre el panel denso. Cuantifican el cambio mes a mes y capturan
estacionalidad. `NULL` si no hay historia suficiente → el preprocessing las
imputa a `0`. Valor `> 0` = el mes vigente supera al mes de referencia.

| Variable | Fórmula | Descripción |
|---|---|---|
| `d_monto_m1` | `monto(mt) − monto(mt−1)` | Cambio de monto vs. el mes anterior. |
| `d_monto_m3` | `monto(mt) − monto(mt−3)` | Cambio de monto vs. hace un trimestre. |
| `d_monto_m6` | `monto(mt) − monto(mt−6)` | Cambio de monto vs. hace 2 trimestres. |
| `d_monto_m9` | `monto(mt) − monto(mt−9)` | Cambio de monto vs. hace 3 trimestres. |
| `d_monto_m12` | `monto(mt) − monto(mt−12)` | Cambio de monto vs. hace un año. |
| `d_nped_m1` | `n_ped(mt) − n_ped(mt−1)` | Cambio de nº de pedidos vs. el mes anterior. |
| `d_nped_m3` | `n_ped(mt) − n_ped(mt−3)` | Cambio de nº de pedidos vs. hace un trimestre. |
| `d_nped_m6` | `n_ped(mt) − n_ped(mt−6)` | Cambio de nº de pedidos vs. hace 2 trimestres. |
| `d_nped_m9` | `n_ped(mt) − n_ped(mt−9)` | Cambio de nº de pedidos vs. hace 3 trimestres. |
| `d_nped_m12` | `n_ped(mt) − n_ped(mt−12)` | Cambio de nº de pedidos vs. hace un año. |

> Las ventanas `u3/u6/u12` y las tendencias `tend_*` cubren la parte de *niveles*
> de DF del paper; estas columnas `d_*` agregan las **diferencias explícitas**
> mes-a-mes que el paper distingue.

### 3.8 Contexto / master data (categóricas, snapshot SCD-1)

Antes de codificar (ver §5). En el SQL llegan como texto.

| Variable | Tipo | Descripción |
|---|---|---|
| `sexo` | categórica | Sexo de la vendedora: `F`, `M` u `OTRO`. |
| `tipo_vendedor` | categórica | Rol comercial: `Asesora`, `Líder`, `DESCONOCIDO`. |
| `departamento` | categórica | Departamento de ubicación de la vendedora. |

> `edad` y `provincia` se calculan en el SQL pero no se usan en este notebook.

---

## 4 · Features derivadas (creadas en el notebook, §1)

Ratios e intensidades construidas con columnas base, mirando solo hacia atrás
(sin leakage). `safe_div(a, b)` devuelve 0 cuando el denominador es 0.

| Variable | Fórmula | Descripción |
|---|---|---|
| `ticket_prom_u12` | `monto_u12 / n_ped_u12` | Valor promedio por pedido en 12 meses. |
| `ticket_prom_u3` | `monto_u3 / n_ped_u3` | Valor promedio por pedido en 3 meses. |
| `intensidad_u3` | `n_ped_u3 / meses_activos_u3` | Pedidos por mes activo (¿concentra compras?). |
| `basket_size_u12` | `n_prod_u12 / n_ped_u12` | Productos por pedido (tamaño de canasta). |
| `recencia_norm` | `meses_desde_compra_previa × (meses_activos_u12 / 12)` | Recencia ponderada por la actividad reciente. |
| `tasa_act_reciente_vs_hist` | `(meses_activos_u3 / 3) / (meses_activos_u12 / 12)` | Ritmo de actividad reciente vs. el de 12 meses (>1 acelerando). |

> De las 6 derivadas, `tasa_act_reciente_vs_hist` es la única que **no** quedó
> seleccionada (importancia ≤ 0 en train). Las otras 5 sí entran al modelo final.

---

## 5 · Variables codificadas (one-hot)

Las categóricas de §3.6 se expanden a columnas binarias (0/1) en preprocessing.
Patrón de nombre: `<variable>_<valor>`.

- **`sexo_*`** → `sexo_F`, `sexo_M`, `sexo_OTRO`.
- **`tipo_vendedor_*`** → `tipo_vendedor_Asesora`, `tipo_vendedor_Líder`, `tipo_vendedor_DESCONOCIDO`.
- **`departamento_*`** → una columna por departamento (ej. `departamento_lima`,
  `departamento_arequipa`, …) más `departamento_DESCONOCIDO` y
  `departamento_sin departamento`.

> Nota de calidad de datos: existen duplicados por inconsistencia de mayúsculas
> en el origen (ej. `departamento_lima` y `departamento_Lima`, `departamento_loreto`
> y `departamento_Loreto`). El modelo de árboles los tolera, pero conviene
> normalizar el texto en una futura iteración del preprocessing.

---

## 6 · Salida del notebook

- **`churn_dataset_features.csv`** — `id` + `churn` + features **seleccionadas**
  (importancia por permutación > 0 en train).
- **`selected_features.json`** — lista de las features seleccionadas que usa
  `05_modelling`.
