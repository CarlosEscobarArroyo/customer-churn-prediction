# Evaluación exploratoria de fuentes externas

Se utilizaron las tablas del Excel `data/fuentes_datos_externos_glamour.xlsx`:
120 meses con 11 indicadores nacionales y 3.000 observaciones de crédito regional,
25 ámbitos (24 departamentos y Callao), entre enero de 2016 y diciembre de 2025.

## Resultados

La mayor media de AUC se obtuvo añadiendo comercio: **0,79005**, frente a **0,78988** de la base (**+0,00017**). Mejoró en tres de los cuatro bloques, con diferencias pequeñas. No se ha demostrado una mejora relevante o estadísticamente significativa; se mantiene el modelo previo.

Con todos los indicadores, el AUC por bloques fue **0,78947**, pero el AUC mensual medio pasó de **0,79038 a 0,79052**. En el top 10 % se identificaron **297 casos frente a 295**, sobre 431 observaciones priorizadas. En el top 20 %, **549 frente a 546**, sobre 870. Los criterios de evaluación no se mueven todos en el mismo sentido.

| variant                  |   auc_mean |   delta_auc |   auc_monthly_mean |   delta_auc_monthly |   precision_top10 |   recall_top10 |   lift_top10 |
|:-------------------------|-----------:|------------:|-------------------:|--------------------:|------------------:|---------------:|-------------:|
| mas_comercio             |    0.79005 |     0.00017 |            0.79046 |             0.00008 |           0.68213 |        0.21826 |      2.24996 |
| cuatro_fuentes           |    0.78990 |     0.00002 |            0.79050 |             0.00012 |           0.68677 |        0.21975 |      2.26592 |
| transaccional            |    0.78988 |     0.00000 |            0.79038 |             0.00000 |           0.68445 |        0.21901 |      2.25675 |
| mas_inflacion            |    0.78984 |    -0.00004 |            0.79046 |             0.00007 |           0.68445 |        0.21901 |      2.25379 |
| mas_credito_regional     |    0.78966 |    -0.00022 |            0.79037 |            -0.00002 |           0.68445 |        0.21901 |      2.24981 |
| mas_expectativas_demanda |    0.78962 |    -0.00026 |            0.78996 |            -0.00042 |           0.68677 |        0.21975 |      2.26340 |
| todos_externos           |    0.78947 |    -0.00041 |            0.79052 |             0.00014 |           0.68910 |        0.22049 |      2.28042 |

## AUC por bloque

| variant                  |       0 |       1 |       2 |       3 |
|:-------------------------|--------:|--------:|--------:|--------:|
| cuatro_fuentes           | 0.78375 | 0.79592 | 0.78287 | 0.79704 |
| mas_comercio             | 0.78466 | 0.79552 | 0.78222 | 0.79781 |
| mas_credito_regional     | 0.78457 | 0.79507 | 0.78180 | 0.79720 |
| mas_expectativas_demanda | 0.78466 | 0.79516 | 0.78200 | 0.79665 |
| mas_inflacion            | 0.78528 | 0.79485 | 0.78211 | 0.79710 |
| todos_externos           | 0.78359 | 0.79213 | 0.78440 | 0.79776 |
| transaccional            | 0.78535 | 0.79493 | 0.78187 | 0.79737 |

## Comparación y variables

Ensemble 50/50 de logística (36 meses) y XGBoost (historia permitida completa),
42 variables transaccionales y los parámetros ya exportados. **No se ejecutó Optuna.**
Cuatro bloques entre octubre de 2023 y enero de 2025, gap de seis meses,
4.379 observaciones de validación. La base reutiliza predicciones verificadas con los
mismos datos, parámetros y versiones. No se añadieron sexo, edad, antigüedad ni códigos
de ubicación como predictores; departamento solo se utiliza para vincular crédito.

- `transaccional`: base sin variables externas.
- `mas_inflacion`: `inflacion_nacional_var12_pct`.
- `mas_comercio`: `comercio_var12_pct`.
- `mas_expectativas_demanda`: `expect_demanda_3m_indice`.
- `mas_credito_regional`: `credito_regional_var12_pct`.
- `cuatro_fuentes`: `inflacion_nacional_var12_pct`, `comercio_var12_pct`, `expect_demanda_3m_indice`, `credito_regional_var12_pct`.
- `todos_externos`: `ipc_nacional_base_dic2021_100`, `inflacion_nacional_mensual_pct`, `inflacion_nacional_var12_pct`, `comercio_var12_pct`, `pbi_var12_pct`, `expect_economia_3m_indice`, `expect_demanda_3m_indice`, `empleo_formal_var12_pct`, `ingreso_formal_nominal_var12_pct`, `tipo_cambio_soles_usd`, `credito_consumo_var12_pct`, `credito_regional_var12_pct`.

Para una observación en el mes `t`, se usa el indicador de `t−3` meses.
Crédito regional es variación porcentual interanual dentro del mismo departamento:
`100 × (saldo(t−3) / saldo(t−3−12) − 1)`; el saldo bruto no entra al modelo.
Callao se conserva separado de Lima. Se normalizan espacios, mayúsculas y tildes.
Los crecimientos ausentes se imputan con la mediana de cada entrenamiento y se
añade un indicador de ausencia. Los porcentajes conservan su escala: 4,35 significa 4,35 %.

## Cobertura

| partition   |   rows |   missing_region_key |   missing_regional_growth |   missing_macro_rows |
|:------------|-------:|---------------------:|--------------------------:|---------------------:|
| development |  28311 |                   70 |                      2088 |                    0 |
| validation  |   4379 |                    3 |                         3 |                    0 |

La variación anual requiere un año previo; por ello falta en algunos meses iniciales
del entrenamiento. También queda ausente cuando el departamento no se puede vincular.
La imputación nunca usa validación y no se eliminan observaciones para favorecer una variante.

## Alcance de la evidencia

- **Snapshot revisado al 20 de septiembre de 2026.** El Excel no incluye fechas de
  publicación históricas ni versiones de lo conocido en cada fecha. El rezago de 3
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

Artefactos en `reports/external_features_v1`: protocolo con hashes, parámetros, variables,
cobertura, tablas fuente, dataset vinculado, predicciones por fold y métricas.
Ejecutar `python -m scripts.external_features_ablation`; un bloqueo evita ejecuciones
simultáneas y el manifiesto impide mezclar protocolos. Las predicciones terminadas se reutilizan.
El Excel original permanece intacto.

Verificación: la extracción se contrastó celda a celda con Artifact Tool. Las pruebas revisan la unión por mes rezagado, la separación Lima/Callao, la variación anual por mes exacto, claves duplicadas y la exclusión de columnas no autorizadas como predictores.

[Notebook ejecutable de resultados](../../05_modelling/09_fuentes_externas.ipynb).
