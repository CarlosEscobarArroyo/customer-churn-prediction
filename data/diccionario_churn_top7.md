# Diccionario resumido — Vista general del dataset (v5)

> Selección de **7 columnas representativas** del dataset de silent churn pensada para una **presentación general**: las que permiten entender de un vistazo qué es una fila (vendedora · mes), cómo se la describe y cuál es el resultado a predecir.
>
> Para el diccionario completo (todas las features del modelo, leakage, schema BigQuery), ver `data/diccionario_churn_data.md`.

## ¿Qué representa una fila?

Cada fila corresponde a **una vendedora en un mes** de observación. A partir de su historia hasta ese mes, el modelo predice si dejará de comprar en los **6 meses siguientes**.

## Columnas (vista general)

| # | Columna | Tipo | Descripción |
|---|---|---|---|
| 1 | `id_vendedor` | identificador | Identificador único de la vendedora. No se usa como feature del modelo; sirve para trazabilidad. |
| 2 | `mes_obs` | fecha | Mes calendario desde el cual se observa a la vendedora y se predice el churn (primer día del mes). |
| 3 | `fecha_ingreso` | fecha | Fecha en que la vendedora ingresó al sistema. Sirve para calcular antigüedad. |
| 4 | `edad_vendedor` | numérica | Edad de la vendedora en años. |
| 5 | `provincia` | categórica | Provincia de residencia de la vendedora. Da contexto geográfico (segmentación). |
| 6 | `num_pedidos_obs` | numérica | Cantidad de pedidos que la vendedora hizo en el **mes observado**. Refleja la actividad puntual del mes. |
| 7 | `monto_total_obs` | numérica (S/) | Monto total comprado por la vendedora en el **mes observado**. |
| 8 | `compras_historicas` | numérica | Total de **meses con compra** acumulados por la vendedora hasta el mes observado. Es el indicador más fuerte del modelo: vendedoras con mayor historial son menos propensas a hacer churn. |

## Variable objetivo

| Columna | Tipo | Descripción |
|---|---|---|
| `churn` | binaria (0 / 1) | **1** si la vendedora **no compra en ninguno de los 6 meses siguientes** al mes observado · **0** si compra al menos una vez en esa ventana. Es lo que el modelo aprende a predecir. |

## Notas para la presentación

- El dataset real tiene **~50 columnas**: además de las que se muestran arriba, incluye features de comportamiento reciente (compras y montos en ventanas de 3, 6 y 12 meses), tendencias (¿está gastando más o menos que antes?), recencia (cuántos meses desde la última compra) y diversidad de producto.
- El modelo final usa esas señales en conjunto. Las columnas mostradas aquí son las más fáciles de leer "a ojo" y sirven para entender qué tipo de información alimenta al modelo.
