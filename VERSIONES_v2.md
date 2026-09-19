# Versiones del dataset de churn — v2 (explicativa)

Segunda versión de `VERSIONES.md`. La v1 era una bitácora cronológica densa;
esta versión está pensada para alguien que llega al proyecto y necesita
entender **qué dataset usar, por qué, y cómo se midió**. La bitácora
histórica completa (v1..v6) sigue en `VERSIONES.md`.

Esta v2 se concentra en:

1. La versión vigente (**v5**) y sus resultados.
2. Qué significan las métricas que reportamos (AUC GroupKFold y AUC forward).
3. Por qué la ventana de churn quedó fijada en **6 meses**.

---

## TL;DR

- **Dataset vigente**: `glamour-peru-dw.glamour_dw.training_churn_v5`.
- **Granularidad**: una fila por (`id_vendedor`, `mes_obs`).
- **Target**: `churn = 1` si la vendedora **no compra en los 6 meses
  posteriores** al mes observado.
- **Filtro de población**: `compras_historicas >= 3`.
- **Resultados (HGB balanceado)**:
  - AUC GroupKFold = **0.7465**
  - AUC split forward = **0.7509**, std por mes test = 0.034
  - PR-AUC / prevalencia (lift) = **1.84×**

---

## 1. Resultados de v5

| Métrica | v4 (anterior) | v5 (vigente) | Δ |
|---|---:|---:|---:|
| **AUC GroupKFold (HGB)** | 0.7413 | **0.7465** | +0.0052 |
| **AUC split forward (HGB)** | 0.7286 | **0.7509** | +0.0223 |
| Std AUC por mes (bloque test) | 0.030 | 0.034 | +0.004 |
| PR-AUC GroupKFold (HGB) | 0.5531 | 0.5029 | −0.0502 |
| Lift PR-AUC / prevalencia | 1.69× | **1.84×** | +0.15× |
| Δ churn rate train↔test | 0.36pp | 1.9pp | +1.5pp |
| Falsos churn (gaps que sí vuelven) | 24.5% | 16.1% | −8.4pp |
| n filas | 24 159 | 23 684 | −475 |
| Prevalencia (churn rate) | 32.6% | 27.5% | −5.1pp |

**Lectura corta**: el AUC sube en los dos protocolos (GroupKFold y forward),
y el AUC forward sube fuerte (+2.2pp) — el modelo de v5 generaliza mejor en
el tiempo. La PR-AUC absoluta baja porque la prevalencia bajó, pero el
**lift sobre prevalencia sube** (1.84× vs 1.69×), que es la métrica que
importa para comparar entre datasets con distinta tasa base.

---

## 2. Qué significan las métricas que reportamos

Toda métrica se reporta bajo **dos protocolos de validación distintos**
porque cada uno responde una pregunta diferente. No son redundantes; son
complementarios.

### 2.1. AUC GroupKFold

**Setup**: 5 folds, agrupando por `id_vendedor` (ninguna vendedora aparece
en train y test al mismo tiempo).

**Qué mide**: capacidad del modelo para **separar vendedoras nuevas**
(que nunca vio) en las mismas condiciones temporales del entrenamiento.
Es decir: ¿generaliza a personas nuevas dentro del mismo régimen de negocio?

**Por qué GroupKFold y no KFold normal**: una misma vendedora aparece
en ~10-20 filas del dataset (una por mes observado). Si la dejamos
mezclar entre train y test, el modelo aprende sus patrones individuales
y mide algo trivial. GroupKFold por `id_vendedor` rompe esa fuga: train y
test son personas distintas.

**Cómo leerlo**:
- `0.5` = aleatorio.
- `0.74` = el modelo, dadas dos vendedoras (una que va a hacer churn y una
  que no), las ordena correctamente el 74% de las veces.
- En este dominio (churn de catálogo, target con ~16% de ruido inherente)
  un AUC ≥ 0.74 es razonable; mejorarlo más probablemente requiere features
  externas (datos de coordinadora, demográficos), no más iteración sobre
  los pedidos.

**Limitación**: no testea generalización **temporal**. Un fold puede
mezclar datos de 2023 y 2025, y el modelo que aprenda peculiaridades
de un período las usa libremente.

### 2.2. AUC forward (split temporal)

**Setup**:
- **Test**: el último bloque de **6 meses observados** del dataset.
- **Train**: todo lo que ocurrió antes de un **GAP** de seguridad.
- **GAP** = `HORIZON_CHURN + 1 = 7 meses`. Aseguramos que el target del
  último mes de train (que mira 6 meses al futuro) **nunca toca** el
  período de test. Sin este gap habría leakage temporal por solapamiento
  de la ventana del target.

**Qué mide**: capacidad del modelo para predecir **el futuro** con datos
del **pasado** — el caso real de uso. Cuando el modelo se despliegue, la
data de entrenamiento siempre será más vieja que la data sobre la que
predice, y posiblemente con un régimen de negocio distinto (cambio de
catálogo, cambio de campañas, estacionalidad).

**Cómo leerlo**:
- AUC forward ≈ AUC GroupKFold → el modelo **no depende del régimen**;
  generaliza temporalmente. Es el caso bueno.
- AUC forward < AUC GroupKFold (>1pp) → hay **drift**; el régimen actual
  difiere del histórico y el modelo se apoya en señales que no se
  conservan. Caso a investigar.
- En v5: forward 0.7509 vs GKF 0.7465 → **prácticamente iguales** (forward
  incluso un poco arriba). v4 tenía gap de 1.3pp (forward por debajo).
  v5 es temporalmente más estable.

**Métricas auxiliares**:
- **Std AUC por mes test**: dentro del bloque de 6 meses, calculamos AUC
  por cada mes. La std mide **estabilidad mensual**. v5 = 0.034 → el AUC
  varía ±3.4pp entre meses del bloque test. Si fuera 0.10+ habría un mes
  rompedor (probable cambio de régimen interno).
- **Δ churn rate train↔test**: cuánto cambió la prevalencia entre los dos
  bloques. v5 = 1.9pp (sube de ~26% a ~28%). Bajo es bueno; alto avisa de
  drift en el target.

**Por qué los dos**: si solo miramos GroupKFold, no detectamos drift.
Si solo miramos forward, perdemos potencia (n_test es ~1/6 del dataset y
los intervalos de confianza son anchos). Los dos juntos: GroupKFold da el
techo de capacidad, forward da el realismo operacional.

### 2.3. PR-AUC y lift sobre prevalencia

**Por qué reportamos lift y no solo PR-AUC**: PR-AUC depende de la
prevalencia. Si la prevalencia baja (de 32.6% en v4 a 27.5% en v5), una
PR-AUC menor no significa peor modelo — significa menos casos positivos
para encontrar. El **lift = PR-AUC / prevalencia** normaliza esto.

- Lift `1.0` = igual que predecir aleatoriamente con la tasa base.
- Lift `1.84×` (v5) = el ranking del modelo es 1.84 veces mejor que la
  tasa base para identificar churners.

---

## 3. Por qué la ventana de churn es de 6 meses

La definición operativa es:

> `churn = 1` si la vendedora **no realiza ninguna compra en los próximos
> 6 meses** después del mes observado.

Esta ventana (`HORIZON_CHURN = 6`) no es arbitraria. Está validada en
`notebooks/06_horizonte_v4.ipynb`, que repite el análisis del NB 01
(originalmente en granularidad campaña) sobre la granularidad mensual de
v4/v5. Resumen del razonamiento:

### 3.1. Codo automático en la curva de churn vs k

Para cada candidato `k ∈ {2,3,4,5,6,7,8}`, se calcula la tasa de churn
correspondiente y la diferencia con `k+1`:

| k (meses) | tasa churn | Δ vs k+1 | κ vs k=4 | falsos churn |
|---|---:|---:|---:|---:|
| 2 | 43.9% | 7.0pp | 0.76 | — |
| 3 | 36.9% | 4.3pp | 0.91 | 32.8% |
| 4 (v4) | 32.6% | 3.0pp | 1.00 | 24.5% |
| 5 | 29.7% | 2.2pp | 0.93 | 19.5% |
| **6 (v5)** | **27.5%** | **1.6pp** | **0.88** | **16.1%** |
| 7 | 25.9% | 1.5pp | 0.84 | ~12% |
| 8 | 24.4% | — | 0.80 | — |

Criterio de codo: el primer `k` con `Δ tasa < 2pp` → **k=6**. Antes de 6 meses
estamos sumando churners que en realidad iban a volver pronto; después
de 6 meses la tasa ya casi no cambia (estamos solo recortando ruido
diminuto a costa de información).

### 3.2. Hazard de retorno

El hazard mide la probabilidad de que una vendedora vuelva a comprar
exactamente en el mes `t+k`, dado que estuvo silente hasta `t+k−1`. Se
aplana entre `k=6` (7.3%) y `k=7` (5.9%). Lectura: **después de 6 meses
sin compra, la probabilidad de retorno es marginal**. Esperar más no
añade información, solo retrasa la señal.

### 3.3. Falsos churn (ruido del target)

Cuántas observaciones marcadas como churn(k) tienen un gap que sí termina
en compra (la vendedora vuelve, pero después de `k`). Es el ruido del target.

- k=4 → 24.5% del target es ruido.
- k=6 → 16.1% (−8.4pp). Target más limpio.
- k=7 → ~12%, pero a costa de un horizonte muy largo para retención.

### 3.4. Cohortes pre-2025 vs post-2025

Sospecha inicial: el cambio de régimen en 2025 (de 1 campaña/mes a 2-3
simultáneas) podría haber acortado el ciclo natural de compra y exigir un
`k` más corto.

Resultado del análisis: las dos cohortes **convergen exactamente en k=6**
(27.5% vs 27.5%). La diferencia entre cohortes se concentra en `k` bajos
(`k=4`: 34.0% post vs 32.5% pre). En `k=6` el régimen no importa.

→ k=4 estaba sobre-ajustado al supuesto de cambio de régimen. **k=6 es
robusto a la cohorte temporal**.

### 3.5. Coste de migración bajo

`κ(k=4, k=6) = 0.88` → el 94% de las etiquetas se mantienen idénticas
entre los dos targets. Cambiar de k=4 a k=6 **no es un re-etiquetado
masivo**, es un ajuste fino que limpia el 6% de casos ambiguos.

### 3.6. Resumen del por qué

| Argumento | Conclusión |
|---|---|
| Codo automático (Δ < 2pp) | k=6 |
| Hazard se aplana | k=6-7 |
| Falsos churn caen | k=6 da −8pp vs k=4 |
| Cohortes pre/post-2025 | convergen en k=6 |
| κ vs k=4 | 0.88 (cambio moderado, no masivo) |
| Coherencia con NB 01 (v3) | El codo en granularidad campaña también era k=6 |

**Por qué no k=5**: candidato razonable (κ=0.93, lift forward similar),
pero el codo automático no se activa (Δ5→6 = 2.2pp queda apenas afuera
del threshold de 2pp), el hazard todavía no se aplanó, y la cohorte
post-2025 todavía está 1pp arriba de la pre. Si negocio pidiera señal
explícitamente más temprana para activar retención, k=5 sería la
alternativa; mientras tanto, k=6 gana en robustez metodológica.

**Por qué no k=7+**: target casi idéntico (Δ < 2pp), pero el horizonte
empieza a ser tarde para retención y se pierde una columna entera del
dataset por el GAP del split forward.

---

## 4. Versiones históricas (resumen)

Detalle completo en `VERSIONES.md`. Aquí solo el mapa para entender de
dónde venimos:

- **v1** (retirada): k=3 campañas, sin filtro de población. El 45% de las
  vendedoras tenía 1 sola compra y dominaba el target → trivial.
- **v2** (retirada): k=6 campañas + `hist >= 4`. Múltiples leakages
  (`compro_t1..t6`, `monto_t1..t6`, `estado_coordinadora` SCD-1).
- **v3** (histórica): v2 sin leakages, tendencias normalizadas.
  Granularidad campaña, k=6 campañas.
- **v4** (reemplazada por v5): granularidad mensual, k=4 meses,
  `hist >= 3`. El cambio de granularidad fue correcto; el `k=4` estaba
  sin validar. Eso motivó v5.
- **v5** (vigente): igual que v4 pero con k=6 meses. AUC sube en ambos
  protocolos, target con menos ruido.
- **v6 / v6_all** (descartadas): experimento de filtro post-pandemia. v6
  no domina v5 (mejor GroupKFold pero peor forward por menor volumen).
  v6_all tiene AUC engañosamente alto por mezclar dos poblaciones —
  `hist >= 3` sigue siendo necesario.

---

## 5. Cómo reproducir las métricas

```bash
# Levantar el notebook que produce los números de v5
uv run jupyter lab notebooks/07_baselines_v5.ipynb

# Validación del horizonte k=6
uv run jupyter lab notebooks/06_horizonte_v4.ipynb
```

Ambos notebooks leen directamente de BigQuery; basta con `gcloud auth
application-default login` previo.