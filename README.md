# Predicción de churn — Glamour

El flujo vigente se estudia y ejecuta desde notebooks, con Python 3.12 y las dependencias de `pyproject.toml` / `uv.lock`. Los scripts sirven de soporte para validaciones, persistencia y ejecución en segundo plano. La auditoría está en `07_results/auditoria_estado_2026-09-19.md`.

## Archivos para leer, en orden

1. [Verificación de la reparación](03_preprocessing/02_verificacion_reparacion.ipynb): entradas, tipos, reconstrucción de variables y compatibilidad con los modelos previos. Ejecutado y con resultados guardados.
2. [Optuna con validación temporal](05_modelling/06_optuna_validacion_temporal.ipynb): notebook principal. Contiene las funciones de partición, selección, espacios de búsqueda, objetivo, exportación y el bucle de entrenamiento, además de tablas y curvas de seguimiento. Por defecto consulta los estudios activos; su sección 8 permite entrenar o reanudar en Jupyter.
3. `00_dataset_construction/qry_churn.sql`: origen de una fila por vendedora/mes. Churn significa ausencia de compras durante los seis meses posteriores. Población: compró en el mes y tiene compra en al menos un mes previo.
4. `reports/optuna_temporal_v1/protocol.json` y `folds.json`: configuración y fechas efectivas; generados y no versionados.

`scripts/temporal_optuna.py` es el backend validado para ejecución desatendida y las dos transformaciones importables que permiten recargar pipelines. `tests/test_temporal_optuna.py` verifica fechas e integridad. `scripts/build_temporal_notebooks.py` genera las celdas a partir del código validado; regenerar elimina salidas, por lo que debe volver a ejecutarse el notebook después. Los notebooks anteriores se conservan como referencia y no son la entrada de Optuna temporal.

Para pasar del proceso de segundo plano al notebook, solicitar parada mediante `STOP`, esperar al final del trial, retirar `STOP` y activar `EJECUTAR_ENTRENAMIENTO=True`. El bloqueo evita dos entrenamientos simultáneos en el mismo estudio; el historial se conserva.

## Entradas y reparación

El entrenamiento nuevo lee `data/processed/churn_dataset.csv`. Excluye identificadores, objetivo y datos maestros (`sexo`, `edad`, `antiguedad_meses`, `tipo_vendedor`, `departamento`, `provincia`) para limitarse a señales transaccionales históricas. La comparación con los modelos anteriores incluye, por tanto, un cambio de variables además de validación.

La reparación del CSV antiguo utiliza `churn_dataset_processed.csv`, `selected_features.json` y `oot_split.csv`. Reconstruye las seis variables derivadas y mantiene las 68 columnas y su orden para conservar compatibilidad con los modelos existentes. Verifica primero que el preprocesado coincide con su reconstrucción desde el original. Si encuentra diferencias, respalda el CSV anterior y registra hashes y posiciones modificadas en `reports/data_repair/`.

```bash
.venv/bin/python -m scripts.temporal_optuna repair
.venv/bin/python -m pytest tests/test_temporal_optuna.py -q
```

## Validación temporal

Se reserva del tuning tanto el OOT histórico (agosto–noviembre de 2025) como el gap de febrero–julio de 2025. El pool de desarrollo termina en enero de 2025. El OOT ya fue consultado en experimentos anteriores: no constituye una prueba final intacta y el nuevo script no lo evalúa.

Cuatro bloques de validación de cuatro meses cubren octubre de 2023 a enero de 2025. Para cada bloque, entrenamiento expansivo con seis meses completos de separación. El objetivo es la **media no ponderada del ROC AUC de los cuatro bloques**, con average precision, lift del decil superior, recall a 0,5 y AUC mensual como diagnósticos. Las etiquetas de bloques de validación próximos pueden compartir meses futuros; no son cuatro experimentos independientes y su desviación no es un intervalo de confianza.

Cada entrenamiento contiene otra partición temporal con el mismo gap para seleccionar variables mediante importancia por permutación, explícitamente `scoring='roc_auc'`. Solo se retienen importancias positivas; esto es una heurística, no una prueba de significancia. Optuna compara esa selección con usar todas las variables transaccionales. El selector fijo se calcula una vez por fold y se reutiliza entre trials. Ninguna etiqueta de la validación exterior participa en la selección. Escalado y pesos de clase se ajustan en cada entrenamiento.

La configuración ganadora sigue seleccionada sobre validación. Sus resultados no sustituyen una evaluación final sobre datos nuevos no utilizados para tomar decisiones.

## Ejecutar, observar y reanudar

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -u -m scripts.temporal_optuna run \
  --trials 100 --threads 4 --output reports/optuna_temporal_v1
```

Los modelos se alternan, un trial por turno: regresión logística, Random Forest, XGBoost y CatBoost. Presupuesto: 100 trials por modelo, contando completados, podados y fallidos; no son 100 adicionales al reanudar. Un único trial se ejecuta a la vez, con hasta cuatro hilos configurados por estimador. La regresión logística no aprovecha necesariamente todos esos hilos. No hay un límite de duración global: los tiempos dependen de las configuraciones probadas.

- `studies.sqlite3`: historial completo de Optuna; se reutiliza al ejecutar el mismo comando.
- `*_trials.csv`: tabla exportada después de cada trial.
- `*_best.json`: parámetros, métricas, variables y hash del mejor candidato exportado.
- `*_pipeline.joblib`: transformación, columnas elegidas y estimador, entrenados hasta enero de 2025. Se actualiza cuando mejora el AUC. No son modelos aprobados para producción.
- `protocol.json`: hashes de datos/código, versiones de librerías y protocolo. Se rechaza reanudar si cambian; usar otro directorio para un experimento nuevo.
- `folds.joblib`: cache local de particiones y selección.
- `process.json`: PID, inicio y presupuesto solicitado.

Para detener limpiamente después del trial actual: `touch reports/optuna_temporal_v1/STOP`. Para reanudar, retirar ese archivo y ejecutar el mismo comando. Un bloqueo de archivo impide dos coordinadores simultáneos sobre el mismo directorio. Trials interrumpidos en estado RUNNING se marcan FAIL al reiniciar.

Si se ejecuta en tmux, la sesión se llama `churn-optuna` y el log queda en `reports/optuna_temporal_v1/run.log`:

```bash
tmux attach -t churn-optuna
tail -n 30 reports/optuna_temporal_v1/run.log
.venv/bin/python -m scripts.optuna_status
```

Tmux mantiene el proceso al desconectar el terminal, pero no ante un apagado o suspensión del equipo. SQLite permite reanudar posteriormente. Los pipelines se cargan desde la raíz del repositorio con `joblib.load(...)`, de modo que Python pueda importar `scripts.temporal_optuna`.

## Ventanas recientes y ensembles

El [notebook 07](05_modelling/07_ventanas_recientes_ensemble.ipynb) compara toda la historia con 24, 36 y 48 meses de entrenamiento. Conserva los cuatro bloques, el gap de seis meses, las 42 variables transaccionales y los hiperparámetros ganadores anteriores: es una comparación controlada de ventanas, sin retuning. Luego compara promedios iguales de todas las parejas y de los cuatro algoritmos, usando la mejor ventana de cada uno.

Los checkpoints y tablas se guardan en `reports/recent_windows_v1/`. Reejecutar reutiliza las predicciones de variantes terminadas; un hash del protocolo impide mezclarlas con código o parámetros distintos. El notebook incluye entrenamiento, selección, gráficas y exportación del candidato. `scripts/recent_windows.py` solo aporta el recorte por meses y la clase persistible del ensemble; el experimento está en las celdas del notebook.

Los AUC de este análisis siguen siendo métricas de selección sobre las mismas validaciones, no una evaluación final independiente.

## Aporte de datos maestros al ensemble

El [notebook 08](05_modelling/08_ablacion_datos_maestros.ipynb) compara el ensemble de logística
(36 meses) y XGBoost (toda la historia), ponderados al 50 %, con versiones que añaden sexo,
edad, antigüedad desde el ingreso, departamento y provincia. Conserva explícitamente las
42 variables transaccionales originales y los bloques de octubre de 2023 a enero de 2025,
aunque el CSV actual contenga variables de campañas y meses adicionales.

La mediana para valores ausentes, el vocabulario de categorías y el escalado se aprenden
solo en cada entrenamiento. Se normalizan mayúsculas, espacios y tildes en las categorías.
Ubicación sigue siendo un snapshot actual: su aporte retrospectivo exige verificar la
disponibilidad histórica antes de usarlo como evidencia de desempeño operativo.

La comparación **no requiere una nueva búsqueda Optuna**: usa parámetros exportados y
fijos para las nueve variantes. El comando por defecto los lee de
`reports/master_features_v1/reconstructed_tuning/`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m scripts.master_features_ablation
```

Para usar otros JSON temporales, proporcionar `--params-dir` y un directorio `--output`
nuevo. La búsqueda de parámetros solo se activa explícitamente con `--reconstruct`
y el presupuesto `--trials`, usando otro directorio de salida.

En esta sesión se reconstruyeron 100 trials de logística y 85 finalizados de XGBoost;
se interrumpió el siguiente para pasar a la comparación solicitada. Se conservan los
estudios y el motivo de interrupción. Se tomó el mejor trial completado con las 42 variables.
Esta reconstrucción no garantiza identidad numérica con la corrida histórica: la
referencia es la base transaccional de la nueva ejecución.

Los artefactos de comparación se guardan en `reports/master_features_comparison_v1/`, con
protocolo, hashes, versiones, parámetros, predicciones por fold y candidato experimental.
La ejecución es reanudable y rechaza cambios de protocolo; no ejecutar dos copias a la vez.
Al terminar genera `07_results/ablacion_datos_maestros.md`. El notebook permite consultar
resultados sin entrenar y activar el ajuste de las variantes cuando sea necesario.

## Indicadores externos del Excel

El [notebook 09](05_modelling/09_fuentes_externas.ipynb) utiliza las tablas de
`data/fuentes_datos_externos_glamour.xlsx` para comparar inflación, comercio,
expectativas de demanda y crecimiento interanual del crédito regional, por separado
y en conjunto, contra las 42 variables transaccionales. También prueba los 11
indicadores nacionales junto con el crecimiento regional. Reutiliza los parámetros
exportados y las particiones originales; no ejecuta Optuna.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m scripts.external_features_ablation
```

El rezago predeterminado es de tres meses (`--lag 3`). Es una hipótesis exploratoria:
el Excel contiene series revisadas y carece de fechas históricas de publicación.
El rezago no acredita disponibilidad histórica ni elimina revisiones posteriores.
Departamento sirve para vincular el crédito, conservando Callao separado de Lima.
Se evalúan tanto AUC por bloques como AUC y priorización dentro de cada mes.

Resultados en `reports/external_features_v1/` y
[reporte de fuentes externas](07_results/ablacion_fuentes_externas.md). El libro
original se conserva intacto. Cambiar el libro, el código o el rezago requiere otro
directorio de resultados mediante `--output`; las variantes terminadas se reutilizan
solo si coinciden los hashes del protocolo.

## Clima provincial con NASA POWER

El [notebook 10](05_modelling/10_nasa_power.ipynb) consulta y visualiza la nueva
fuente climática. El extractor usa la
[API diaria de NASA POWER](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)
para enero de 2016 a diciembre de 2025, el mismo periodo del Excel económico:

```bash
.venv/bin/python -u -m scripts.nasa_power
.venv/bin/python -m pytest tests/test_nasa_power.py -q
```

Se vinculan departamentos y provincias con las coordenadas de sus capitales del
[catálogo público de UBIGEO aumentado](https://github.com/jmcastagnetto/ubigeo-peru-aumentado).
La capital provincial es una aproximación espacial. El extractor consulta una vez
por celda meteorológica MERRA-2 (0,5° × 0,625°) y comparte la respuesta entre
provincias de la misma celda. Las ubicaciones desconocidas quedan sin dato. Solo
se envían coordenadas y fechas a NASA; los registros del modelo se procesan localmente.

Las seis series diarias son temperatura media, máxima y mínima (`T2M`, `T2M_MAX`,
`T2M_MIN`), humedad (`RH2M`), precipitación (`PRECTOTCORR`) y viento (`WS2M`). Se
construyen diez variables mensuales, dos diferencias interanuales y un indicador
de ausencia. Los valores de relleno se convierten en nulos y un mes incompleto no
produce totales ni medias aparentando cobertura completa. Se usa tiempo solar local
(LST), que no es el horario civil peruano.

Salidas en `data/external/nasa_power/` (no versionadas):

- `clima_diario_celdas.csv` / `.parquet`: observaciones por celda y día.
- `clima_mensual_provincias.csv` / `.parquet`: tabla mensual con provincia,
  departamento, coordenadas, variables y cobertura por parámetro.
- `churn_con_clima.parquet`: dataset original con las variables `nasa_*` y
  `nasa_mes_ref` como metadato de fecha, no predictor.
- `ubicaciones.csv`, `cobertura_ubicaciones.csv` y `resumen.json`: correspondencias
  y cobertura efectiva. `diccionario.json` describe unidades y agregaciones.
- `raw/`, `manifest.json` y `protocol.json`: respuestas originales, URLs,
  fechas de descarga y hashes. El catálogo geográfico se conserva junto a ellos.

Por defecto se une el mes de observación con el clima de tres meses antes (`--lag 3`).
El rezago es una hipótesis: la historia meteorológica puede estar revisada y la
provincia del cliente es un snapshot actual. Las diferencias interanuales quedan
nulas donde no hay un año anterior de historia. Toda imputación posterior debe
aprenderse dentro del entrenamiento de cada fold.

Las descargas son reanudables, con dos consultas simultáneas y reintentos para
errores transitorios. Se puede cambiar `--start`, `--end`, `--lag`, `--dataset` y
`--output`; cambios de protocolo requieren otro directorio de salida. El notebook
permite variar el rezago localmente usando `attach_climate`, sin volver a descargar.
La extracción no reentrena modelos ni acredita mejora predictiva; prepara la
comparación temporal de variables climáticas frente a la base y las fuentes económicas.

## Evaluación de NASA POWER y Google Trends

El [notebook 11](05_modelling/11_ablacion_nasa_google_trends.ipynb) compara la base
transaccional con clima provincial, interés nacional de búsqueda y su combinación.
Utiliza exclusivamente los archivos locales de `data/external/`, los parámetros
exportados de logística y XGBoost y los cuatro bloques temporales originales.
No ejecuta Optuna ni descarga datos.

```bash
.venv/bin/python -m scripts.climate_trends_ablation
.venv/bin/python -m pytest tests/test_climate_trends_ablation.py tests/test_nasa_power.py -q
```

Se mantiene un rezago de tres meses para ambas fuentes. Trends se prueba con los
índices originales y con z-scores expansivos calculados hasta cada mes de referencia;
la normalización no elimina revisiones ni redondeo de la descarga histórica. El
resumen regional de Trends carece de fechas mensuales y se excluye del backtest.
Las fuentes siguen siendo snapshots exploratorios, sin versiones históricas de publicación.

Resultados y predicciones en `reports/climate_trends_v1/`, con hashes del protocolo
y reutilización de resultados solo si coinciden. Se informa AUC por bloques y por
mes, AP y casos detectados al priorizar el 10 %, 20 % y 30 % mensual. El modelo
anterior se conserva. Véase el [informe](07_results/ablacion_nasa_google_trends.md).

## Actividad histórica del departamento

El [notebook 12](05_modelling/12_actividad_departamental.ipynb) compara la base con
indicadores de ventas, pedidos y vendedoras activas del departamento. Parte del
panel mensual completo, incluyendo primeras compras, y descuenta la contribución
de la propia vendedora. Las variables resumen tres meses completos anteriores a
la predicción y ventanas equivalentes de seis y doce meses antes.

```bash
.venv/bin/python -m scripts.regional_activity_ablation
.venv/bin/python -m pytest tests/test_regional_activity_ablation.py -q
```

No se utilizan etiquetas para construir los indicadores. El panel y el maestro
locales se contrastan con el dataset de desarrollo. Las ubicaciones son las del
maestro actual; no se conocen los cambios históricos de departamento. Una ventana
sin historia suficiente queda ausente, mientras que un mes observado sin compras
conserva su cero. Imputación y escalado se ajustan dentro de cada entrenamiento.

Se mantienen los parámetros, los cuatro bloques y las 4.379 observaciones de
validación de las comparaciones anteriores, sin Optuna. Esta evaluación se usa
para desarrollo y no sustituye una prueba OOT final independiente. El nombre del
departamento codificado sirve de control adicional. Resultados, fuentes agregadas,
variables y predicciones: `reports/regional_activity_v1/` y
[reporte](07_results/ablacion_actividad_departamental.md).
