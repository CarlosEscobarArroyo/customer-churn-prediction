# Aclaración: ¿correlación entre filas dentro de train es leakage?

**Respuesta corta:** No es leakage en sentido técnico. Pero sí es un problema que vale la pena vigilar (por eso está en el checklist como punto 2 con marca 🟡).

---

## Qué es leakage técnicamente

Leakage significa: **información del test se filtra al train**, lo que infla las métricas reportadas respecto al desempeño real en producción.

Las dos formas clásicas:
- **Leakage temporal**: features que miran al futuro (ej. usar `compro_t1` como feature). En este proyecto ya excluido.
- **Leakage por entidad**: la misma vendedora aparece en train y test, el modelo "la reconoce". En este proyecto resuelto con GroupKFold.

**El test de leakage es:** ¿las métricas que reporto son optimistas respecto a producción? Si sí → leakage. Si no → no es leakage.

---

## Qué pasa con la correlación entre filas DENTRO de train

Ejemplo: María tiene 3 filas en train (C20, C25, C30), y María NO está en test (gracias a GroupKFold).

¿Hay leakage? Aplicando el test:
- ¿El modelo ve información de test durante el entrenamiento? **No.** María no está en test.
- ¿Las métricas en test están infladas? **No.** El modelo evalúa sobre vendedoras que nunca vio.

Entonces **no es leakage**. Las métricas reportadas (AUC 0.757) son honestas.

---

## ¿Cuál es entonces el problema?

Es un problema **distinto**: afecta **qué aprende el modelo**, no la honestidad de las métricas.

- Una vendedora con 20 filas pesa 20× más que una con 1 fila al entrenar.
- El modelo se sesga hacia patrones de vendedoras muy activas.
- Puede que el modelo sea **subóptimo** porque está sobre-ajustado a vendedoras frecuentes.

Esto es un problema de **calidad del modelo**, no de **honestidad de la evaluación**.

---

## Analogía: dos formas de prepararse para un examen

**Escenario A (leakage real):** Te dan las preguntas del examen para estudiar. Sacas 95. Pero esa nota no refleja lo que sabes — refleja que viste las respuestas. Cuando te pongan otro examen sacarás mucho menos.

**Escenario B (correlación entre filas):** Estudias mucho un tema y poco otro. Sacas 75 en un examen variado. La nota **es honesta** — refleja lo que sabes. Pero estudiaste mal: si hubieras balanceado mejor, hubieras sacado 80.

Este caso es el **escenario B**. La nota (AUC 0.757) es real. Pero quizás se puede mejorar con `sample_weight` para balancear.

---

## Tabla resumen

| Concepto | ¿Es leakage? | ¿Infla métricas? | Estado |
|---|---|---|---|
| Misma vendedora en train Y test (split temporal) | **Sí** | Sí (vimos +5pp) | ✅ Resuelto con GroupKFold |
| Múltiples filas del mismo vendedor solo en train | **No** | No | Riesgo de calidad, no de honestidad |

---

## Cómo defenderlo si alguien dice "es leakage"

Si te dicen *"tener varias filas del mismo vendedor en train es leakage"*, la respuesta correcta es:

> "No es leakage si las filas no se reparten entre train y test. Eso lo controlo con GroupKFold por vendedora. Lo que sí queda es una correlación entre observaciones del mismo vendedor dentro de train, pero eso no infla las métricas reportadas — solo afecta la eficiencia del aprendizaje, y se puede mitigar con `sample_weight`."