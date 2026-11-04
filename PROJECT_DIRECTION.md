# TalkingFaceKit: dirección del proyecto

> **Dirección vigente: 1 de noviembre de 2026.** Este documento registra el alcance y las
> prioridades acordadas. El comportamiento implementado está en [`README.md`](README.md) y los
> contratos y decisiones de arquitectura en [`docs/architecture.md`](docs/architecture.md).
> El reparto entre Juanita, Juan y Matias está en el [plan del equipo](docs/team_tasks.md).

## Objetivo

TalkingFaceKit es una biblioteca de Python para trabajar con video, audio y parámetros de animación
facial sobre una representación temporal común. Debe permitir descubrir segmentos útiles de videos
arbitrarios, extraer parámetros compatibles con FLAME e inspeccionar y exportar resultados
reutilizables para modelos de animación facial a partir de voz.

El preprocesamiento y la selección ya tienen un recorrido funcional. **FLAME vuelve al camino
principal de implementación**, junto con la simplificación de la API pública, la optimización del
rendimiento y la limpieza del código. La prueba inicial de FLAME comienza en paralelo a la
consolidación de la evaluación del preprocesamiento.

El siguiente recorrido de producto es:

```text
Video o carpeta
    → análisis y selección de intervalos útiles
    → extracción de parámetros FLAME sobre los segmentos seleccionados
    → validación temporal e inspección visual
    → guardado, carga y exportación reutilizable
```

SMIRK ya tiene una integración experimental para estimar parámetros por frame con tiempos fuente.
La generación de meshes FLAME, persistencia y fachada pública de reconstrucción siguen pendientes.
El recorrido completo describe el objetivo de la etapa; las firmas futuras se definen a partir de
ejemplos de uso y de la validación de esta primera integración.

## Estado de partida

| Área | Disponible | Pendiente de esta etapa |
| --- | --- | --- |
| Media | Inspección, intervalos y streaming PyAV de video/audio con timestamps fuente | Transformaciones y exportación según el formato de consumo acordado |
| Selección | DeepTalk-ASD experimental, habla, rostros, scores crudos, cortes y continuidad temporal | Evaluación ampliada y ajustes basados en errores reproducibles |
| Sincronización y filtro visual | SyncNet-v2 opcional; pose, tamaño y encuadre con evidencia InspireFace | Validación en más casos; calidad visual adicional según necesidad |
| Colecciones | Descubrimiento y análisis secuencial, progreso, errores y guardado incremental | Carga de análisis guardados, reanudación y optimizaciones medidas |
| Evaluación | Anotaciones versionadas, manifiesto con hashes, métricas y comparador HTML | Completar archivos, revisar anotaciones y actualizar el manifiesto |
| Geometría facial | Landmarks y mesh MediaPipe; estimación experimental de parámetros SMIRK con contratos NumPy | Geometría FLAME, persistencia de parámetros, inspección y validación ampliada |
| Experiencia pública | API Python, CLI y ejemplos ejecutables | Reducir pasos y conceptos, consolidar recorridos y actualizar documentación |

El cambio de Juan a `AdaptiveDetector` quedó integrado en `main`. Al iniciar esta etapa hay 18 casos
anotados y 5 en el manifiesto compartido; parte del desfase es anterior a ese cambio. Los cinco videos
nuevos de Anne y Murphy se incorporarán desde Drive. También deben reconciliarse los archivos y
nombres pendientes de anotaciones anteriores. Los videos originales siguen fuera de Git.

La pose visual admite vistas de tres cuartos según el criterio de entrenamiento acordado. Las
métricas anteriores sobre un conjunto pequeño son una referencia de regresión; la validación final
requiere casos que no se hayan usado para ajustar reglas.

## Prioridades

### 1. Simplificar la API pública

Diseñar primero ejemplos breves para cuatro recorridos: analizar un video, procesar una carpeta,
obtener parámetros FLAME y guardar/recargar resultados. Evaluar cada decisión por la cantidad de
pasos y conceptos que necesita el usuario.

- Elegir un recorrido recomendado y defaults claros para cada operación habitual.
- Mantener diagnósticos y controles avanzados accesibles sin exigir conocer los módulos internos.
- Consolidar recorridos equivalentes entre Python, CLI y ejemplos sobre operaciones compartidas.
- Revisar nombres, configuración, resultados y exports públicos a partir de esos ejemplos.
- Explicitar cambios incompatibles y actualizar ejemplos, documentación y pruebas en conjunto.

La propuesta debe resolver casos reales antes de introducir nuevas capas o abstracciones.

### 2. Integrar extracción de parámetros FLAME

Empezar por un clip corto disponible y un extractor existente. SMIRK y PEAR son candidatos de la
propuesta original; la elección queda pendiente de una prueba de compatibilidad, calidad y costo.
La primera entrega debe demostrar una integración funcional antes de ampliar el soporte a otras.

Matias define, a partir de esa prueba:

- versión de modelo y significado de los parámetros que entrega el backend;
- shapes, dtype, coordenadas, unidades y convenciones de pose y cámara;
- identidad y parámetros por secuencia o por frame, según corresponda;
- timestamps fuente, correspondencia con frames y máscaras de validez;
- provenance de modelos, configuración, transformaciones y dispositivo;
- persistencia y visualización suficientes para guardar, recargar e inspeccionar el resultado.

Los contratos core deben ser independientes del framework. La geometría MediaPipe existente conserva
su propósito; sus landmarks y profundidad relativa no constituyen parámetros FLAME. El trabajo
inicial integra un extractor existente; el entrenamiento de un modelo nuevo queda fuera de esta
etapa.

### 3. Optimizar con mediciones comparables

Medir carga de modelos, decodificación, conversiones, inferencia, guardado, tiempo total y memoria
sobre entradas y configuraciones fijas. Registrar hardware y distinguir primera ejecución de
corridas posteriores.

Investigar primero los costos dominantes: carga repetida de modelos, decodificación repetida,
copias y conversiones, e inferencia. Evaluar reutilización de modelos manteniendo independiente el
estado de cada video. Introducir lotes o concurrencia cuando el perfil y la estabilidad lo justifiquen.

Cada mejora debe comparar tiempos y resultados antes y después. Una reducción de calidad o de
cobertura debe quedar explícita; conservar la semántica de timestamps y la visibilidad de fallos.
Los objetivos numéricos de rendimiento se fijarán después de obtener una línea base comparable.

### 4. Limpiar y consolidar la implementación

Revisar responsabilidades entre selección, extracción facial, persistencia y presentación.
Consolidar lógica repetida de serialización, conversiones, CLI y ejemplos. Retirar abstracciones que
no aporten al recorrido de usuario.

Matias dirige los cambios de arquitectura y superficie pública. Juan desarrolla mejoras acotadas
sobre los contratos y límites acordados. Los cambios se mantienen pequeños y revisables, con
pruebas que protejan el comportamiento y las invariantes relevantes.

### 5. Consolidar evaluación y resultados reutilizables

Completar el inventario y las selecciones humanas, revisar anotaciones y regenerar el manifiesto
cuando estén disponibles los videos fuente necesarios. No ajustar etiquetas para que coincidan
con las predicciones. Reservar videos nuevos para validación.

Comparar precisión, recall, segundos mal conservados y segundos útiles perdidos por video. Revisar
las diferencias en el comparador y distinguir errores de modelo, política temporal, cortes,
sincronización o filtro visual antes de elegir una corrección.

Agregar carga de análisis guardados sin repetir inferencia. Después, diseñar reanudación y caché
con compatibilidad explícita por fuente, intervalo, modelos, configuración y esquema. El formato de
exportación de clips y parámetros se acuerda con el consumidor del dataset.

## Equipo y coordinación

| Responsable | Frente | Resultado esperado |
| --- | --- | --- |
| Juanita | Datos, selecciones humanas, revisión visual y experiencia de uso | Conjunto revisado, incidencias reproducibles y guía de inicio probada |
| Juan | Evaluación automatizada, mediciones, persistencia y mantenimiento acotado | Comparaciones repetibles, resultados recargables y herramientas actualizadas |
| Matias | API pública, FLAME, rendimiento y arquitectura | Recorrido integrado simple, parámetros reutilizables y mejoras de velocidad demostradas |

Las tareas, primeros pasos, entregables y dependencias se detallan en
[`docs/team_tasks.md`](docs/team_tasks.md). Juanita prepara los casos mientras Juan instrumenta
comparaciones y mediciones; Matias inicia la propuesta de API y la prueba FLAME. Juan implementa
persistencia sobre los contratos definidos con Matias y los resultados alimentan la revisión visual
de Juanita.

## Criterios de aceptación del próximo hito

1. Un usuario resuelve los recorridos habituales con una API breve y documentada.
2. Un segmento seleccionado puede producir parámetros FLAME mediante una integración funcional,
   con correspondencia temporal, validez y provenance explícitas.
3. Es posible guardar, recargar e inspeccionar resultados sin volver a ejecutar los modelos.
4. La evaluación ampliada expone errores por video y conserva casos separados de los ajustes.
5. Las optimizaciones muestran tiempos comparables y preservan calidad e invariantes, o documentan
   cualquier cambio de comportamiento.
6. README, ejemplos y arquitectura distinguen capacidades disponibles, experimentales y pendientes.

## Decisiones que se conservan

- Python 3.11 y `uv` para el entorno; dependencias y modelos opcionales en integraciones.
- Streaming con memoria acotada y timestamps sobre el video original.
- Core independiente de PyAV, OpenCV, PyTorch y objetos de backends.
- DeepTalk-ASD como backend experimental; cambios a LR-ASD directo o TalkNet sujetos a evidencia.
- Scores crudos, incertidumbre y razones visibles; pose y sincronización como señales separadas.
- Los modelos, videos, credenciales y resultados generados permanecen fuera de Git.
- Los cambios de dependencias y modelos siguen las reglas de [`AGENTS.md`](AGENTS.md).

## Trabajo posterior

El entrenamiento de modelos de animación, textura, retargeting, ASR, fonemas, identidad entre videos
y ejecución distribuida quedan para necesidades posteriores. Los filtros adicionales de mirada,
nitidez, iluminación u oclusión se priorizan según errores observados y requerimientos de la salida.

La extracción FLAME, la API pública, la limpieza y el rendimiento forman parte de la etapa actual.

## Documentos relacionados

- [Plan de trabajo del equipo](docs/team_tasks.md).
- [Roadmap de implementación](docs/blueprint/roadmap.md).
- [Comportamiento y uso actuales](README.md).
- [Contratos e invariantes de arquitectura](docs/architecture.md).
- [Investigación y dirección de septiembre de 2026](docs/research/project_direction_2026-09.md),
  conservadas como referencia histórica; sus prioridades y estados no describen la etapa actual.
