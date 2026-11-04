# Plan de trabajo del equipo

Actualizado el **1 de noviembre de 2026**. Este reparto acompaña la
[dirección vigente](../PROJECT_DIRECTION.md) y el [roadmap](blueprint/roadmap.md).
Las tareas de esta etapa están pendientes salvo que se indique lo contrario.

## Objetivo compartido

Completar el recorrido **video → segmentos útiles → parámetros FLAME → guardado, carga y
visualización**, con una API pública simple, resultados evaluables y rendimiento medido.

## Juanita — datos, validación visual y experiencia de uso

1. **Organizar el conjunto de evaluación.** Mantener el inventario de videos, nombres, disponibilidad
   y anotaciones. Coordinar las incorporaciones desde Drive y actualizar
   [`benchmark/assignments.md`](../benchmark/assignments.md) con los responsables de cada video.
2. **Completar y revisar las selecciones humanas.** Aplicar el criterio acordado de utilidad para el
   entrenamiento, incluidas vistas de tres cuartos. Anotar antes de mirar las predicciones y reservar
   videos nuevos para la validación final.
3. **Revisar el preprocesamiento en el comparador.** Registrar diferencias con video, timestamp,
   comportamiento esperado y observado. Mantener una lista de ejemplos reproducibles para que el
   equipo investigue sus causas.
4. **Revisar visualmente los resultados FLAME.** Cuando esté disponible el primer prototipo, seguir
   una guía de correspondencia de expresión, apertura de boca, alineación, saltos y pérdida del
   rostro. La revisión visual complementa las comprobaciones técnicas.
5. **Probar instalación y ejemplos.** Seguir la documentación desde un entorno acordado y registrar
   pasos confusos, errores y conceptos que necesiten explicación. Ayudar a preparar la guía de inicio
   y los casos de demostración.

**Entregable:** conjunto de evaluación revisado, lista de errores reproducibles y guía de inicio
probada. Cada incidencia debe incluir fuente, intervalo, resultado esperado y resultado observado.

**Primer trabajo:** inventariar el material disponible y pendiente, revisar el criterio de selección
con el equipo y elegir los casos que se mantendrán fuera de los ajustes.

## Juan — evaluación reproducible, persistencia y mantenimiento

1. **Automatizar las comparaciones.** Preparar una ejecución reproducible del benchmark con métricas
   globales y por video, reportes y diferencias respecto de una versión anterior. Registrar versión
   de código, configuración, modelos y conjunto evaluado.
2. **Preparar las mediciones de rendimiento.** Medir carga de modelos, etapas de análisis, lectura y
   escritura, tiempo total y memoria. Distinguir primera ejecución y ejecuciones posteriores;
   registrar hardware, duración y resolución de los videos.
3. **Implementar guardado y carga de resultados.** Comenzar por los análisis actuales y extenderlo a
   FLAME cuando Matias defina el contrato. Verificar que guardar y recargar conserve timestamps,
   parámetros, validez y metadata, sin repetir inferencia.
4. **Resolver redundancias acotadas.** Consolidar serialización, conversiones y lógica repetida en
   CLI o ejemplos según el alcance acordado con Matias. Hacer cambios pequeños, con verificaciones
   que preserven el comportamiento.
5. **Actualizar ejemplos y documentación.** Adaptarlos a la API acordada, mantenerlos ejecutables y
   retirar instrucciones desactualizadas. Regenerar el manifiesto con las anotaciones revisadas
   cuando estén disponibles los videos fuente necesarios.

**Entregable:** comparaciones y mediciones reproducibles, carga de resultados guardados y mejoras
acotadas de mantenimiento, con ejemplos actualizados.

**Primer trabajo:** automatizar la línea base de evaluación y rendimiento sobre un subconjunto fijo
que ya esté disponible localmente. Identificar y reportar archivos faltantes de forma explícita.

## Matias — API pública, FLAME, rendimiento y arquitectura

1. **Definir la API pública mínima.** Diseñar los recorridos de un video, una carpeta, extracción
   FLAME y guardado/carga. Reducir pasos, conceptos y decisiones del uso habitual; mantener el detalle
   accesible cuando haga falta. Usar ejemplos de usuario para decidir los contratos antes de
   reorganizar módulos o cambiar exports.
2. **Abrir la integración FLAME.** Evaluar extractores existentes, incluidos los candidatos de la
   propuesta original, y obtener un primer recorrido completo sobre un clip corto. Registrar
   requisitos de hardware, dependencias, modelos, tiempos, calidad y limitaciones. Elegir una primera
   integración a partir de esa prueba.
3. **Definir los contratos de parámetros faciales.** Acordar versión de modelo, significado de cada
   parámetro, shapes, dtype, coordenadas, unidades, timestamps, identidad, validez y provenance.
   Establecer la salida reutilizable y los criterios de visualización y validación para que Juan
   implemente su persistencia.
4. **Optimizar los costos dominantes.** Interpretar las mediciones y abordar carga repetida de
   modelos, decodificación, conversiones e inferencia donde tengan impacto. Evaluar reutilización de
   modelos con estado independiente por video; introducir lotes o concurrencia cuando las
   mediciones y la estabilidad lo justifiquen. Comparar calidad y tiempos antes y después.
5. **Dirigir la limpieza estructural.** Acordar responsabilidades entre selección, extracción facial,
   persistencia y presentación. Consolidar recorridos duplicados y retirar abstracciones que no
   aporten al flujo de usuario. Delimitar con Juan los cambios que pueda desarrollar de manera
   independiente.
6. **Integrar y validar la entrega.** Revisar cambios de API, backend, contratos y rendimiento;
   comprobar alineación temporal, regresiones y utilidad de la salida FLAME. Documentar las
   decisiones aceptadas en la arquitectura.

**Entregable:** recorrido completo con una API simple, integración FLAME funcional y mejoras de
rendimiento demostradas sobre entradas y configuraciones comparables.

**Primer trabajo:** escribir los ejemplos de la experiencia pública deseada, iniciar la prueba FLAME
y usar las primeras mediciones para elegir una optimización concreta.

## Coordinación y orden de trabajo

- Juanita prepara y revisa los casos mientras Juan automatiza las mediciones y Matias inicia la API
  y la prueba FLAME. El prototipo FLAME puede empezar con un clip disponible sin esperar a completar
  el benchmark.
- Matias define los contratos de API y datos; Juan implementa persistencia y adapta herramientas a
  esos contratos. Ambos acuerdan los archivos y responsabilidades antes de cambios que se crucen.
- Juanita revisa los resultados y ejemplos; Juan convierte los hallazgos en comparaciones
  reproducibles; Matias prioriza cambios de política, backend o arquitectura a partir de la evidencia.
- La descarga de los videos nuevos queda pendiente de su incorporación desde Drive. El manifiesto
  se regenera desde las anotaciones revisadas y los videos originales; sus hashes no se editan a mano.
- Reanudación de corridas y caché se abordan después de contar con carga de resultados y una regla
  explícita de compatibilidad por fuente, intervalo, configuración, modelos y esquema.
- Cada cambio debe tener un objetivo revisable y evidencia proporcional: comparación de métricas
  para selección, mediciones comparables para rendimiento y guardado/carga para persistencia.

## Criterio de cierre de la etapa

- Un usuario puede analizar un video o carpeta y obtener los segmentos seleccionados con pocos pasos.
- Un segmento seleccionado puede alimentar una integración FLAME con tiempos y validez explícitos.
- Los resultados pueden guardarse, recargarse e inspeccionarse sin volver a ejecutar los modelos.
- La evaluación registra precisión, recall, segundos mal conservados y útiles perdidos por video.
- Las mejoras de velocidad tienen mediciones comparables y no introducen regresiones sin explicar.
- La documentación distingue las capacidades disponibles de las que continúan en investigación.
