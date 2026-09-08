# Documentación de TalkingFaceKit

Esta carpeta contiene documentación con propósitos distintos:

- [`architecture.md`](architecture.md) registra las decisiones e invariantes vigentes del código.
- [`../PROJECT_DIRECTION.md`](../PROJECT_DIRECTION.md) explica la orientación del producto, el mapa
  completo del problema y las alternativas técnicas consideradas.
- [`research/deeptalk_asd_compatibility.md`](research/deeptalk_asd_compatibility.md) registra el spike
  que llevó a elegir DeepTalk-ASD como primer backend experimental para la tesis.
- [`blueprint/`](blueprint/README.md) describe cómo podría verse la documentación final de una
  versión madura de TalkingFaceKit y funciona como guía de implementación.

El blueprint es deliberadamente más amplio que la versión `0.1.0`. No debe usarse para asumir que
una API ya existe. Cada página distingue lo que está disponible de lo que todavía es un objetivo o
una línea de investigación.

La decisión vigente es probar primero un flujo completo para **un video**: audio con timestamps,
DeepTalk-ASD detrás de un adapter, reporte de hablante activo y overlay de diagnóstico. MediaPipe
queda como backend de pose/calidad, SyncNet como candidato posterior de sincronización, y el
contenedor de carpetas se implementa sólo después de validar ese resultado. El orden y los criterios
de aceptación están en el [`roadmap`](blueprint/roadmap.md).

Para instalar, ejecutar y contribuir al estado actual del repositorio, consultar el
[`README.md`](../README.md) y las reglas de [`AGENTS.md`](../AGENTS.md).
