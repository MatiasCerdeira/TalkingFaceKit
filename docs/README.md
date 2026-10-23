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

El flujo vigente ya ofrece análisis tipado de una secuencia y colecciones secuenciales desde
carpeta. DeepTalk-ASD permanece experimental detrás de una integración; SyncNet y el filtro visual
son etapas opcionales. `VideoAnalysisResult` reúne evidencia y selección, y la CLI usa el mismo
analizador que la API Python. El orden de implementación y los pendientes se registran en el
[`roadmap`](blueprint/roadmap.md).

Para instalar, ejecutar y contribuir al estado actual del repositorio, consultar el
[`README.md`](../README.md) y las reglas de [`AGENTS.md`](../AGENTS.md).
