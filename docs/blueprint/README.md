# TalkingFaceKit: blueprint de documentación final

> **Documento de diseño, no referencia de la versión publicada.** Esta sección presenta una
> documentación futura y ficticia para orientar el crecimiento de la biblioteca. La API real de
> la versión `0.1.0` está documentada en el [`README.md`](../../README.md) y sus decisiones vigentes
> en [`docs/architecture.md`](../architecture.md).

TalkingFaceKit aspira a ser una biblioteca modular de Python para representar, procesar, analizar,
animar y exportar secuencias de personas hablando. Su unidad central es una secuencia temporal que
puede reunir video, audio, identidades faciales, landmarks, geometría, pose, parámetros de
animación y señales de habla sin quedar atada a un dataset, tracker, modelo neuronal o renderer.

La documentación imaginada aquí cumple dos funciones:

1. mostrar la experiencia de uso que debería ofrecer una versión madura;
2. convertir esa experiencia en contratos y entregas verticales implementables.

## Leyenda de estado

Todas las páginas usan las siguientes etiquetas:

| Estado | Significado |
| --- | --- |
| **Disponible** | Existe en el repositorio y está cubierto por tests. |
| **Objetivo** | Forma parte de la dirección propuesta, pero todavía no es API pública. |
| **Investigación** | Requiere experimentación o una decisión arquitectónica antes de comprometer un contrato. |

Una firma marcada como objetivo es ilustrativa. Antes de implementarla hay que validar el caso de
uso con una entrega vertical, escribir sus contratos en la arquitectura vigente y agregar tests.

## Recorrido recomendado

1. [Inicio rápido](getting-started.md): cómo debería sentirse instalar y usar la biblioteca.
2. [Conceptos fundamentales](concepts.md): secuencias, timelines, coordenadas, resultados y
   provenance.
3. [Guía de flujos](workflows.md): video, audio, tracking, habla, geometría, animación y exportación.
4. [Contratos de datos](data-contracts.md): shapes, dtypes, unidades, rangos y valores faltantes.
5. [Referencia de API propuesta](api-reference.md): superficie pública actual y futura.
6. [Referencia de CLI propuesta](cli-reference.md): comandos interactivos y por lotes.
7. [Integraciones y backends](integrations.md): dependencias opcionales y límites de framework.
8. [Arquitectura objetivo](target-architecture.md): módulos, dependencias y persistencia.
9. [Roadmap de implementación](roadmap.md): orden de entregas y criterios de aceptación.
10. [Guía de desarrollo](development.md): cómo convertir una página del blueprint en código.

## Promesa de producto

La biblioteca debería permitir que un usuario:

- abra una fuente de media sin decodificarla completa;
- seleccione un intervalo temporal sin inventar timestamps;
- procese video y audio por streaming con memoria acotada;
- obtenga uno o varios rostros con identidad temporal estable;
- calcule landmarks 2D/3D, pose, mirada, máscaras, blendshapes o parámetros FLAME;
- extraiga voz, actividad vocal, transcripción, fonemas, visemas y prosodia;
- alinee señales de distintas frecuencias sobre una línea temporal explícita;
- suavice, interpole o edite resultados conservando una máscara de validez y provenance;
- conduzca un avatar o retargetee animación facial sin acoplar el core al motor de destino;
- renderice overlays, meshes y comparaciones;
- exporte resultados interoperables y archivos nativos versionados;
- procese datasets de manera reproducible, reiniciable y observable;
- mida cobertura, jitter, sincronía labial y errores de fitting;
- elija de forma explícita backend, dispositivo, modelo y política de precisión.

## Qué no debería prometer

Incluso en una versión madura, TalkingFaceKit no debería:

- descargar modelos, subir media o acceder a la red silenciosamente;
- afirmar reconstrucción métrica cuando el backend sólo entrega profundidad relativa;
- inferir FPS, sample rate, color, escala, identidad o sincronía sin registrar la decisión;
- esconder frames o muestras fallidas para fabricar una secuencia aparentemente continua;
- garantizar la calidad de un modelo externo o ignorar sus condiciones de licencia;
- convertir el core en un wrapper de PyAV, OpenCV, PyTorch, MediaPipe, FFmpeg o un motor 3D;
- convertirse en un editor de video general, un framework de entrenamiento o un servicio cloud;
- exponer como emoción objetiva una clasificación afectiva incierta. Los descriptores de expresión
  podrán existir como señales experimentales, con limitaciones explícitas.

## Mapa de capacidades

| Área | Disponible | Objetivo | Investigación |
| --- | --- | --- | --- |
| Media | metadata y frames RGB por streaming | rotación, pixel aspect ratio, audio, clips | cámara o streams en vivo |
| Rostro | MediaPipe, un rostro, 478 landmarks | multi-rostro, identidades, pose, gaze, máscaras | oclusión y confianza por punto |
| Geometría | mesh MediaPipe 468/852 | normales, smoothing, exportación glTF | fitting FLAME y textura |
| Habla | presencia de stream de audio | waveform, VAD, ASR, fonemas, visemas, prosodia | diarización audiovisual |
| Animación | renderer Plotly de mesh | blendshapes, curves, retargeting | audio a expresión y edición semántica |
| Persistencia | landmarks NPZ v1 | proyecto nativo versionado y otros tracks | almacenamiento por chunks a gran escala |
| Operación | tres comandos CLI | pipeline, batch, cache, reportes | ejecución distribuida |
| Evaluación | validación de contratos | cobertura, jitter, fitting y lip-sync | benchmarks comparables entre modelos |

La amplitud del mapa no autoriza a crear módulos vacíos. El orden y la definición de terminado se
encuentran en el [roadmap](roadmap.md).
