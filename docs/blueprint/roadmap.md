# Roadmap de implementación

> **Dirección vigente desde 2026-09-08.** Este roadmap ordena el trabajo alrededor del nuevo
> objetivo: encontrar intervalos de video útiles para datasets donde un rostro visible realmente
> produce el habla audible. Distingue capacidades disponibles, spikes terminados y código todavía
> no implementado.

## Resultado de producto

La biblioteca debe avanzar desde un archivo arbitrario hasta una respuesta explicable como:

```text
video_005.mp4
00:04.000–00:06.200  candidate  face_2  visible speaker likely
00:06.200–00:08.050  rejected   none    off-screen speech likely
00:08.050–00:13.000  uncertain  face_2  competing face score
```

Más adelante, orientación a cámara, calidad visual y sincronización A/V convertirán los segmentos
candidatos en segmentos aceptados según una política configurable. El análisis de un video debe
probarse antes de crear un contenedor de carpetas o un sistema batch.

## Regla de avance

Una capacidad pasa de **objetivo** a **disponible** sólo cuando:

- existe un flujo ejecutable desde Python;
- tiene CLI cuando mejora la inspección o automatización;
- los contratos validan dtype, shape, unidades, rango, ejes, tiempo y faltantes;
- I/O, modelos y frameworks permanecen en integraciones;
- hay tests unitarios y de integración proporcionales al riesgo;
- README y arquitectura describen exactamente el comportamiento real;
- ningún test o import descarga modelos o datasets;
- Ruff, mypy y pytest pasan;
- limitaciones, fallos y provenance quedan visibles.

Un score de un modelo no se transforma en probabilidad ni en “válido” sin evidencia de calibración.

## Mapa de dependencias

```text
Base disponible: source timeline + video frames + MediaPipe landmarks/mesh
                              |
                              v
1. audio con timestamps de fuente                  [completo]
                              |
                              v
2. adapter experimental DeepTalk-ASD
                              |
                              v
3. reporte + overlay de un video
                              |
                              v
4. evaluación de seis casos y gate de backend
                  /-----------+-----------\
                 v                       v
5. pose/calidad MediaPipe         6. sync explícito SyncNet
                  \-----------+-----------/
                              v
7. política final de segmentos
                              |
                              v
8. colección, folder workflow y batch
                              |
                              v
9. normalización/export reproducible
```

Direct LR-ASD ONNX y TalkNet son rutas condicionales dentro del gate, no trabajos paralelos.

## Estado 0 — base implementada

El repositorio ya ofrece:

- `TalkingFaceSequence.from_video` y clips sobre la línea temporal fuente;
- `VideoSource` y `VideoMetadata`;
- `DecodedVideoFrame` y `stream_video_frames` RGB por PyAV;
- `FaceLandmarkTrack` y tracking MediaPipe de un rostro;
- persistencia NPZ v1 de landmarks;
- `FaceMeshTrack` y conversión MediaPipe 468/852;
- renderer Plotly offline;
- CLI para extraer, inspeccionar y visualizar landmarks;
- tests unitarios y fixtures pequeños.

Estas capacidades se conservan. Landmarks, mesh y rendering pasan a ser herramientas visuales
opcionales; ya no definen la promesa principal del producto.

Limitaciones relevantes para la nueva dirección:

- no hay decode de audio como contrato público;
- no hay tracking multi-rostro en el core;
- no hay VAD ni atribución de hablante visible;
- no hay reporte de segmentos;
- no se mide orientación a cámara ni sincronización;
- no hay folder workflow.

## Spike 1 — DeepTalk-ASD, completo

El spike aislado de DeepTalk-ASD 0.3.1 demostró que, en Apple Silicon, Python 3.11 y CPU:

- InspireFace detecta y sigue múltiples rostros;
- Silero VAD produce turnos de habla;
- LR-ASD ONNX produce scores por rostro;
- el flujo funciona offline después de adquirir los modelos explícitamente;
- el rendimiento observado es suficiente para continuar con un prototipo.

También detectó riesgos que el adapter debe contener:

- conflicto de versión exacta de NumPy;
- speaker embeddings rotos por linkage nativo en macOS;
- posible `SIGABRT` ante assets inválidos;
- resampling de 25 FPS no determinista por comparación `float`;
- reset de VAD según wall clock;
- buffers visuales sin límite;
- timestamps sintéticos en el demo upstream;
- valores LR-ASD crudos, no probabilidades calibradas.

Decisión: usar DeepTalk-ASD como backend **experimental de tesis**, no como modelo de dominio ni
dependencia central. El uso no comercial permite usar los pesos InspireFace actuales en este
alcance. Ver el [reporte completo](../research/deeptalk_asd_compatibility.md).

## Entrega 1 — audio PyAV con timestamps, completa

Implementada sin agregar dependencias runtime.

### Resultado de usuario

Recorrer el audio de una fuente o intervalo con memoria acotada, conservando sample rate, canales y
tiempo de presentación.

### Entregado

1. Crear `DecodedAudioChunk` como valor core independiente de PyAV.
2. Definir y validar shape, dtype, orden de ejes, sample rate, canales y amplitud.
3. Implementar `stream_audio_chunks(path, start_seconds, end_seconds)` en la frontera PyAV.
4. Usar el mismo intervalo semiabierto y timeline fuente que video.
5. Definir qué ocurre con un chunk que cruza el inicio o final solicitado.
6. Fallar claramente si no existe audio o faltan timestamps necesarios.
7. Agregar tests pequeños para mono, stereo, sample rates distintos, intervalos y ausencia de audio.
8. Documentar el contrato real en README y arquitectura.

### No incluido

- DeepTalk-ASD;
- materializar la pista completa;
- WAV export;
- resample, downmix o normalización;
- VAD;
- colección o pipeline genérico.

### Criterio de terminado

Los tests verifican timestamps e índices fuente, límites con precisión de sample, planar stereo,
PCM entero mono, normalización, ausencia de audio e intervalos vacíos. El productor no retiene la
pista completa.

## Entrega 2 — adapter experimental DeepTalk-ASD

### Resultado de usuario

Analizar un video y obtener evidencia temporal de rostros, habla y hablante activo sin exponer
objetos DeepTalk en la API core.

### Antes de implementar

Agregar DeepTalk, ONNX Runtime, OpenCV, InspireFace o assets requiere aprobación explícita del
equipo. La implementación debe decidir, con un pequeño experimento técnico, entre:

- extra opcional dentro del entorno del proyecto; o
- worker/proceso aislado con su propio entorno, preferible si el conflicto de NumPy o el crash
  nativo no puede contenerse razonablemente.

### Alcance mínimo

1. Validar Python, plataforma, modelo, path y hash antes de entrar en código nativo.
2. Mantener los modelos fuera de Git y sin descarga durante import o tests.
3. Convertir RGB a la entrada visual esperada sólo dentro del adapter.
4. Convertir audio a mono 16 kHz `int16` sólo dentro del adapter y registrar la transformación.
5. Resamplear a 25 FPS con una política determinista basada en PTS fuente.
6. Administrar VAD por tiempo de media, no por wall clock, y resolver EOF explícitamente.
7. Acotar ventanas y buffers de cada rostro.
8. Devolver IDs locales, bounding boxes, speech intervals y `raw_score` sobre segundos fuente.
9. Reportar speaker embeddings como `unavailable` en macOS durante el primer slice.
10. Traducir fallos a errores claros; considerar proceso separado para convertir un crash nativo en
    un fallo estructurado.

### No incluido

- confidence calibrada;
- orientación a cámara;
- exactitud de lip-sync;
- identidad de una persona entre videos;
- selección final para entrenamiento.

### Criterio de terminado

El mismo archivo produce resultados temporalmente equivalentes en ejecuciones repetidas; el
adapter no materializa la fuente completa ni filtra tipos externos al core.

## Entrega 3 — reporte y demostración de un video

### Resultado de usuario

Ejecutar un comando o función sobre un video y recibir:

- metadata y estado de decode;
- intervalos de habla;
- face tracks y boxes;
- observaciones `raw_score` por rostro y ventana;
- segmentos `candidate`, `rejected` o `uncertain`;
- razones como `no_speech`, `no_visible_face`, `offscreen_speech_likely` o
  `multiple_faces_ambiguous`;
- capacidades no ejecutadas como `camera_facing: not_evaluated` y `av_sync: not_evaluated`;
- backend, versiones, hashes, configuración, plataforma y tiempos de ejecución.

### Salidas

1. JSON versionado y machine-readable.
2. Resumen legible en terminal.
3. Video anotado o reporte visual con boxes, IDs, speech state y scores.

La política vive en TalkingFaceKit. Debe conservar top score, segundo score, margen, ventanas
originales y decisiones temporales. `raw_score > 0` no es una regla suficiente.

### Criterio de terminado

Una persona sin conocimientos de ML puede mirar el overlay, leer el JSON y entender qué encontró el
sistema, qué no midió y por qué tomó cada decisión preliminar.

## Entrega 4 — evaluación mínima y gate

Crear y anotar seis clips o casos:

1. un rostro visible hablando;
2. rostro visible silencioso con narrador o entrevistador fuera de cámara;
3. dos rostros visibles alternando habla;
4. habla sin rostro visible;
5. audio deliberadamente desplazado;
6. rostro visible sin habla.

Medir aciertos por intervalo, falsos positivos de narración, ambigüedad multi-rostro, estabilidad de
IDs, error de límites, repetibilidad, tiempo y memoria. No es necesario publicar un benchmark grande
para tomar esta primera decisión.

### Gate de backend

| Resultado observado | Siguiente decisión |
| --- | --- |
| LR-ASD útil y wrapper manejable | conservar DeepTalk experimental |
| LR-ASD útil pero wrapper problemático | adapter enfocado sobre LR-ASD ONNX |
| LR-ASD insuficiente | comparar TalkNet con los mismos seis casos |
| errores de política, no de modelo | ajustar agregación/umbrales con evidencia etiquetada |

El gate debe dejar una nota de decisión con ejemplos, métricas y limitaciones. No se reemplaza una
mala atribución audiovisual con heurísticas de movimiento de boca.

## Entrega 5 — orientación a cámara y calidad visual

Una vez que el hablante visible funciona:

1. reutilizar MediaPipe para pose de cabeza y señales visuales;
2. medir tamaño de rostro, cobertura, oclusión y proximidad al borde;
3. definir tolerancias de yaw, pitch y roll con ejemplos etiquetados;
4. decidir por evidencia si head pose alcanza o si gaze necesita un modelo separado;
5. incorporar razones visuales a la política sin alterar los scores ASD.

Resultado: los segmentos candidatos pueden clasificarse por suitability visual. MediaPipe sigue sin
ser prueba de quién habla.

## Entrega 6 — sincronización A/V explícita

Evaluar SyncNet con desplazamientos conocidos, por ejemplo ±40, ±80, ±160 y ±320 ms. El resultado
debe incluir offset estimado, score/confianza con semántica documentada, ventana analizada y estados
`acceptable`, `out_of_sync`, `uncertain` o `not_measurable`.

Aplicar una corrección temporal es una operación separada. ASD y mouth motion no sustituyen esta
medición.

## Entrega 7 — política final de segmentos

Combinar observaciones independientes:

```text
speech present
AND one visible active speaker is sufficiently unambiguous
AND face quality passes
AND camera orientation passes
AND sync passes or policy allows not_measurable
```

La configuración define duración mínima, unión de gaps, tolerancias, comportamiento multi-rostro y
offset aceptable. El reporte conserva observaciones y razones, no sólo el booleano final.

## Entrega 8 — folder workflow, collection y batch

Sólo cuando el resultado de un video sea estable:

1. descubrir extensiones y paths de forma determinista;
2. representar fuentes pendientes, no videos precargados;
3. analizar cada archivo de manera independiente;
4. persistir resultados por archivo atómicamente;
5. continuar o detener ante error mediante política explícita;
6. soportar resume por fingerprint, configuración, backend, modelo y schema;
7. agregar concurrencia acotada después de medir CPU, memoria y thread/process safety;
8. generar un reporte agregado de segmentos, rechazos, errores y runtime.

`TalkingFaceCollection.from_folder(...)` es una posible fachada, no un contrato comprometido. La
colección no se implementa primero porque todavía no existe el resultado estable que contendría.

## Entrega 9 — normalización y export

Cuando el cliente defina el formato de entrenamiento:

- exportar manifest de intervalos o clips físicos;
- mantener referencias a timestamps y fingerprints fuente;
- declarar codec, frame rate, resolución, pixel format, sample rate, canales y loudness;
- distinguir normalización temporal de inputs de modelo de normalización permanente del dataset;
- no sobrescribir media fuente por defecto.

## Trabajo postergado

No pertenece al camino crítico actual:

- ASR, forced alignment, fonemas, visemas y prosodia;
- FLAME, textura, retargeting y animación generativa;
- reconocimiento de identidad entre videos;
- diarización completa si ASD ya resuelve el caso;
- scene detection hasta que los errores lo justifiquen;
- pipeline/plugin system genérico;
- persistencia multipista `.tfk` antes de necesitar más de un artefacto estable;
- scheduler distribuido, cloud service o UI web.

Estas capacidades no se eliminan. Se evitan hasta que aporten valor a la validación del dataset o a
un requerimiento posterior de la tesis.

## Próximas cuatro unidades de trabajo

| Orden | Cambio | Demuestra |
| ---: | --- | --- |
| Completo | `DecodedAudioChunk` + `stream_audio_chunks` + tests | audio/video comparten tiempo fuente |
| 1 | adapter DeepTalk experimental + tests con fake/integración marcada | se obtienen face/VAD/ASD sin contaminar el core |
| 2 | JSON versionado + overlay de un video | el resultado ya es visible y explicable |
| 3 | manifest y evaluación de seis casos | existe evidencia para conservar o cambiar backend |

El próximo trabajo es decidir con el equipo cómo aislar/agregar la dependencia DeepTalk e implementar
su adapter. No comenzar MediaPipe pose, SyncNet o `from_folder` hasta que el adapter y el reporte
produzcan un flujo inspeccionable y la evaluación identifique sus errores reales.
