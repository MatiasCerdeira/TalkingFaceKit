# Roadmap de implementación

Este roadmap transforma la visión en entregas verticales. El orden es una propuesta y puede cambiar
por necesidades del proyecto. Cada etapa debe cerrar un flujo útil de punta a punta; no se crean
todos los tipos o módulos de una fase por adelantado.

## Regla de avance

Una capacidad pasa de **objetivo** a **disponible** sólo cuando:

- existe un caso de uso ejecutable desde Python;
- tiene CLI si el flujo beneficia a usuarios no programadores;
- sus contratos validan dtype, shape, unidades, rango, ejes, tiempo y faltantes;
- los efectos están en una integración;
- hay tests unitarios y de integración apropiados;
- la API pública tiene docstrings NumPy;
- README y arquitectura vigente fueron actualizados;
- no descarga assets durante tests o import;
- Ruff, mypy y pytest pasan;
- se documentan limitaciones y failure modes.

## Dependencias entre etapas

```text
0. base actual
   |
   +--> 1. estabilizar media/timeline
           |
           +--> 2. audio core --------> 5. habla/alineación ----+
           |                                                |
           +--> 3. calidad landmarks --> 4. multi-rostro    |
           |                     |            |              |
           |                     +--> 6. pose/blendshapes    |
           |                                  |              |
           +--> 7. proyecto multipista <------+--------------+
                              |
                              +--> 8. FLAME/3D
                              +--> 9. render/export
                              +--> 10. pipeline/batch/cache
```

El proyecto multipista puede comenzar antes si dos familias de tracks ya demuestran su necesidad.

## Etapa 0 — base disponible

### Resultado

Video local a landmarks MediaPipe persistidos y mesh HTML.

### Capacidades existentes

- `TalkingFaceSequence.from_video`;
- `TalkingFaceSequence.clip` con intervalos sobre la línea temporal fuente;
- `VideoMetadata` con validación de dimensiones, FPS y duración;
- `DecodedVideoFrame` y `stream_video_frames`;
- `FaceLandmarkTrack` y `LandmarkTracker`;
- `MediaPipeFaceTracker` para un rostro;
- NPZ de landmarks v1;
- `FaceMeshTrack` y conversión MediaPipe 468/852;
- renderer Plotly offline;
- CLI `extract-landmarks` con intervalo, `inspect-landmarks`, `render-mesh`;
- tests con fixtures chicos y backends fake.

### Deuda visible

- la CLI de tracking no expone thresholds;
- provenance persistida es mínima;
- sólo hay un rostro y un backend;
- no existe audio core;
- algunos contratos actuales retienen arrays sin impedir mutación;
- el renderer materializa todos los frames en el HTML y no escala a secuencias largas.

La deuda no invalida el slice; ayuda a priorizar.

## Etapa 1 — estabilizar media e intervalos

### Resultado de usuario

Inspeccionar correctamente fuentes comunes, seleccionar un clip y ejecutar tracking sólo sobre ese
intervalo desde Python y CLI.

### Entrega mínima

1. Validar `VideoMetadata`: dimensiones positivas, FPS/duración finitos y positivos cuando existen.
2. Agregar flags `--start-seconds` y `--end-seconds` a `extract-landmarks`.
3. Agregar una operación barata `sequence.clip(...)` o validar primero el uso directo de intervalos.
4. Testear VFR, timestamps no-cero, duración desconocida y media rotada si hay fixtures mínimos.
5. Documentar claramente selección del primer stream y metadata no soportada.

### Decisiones necesarias

- si `end_seconds` de una secuencia creada desde video debe usar duración reportada o quedar
  ilimitado cuando la duración es dudosa;
- cómo exponer múltiples streams sin romper `VideoMetadata`.

### Fuera de alcance

Resize, sampling por FPS, cámara en vivo y materialización de video.

## Etapa 2 — audio core y decode por streaming

### Resultado de usuario

Inspeccionar y extraer audio preservando sample rate, canales y timestamps; guardar un WAV explícito.

### Entrega 2A: contrato y streaming

1. `AudioStreamMetadata` o ampliación estructurada de metadata.
2. `DecodedAudioChunk` con shape/dtype/layout definidos.
3. `stream_audio_chunks` en boundary PyAV.
4. Intervalo semiabierto y manejo documentado de muestras que cruzan límites.
5. Tests sintéticos mono/stereo, distintos sample rates, sin audio, timestamps y chunks.

### Entrega 2B: materialización acotada y export

1. `AudioTrack` o helper equivalente.
2. `decode_audio(..., max_bytes=...)`.
3. writer WAV con overwrite transaccional.
4. CLI `extract audio`.
5. Round-trip y tests de amplitud/canales.

### Entrega 2C: conversiones explícitas

1. resample;
2. downmix/remix;
3. normalización opcional;
4. provenance de cada transformación.

### Decisiones necesarias

- dtype canónico;
- orientación `(samples, channels)`;
- rango de amplitud;
- layout y nombres de canales;
- timestamps tras resample;
- PyAV solo o dependencia adicional.

## Etapa 3 — calidad y procesamiento de landmarks

### Resultado de usuario

Medir, suavizar e interpolar un track sin perder gaps ni timeline.

### Entrega 3A: reporte de calidad

- cobertura;
- cantidad y duración de gaps;
- velocidad/jitter robustos;
- detección de outliers simple;
- summary Python y CLI/HTML pequeño.

### Entrega 3B: smoothing consciente del tiempo

- primer método con config dataclass tipado;
- usa `timestamps_seconds`, no FPS;
- no cruza gaps largos;
- produce track nuevo y conserva original.

### Entrega 3C: interpolación marcada

- máximo gap explícito;
- máscara `observed/interpolated/valid` resuelta mediante un contrato nuevo o extensión versionada;
- tests de gaps al comienzo, medio y final.

### Decisión crítica

Extender `FaceLandmarkTrack` puede romper construcción y NPZ v1. Evaluar un nuevo tipo procesado o
campos opcionales con schema v2; no reinterpretar `detected`.

## Etapa 4 — multi-rostro e identidad local

### Resultado de usuario

Seguir participantes de una entrevista y extraer resultados por ID local.

### Entrega 4A: detecciones

- contrato de bounding boxes y scores;
- backend concreto;
- renderer overlay de boxes;
- fixture sintético de dos rostros sin modelo pesado en unit tests.

### Entrega 4B: asociación temporal

- IDs locales;
- births/deaths/gaps;
- ambigüedad explícita;
- test de cruce, desaparición y reaparición.

### Entrega 4C: landmarks por track

- crop/ROI hacia tracker;
- transformación invertible al frame fuente;
- API para seleccionar un `face_id`;
- comparación visual.

### Fuera de alcance inicial

Reconocimiento entre videos, base biométrica y diarización.

## Etapa 5 — habla y alineación

### Resultado de usuario

Audio a voz/segmentos/texto/fonemas/visemas con tiempos explícitos.

### Entrega 5A: features y VAD

- RMS/energía por ventana;
- `VoiceActivityTrack`;
- backend local pequeño o algoritmo documentado;
- evaluación sobre fixtures sintéticos.

### Entrega 5B: transcripción

- un backend concreto opcional;
- segmentos y palabras con timestamps;
- idioma y confidence;
- tests con backend fake, integration test marcado para modelo real.

### Entrega 5C: forced alignment y visemas

- inventario fonético versionado;
- regiones no alineadas;
- un mapping idioma/rig concreto;
- curvas con transiciones explícitas.

### Dependencias y permisos

Cualquier modelo o dependencia de speech requiere acuerdo del equipo. No se descarga durante tests.

## Etapa 6 — pose, mirada y parámetros faciales

### Resultado de usuario

Obtener movimiento de cabeza y señales de animación backend-independent.

### Orden propuesto

1. `HeadPoseTrack` desde landmarks y cámara aproximada documentada.
2. renderer de ejes para validación.
3. blendshapes MediaPipe si el backend/version los soporta de manera estable.
4. `GazeTrack` aproximado y blinks, con limitaciones claras.
5. retargeting hacia un schema de rig concreto.

### Criterios

- convenciones de rotación y cámara testeadas;
- units explícitas;
- canales únicos/versionados;
- ninguna etiqueta afectiva presentada como verdad objetiva.

## Etapa 7 — proyecto multipista y provenance

### Trigger para iniciar

Al menos dos familias persistibles —por ejemplo landmarks y audio/speech— necesitan viajar juntas y
ser abiertas parcialmente.

### Entrega 7A: manifest y lectura

- elegir directory bundle o formato existente;
- manifest v1;
- metadata, source fingerprint e intervalo;
- catálogo de artefactos sin cargar arrays;
- validación shallow/deep.

### Entrega 7B: escritura transaccional

- temporales sibling;
- overwrite explícito;
- arrays/chunks;
- checksums;
- limpieza ante fallo;
- tests de corrupción y archivo futuro.

### Entrega 7C: provenance graph y migración

- backend/model/config/padres;
- loader de schema v1;
- migrador de al menos un fixture v0 sintético o postergar la API hasta existir v2;
- export/import de NPZ landmarks sin perder datos.

### Decisiones

Consultar todas las abiertas en [arquitectura objetivo](target-architecture.md). No adoptar una
dependencia de storage sin aprobación.

## Etapa 8 — fitting FLAME y geometría 3D

### Resultado de usuario

Ajustar un modelo paramétrico a landmarks y exportar parámetros/mesh con diagnósticos.

### Spike de investigación

- verificar licencias y assets;
- elegir implementación/model version;
- medir CPU, CUDA y MPS;
- fijar convenciones de cámara y parámetros;
- probar un clip corto fuera de tests unitarios;
- decidir si el output puede ser estable.

Un spike no publica API.

### Slice publicable

- adapter opcional `FlameFitter`;
- config tipada;
- parámetros de identidad vs temporales;
- convergencia/loss por frame;
- resultado NumPy core;
- persistencia en proyecto;
- renderer/export simple;
- tests unitarios con optimizer/model fake e integración marcada.

### Fuera de alcance inicial

Textura fotorrealista, relighting, entrenamiento y generación neuronal completa.

## Etapa 9 — rendering, reportes y export

### Resultado de usuario

Inspeccionar visualmente cualquier etapa y llevar geometría/curvas a herramientas externas.

### Orden propuesto

1. overlay de landmarks/boxes/pose a video corto;
2. reporte HTML de calidad y provenance;
3. comparación de dos tracks;
4. glTF/GLB de mesh estático;
5. animación si el formato conserva timeline/topología;
6. adaptadores Blender/Unity/Unreal sólo por demanda real.

### Escalabilidad Plotly

Definir límite o estrategia para no generar HTML enorme: sampling de display, compresión, assets
separados u otro renderer. Nunca modificar el track sólo para ocultar el costo.

## Etapa 10 — pipeline, cache y batch

### Trigger para iniciar

Al menos tres operaciones reales se repiten manualmente y la persistencia multipista ya tiene schema
estable.

### Entrega 10A: pipeline lineal

- lista de steps conocidos;
- validación previa de inputs/outputs;
- config TOML/JSON sin código arbitrario;
- progreso/cancelación;
- ejecución secuencial.

### Entrega 10B: cache

- claves explicables;
- invalidación por source/config/backend/model/schema;
- stats hit/miss;
- eviction manual primero;
- cache separada del output.

### Entrega 10C: dataset y batch

- manifest JSONL;
- ejecución secuencial resumible;
- reporte por item;
- luego workers CPU y scheduling GPU basados en perfiles.

### Fuera de alcance inicial

Scheduler distribuido, cluster, UI web y marketplace de plugins.

## Backlog de investigación

Estas líneas no tienen compromiso ni orden:

- fuentes en vivo y streaming de baja latencia;
- diarización audiovisual;
- compensación de drift A/V;
- estimación de textura/iluminación;
- audio a expresión o talking-head generation;
- evaluación perceptual;
- oclusión y confidence por landmark;
- tracking corporal/manos para contexto;
- calibración multi-cámara;
- soporte de formatos profesionales de animación;
- backend remoto con privacidad explícita;
- ejecución distribuida.

Una investigación puede terminar en “no implementar”. Su salida mínima es evidencia, límites y una
decisión registrada.

## Próxima entrega recomendada

Después de estabilizar el frame streaming actualmente en desarrollo, el siguiente slice más
coherente es **audio core y decode por streaming**:

- completa la segunda modalidad central de TalkingFaceKit;
- fuerza a resolver sincronización sin inventar una arquitectura genérica;
- reutiliza PyAV y el patrón de streaming existente;
- habilita VAD, fonemas, visemas y lip-sync;
- puede testearse con audio sintético pequeño;
- no requiere todavía PyTorch, CUDA ni descargar modelos.

Antes de implementarlo hay que acordar dtype, layout, amplitud y semántica temporal.
