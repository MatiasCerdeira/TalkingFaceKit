# Guía de flujos propuesta

Esta guía está organizada por resultados de usuario. Las APIs marcadas como objetivo son bocetos
para discutir y validar antes de implementarlas.

## 1. Inspección y selección de media

### Inspeccionar el primer stream — disponible

```python
from talkingfacekit import TalkingFaceSequence

sequence = TalkingFaceSequence.from_video("session.webm")
metadata = sequence.source.metadata
```

Actualmente se informa ancho, alto, FPS promedio opcional, duración opcional y presencia de audio.
La metadata describe siempre el stream completo; el intervalo pertenece a `sequence`.

### Inspección completa — objetivo

La metadata futura debería describir streams sin decodificarlos:

```python
from talkingfacekit.io import inspect_media

media = inspect_media("session.mov")
for stream in media.video_streams:
    print(stream.index, stream.codec, stream.width, stream.height, stream.rotation_degrees)
for stream in media.audio_streams:
    print(stream.index, stream.codec, stream.sample_rate_hz, stream.channel_layout)
```

La selección de stream debe ser explícita si hay más de uno. Rotación, pixel aspect ratio, color
primaries, transfer function y time base no pueden aplicarse o descartarse silenciosamente.

### Recortar una vista — disponible

```python
clip = sequence.clip(start_seconds=12.5, end_seconds=18.0)
```

`clip` crea una secuencia barata que comparte el mismo `VideoSource`, conserva timestamps de origen,
no decodifica y comienza sin resultados adjuntos. Una operación futura separada puede rebasing el
timeline a cero.

## 2. Frames de video

### Streaming RGB — disponible

`stream_video_frames` produce `DecodedVideoFrame` en orden temporal. Es la frontera canónica para
trackers visuales.

### Transformaciones de decode — objetivo

```python
frames = stream_video_frames(
    "session.mov",
    start_seconds=4.0,
    end_seconds=8.0,
    video_stream=0,
    output_size=(640, 360),
    pixel_format="rgb24",
    rotation="apply",
    sampling="source",
)
```

Cada transformación debe quedar declarada. `sampling="source"` conserva todos los frames;
alternativas como `fps=10` requieren una política determinista de selección y timestamps de origen.
No se debería usar el FPS promedio para fabricar tiempos.

### Materializar — objetivo condicionado

Para clips chicos podría existir `decode_video`, pero debe exigir o advertir un límite de memoria:

```python
video = decode_video(clip, max_bytes=512 * 1024 * 1024)
```

El contrato de un tensor de video completo sólo se publicará si casos reales justifican el costo.

## 3. Audio

### Decodificar chunks — objetivo, siguiente gran frontera

```python
from talkingfacekit.audio import stream_audio_chunks

for chunk in stream_audio_chunks(
    "session.webm",
    start_seconds=2.0,
    end_seconds=5.0,
    sample_rate_hz=None,
    channel_layout="source",
):
    consume(chunk.samples, chunk.start_timestamp_seconds)
```

El primer contrato debería conservar el sample rate y layout originales. El resample, downmix y
normalización deben ser operaciones explícitas que registren sus parámetros.

### Transformaciones de audio — objetivo

```python
mono_16k = audio.resample(sample_rate_hz=16_000).remix(channel_layout="mono")
normalized = mono_16k.normalize_peak(target_dbfs=-1.0)
```

La normalización produce un track nuevo. Nunca modifica muestras en sitio ni confunde amplitud
lineal, dBFS, LUFS y presión sonora física.

### Features acústicos — objetivo

- RMS, peak y energía por ventana;
- log-mel spectrogram y MFCC con configuración explícita;
- pitch/F0, voicing probability e intensidad;
- spectral centroid y medidas útiles para animación;
- loudness integrada cuando el backend soporte la norma declarada.

Todos conservan ventanas, hop, sample rate y convención de padding.

## 4. Detección, tracking e identidad facial

### Un rostro con landmarks — disponible

`MediaPipeFaceTracker` procesa un rostro y devuelve 478 puntos por frame. Las faltas de detección se
conservan como filas `NaN`.

### Detecciones y bounding boxes — objetivo

```python
detections = sequence.detect_faces(detector, name="faces")
```

Un `FaceDetectionTrack` debería incluir boxes, score, máscara de validez y, cuando corresponda,
landmarks livianos de detección. El formato de box debe declarar si usa `(x_min, y_min, width,
height)` o esquinas, y si sus unidades son pixels o coordenadas normalizadas.

### Asociación multi-rostro — objetivo

```python
face_tracks = sequence.track_faces(
    tracker,
    detections="faces",
    name="participants",
)
```

La asociación mantiene IDs locales, eventos de nacimiento/fin, gaps y ambigüedad. No debe rellenar
un cruce de identidades con una asignación arbitraria.

### Landmarks por identidad — objetivo

El tracker de landmarks puede recibir una región o un `face_id` y producir un
`FaceLandmarkTrack`. Esto permite reutilizar el contrato actual sin agregar un eje de rostro a todos
los arrays existentes.

## 5. Limpieza y control de calidad

### Métricas — objetivo

```python
quality = evaluate_landmarks(landmarks)
print(quality.detection_coverage)
print(quality.temporal_jitter)
print(quality.longest_missing_gap_seconds)
```

Las métricas propuestas incluyen cobertura, gaps, velocidad/aceleración robusta, outliers,
consistencia de escala, reprojection error y estabilidad de identidad. Toda métrica debe documentar
si un valor mayor es mejor y en qué unidades está expresada.

### Smoothing — objetivo

```python
smoothed = smooth_landmarks(
    landmarks,
    method="one-euro",
    min_cutoff_hz=1.0,
    beta=0.05,
)
```

El filtro usa diferencias reales de timestamps y no supone frame rate constante. Conserva los
datos originales, crea provenance y no cruza gaps o cortes de identidad sin una política explícita.

### Interpolación — objetivo

```python
filled = interpolate_landmarks(
    smoothed,
    method="linear",
    max_gap_seconds=0.12,
)
```

El resultado distingue valores observados e interpolados. Los gaps mayores siguen faltantes.

## 6. Pose, mirada y movimiento facial

### Pose de cabeza — objetivo

Un `HeadPoseTrack` debería separar:

- rotación con convención explícita (`rotation_matrix`, quaternion o axis-angle);
- traslación, con unidades métricas sólo si el método realmente las estima;
- cámara o intrinsics usados;
- confidence/reprojection error.

Ángulos yaw/pitch/roll pueden ser propiedades derivadas, pero deben declarar orden y convención.

### Gaze y párpados — objetivo

`GazeTrack` registra dirección por ojo, validez y sistema de coordenadas. Los eventos de blink son
segmentos temporales derivados y no reemplazan la señal continua. Un resultado MediaPipe basado en
iris debe declarar que es una aproximación 2.5D, no eye tracking calibrado.

### Blendshapes y action units — objetivo

`BlendshapeTrack` contiene nombres estables y pesos `float32`. La escala debe distinguir pesos
normalizados, logits y unidades específicas del rig. Los Facial Action Coding System action units
requieren un schema y backend propios; no son sinónimos automáticos de blendshapes.

Clasificadores afectivos permanecen en investigación y deben evitar etiquetas deterministas sobre
estados internos de una persona.

## 7. Segmentación y regiones de interés

### Máscaras — objetivo

Posibles tracks: rostro, piel visible, pelo, boca, labios, dientes y oclusores. Una máscara debe
declarar resolución, alineación con el frame, dtype y si contiene booleanos, clases o probabilidades.

Por su tamaño, las máscaras deberían persistirse por chunks o mediante un codec; no como un único
NPZ enorme por defecto.

### Crops estabilizados — objetivo

```python
crops = stream_face_crops(
    sequence,
    face_track=face_tracks["face-0001"],
    output_size=(256, 256),
    padding=0.2,
    stabilization="similarity",
)
```

Cada crop incluye la transformación invertible al frame original. El crop no reemplaza la
geometría en coordenadas de fuente.

## 8. Geometría 3D

### Mesh de MediaPipe — disponible

`build_mediapipe_face_mesh` produce 468 vértices y 852 triángulos. Es útil para visualización y
animación relativa.

### Operaciones de mesh — objetivo

- normales por vértice y por cara;
- smoothing temporal sin cambiar topología;
- cálculo de regiones y contornos;
- transformación rígida y cambio explícito de coordenadas;
- extracción de frames individuales;
- comparación entre meshes de la misma topología;
- exportación de geometría y curvas de animación.

### Fitting FLAME — investigación prioritaria

```python
from talkingfacekit.models.flame import FlameFitter

fit = sequence.fit_face_model(
    FlameFitter(model_path="models/flame.pkl", device="cuda:0"),
    landmarks="mediapipe-smoothed",
    name="flame",
)
```

El contrato debería separar parámetros constantes de identidad de parámetros temporales:

- `shape` por identidad;
- `expression` por frame;
- pose global, cuello, mandíbula y ojos;
- cámara e iluminación cuando se estimen;
- pérdida total y términos de diagnóstico;
- máscara de convergencia.

Los archivos de FLAME tienen licencias y distribución propias; TalkingFaceKit no debe incluirlos.
PyTorch y CUDA permanecen extras opcionales y encapsulados en el adapter.

### Textura y fotometría — investigación

La estimación de albedo, iluminación y textura requiere convenciones de color, cámara y visibilidad
mucho más fuertes. Debe llegar después del fitting geométrico y no mezclarse en el contrato base de
mesh.

## 9. Voz, texto, fonemas y visemas

### Voice activity detection — objetivo

`VoiceActivityTrack` es una lista de intervalos con probabilidad. Mantiene silencios; no recorta el
audio automáticamente.

### Transcripción — objetivo

```python
transcript = sequence.transcribe(
    recognizer,
    language="es",
    word_timestamps=True,
    name="asr",
)
```

Cada segmento o palabra registra inicio, fin, texto y confianza si el backend la provee. La
normalización del texto original debe ser una transformación separada.

### Forced alignment — objetivo

El alineador recibe audio y texto conocido, y produce palabras/fonemas temporizados. Debe registrar
inventario fonético, idioma, pronunciaciones desconocidas y regiones no alineadas.

### Visemas — objetivo

```python
visemes = phonemes.to_visemes(mapping="arkit-en-v1")
```

El mapping es versionado y dependiente de idioma/rig. Un visema puede ser una categoría por
intervalo o una curva de pesos con transiciones; ambos contratos no deben confundirse.

### Diarización audiovisual — investigación

La diarización de audio y la actividad de labios podrían asociar turnos de habla con rostros. El
resultado debe expresar incertidumbre, superposición y hablante fuera de cámara.

## 10. Sincronización audiovisual

### Estimar offset — objetivo

```python
sync = estimate_av_sync(
    mouth_motion=landmarks,
    audio_energy=features,
    search_range_seconds=(-0.5, 0.5),
)
```

El resultado incluye offset, score, rango analizado y método. Aplicar la corrección es otra
operación para evitar modificar timestamps por sorpresa.

### Drift y discontinuidades — investigación

Fuentes largas pueden tener drift, cortes o timestamps discontinuos. La corrección requiere una
transformación temporal por tramos, no sólo un offset escalar.

## 11. Animación, edición y retargeting

### Curvas de animación — objetivo

`AnimationCurveTrack` representa canales nombrados, timestamps, valores y unidades. Puede recibir
blendshapes, pose o visemas, pero no pierde su schema de origen.

### Retargeting — objetivo

```python
rig_animation = retarget_blendshapes(
    source=blendshapes,
    target_schema="arkit-52",
    mapping=Path("rigs/character-a.json"),
)
```

El mapping incluye escalas, offsets, clamps, combinaciones y canales no soportados. La exportación
no debe suponer que todos los rigs comparten nombres o rango `[0, 1]`.

### Audio a animación — investigación

Un backend generativo podría producir visemas, blendshapes o parámetros 3D a partir de audio. Sus
salidas usan los mismos contratos que resultados observados, pero provenance debe indicar que son
predicciones. Identidad, estilo y neutral pose requieren entradas explícitas.

### Edición temporal — objetivo tardío

Recorte, concatenación, crossfade, time-warp y mezcla de tracks deben definir qué ocurre con
timestamps, gaps, IDs, provenance y señales categóricas. No deberían implementarse como mutaciones
destructivas sobre resultados originales.

## 12. Rendering y visualización

### HTML de mesh — disponible

El renderer Plotly escribe un archivo autocontenido con cámara orbital y controles temporales. Usa
un material neutro y un multiplicador de profundidad sólo visual.

### Overlays de diagnóstico — objetivo

```python
render_overlay_video(
    sequence,
    output_path="outputs/debug.mp4",
    layers=["face-boxes", "landmarks", "head-pose", "speaker"],
    audio="copy",
)
```

El renderer debe declarar codec, pixel format, dimensiones, FPS/timestamps y tratamiento del audio.
Los overlays nunca deben alterar el track de origen.

### Comparadores — objetivo

- lado a lado de backends;
- error por punto o región;
- curvas temporales sincronizadas con video/audio;
- vista de gaps y confidence;
- reportes HTML autocontenidos para inspección.

### Render 3D — objetivo/investigación

Una integración 3D puede renderizar mesh, cámara y luces, pero el core no adopta objetos del motor.
Exportar a glTF/GLB es preferible como primer puente portable; Blender, Unreal o Unity deberían ser
adaptadores separados si aparecen casos reales.

## 13. Persistencia y exportación

### NPZ de landmarks — disponible

`save_landmark_track` y `load_landmark_track` usan schema v1, no permiten overwrite implícito y
revalidan al cargar.

### Proyecto nativo `.tfk` — objetivo

Un contenedor de proyecto debería almacenar un manifest legible, arrays por tipo, schemas
versionados, provenance y referencias a la fuente. Debe permitir abrir metadata sin cargar todos
los datos y migrar versiones explícitamente. Ver [arquitectura objetivo](target-architecture.md).

### Exportadores interoperables — objetivo

- CSV/JSON para segmentos y tablas chicas;
- NPZ o NPY para arrays NumPy;
- WAV/FLAC para audio procesado;
- PNG o secuencias codificadas para máscaras;
- glTF/GLB para mesh y animación;
- formatos de curvas específicos sólo mediante integraciones dedicadas.

La exportación debe avisar cuando un formato no puede conservar topología, gaps, precisión,
provenance o time base.

## 14. Datasets, batch y cache

### Manifest de dataset — objetivo

```python
dataset = TalkingFaceDataset.from_manifest("dataset.jsonl")
for item in dataset:
    pipeline.run(item.sequence)
```

Cada entrada debería contener ID estable, fuente, intervalo, splits y metadata del usuario sin
acoplar el core a un dataset conocido.

### Ejecución por lotes — objetivo tardío

- límite explícito de workers y dispositivos;
- progreso y logs estructurados;
- continuar ante errores configurables;
- resultados transaccionales por item;
- resume basado en configuración y fingerprint;
- reporte final con éxitos, fallos y skips.

### Cache — objetivo tardío

La clave de cache debe incluir fuente, intervalo, operación, parámetros, backend, modelo y versión
del schema. Un cambio relevante invalida el resultado. La cache no es el mismo concepto que un
resultado nombrado por el usuario.

## 15. Evaluación y benchmarks

### Reporte de calidad — objetivo

Un reporte combina métricas sin convertirlas en un score opaco:

- cobertura y gaps;
- jitter y outliers;
- reprojection/fitting error;
- estabilidad de identidad;
- offset y confianza de lip-sync;
- real-time factor, memoria pico y dispositivo;
- versión de backend/modelo y configuración.

### Ground truth — objetivo tardío

Las métricas contra ground truth deben declarar formato, sistema de coordenadas, regiones incluidas,
alineación previa y tratamiento de faltantes. Los tests del repositorio seguirán usando fixtures
sintéticos pequeños; los benchmarks externos no se descargan durante pytest.

## 16. Fuentes en vivo y servicios remotos

**Investigación.** Cámara, RTSP, WebRTC o inferencia remota requieren backpressure, latencia,
reconexión y timestamps distintos del flujo de archivos. No deberían forzarse dentro de
`stream_video_frames` hasta que exista un caso real. Una futura `LiveTalkingFaceSession` puede
compartir tipos de frames y resultados, pero tendrá un ciclo de vida explícito y contratos propios.
