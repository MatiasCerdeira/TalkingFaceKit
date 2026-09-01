# Inicio rápido imaginado

Esta página muestra la experiencia de una versión futura. Cada bloque indica si puede ejecutarse
con el repositorio actual.

## Instalación

### Núcleo mínimo — disponible

El núcleo inspecciona y decodifica video con PyAV, y representa datos numéricos con NumPy:

```bash
uv sync
```

En un paquete publicado, el equivalente esperado sería:

```bash
uv add talkingfacekit
```

### Capacidades opcionales — parcialmente disponible

La instalación futura debería mantener extras pequeños y orientados a capacidad:

```bash
uv add "talkingfacekit[tracking-mediapipe]"  # Disponible en pyproject, pensado para distribución.
uv add "talkingfacekit[rendering]"           # Disponible en pyproject, pensado para distribución.
uv add "talkingfacekit[audio]"               # Objetivo.
uv add "talkingfacekit[speech]"              # Objetivo.
uv add "talkingfacekit[flame]"               # Investigación.
uv add "talkingfacekit[all]"                 # Objetivo; sólo cuando la matriz sea mantenible.
```

Dentro de este repositorio se usa `uv sync --extra <nombre>`. TalkingFaceKit nunca debería instalar
o descargar modelos en el momento del import.

## Cinco minutos: video a landmarks y mesh

### Flujo actual — disponible

```python
from pathlib import Path

from talkingfacekit import (
    TalkingFaceSequence,
    build_mediapipe_face_mesh,
    render_face_mesh_html,
    save_landmark_track,
)
from talkingfacekit.tracking.mediapipe import MediaPipeFaceTracker

video_path = Path("portrait.webm")
sequence = TalkingFaceSequence.from_video(video_path)
tracker = MediaPipeFaceTracker(Path("models/face_landmarker.task"))

landmarks = sequence.track_landmarks(tracker, name="mediapipe")
save_landmark_track(landmarks, Path("outputs/portrait-landmarks.npz"))

mesh = build_mediapipe_face_mesh(
    landmarks,
    image_width=sequence.source.metadata.width,
    image_height=sequence.source.metadata.height,
)
render_face_mesh_html(mesh, Path("outputs/portrait-mesh.html"))
```

El flujo conserva timestamps y frames sin detección. No conserva pixels RGB dentro de la
secuencia. La profundidad del mesh sigue siendo relativa a MediaPipe.

### Flujo unificado propuesto — objetivo

Una API madura debería permitir expresar un pipeline completo sin perder la posibilidad de usar
las funciones pequeñas:

```python
from pathlib import Path

from talkingfacekit import TalkingFaceSequence
from talkingfacekit.pipeline import Pipeline
from talkingfacekit.tracking.mediapipe import MediaPipeFaceTracker

sequence = TalkingFaceSequence.from_video(Path("portrait.webm")).clip(0.5, 8.0)

pipeline = Pipeline(
    steps=[
        MediaPipeFaceTracker(Path("models/face_landmarker.task")),
        "head-pose",
        "face-mesh",
        "audio",
        "voice-activity",
        "visemes",
    ]
)

result = pipeline.run(sequence, cache_dir=Path(".tfk-cache"))
result.save(Path("outputs/portrait.tfk"))
result.report(Path("outputs/portrait-report.html"))
```

`Pipeline` no debería ser el primer paso de implementación. Sólo tiene sentido después de que
existan varias operaciones reales con una interfaz estable y semántica transaccional común.

## Inspeccionar sin procesar — disponible

```python
from talkingfacekit import TalkingFaceSequence

sequence = TalkingFaceSequence.from_video("interview.webm")
metadata = sequence.source.metadata

print(sequence.source.path)
print(metadata.width)
print(metadata.height)
print(metadata.average_fps)
print(metadata.stream_duration_seconds)
print(metadata.has_audio)
print(sequence.duration_seconds)  # None: intervalo abierto hasta EOF
```

Los valores ausentes permanecen como `None`; no se deducen a partir de otros campos.

## Decodificar con memoria acotada — disponible

```python
from talkingfacekit import stream_video_frames

for frame in stream_video_frames(
    "interview.webm",
    start_seconds=2.0,
    end_seconds=3.0,
):
    consume(frame.rgb, timestamp_seconds=frame.timestamp_seconds)
```

El intervalo es semiabierto: `[2.0, 3.0)`. Cada `rgb` tiene shape `(height, width, 3)`, dtype
`uint8`, canales RGB y rango `[0, 255]`.

## Audio y habla — objetivo

El diseño esperado mantiene audio y video sobre timestamps de origen:

```python
from talkingfacekit.audio import stream_audio_chunks
from talkingfacekit.speech import VoiceActivityDetector, align_phonemes

chunks = stream_audio_chunks(
    "interview.webm",
    sample_rate_hz=16_000,
    channel_layout="mono",
)
voice = VoiceActivityDetector().detect(chunks)
phonemes = align_phonemes(
    audio=chunks,
    transcript="Welcome to TalkingFaceKit",
    language="en",
)
```

El resample y el downmix son parámetros explícitos. El resultado registra sample rate original,
sample rate de salida, backend, versión, configuración y tiempos de cada segmento.

## Multi-rostro — objetivo

El track actual procesa un solo rostro. La extensión propuesta agrega una colección sin cambiar el
significado de `FaceLandmarkTrack`:

```python
from talkingfacekit.tracking import MultiFaceTracker

faces = sequence.track_faces(
    MultiFaceTracker(max_faces=4),
    name="participants",
)

for face_id, face_track in faces.items():
    print(face_id, face_track.first_timestamp_seconds, face_track.coverage)
```

La identidad es local a la secuencia y no implica reconocimiento de la persona. Un backend debe
registrar cortes, reapariciones y ambigüedades; no puede reasignar una identidad en silencio.

## Próximos pasos

- Leer [conceptos fundamentales](concepts.md) antes de diseñar un nuevo track.
- Consultar [contratos de datos](data-contracts.md) antes de exponer arrays.
- Elegir una entrega del [roadmap](roadmap.md) antes de crear módulos nuevos.
- Usar la [referencia de API propuesta](api-reference.md) como dirección, no como compatibilidad
  garantizada.
