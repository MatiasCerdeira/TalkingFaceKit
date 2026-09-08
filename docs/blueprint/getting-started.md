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
uv add "talkingfacekit[speech]"              # Objetivo.
uv add "talkingfacekit[flame]"               # Investigación.
uv add "talkingfacekit[all]"                 # Objetivo; sólo cuando la matriz sea mantenible.
```

Dentro de este repositorio se usa `uv sync --extra <nombre>`. TalkingFaceKit nunca debería instalar
o descargar modelos en el momento del import.

## Cinco minutos: video a segmentos candidatos — objetivo prioritario

La experiencia principal futura será analizar primero un video:

```python
from pathlib import Path

from talkingfacekit.analysis import analyze_video
from talkingfacekit.integrations.deeptalk import DeepTalkAnalyzer

report = analyze_video(
    "interview.mp4",
    analyzer=DeepTalkAnalyzer(model_dir=Path("models/deeptalk")),
)

for segment in report.segments:
    print(segment.start_seconds, segment.end_seconds, segment.status, segment.reasons)

report.save_json("outputs/interview-analysis.json")
report.render_overlay("outputs/interview-analysis.mp4")
```

Esta API es ilustrativa y todavía no funciona. El audio PyAV con timestamps ya está disponible; el
orden restante es adapter DeepTalk experimental, reporte/overlay de un video y evaluación de seis
casos. El primer reporte marca pose y sync como `not_evaluated`; no promete segmentos completamente
válidos antes de medirlos.

La instalación del backend y la forma de obtener sus modelos se definirán con su adapter. No se
agregará una dependencia ni se descargarán assets implícitamente.

## Cinco minutos: video a landmarks y mesh — capacidad visual existente

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

## Audio disponible; habla como próximo objetivo

El flujo disponible mantiene audio y video sobre timestamps de origen:

```python
from talkingfacekit import stream_audio_chunks

chunks = stream_audio_chunks(
    "interview.webm",
)
for chunk in chunks:
    consume(chunk.samples, chunk.start_timestamp_seconds)
```

El stream conserva sample rate y canales decodificados. El adapter DeepTalk convertirá después a
mono 16 kHz de manera explícita dentro de su boundary. ASR, forced alignment, fonemas y visemas dejan
de ser prioridad inmediata.

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

- Seguir el orden obligatorio del [roadmap](roadmap.md): DeepTalk, reporte y evaluación; audio ya
  está completo.
- Leer [conceptos fundamentales](concepts.md) antes de diseñar un nuevo track.
- Consultar [contratos de datos](data-contracts.md) antes de exponer arrays.
- Elegir una entrega del [roadmap](roadmap.md) antes de crear módulos nuevos.
- Usar la [referencia de API propuesta](api-reference.md) como dirección, no como compatibilidad
  garantizada.
