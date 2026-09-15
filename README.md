# TalkingFaceKit

TalkingFaceKit es una biblioteca modular en Python para analizar videos y localizar intervalos útiles
para datasets de personas hablando.

## Dirección actual del proyecto

El objetivo principal ya no es asumir que cada entrada es un video de tipo *talking head*. La
biblioteca debe poder recibir videos arbitrarios y responder, sobre la línea temporal original:

- cuándo hay habla audible;
- qué rostros aparecen y cómo se mantienen sus identidades locales;
- qué rostro visible, si alguno, parece producir esa voz;
- qué intervalos cumplen criterios posteriores de orientación a cámara, calidad y sincronización;
- por qué un intervalo fue aceptado, rechazado o marcado como incierto.

El primer milestone será un analizador de **un solo video** que genere resultados estructurados y
un video de diagnóstico. DeepTalk-ASD es el backend experimental elegido para obtener rápidamente
detección/tracking de rostros, VAD y scores de hablante activo. No será parte del modelo de dominio
ni se considera todavía una dependencia estable. TalkingFaceKit conservará los timestamps fuente,
traducirá los outputs a contratos propios y decidirá la política de segmentos. MediaPipe se reserva
para pose y calidad visual; SyncNet se evaluará después para sincronización A/V explícita.

Orden de implementación acordado:

1. ~~agregar `DecodedAudioChunk` y audio PyAV por streaming con timestamps de fuente~~ — completo;
2. construir un adapter experimental y acotado para DeepTalk-ASD — implementado con tests
   sintéticos; falta validarlo end-to-end con modelos reales;
3. producir para un video speech intervals, face tracks, `raw_score`, segmentos candidatos,
   razones y provenance, además de un overlay inspeccionable;
4. evaluar seis casos mínimos y decidir si conservar DeepTalk, adaptar LR-ASD directamente o
   comparar TalkNet;
5. agregar orientación/calidad con MediaPipe y sincronización exacta con SyncNet;
6. recién entonces generalizar el flujo a carpetas, datasets y ejecución batch.

La generación de landmarks y meshes ya implementada se conserva como capacidad visual opcional;
no es el centro del producto. El estado detallado, la justificación y los criterios de decisión se
encuentran en [`PROJECT_DIRECTION.md`](PROJECT_DIRECTION.md), el
[`roadmap`](docs/blueprint/roadmap.md) y el
[`spike de DeepTalk-ASD`](docs/research/deeptalk_asd_compatibility.md).

> **Estado:** el flujo de análisis descripto arriba es dirección de producto, no API disponible.
> La sección siguiente enumera únicamente lo que funciona hoy.

## Funcionalidad actual

La primera versión inspecciona un archivo de video local y expone la metadata de su primer stream
de video:

- ancho y alto en píxeles;
- FPS promedio, cuando el archivo lo informa;
- duración en segundos, cuando el stream o el contenedor la informan;
- presencia de al menos un stream de audio.

```python
from talkingfacekit import TalkingFaceSequence

sequence = TalkingFaceSequence.from_video("video.webm")
source_metadata = sequence.source.metadata

print(sequence.source.path)
print(source_metadata.width)
print(source_metadata.height)
print(source_metadata.average_fps)
print(source_metadata.stream_duration_seconds)
print(source_metadata.has_audio)
```

Los FPS y la duración son `None` cuando el archivo no informa esos valores. `from_video` valida que
la ruta exista, que sea un archivo regular y que el contenedor tenga al menos un stream de video.
`VideoSource` separa explícitamente la identidad del archivo y la metadata del stream completo. Una
o más `TalkingFaceSequence` pueden compartir esa fuente y representar intervalos diferentes. La
clase ofrece la API principal, pero delega la inspección a la integración con PyAV; construir los
modelos directamente no abre archivos ni realiza otras operaciones de entrada/salida. Ninguno
conserva frames decodificados ni audio en memoria.

`VideoMetadata` exige dimensiones enteras positivas y presencia de audio booleana. Cuando los FPS
o la duración están disponibles, también deben ser valores finitos y positivos.

### Decodificar frames por streaming

La decodificación también está centralizada en la integración de video. `stream_video_frames`
abre el primer stream de video con PyAV y entrega un frame RGB por vez, sin materializar el video
completo en memoria:

```python
from talkingfacekit import stream_video_frames

for frame in stream_video_frames("portrait.webm", start_seconds=0.0, end_seconds=1.0):
    print(frame.frame_index)
    print(frame.timestamp_seconds)
    print(frame.rgb.shape)  # (alto, ancho, 3), RGB uint8
```

El intervalo es semiabierto: `[start_seconds, end_seconds)`. Los índices y timestamps provienen del
stream original; la función no usa los FPS para inventar tiempos. La integración conserva como
máximo el frame que está entregando en ese momento. Un consumidor puede retener explícitamente sus
arrays, pero hacerlo para muchos frames aumenta el uso de memoria.

`TalkingFaceSequence` usa las mismas reglas: el inicio debe ser finito y no negativo, y el final,
cuando existe, debe ser finito y posterior al inicio.

Un clip lógico reutiliza el mismo `VideoSource` sin decodificar ni copiar el video. Sus timestamps
permanecen sobre la línea temporal original, comienza sin tracks adjuntos y expone la duración de
su propio intervalo por separado de la duración informada por la fuente:

```python
sequence = TalkingFaceSequence.from_video("portrait.webm")
clip = sequence.clip(0.5, 1.5)

print(clip.source is sequence.source)  # True
print(clip.duration_seconds)  # 1.0
print(clip.source.metadata.stream_duration_seconds)  # duración informada del video completo
```

La fuente y los límites de una secuencia no se pueden reasignar después de construirla. Una
secuencia creada con `from_video` queda abierta hasta EOF (`end_seconds=None`): la duración
informada permanece como metadata y no se usa como si fuera un timestamp final exacto.

### Decodificar audio por streaming

`stream_audio_chunks` abre el primer stream de audio y entrega chunks sobre el mismo timeline de
fuente, sin cargar la pista completa:

```python
from talkingfacekit import stream_audio_chunks

for chunk in stream_audio_chunks("portrait.webm", start_seconds=0.5, end_seconds=1.5):
    print(chunk.start_sample_index)
    print(chunk.start_timestamp_seconds)
    print(chunk.samples.shape)  # (cantidad_de_samples, cantidad_de_canales)
    print(chunk.sample_rate_hz, chunk.channel_layout)
```

Cada buffer es C-contiguous, sample-major y `float32`. PCM entero se normaliza a full scale;
fuentes de punto flotante conservan su amplitud y no se recortan. La función no cambia sample rate,
no mezcla canales y recorta los chunks que cruzan los límites solicitados con precisión de sample.
Los índices continúan referidos al stream decodificado completo y los timestamps son PTS de fuente,
no valores calculados desde la duración del video.

### Adapter experimental DeepTalk-ASD

El primer adapter offline de hablante activo se instala mediante su extra opcional:

```bash
uv sync --extra active-speaker-deeptalk
```

El adapter prepara una sola secuencia para DeepTalk y copia observaciones de rostros y ventanas
diagnósticas con los scores finales devueltos por el backend, sin convertirlos en probabilidades ni
decisiones de speaking:

```python
from talkingfacekit import TalkingFaceSequence
from talkingfacekit.integrations.deeptalk import analyze_sequence

sequence = TalkingFaceSequence.from_video("conversation.mp4")
result = analyze_sequence(sequence)

print(result.face_observations)
print(result.score_windows)
```

El código de adaptación muestrea video a 25 Hz y convierte audio a mono PCM `int16` de 16 kHz dentro
del boundary experimental. Usa únicamente tiempo de medio relativo para alimentar el detector y
devuelve todos los timestamps sobre la timeline fuente. Esta ruta sólo está validada con media
sintética y un detector fake: todavía no se ejecutó end-to-end un MP4 con los modelos reales.
DeepTalk administra sus propios modelos al construir el detector; TalkingFaceKit no los incluye.

Los trackers consumen este stream compartido en lugar de abrir el video por su cuenta. Así, la
selección del stream, la conversión a RGB y las reglas de timestamps permanecen iguales para
MediaPipe y futuros backends como FLAME.

También está disponible un primer flujo vertical opcional de tracking facial con MediaPipe Face
Landmarker. MediaPipe se instala por separado porque no es necesario para inspeccionar metadata:

```bash
uv sync --extra tracking-mediapipe
```

TalkingFaceKit no descarga ni incluye modelos. El usuario debe proporcionar un modelo compatible
de Face Landmarker en formato `.task`:

```python
from pathlib import Path

from talkingfacekit import TalkingFaceSequence
from talkingfacekit.tracking.mediapipe import MediaPipeFaceTracker

sequence = TalkingFaceSequence.from_video("portrait.webm")
tracker = MediaPipeFaceTracker(Path("models/face_landmarker.task"))

landmarks = sequence.track_landmarks(tracker, name="mediapipe")

print(landmarks.landmarks.shape)  # (cantidad_de_frames, 478, 3)
print(landmarks.detected)  # una máscara booleana por frame
print(sequence.landmark_tracks["mediapipe"] is landmarks)
```

El tracker procesa un solo rostro. Consume los frames RGB de la integración compartida, los envía a
MediaPipe y no conserva sus píxeles. El resultado sí se conserva: índices de frame, timestamps
originales en segundos, 478 puntos `(x, y, z)` en `float32` y una máscara que indica en qué frames
se detectó el rostro. Los frames sin detección permanecen en la línea temporal y contienen `NaN` en
sus landmarks.

Los resultados se guardan por nombre para permitir futuras comparaciones entre backends o
configuraciones. Un nombre existente no se sobrescribe salvo que se use `overwrite=True`. El
resultado se adjunta a la secuencia solamente cuando el tracking termina correctamente; un error no
deja estado parcial.

### Extraer los landmarks de un video completo

La biblioteca incluye una CLI para ejecutar el flujo real de punta a punta y conservar su resultado.
Primero hay que instalar el extra opcional, crear las carpetas locales ignoradas por Git y colocar un
modelo Face Landmarker compatible dentro de `models/`:

```powershell
uv sync --extra tracking-mediapipe
New-Item -ItemType Directory -Force models, outputs
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" `
  -OutFile models/face_landmarker.task
```

Después se procesa el video completo:

```powershell
uv run python -m talkingfacekit extract-landmarks `
  tests/fixtures/example1.webm `
  --model models/face_landmarker.task `
  --output outputs/example1-landmarks.npz
```

Para procesar solamente un intervalo, se agregan límites sobre la línea temporal original:

```powershell
uv run python -m talkingfacekit extract-landmarks `
  tests/fixtures/example1.webm `
  --model models/face_landmarker.task `
  --output outputs/example1-clip-landmarks.npz `
  --start-seconds 0.5 `
  --end-seconds 1.5
```

El archivo comprimido conserva una entrada de landmarks por cada frame decodificado dentro del
intervalo, incluidos los frames sin detección. No conserva los píxeles RGB. Para comprobar y
resumir un resultado guardado:

```powershell
uv run python -m talkingfacekit inspect-landmarks outputs/example1-landmarks.npz
```

También se puede volver a cargar desde Python sin ejecutar MediaPipe otra vez:

```python
from talkingfacekit import load_landmark_track

track = load_landmark_track("outputs/example1-landmarks.npz")

print(track.landmarks.shape)
print(track.timestamps_seconds)
print(track.detected)
```

Los modelos y resultados se mantienen fuera de Git mediante `models/` y `outputs/`. La CLI no
descarga modelos automáticamente ni sobrescribe un archivo existente salvo que se agregue
`--overwrite`.

### Construir una superficie facial triangular

Un track de MediaPipe puede convertirse en una secuencia de meshes en memoria sin volver a
procesar el video. El ancho y el alto se proporcionan explícitamente porque son necesarios para
corregir la relación de aspecto de las coordenadas normalizadas:

```python
from talkingfacekit import build_mediapipe_face_mesh, load_landmark_track
from talkingfacekit.io.video import inspect_video_metadata

video_path = "tests/fixtures/example1.webm"
metadata = inspect_video_metadata(video_path)
landmarks = load_landmark_track("outputs/example1-landmarks.npz")

mesh = build_mediapipe_face_mesh(
    landmarks,
    image_width=metadata.width,
    image_height=metadata.height,
)

print(mesh.vertices.shape)  # (cantidad_de_frames, 468, 3)
print(mesh.triangles.shape)  # (852, 3)
```

Los vértices 0 a 467 forman la superficie triangular oficial de MediaPipe. Los diez landmarks de
iris quedan fuera porque MediaPipe los expone como contornos separados, no como parte de la piel
teselada. La conversión centra las coordenadas, orienta `y` hacia arriba y corrige `x` y `z` usando
`image_width / image_height`. El resultado conserva índices de frame, timestamps y detecciones,
pero sigue usando profundidad relativa de MediaPipe: es adecuado para visualización y animación,
no una reconstrucción métrica de la cabeza.

### Renderizar el mesh animado

El renderer interactivo es opcional. Se instala junto con MediaPipe sin convertir Plotly en una
dependencia del núcleo de la biblioteca:

```powershell
uv sync --extra tracking-mediapipe --extra rendering
```

El comando siguiente carga los landmarks ya extraídos, usa el video solamente para recuperar su
ancho y alto, construye el mesh y escribe un único HTML autocontenido que funciona offline:

```powershell
uv run python -m talkingfacekit render-mesh `
  outputs\example1-landmarks.npz `
  --video tests\fixtures\example1.webm `
  --output outputs\example1-mesh.html
```

El visor ofrece reproducción, pausa, selección temporal, zoom y cámara orbital. Usa un material
uniforme con iluminación virtual; no vuelve a decodificar los píxeles ni toma colores del video.
Por defecto conserva la profundidad relativa con `--depth-scale 1.0`. Un valor distinto modifica
sólo la presentación, no el `FaceMeshTrack`:

```powershell
uv run python -m talkingfacekit render-mesh `
  outputs\example1-landmarks.npz `
  --video tests\fixtures\example1.webm `
  --output outputs\example1-mesh-depth-2.html `
  --depth-scale 2.0
```

## Regla principal

El proyecto usa **Python 3.11** y **uv**. No usamos `pip`, Conda ni entornos creados manualmente para este repositorio.

`uv` crea una `.venv` local y se asegura de que todos usemos las mismas versiones de Python y de las librerías.

## Preparación inicial

Esto se hace una sola vez por computadora.

### 1. Instalar Git

Descargar Git desde [git-scm.com](https://git-scm.com/downloads) si todavía no está instalado.

### 2. Instalar uv

macOS y Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Después de instalarlo, cerrar y volver a abrir la terminal.

### 3. Clonar y preparar el proyecto

```bash
git clone https://github.com/MatiasCerdeira/TalkingFaceKit.git
cd TalkingFaceKit
uv sync
uv run pytest
uv run mypy
```

`uv sync` crea `.venv`, obtiene Python 3.11 si hace falta e instala las dependencias del proyecto.

## Cómo empezar a trabajar cada día

Antes de modificar archivos:

```bash
git switch main
git pull --ff-only
uv sync
```

Esto trae los últimos cambios del equipo y actualiza las librerías locales.

## Cómo ejecutar Python

No hace falta activar `.venv`. Usar `uv run` delante del comando:

```bash
uv run python mi_script.py
uv run python
uv run pytest
uv run ruff check .
```

Por ejemplo, en lugar de:

```bash
python main.py
```

usar:

```bash
uv run python main.py
```

## Cómo agregar o eliminar librerías

Agregar una librería necesaria para el proyecto:

```bash
uv add numpy
```

Agregar una herramienta usada solamente para desarrollar:

```bash
uv add --dev nombre-de-la-herramienta
```

Eliminar una librería:

```bash
uv remove numpy
```

`uv add` y `uv remove` actualizan `pyproject.toml` y `uv.lock`. Estos dos archivos deben subirse a Git.

No usar:

```bash
pip install nombre-de-la-libreria
conda install nombre-de-la-libreria
```

Antes de agregar dependencias grandes como PyTorch, modelos de tracking o librerías con CUDA, hablarlo con el equipo. Pueden necesitar configuraciones diferentes en macOS y Windows.

## Estilo, tipado y arquitectura

El código, los identificadores, comentarios, docstrings y mensajes de error se escriben en inglés.

El proyecto usa:

- **Ruff** para formato, imports y errores comunes.
- **mypy en modo estricto** para verificar type hints.
- **pytest** para verificar comportamiento.
- Docstrings estilo **NumPy** para la API pública.

Las reglas completas están en [`AGENTS.md`](AGENTS.md). Las decisiones de diseño están en
[`docs/architecture.md`](docs/architecture.md).

Codex lee `AGENTS.md` directamente. Claude Code lee [`CLAUDE.md`](CLAUDE.md), que importa las
mismas reglas. Cuando el equipo cambia una regla o decisión arquitectónica, debe actualizar esos
documentos en el mismo commit que el código afectado.

Reglas básicas de tipado:

- Tipar parámetros y retornos de todas las funciones de producción.
- Usar sintaxis de Python 3.11 como `list[str]` y `Value | None`.
- Evitar `Any` y no desactivar mypy globalmente por una librería problemática.
- Documentar y validar shape, dtype, unidades, rangos y ejes de los arrays NumPy.
- Usar `pathlib.Path` para rutas públicas del filesystem.

## Cómo subir cambios

Primero revisar y probar:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest
git status
```

Si todo funciona:

```bash
git add .
git commit -m "descripción corta del cambio"
git pull --rebase
git push
```

Ejemplos de mensajes de commit:

```text
Add video loading function
Fix audio duration validation
Add numpy dependency
Update setup instructions
```

## Cómo recibir cambios de otra persona

```bash
git pull --ff-only
uv sync
```

Siempre ejecutar `uv sync` después de un pull que haya modificado `pyproject.toml` o `uv.lock`.

## Reglas para no romper el proyecto

- Avisar al equipo qué archivos o funcionalidad está tocando cada uno.
- Hacer `git pull --ff-only` antes de empezar a trabajar.
- Hacer commits chicos y con mensajes claros.
- Ejecutar Ruff, mypy y pytest antes de cada push.
- No usar `git push --force` sobre `main`.
- No editar manualmente `uv.lock`.
- No subir `.venv`, datasets, videos generados, modelos, outputs ni archivos `.env`.
- No guardar contraseñas, tokens o credenciales en el repositorio.
- No instalar dependencias del proyecto con `pip` o Conda.
- Si aparece un conflicto de Git y no está claro cómo resolverlo, no adivinar: hablar con la persona que modificó ese archivo.

## Archivos importantes

```text
.python-version    Versión de Python usada por el proyecto
pyproject.toml     Configuración y lista de dependencias
uv.lock            Versiones exactas resueltas por uv
AGENTS.md           Reglas compartidas para humanos y agentes
CLAUDE.md           Importa AGENTS.md para Claude Code
PROJECT_DIRECTION.md Orientación de producto, opciones y decisiones de backend
docs/architecture.md Decisiones arquitectónicas vigentes
docs/blueprint/      Contratos objetivo y roadmap de implementación
src/               Código de TalkingFaceKit
tests/             Tests automáticos
.venv/             Entorno local; nunca se sube a Git
```

## Comprobación rápida

Si hay dudas sobre el entorno:

```bash
uv run python --version
uv run ruff check .
uv run mypy
uv run pytest
```

Python debe mostrar una versión `3.11.x` y los tests deben terminar correctamente.
