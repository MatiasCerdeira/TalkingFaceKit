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
para pose y calidad visual. SyncNet-v2 verifica sincronización A/V explícita como etapa opcional
posterior a los candidatos.

Orden de implementación acordado:

1. ~~agregar `DecodedAudioChunk` y audio PyAV por streaming con timestamps de fuente~~ — completo;
2. construir un adapter experimental y acotado para DeepTalk-ASD — implementado y validado con
   tests sintéticos y un smoke test end-to-end con modelos reales;
3. producir para un video speech intervals, face tracks, `raw_score`, segmentos candidatos,
   razones y provenance, además de un overlay inspeccionable;
4. evaluar seis casos mínimos y decidir si conservar DeepTalk, adaptar LR-ASD directamente o
   comparar TalkNet;
5. verificar sincronía con SyncNet — implementado como etapa experimental opcional — y agregar
   orientación/calidad con MediaPipe;
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

El adapter prepara una sola secuencia para DeepTalk y copia observaciones de rostros —incluidos
bounding box, pose cruda `yaw/pitch/roll` en grados y score crudo de calidad—, intervalos VAD y
ventanas diagnósticas con los scores finales devueltos por el backend, sin convertirlos en
probabilidades ni decisiones finales de speaking o suitability visual:

```python
from talkingfacekit import TalkingFaceSequence
from talkingfacekit.integrations.deeptalk import analyze_sequence

sequence = TalkingFaceSequence.from_video("conversation.mp4")
result = analyze_sequence(sequence)

print(result.face_observations)
print(result.speech_intervals)
print(result.score_windows)
print(result.shot_boundaries)
print(result.preliminary_segments)
print(result.provenance)
print(result.audio_timeline_repairs)
print(result.issues)
```

Para probar distintos videos durante una demo sin editar código, la CLI imprime un resumen de
rostros, intervalos VAD y scores crudos por ventana:

```bash
uv run --extra active-speaker-deeptalk python -m talkingfacekit analyze-video \
  /ruta/al/video.mp4
```

También se puede limitar el análisis a un fragmento sobre la timeline original:

```bash
uv run --extra active-speaker-deeptalk python -m talkingfacekit analyze-video \
  /ruta/al/video.mp4 \
  --start-seconds 5 \
  --end-seconds 15
```

Para inspeccionarlo visualmente durante una demo, `--report` genera una página HTML sincronizada que
referencia el video original sin copiarlo ni recodificarlo:

```bash
uv run --extra active-speaker-deeptalk python -m talkingfacekit analyze-video \
  /ruta/al/video.mp4 \
  --report /ruta/al/reporte.html
```

Al abrir el HTML en el navegador, los boxes, IDs, pose, calidad, scores crudos, estado de voz,
cortes de plano y cursor de la timeline siguen el tiempo del reproductor. El rostro con mayor score
durante un intervalo de voz se marca como `top candidate`, nunca como probabilidad o decisión
final. Los ángulos y la calidad también se muestran como evidencia cruda, todavía sin thresholds
de aceptación. Hacer click en el gráfico mueve el video a ese instante. El archivo depende de que
el video original continúe en la misma ruta. El panel **Decisiones por segmento** distingue
decisiones `INSTANTÁNEO` y `FINAL`, muestra `CONSERVAR`, `DESCARTAR` o `REVISAR`, resume las razones
y métricas, y permite saltar al comienzo de cada región con un click. La banda inferior de la
timeline conserva la clasificación preliminar, incluidos los scores ASD negativos; una banda
separada justo encima muestra el candidate run final. Si el destino ya existe, la CLI crea
automáticamente un hermano numerado como `reporte-2.html`; `--overwrite` permite reemplazar
deliberadamente el nombre solicitado.

El resultado también contiene una primera clasificación conservadora por subintervalo:

- `rejected` cuando no hay habla, no se observa una cara, aparecen varias caras, cambia la
  identidad dentro de la ventana o el único rostro tiene score no positivo;
- `uncertain` cuando hay voz y un único rostro, pero DeepTalk no devolvió su score;
- `candidate` únicamente cuando hay voz, se observa una sola identidad y su score crudo es
  positivo.

Cada segmento incluye razones machine-readable, los IDs observados y el score sin modificar. Un
`candidate` preliminar no es todavía un clip aceptado. TalkingFaceKit detecta cortes de escena con
`ContentDetector` de PySceneDetect usando sus valores por defecto. El detector recibe los mismos
frames ya decodificados y muestreados a 25 Hz para DeepTalk, sin abrir ni decodificar el video otra
vez, y compara cambios de color e intensidad en vez de limitarse a frames negros. El timestamp de
fuente queda en `shot_boundaries`; los intervalos preliminares se dividen allí y dos candidatos
nunca se unen a través del corte, aunque DeepTalk reutilice el mismo `face_id`.

Los candidatos adyacentes de la misma identidad dentro de un mismo plano se unen luego en
`candidate_runs` continuos. La política temporal también puede unir dos candidatos separados por un
hueco de hasta 1,25 segundos, pero solamente si el hueco contiene decisiones ASD no positivas o una
interrupción breve de VAD, se observa únicamente el mismo `face_id` con al menos 90% de cobertura,
no hay un hueco de detección mayor a 200 ms y no existe un corte de plano. Debe haber evidencia ASD
positiva inmediatamente antes y después: una región negativa al final no se agrega al run. Por eso
el límite funciona como una gracia de decisión offline, no como padding de silencio.

Los intervalos tolerados quedan explícitos en `bridged_gap_intervals`. La clasificación preliminar y
los scores crudos no se modifican; durante un puente el HTML puede mostrar simultáneamente el rechazo
ASD instantáneo y la conservación final por continuidad. El resumen mínimo/promedio/máximo del run
usa sólo los segmentos con score positivo, mientras los valores negativos siguen visibles en las
ventanas y segmentos preliminares.

Cada run mide duración, cobertura de detecciones a 25 Hz, mayor hueco sin rostro y score crudo
mínimo/promedio temporal/máximo de su evidencia positiva. La política estructural inicial exige al
menos 2 segundos, 90% de cobertura y ningún hueco visual mayor a 200 ms; un run que falla queda
preservado como `rejected` con razones explícitas. Estos valores son puntos de partida conservadores,
no thresholds calibrados.

Incluso un run que pasa esta etapa sigue sin ser un clip aceptado: faltan calibración del score,
pose, calidad visual y sincronización. La regla positiva sólo conserva candidatos para esas etapas;
no presenta `raw_score > 0` como garantía final. La CLI y el HTML muestran tanto el segmento
preliminar como el resultado estructural continuo para que la demo permita entender cada rechazo.

La primera ejecución adquiere los modelos faltantes mediante el model manager de DeepTalk. El
resumen es diagnóstico: muestra los valores LR-ASD sin convertirlos en probabilidades ni elegir un
hablante final. La CLI oculta los logs internos del backend para que esa salida sea fácil de leer.

El código de adaptación muestrea video a 25 Hz y convierte audio a mono PCM `int16` de 16 kHz dentro
del boundary experimental. Usa únicamente tiempo de medio relativo para alimentar el detector y
devuelve todos los timestamps sobre la timeline fuente. Desactiva las expiraciones por wall clock
del flujo offline, conserva sólo los frames visuales necesarios después de evaluar cada ventana y
reporta audio/video incompletos mediante `issues`. Los intervalos VAD conservan si DeepTalk los
confirmó, rechazó o si quedaron incompletos al llegar a EOF.

Antes del resampling, el adapter normaliza el audio sobre una grilla de samples anclada al intervalo
fuente. Los gaps de timestamps se rellenan con silencio y los samples solapados se recortan, siempre
por streaming. Cada corrección queda visible en `audio_timeline_repairs` con tipo, intervalo, sample
rate y cantidad de samples ajustados; `issues` incluye `audio_gap_filled` o
`audio_overlap_trimmed`. La cuantización menor a un milisegundo se trata como jitter de timestamps y
no genera una reparación.

TalkingFaceKit adquiere mediante el model manager público de DeepTalk solamente los cinco assets que
esta ruta usa: InspireFace, Silero VAD y los tres modelos LR-ASD. Los hashes del registry upstream se
verifican y los archivos quedan fuera del repositorio. El modelo de voiceprints no se adquiere y los
speaker embeddings se desactivan deliberadamente para que el fallo de linkage de `sherpa-onnx` en
macOS no cambie silenciosamente los scores. `result.provenance` registra esa capacidad como no
disponible.

Además de los tests sintéticos, esta ruta se validó en CPU con DeepTalk-ASD 0.3.1 y sus modelos
oficiales sobre un MP4 H.264/AAC de 10,6 segundos con una sola persona frontal. El smoke test obtuvo
una identidad estable y scores finitos; no evalúa todavía precisión, calibración ni escenas con
varias personas.

El test con modelos y video reales es explícitamente opt-in y se omite en la suite normal:

```bash
TALKINGFACEKIT_DEEPTALK_TEST_VIDEO=/ruta/al/video.mp4 \
  uv run --extra active-speaker-deeptalk pytest tests/test_deeptalk_e2e.py
```

### Verificar sincronización de candidatos con SyncNet

La etapa opcional recibe los `candidate_runs` de DeepTalk. Reutiliza sus IDs y boxes: no vuelve a
detectar caras ni escenas. Lee ventanas acotadas del original con PyAV, prepara recortes BGR de
224×224 a 25 Hz y audio mono PCM de 16 kHz, y ejecuta el núcleo SyncNet-v2 en CPU. PyTorch se
instala únicamente con el extra; no se incorpora el pipeline completo `syncnet-python` ni su
dependencia incompatible de PySceneDetect.
Dentro de cada video, la verificación avanza una sola vez por cada stream y conserva únicamente
las ventanas y frames de borde necesarios, sin volver a decodificar el prefijo para cada candidato.

```bash
uv sync --extra active-speaker-deeptalk --extra synchronization-syncnet
mkdir -p models reports/syncnet
curl -fL https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model \
  -o models/syncnet_v2.model

uv run --extra active-speaker-deeptalk --extra synchronization-syncnet \
  python -m talkingfacekit analyze-video data/eval/videos/Entrevista_01.webm \
  --start-seconds 0 --end-seconds 38 --syncnet-model models/syncnet_v2.model \
  --report reports/syncnet/entrevista.html --output reports/syncnet/entrevista.json
```

El modelo oficial se valida por SHA-256 antes de cargar tensores con `weights_only=True`. No se
descarga automáticamente ni se guarda en Git. La atribución del código adaptado se conserva en
`THIRD_PARTY_NOTICES.md` y en la licencia incluida con el paquete.

Un offset **positivo** significa que el audio llega **tarde** respecto de la boca. Las mediciones
se expresan en segundos y tienen pasos de 40 ms; la confianza es una diferencia cruda de distancias,
no una probabilidad. Los defaults experimentales son tolerancia de ±80 ms y confianza mínima de
3.0, configurables con `--sync-tolerance-ms` y `--sync-minimum-confidence`. El rango buscado es de
±1 segundo (`--sync-maximum-offset-seconds`), reducido en ventanas cortas para conservar al menos
un segundo de evidencia común a todos los desplazamientos. Un óptimo en el borde queda incierto.

Las ventanas particionan completamente cada candidato, con un máximo de 5 segundos y mínimo de
2 segundos para intentar medir. Nunca se unen a través de un corte ni otra identidad. Un desfase
confiable en cualquier ventana rechaza todo ese candidato. Sólo pasa sincronía cuando todas sus
ventanas son verificables y están dentro de la tolerancia; los casos cortos, discontinuos, débiles
o incompletos quedan inciertos y fuera de la selección automática. Pasar sincronía aún requiere
los filtros pendientes de orientación/calidad; no es una aceptación final para entrenamiento.

La salida conserva las decisiones estructurales y añade mediciones y decisiones de sincronía,
hash del modelo, política y tiempos. El HTML muestra la evidencia por ventana; no corrige ni
reescribe el video. Cuando los offsets confiables cambian entre ventanas, no publica un offset
global como si bastara desplazar toda la pista. No hay corrección automática en esta versión.

Para comparar contra las anotaciones humanas fijas:

```bash
uv run --extra active-speaker-deeptalk --extra synchronization-syncnet \
  python -m talkingfacekit evaluate-benchmark benchmark/manifest.json \
  --videos-dir data/eval/videos --syncnet-model models/syncnet_v2.model \
  --output reports/syncnet/benchmark-run01.json
uv run python -m talkingfacekit render-benchmark reports/syncnet/benchmark-run01.json \
  --videos-dir data/eval/videos --output reports/syncnet/comparison.html
```

Con el flag, las predicciones incluyen únicamente candidatos que pasaron sincronía. Sin el flag se
conserva la evaluación estructural anterior. `--verify-only` y `render-benchmark` no cargan modelos.
Los resultados antiguos siguen siendo legibles. La validación real, sus límites y los tiempos
observados se documentan en [el spike de SyncNet](docs/research/syncnet_verification.md).

La prueba diferencial con modelos reales es opt-in; requiere modelos de DeepTalk ya cacheados y un
video de habla clara. Comprueba la recuperación de adelantos y atrasos PCM conocidos respecto del
offset original del clip, sin afirmar que ese offset original sea exactamente cero:

```bash
TALKINGFACEKIT_SYNCNET_TEST_MODEL=models/syncnet_v2.model \
TALKINGFACEKIT_SYNCNET_TEST_VIDEO=data/eval/videos/Piers_Sellers_JSC_2016_interview.mp4 \
TALKINGFACEKIT_SYNCNET_TEST_START_SECONDS=20 \
uv run --extra active-speaker-deeptalk --extra synchronization-syncnet \
  pytest tests/test_syncnet_e2e.py
```

Los trackers consumen este stream compartido en lugar de abrir el video por su cuenta. Así, la
selección del stream, la conversión a RGB y las reglas de timestamps permanecen iguales para
MediaPipe y futuros backends como FLAME.

También está disponible un primer flujo vertical opcional de tracking facial con MediaPipe Face
Landmarker. MediaPipe se instala por separado porque no es necesario para inspeccionar metadata:

```bash
uv sync --extra tracking-mediapipe
```

Ambos backends pueden instalarse para usarlos en etapas diferentes del mismo proceso:

```bash
uv sync --extra active-speaker-deeptalk --extra tracking-mediapipe
```

DeepTalk declara `opencv-python` y MediaPipe declara `opencv-contrib-python`; el lock los fija en la
misma versión de OpenCV y la suite combinada verifica que `cv2`, DeepTalk y MediaPipe se importen en
el mismo entorno. Esta convivencia sigue siendo una restricción del packaging upstream y debe
revalidarse al actualizar cualquiera de los dos backends.

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

### Benchmark con selecciones humanas

El equipo completa un JSON simple con cuatro campos por caso: `file`, `start_seconds`,
`end_seconds` y `selected`. La [guía de recolección](benchmark/README.md) explica cómo elegir y anotar
los videos e incluye una [plantilla](benchmark/annotations.example.json). Las anotaciones se guardan
en Git, un JSON por video dentro de [`benchmark/annotations/`](benchmark/annotations/README.md).
El [reparto](benchmark/assignments.md) coordina quién anota cada video. Los videos siguen en Drive;
cada persona los descarga a `data/eval/videos/` conservando sus nombres.

`benchmark/manifest.json` también está versionado. Después de revisar cambios en las anotaciones,
una persona lo actualiza con identificadores y hashes automáticos y sube ambos cambios juntos:

```bash
uv run python -m talkingfacekit prepare-benchmark benchmark/annotations \
  --videos-dir data/eval/videos --output benchmark/manifest.json --overwrite
```

Para usar la versión compartida, actualizar el repositorio y comprobar que las anotaciones, el
manifiesto y los videos coincidan. `--check` no escribe archivos y falla si el manifiesto quedó
desactualizado. La verificación no carga modelos:

```bash
uv run python -m talkingfacekit prepare-benchmark benchmark/annotations \
  --videos-dir data/eval/videos --output benchmark/manifest.json --check

uv run python -m talkingfacekit evaluate-benchmark benchmark/manifest.json \
  --videos-dir data/eval/videos --verify-only
```

La corrida completa requiere el extra DeepTalk y puede guardar métricas y predicciones en JSON:

```bash
uv run --extra active-speaker-deeptalk python -m talkingfacekit evaluate-benchmark \
  benchmark/manifest.json --videos-dir data/eval/videos \
  --output reports/benchmark-run.json
```

La corrida completa compara los `candidate_runs` con la selección humana del manifiesto y verifica
los videos antes de ejecutar modelos. Su JSON registra el hash del manifiesto y la versión de código.
Los JSON únicos del flujo anterior siguen admitidos por `prepare-benchmark`.

Para inspeccionar dónde difieren las selecciones humanas y la librería, generar un comparador a
partir del JSON de una corrida. No vuelve a ejecutar modelos ni modifica las anotaciones:

```bash
uv run python -m talkingfacekit render-benchmark reports/benchmark-run.json \
  --videos-dir data/eval/videos --output reports/comparison.html
```

El HTML permite elegir un caso, reproducir su video original, comparar las bandas **Humano** y
**Librería**, y navegar las diferencias **Conservado de más** y **Útil perdido**. Seleccionar una
diferencia amplía su entorno; **Reproducir diferencia con contexto** reproduce desde 1,5 segundos
antes hasta 1,5 segundos después, limitado a la ventana evaluada. El tiempo `ignore` queda excluido.
Los límites de cada intervalo siguen visibles, incluso cuando son contiguos.
Cuando una selección está vacía, su banda muestra **Sin intervalos conservados**. La comparación
**Ambos descartan** indica acuerdo en el descarte; **Sin evaluar** identifica tiempo excluido.

Las nuevas corridas guardan automáticamente decisiones preliminares, segmentos candidatos,
motivos y cortes para inspeccionar qué regla se aplicó en ese instante. Los JSON anteriores también
funcionan, pero el comparador indica cuando no tienen ese detalle. No infiere motivos ausentes.
La tabla muestra métricas por caso y globales, recalculadas con los intervalos guardados.

El comando verifica los hashes de los videos y el HTML referencia sus rutas locales sin copiarlos.
Abrirlo como archivo local en un navegador con soporte para el formato de video y mantener las
rutas de los videos. Su carpeta de destino debe existir; `--overwrite` permite reemplazar el HTML.
El comparador no requiere el extra de DeepTalk. Ver la [guía de demo](docs/demo_friday.md).

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
