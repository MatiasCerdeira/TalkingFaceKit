# Referencia de API propuesta

Esta no es una referencia importable completa. Las firmas disponibles reflejan el código actual;
las firmas objetivo fijan intención y vocabulario para futuras entregas.

## Filosofía de la API pública

- El paquete raíz expone tipos y operaciones comunes e independientes de backend.
- Los adapters concretos se importan desde su integración, por ejemplo
  `talkingfacekit.tracking.mediapipe`.
- Una operación costosa es explícita; acceder a una propiedad no ejecuta inferencia o decode.
- Las rutas públicas aceptan `str | Path` y se normalizan a `Path` en el boundary.
- No hay overwrite, descarga, upload, resample, conversión de color o selección de dispositivo
  implícitos.
- Los resultados se validan antes de adjuntarse o persistirse.
- Una API backend-agnostic no incluye tensores de PyTorch, frames de PyAV u objetos MediaPipe.

## Exportaciones del paquete raíz — disponible

La versión actual exporta:

```python
from talkingfacekit import (
    DecodedVideoFrame,
    FaceLandmarkTrack,
    FaceMeshTrack,
    LandmarkTracker,
    TalkingFaceSequence,
    VideoMetadata,
    build_mediapipe_face_mesh,
    load_landmark_track,
    render_face_mesh_html,
    save_landmark_track,
    stream_video_frames,
)
```

No toda capacidad futura debe agregarse al root. Las familias especializadas deberían conservar
namespaces claros.

## `TalkingFaceSequence` — disponible

```python
TalkingFaceSequence(
    path: Path,
    metadata: VideoMetadata,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
)
```

### `from_video`

```python
@classmethod
def from_video(cls, video_path: str | Path) -> TalkingFaceSequence: ...
```

Inspecciona metadata del primer stream de video. No decodifica frames o audio.

### `landmark_tracks`

```python
@property
def landmark_tracks(self) -> Mapping[str, FaceLandmarkTrack]: ...
```

Devuelve una vista read-only viva de resultados nombrados.

### `track_landmarks`

```python
def track_landmarks(
    self,
    tracker: LandmarkTracker,
    *,
    name: str,
    overwrite: bool = False,
) -> FaceLandmarkTrack: ...
```

Ejecuta el backend sobre el intervalo de la secuencia y adjunta el resultado transaccionalmente.

## `TalkingFaceSequence` — extensiones objetivo

Las siguientes firmas muestran el estilo deseado, no una obligación de implementarlas todas como
métodos. Una función libre es preferible cuando no necesita coordinar estado del agregado.

```python
def clip(
    self,
    start_seconds: float,
    end_seconds: float | None,
    *,
    time_origin: Literal["source", "zero"] = "source",
) -> TalkingFaceSequence: ...


def attach_landmark_track(
    self,
    track: FaceLandmarkTrack,
    *,
    name: str,
    overwrite: bool = False,
) -> None: ...


def track_faces(
    self,
    tracker: FaceTracker,
    *,
    name: str,
    overwrite: bool = False,
) -> FaceTrackSet: ...


def decode_audio(
    self,
    *,
    stream_index: int | None = None,
    max_bytes: int | None = None,
) -> AudioTrack: ...


def transcribe(
    self,
    recognizer: SpeechRecognizer,
    *,
    name: str,
    overwrite: bool = False,
) -> TranscriptTrack: ...
```

Agregar un método requiere que la secuencia realmente coordine fuente, intervalo y ownership. Los
algoritmos puros de transformación deberían seguir siendo funciones.

## Video — disponible

### `inspect_video_metadata`

```python
def inspect_video_metadata(video_path: str | Path) -> VideoMetadata: ...
```

Vive en `talkingfacekit.io.video` y se reexporta desde `talkingfacekit.io`.

### `stream_video_frames`

```python
def stream_video_frames(
    video_path: str | Path,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> Iterator[DecodedVideoFrame]: ...
```

Valida ruta e intervalo al llamarse; abre y decodifica al consumir el iterador. Selecciona el
primer stream, entrega RGB y rechaza timestamps ausentes o no crecientes.

## Video — objetivos

```python
def inspect_media(path: str | Path) -> MediaMetadata: ...


def stream_video_frames(
    path: str | Path,
    *,
    video_stream: int | None = None,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    output_size: tuple[int, int] | None = None,
    rotation: Literal["preserve", "apply"] = "preserve",
) -> Iterator[DecodedVideoFrame]: ...
```

Modificar la firma existente requiere preservar sus defaults y semántica. Selección múltiple de
streams y rotación llegan con metadata ampliada y tests de compatibilidad.

## Audio — objetivo

Namespace: `talkingfacekit.audio` para contratos y transformaciones puras;
`talkingfacekit.io.audio` para decode/encode.

```python
def stream_audio_chunks(
    media_path: str | Path,
    *,
    audio_stream: int | None = None,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> Iterator[DecodedAudioChunk]: ...


def decode_audio(
    media_path: str | Path,
    *,
    audio_stream: int | None = None,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    max_bytes: int | None = None,
) -> AudioTrack: ...


def resample_audio(audio: AudioTrack, *, sample_rate_hz: int) -> AudioTrack: ...


def remix_audio(audio: AudioTrack, *, channel_layout: str) -> AudioTrack: ...


def compute_audio_features(
    audio: AudioTrack,
    config: AudioFeatureConfig,
) -> AudioFeatureTrack: ...
```

Las primeras entregas deberían implementar streaming, materialización acotada y round-trip antes
de VAD o ASR.

## Landmark tracking — disponible

```python
class LandmarkTracker(Protocol):
    def track(
        self,
        video_path: Path,
        *,
        start_seconds: float,
        end_seconds: float | None,
    ) -> FaceLandmarkTrack: ...
```

Adapter disponible:

```python
from talkingfacekit.tracking.mediapipe import MediaPipeFaceTracker

MediaPipeFaceTracker(
    model_path: Path,
    min_face_detection_confidence: float = 0.5,
    min_face_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
)
```

El protocolo permanece pequeño. Sólo debería cambiar si un segundo backend real demuestra una
necesidad compartida.

## Tracking multi-rostro — objetivo

```python
class FaceTracker(Protocol):
    def track_faces(
        self,
        frames: Iterable[DecodedVideoFrame],
    ) -> FaceTrackSet: ...


@dataclass(frozen=True, slots=True)
class FaceTrackSet:
    tracks: Mapping[str, FaceTrack]
    association_method: str
    events: tuple[FaceTrackingEvent, ...]
```

La forma final depende de una primera implementación multi-rostro y no debe publicarse sólo desde
este boceto.

## Procesamiento de tracks — objetivo

Namespace candidato: `talkingfacekit.processing`.

```python
def smooth_landmarks(
    track: FaceLandmarkTrack,
    *,
    method: Literal["one-euro", "savgol", "ema"],
    **method_parameters: object,
) -> FaceLandmarkTrack: ...


def interpolate_landmarks(
    track: FaceLandmarkTrack,
    *,
    method: Literal["linear", "cubic"],
    max_gap_seconds: float,
) -> InterpolatedLandmarkTrack: ...


def evaluate_landmarks(track: FaceLandmarkTrack) -> LandmarkQualityReport: ...
```

La firma con `**method_parameters: object` es sólo conceptual y no cumple el estándar final de
tipado. Cada método deberá recibir un config dataclass tipado o una función dedicada.

## Mesh — disponible

```python
def build_mediapipe_face_mesh(
    track: FaceLandmarkTrack,
    *,
    image_width: int,
    image_height: int,
) -> FaceMeshTrack: ...
```

Requiere topología MediaPipe 478 e instalación del extra correspondiente.

## Mesh y modelos paramétricos — objetivos

Namespace de operaciones core: `talkingfacekit.geometry`. Adapter investigado:
`talkingfacekit.models.flame`.

```python
def compute_vertex_normals(mesh: FaceMeshTrack) -> VertexNormalTrack: ...


def transform_mesh(
    mesh: FaceMeshTrack,
    transform: CoordinateTransformTrack,
) -> FaceMeshTrack: ...


class FlameFitter:
    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cpu",
        dtype: Literal["float32", "float64"] = "float32",
        config: FlameFitConfig | None = None,
    ) -> None: ...

    def fit(
        self,
        landmarks: FaceLandmarkTrack,
        *,
        camera: CameraModel | None = None,
    ) -> FlameParameterTrack: ...
```

No se define una clase genérica `FaceModel` hasta contar con al menos dos modelos o consumidores
intercambiables.

## Habla — objetivo

Namespace: `talkingfacekit.speech` para contratos; submódulos por backend.

```python
class VoiceActivityDetector(Protocol):
    def detect(self, audio: AudioTrack) -> VoiceActivityTrack: ...


class SpeechRecognizer(Protocol):
    def transcribe(
        self,
        audio: AudioTrack,
        *,
        language: str | None,
    ) -> TranscriptTrack: ...


class PhonemeAligner(Protocol):
    def align(
        self,
        audio: AudioTrack,
        transcript: str | TranscriptTrack,
        *,
        language: str,
    ) -> PhonemeTrack: ...


def phonemes_to_visemes(
    phonemes: PhonemeTrack,
    *,
    mapping: VisemeMapping,
) -> VisemeTrack: ...
```

Un protocolo aparece cuando existe reemplazo real. La primera integración puede ser una clase
concreta aislada.

## Sincronización — objetivo

Namespace: `talkingfacekit.sync`.

```python
def estimate_av_offset(
    visual_signal: TemporalSignal,
    audio_signal: TemporalSignal,
    *,
    search_range_seconds: tuple[float, float],
) -> SynchronizationEstimate: ...


def shift_timeline(track: TrackT, *, offset_seconds: float) -> TrackT: ...


def resample_track(
    track: TrackT,
    target_timestamps_seconds: NDArray[np.float64],
    *,
    policy: ResamplingPolicy,
) -> TrackT: ...
```

`TrackT` y `TemporalSignal` son placeholders de diseño; no se deben implementar con `Any`. Primero
se validará el caso concreto de sincronía boca/audio.

## Animación y retargeting — objetivo

Namespace: `talkingfacekit.animation`.

```python
def phonemes_to_viseme_curves(
    phonemes: PhonemeTrack,
    *,
    mapping: VisemeMapping,
    transition_seconds: float,
) -> AnimationCurveTrack: ...


def retarget_blendshapes(
    source: BlendshapeTrack,
    mapping: BlendshapeMapping,
) -> AnimationCurveTrack: ...
```

Los mappings son datos versionados y validables, no diccionarios ad hoc escondidos en un backend.

## Persistencia — disponible

```python
def save_landmark_track(
    track: FaceLandmarkTrack,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path: ...


def load_landmark_track(input_path: str | Path) -> FaceLandmarkTrack: ...
```

El schema NPZ actual es independiente de la versión Python y vale `1`.

## Persistencia — objetivos

```python
def save_project(
    sequence: TalkingFaceSequence,
    output_path: str | Path,
    *,
    include_source_media: bool = False,
    overwrite: bool = False,
) -> Path: ...


def load_project(
    input_path: str | Path,
    *,
    lazy: bool = True,
) -> TalkingFaceSequence: ...


def validate_artifact(path: str | Path) -> ValidationReport: ...


def migrate_artifact(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_version: int,
    overwrite: bool = False,
) -> Path: ...
```

La extensión `.tfk`, su formato físico y semántica de lazy loading deben probarse antes de ser API.

## Rendering — disponible

```python
def render_face_mesh_html(
    mesh: FaceMeshTrack,
    output_path: str | Path,
    *,
    title: str = "TalkingFaceKit animated face mesh",
    depth_scale: float = 1.0,
    overwrite: bool = False,
) -> Path: ...
```

Escribe HTML offline mediante Plotly y requiere al menos un frame detectado.

## Rendering y export — objetivos

```python
def render_overlay_video(
    sequence: TalkingFaceSequence,
    output_path: str | Path,
    *,
    layers: Sequence[OverlayLayer],
    video_config: VideoEncodeConfig,
    audio_policy: Literal["copy", "reencode", "omit"] = "copy",
    overwrite: bool = False,
) -> Path: ...


def export_mesh_gltf(
    mesh: FaceMeshTrack,
    output_path: str | Path,
    *,
    animation: bool = True,
    overwrite: bool = False,
) -> Path: ...


def render_quality_report(
    report: QualityReport,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path: ...
```

## Pipeline y datasets — objetivos tardíos

```python
pipeline = Pipeline(steps=[...])
result = pipeline.run(sequence, cache_dir=Path(".tfk-cache"))

dataset = TalkingFaceDataset.from_manifest(Path("dataset.jsonl"))
batch_result = pipeline.run_batch(dataset, workers=4, on_error="continue")
```

Estas APIs llegan después de operaciones concretas y persistencia multipista. El pipeline no debe
ser un plugin system genérico ni aceptar objetos sin tipar. Cada step declara inputs, outputs,
configuración normalizada y side effects.

## Excepciones — dirección propuesta

Por ahora se usan excepciones estándar con mensajes específicos. Clases de dominio sólo se agregan
cuando el caller necesita distinguir categorías:

```python
class TalkingFaceKitError(Exception): ...


class MediaDecodeError(TalkingFaceKitError): ...


class ContractValidationError(TalkingFaceKitError): ...


class BackendUnavailableError(TalkingFaceKitError): ...


class ModelAssetError(TalkingFaceKitError): ...


class ArtifactSchemaError(TalkingFaceKitError): ...
```

No se debe envolver indiscriminadamente toda excepción. `FileNotFoundError`, `FileExistsError` e
`IsADirectoryError` siguen siendo útiles y esperables en boundaries de archivos.

## Estabilidad y deprecaciones — objetivo

- Versionado semántico una vez publicada una API estable.
- Dos releases menores de aviso antes de retirar una API, cuando sea posible.
- Warning con alternativa y versión prevista de retiro.
- Migradores explícitos para schemas persistidos.
- Cambios de dtype, shape, unidades, topología o timeline se consideran incompatibles aunque la
  firma Python no cambie.
