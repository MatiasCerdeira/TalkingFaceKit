# Contratos de datos propuestos

Este documento reúne contratos actuales y candidatos. Un contrato objetivo no se considera
aceptado hasta incorporarse a [`docs/architecture.md`](../architecture.md) junto con código y tests.

## Reglas transversales

- NumPy es la representación numérica independiente de frameworks.
- Los parámetros y retornos públicos tienen type hints; los arrays usan `NDArray`.
- Shapes, dtypes, ejes, unidades, rangos y orden de canales son parte de la API.
- Los tiempos públicos se expresan en segundos `float64`, salvo APIs de audio que además expongan
  índices de muestra enteros.
- Los timelines por frame son estrictamente crecientes.
- Los arrays se retienen sin copia cuando se documenta y se tratan como inmutables.
- `NaN` representa valores espaciales/continuos faltantes sólo cuando una máscara lo confirma.
- Los datos categóricos faltantes usan una máscara o ausencia estructural; no strings mágicos.
- Las conversiones producen valores nuevos y provenance; no reinterpretan arrays en sitio.
- Unidades métricas sólo se declaran cuando hay calibración suficiente.

## Contratos disponibles

### `VideoMetadata`

| Campo | Tipo | Semántica |
| --- | --- | --- |
| `width` | `int` | ancho codificado del primer stream, pixels |
| `height` | `int` | alto codificado del primer stream, pixels |
| `average_fps` | `float | None` | FPS promedio informado por la fuente |
| `stream_duration_seconds` | `float | None` | duración informada por stream o contenedor |
| `has_audio` | `bool` | existe al menos un stream de audio |

Pendiente actual: el dataclass todavía no valida positividad o finitud. Una futura ampliación debe
definir compatibilidad antes de agregar campos.

### `DecodedVideoFrame`

| Campo | Tipo/shape | Invariantes |
| --- | --- | --- |
| `frame_index` | `int` | índice de decode de fuente, base cero, no negativo |
| `timestamp_seconds` | `float` | PTS finito en segundos |
| `rgb` | `uint8[height, width, 3]` | RGB, dimensiones positivas, rango `[0, 255]` |

La integración exige timestamps estrictamente crecientes entre frames entregados. El buffer no se
copia y debe tratarse como read-only.

### `FaceLandmarkTrack`

| Campo | Tipo/shape | Invariantes |
| --- | --- | --- |
| `tracker_name` | `str` | no vacío |
| `tracker_version` | `str | None` | versión declarada por backend |
| `topology` | `str` | orden y cantidad de puntos |
| `coordinate_system` | `str` | significado y unidades de `(x, y, z)` |
| `frame_indices` | `int64[F]` | no negativos, estrictamente crecientes |
| `timestamps_seconds` | `float64[F]` | finitos, estrictamente crecientes |
| `landmarks` | `float32[F, L, 3]` | finitos si detectado; todos `NaN` si falta |
| `detected` | `bool[F]` | una entrada por frame |

`F >= 1` y `L >= 1`. MediaPipe usa `L=478` y topología
`mediapipe-face-landmarker-478`.

### `FaceMeshTrack`

| Campo | Tipo/shape | Invariantes |
| --- | --- | --- |
| `topology` | `str` | orden de vértices y conectividad |
| `coordinate_system` | `str` | origen, ejes, escala y profundidad |
| `frame_indices` | `int64[F]` | no negativos, estrictamente crecientes |
| `timestamps_seconds` | `float64[F]` | finitos, estrictamente crecientes |
| `vertices` | `float32[F, V, 3]` | finitos si detectado; todos `NaN` si falta |
| `triangles` | `int32[T, 3]` | índices válidos y no degenerados |
| `detected` | `bool[F]` | una entrada por frame |

`F >= 1`, `V >= 1`, `T >= 1`. La topología de MediaPipe actual usa `V=468`, `T=852`.

## Timeline común — objetivo

Antes de generalizar tracks se propone validar un valor pequeño:

```python
@dataclass(frozen=True, slots=True)
class FrameTimeline:
    frame_indices: NDArray[np.int64]
    timestamps_seconds: NDArray[np.float64]
```

No debería introducirse sólo para reducir repetición. Tiene que resolver operaciones reales de
recorte, alineación o persistencia en al menos dos familias de tracks. El timeline no incluye FPS
porque un stream puede tener frame rate variable.

## `DecodedAudioChunk` — objetivo

Contrato inicial propuesto, conservando la fuente:

| Campo | Tipo/shape | Invariantes |
| --- | --- | --- |
| `start_sample_index` | `int` | índice base cero en el stream decodificado |
| `start_timestamp_seconds` | `float` | timestamp finito del primer sample |
| `sample_rate_hz` | `int` | positivo, constante dentro del stream |
| `channel_layout` | `str` | nombre explícito, por ejemplo `mono` o `stereo` |
| `samples` | `float32[S, C]` | sample-major, amplitud nominal `[-1, 1]` |

Decisiones a validar en la primera entrega de audio:

- si PyAV puede garantizar el timestamp del primer sample después de conversiones;
- cómo representar muestras válidas fuera de `[-1, 1]` y clipping;
- nombres canónicos de canales;
- si el primer decode debe producir formato fuente entero o el canónico `float32`;
- cómo representar discontinuidades y padding del decoder.

No habrá downmix ni resample implícito.

## `AudioTrack` — objetivo

Para materialización acotada:

| Campo | Tipo/shape | Invariantes |
| --- | --- | --- |
| `samples` | `float32[S, C]` | sample-major; contrato de amplitud declarado |
| `sample_rate_hz` | `int` | positivo |
| `channel_layout` | `str` | consistente con `C` |
| `start_timestamp_seconds` | `float` | origen temporal finito |
| `valid` | `bool[S] | None` | opcional para discontinuidades/padding |

El tamaño máximo o la decisión de materializar pertenece al caller.

## `IntervalTrack[T]` — candidato

VAD, palabras, fonemas, visemas categóricos y eventos de blink comparten intervalos, pero una clase
genérica sólo será útil si preserva el tipado sin `Any`.

Invariantes de cualquier track de intervalos:

- `start_seconds` y `end_seconds` finitos;
- `end_seconds > start_seconds`;
- orden determinista;
- superposición permitida o prohibida de manera explícita por schema;
- labels definidas por un vocabulario versionado;
- confidence opcional con semántica documentada.

## `FaceDetectionTrack` — objetivo

Representación candidata por frame:

| Campo | Tipo/shape | Semántica |
| --- | --- | --- |
| timeline | `int64[F]`, `float64[F]` | índices y timestamps de fuente |
| `boxes_xywh` | `float32[F, N, 4]` | pixels, `(x_min, y_min, width, height)` |
| `scores` | `float32[F, N]` | rango y calibración declarados por backend |
| `valid` | `bool[F, N]` | slots reales vs padding |

Como `N` varía, se debe comparar este tensor padded con una estructura por frame antes de publicar
el contrato. La facilidad de vectorización no justifica ambigüedad.

## `FaceTrackSet` — objetivo

Colección ordenada de `face_id -> track`, con:

- IDs estables sólo dentro de la secuencia;
- source intervals de presencia;
- referencia al método de asociación;
- eventos de identidad ambigua o merge/split;
- sin reconocimiento biométrico implícito.

Se prefiere una colección de tracks por identidad a agregar un eje variable de persona en
`FaceLandmarkTrack`.

## `HeadPoseTrack` — objetivo

Propuesta mínima:

| Campo | Tipo/shape | Semántica |
| --- | --- | --- |
| timeline | `int64[F]`, `float64[F]` | fuente |
| `rotation_matrices` | `float32[F, 3, 3]` | convención source-to-camera declarada |
| `translations` | `float32[F, 3] | None` | unidades declaradas; opcional |
| `valid` | `bool[F]` | pose usable |
| `reprojection_error` | `float32[F] | None` | pixels u otra unidad explícita |

Las matrices válidas deben ser finitas y aproximadamente ortonormales. Una función separada puede
derivar quaternions o Euler angles.

## `GazeTrack` — objetivo

| Campo | Tipo/shape | Semántica |
| --- | --- | --- |
| `left_direction` | `float32[F, 3]` | vector unitario en coordenadas declaradas |
| `right_direction` | `float32[F, 3]` | vector unitario en coordenadas declaradas |
| `left_valid`, `right_valid` | `bool[F]` | validez independiente por ojo |
| confidence | `float32[F, 2] | None` | semántica del backend |

Los puntos de mirada en pantalla requieren calibración y un contrato diferente.

## `BlendshapeTrack` — objetivo

| Campo | Tipo/shape | Invariantes |
| --- | --- | --- |
| `channel_names` | `tuple[str, ...]` | únicos, ordenados y schema-versionados |
| timeline | `int64[F]`, `float64[F]` | fuente o timeline generado |
| `weights` | `float32[F, B]` | rango declarado, no necesariamente `[0, 1]` |
| `valid` | `bool[F]` o `bool[F, B]` | elección explícita según backend |
| `schema` | `str` | por ejemplo un schema ARKit versionado |

No se deben mapear canales desconocidos a cero sin reportarlos.

## `FlameParameterTrack` — investigación

Propuesta conceptual:

| Campo | Frecuencia | Shape orientativo |
| --- | --- | --- |
| `shape` | por identidad | `float32[S]` |
| `expression` | por frame | `float32[F, E]` |
| `global_pose` | por frame | `float32[F, 3]` axis-angle |
| `neck_pose` | por frame | `float32[F, 3]` |
| `jaw_pose` | por frame | `float32[F, 3]` |
| `eye_pose` | por frame | shape definido por versión de modelo |
| `translation` | por frame | `float32[F, 3]` |
| `converged` | por frame | `bool[F]` |

La dimensión y convención dependen de la versión/licencia del modelo. El schema debe incluir esa
identidad y no asumir que todos los modelos llamados FLAME son intercambiables.

## `SegmentationMaskTrack` — objetivo

| Campo | Tipo/shape | Semántica |
| --- | --- | --- |
| timeline | por frame | alineación a fuente |
| `masks` | `bool[F,H,W]`, `uint8` o `float32` | tipo determina clase/probabilidad |
| `class_names` | tuple opcional | schema de clases para máscaras categóricas |
| transform | por frame o fija | mapping a coordenadas del video |
| `valid` | `bool[F]` | frame usable |

El storage debe admitir chunks. Una máscara de probabilidad requiere rango `[0, 1]`; una de clases
requiere IDs y palette documentados.

## Tracks de habla — objetivo

### `TranscriptTrack`

Segmentos y palabras con texto, tiempos, idioma y confidence opcional. Se conserva texto bruto y
texto normalizado por separado.

### `PhonemeTrack`

Intervalos con símbolos de un inventario declarado (`ipa`, ARPAbet u otro), idioma, pronunciación y
confidence opcional. Silencio y fonema desconocido tienen etiquetas explícitas.

### `VisemeTrack`

Puede ser:

- intervalos categóricos con mapping versionado; o
- `float32[K, V]` sobre timestamps propios para mezclas de visemas.

Son tipos distintos aunque compartan el nombre de dominio.

### `ProsodyTrack`

Timestamps de ventana más F0 en Hz, probabilidad de voz, energía/loudness con unidad y otros
features. F0 no son semitonos y los frames no-voiced necesitan máscara explícita.

## Provenance — objetivo

Un `ArtifactProvenance` candidato debería ser serializable sin objetos de frameworks:

```text
artifact_id
schema_name + schema_version
talkingfacekit_version
operation_name + normalized_parameters
backend_name + backend_version
model_name + model_version + model_sha256 + declared_license
device + numerical_precision
source_fingerprint + source_interval
parent_artifact_ids
created_at_utc
random_seed
```

Los campos no aplicables pueden ser `None`, pero la ausencia no debe convertirse en un valor
inventado. Rutas locales absolutas y credenciales no deberían persistirse por defecto.

## Compatibilidad y schemas

- Cada formato persistido tiene versión independiente del paquete.
- Un loader rechaza versiones futuras desconocidas con un error claro.
- Una migración es explícita, testeada y conserva el archivo original salvo overwrite solicitado.
- Agregar un campo opcional puede ser compatible; cambiar dtype, unidad, shape o semántica requiere
  nueva versión.
- Los IDs de topología, coordenadas, canales, vocabularios y modelos son parte del schema.
- Exportar a un formato con menos capacidad requiere advertir o exigir una política de pérdida.
