# Arquitectura objetivo

Este documento extiende la arquitectura vigente sin reemplazarla. Sólo
[`docs/architecture.md`](../architecture.md) registra decisiones aceptadas. La estructura objetivo
se materializa por entregas; no se crean carpetas o interfaces vacías para parecer completa.

## Objetivos arquitectónicos

1. Mantener el modelo de dominio independiente de frameworks.
2. Conservar tiempo, coordenadas, unidades, shape, dtype y validez de forma explícita.
3. Procesar media densa por streaming o chunks y materializar sólo resultados justificables.
4. Coordinar operaciones transaccionales desde una fachada sencilla.
5. Permitir backends reemplazables sin diseñar un plugin system prematuro.
6. Hacer reproducibles los resultados mediante schemas y provenance.
7. Separar análisis, transformación, persistencia, rendering y exportación.
8. Escalar de un archivo a batch sin cambiar los contratos core.

## Capas y dirección de dependencias

```text
                             entrypoints
                    Python facade · CLI · notebooks
                                  |
                                  v
                         application workflows
                  sequence operations · future pipeline
                         /                    \
                        v                      v
             core contracts              integration ports
        NumPy · dataclasses · schemas     small Protocols when needed
                        ^                      ^
                        |                      |
                        +----------+-----------+
                                   |
                              integrations
             PyAV · MediaPipe · Plotly · speech · FLAME · exporters
```

Core no depende de integrations. Los workflows conocen contratos y puertos pequeños. Un adapter
convierte objetos externos a tipos core antes de devolver el control.

## Flujo de una operación

```text
source + interval + explicit config
                |
                v
      integration boundary opens resource
                |
                v
       stream/chunks -> backend inference
                |
                v
        complete core result + provenance
                |
                v
             validation
           /            \
        failure        success
           |              |
 no sequence mutation     +--> atomic persistence, if requested
                          |
                          +--> named attachment, if requested
```

La secuencia no se modifica hasta validar. Persistir y adjuntar son decisiones separadas; una API
puede hacer ambas sólo si documenta el orden y conserva semántica transaccional.

## Estado actual del paquete

```text
src/talkingfacekit/
├── metadata.py
├── video.py
├── mesh.py
├── sequence.py
├── cli.py
├── io/
│   ├── video.py
│   └── landmarks.py
├── tracking/
│   ├── landmarks.py
│   ├── mediapipe.py
│   └── mediapipe_mesh.py
└── rendering/
    └── plotly.py
```

Esta disposición ya respeta el principio de depender hacia adentro. Los futuros nombres no
justifican mover los módulos existentes sin un beneficio concreto.

## Mapa de módulos objetivo

Cada entrada marcada como objetivo aparece cuando una entrega real la necesita:

```text
src/talkingfacekit/
├── metadata.py              # Disponible: metadata de video
├── video.py                 # Disponible: fuente inmutable y frame RGB core
├── audio.py                 # Objetivo: chunks/tracks de audio core
├── timeline.py              # Objetivo condicionado: tiempo compartido y alineación
├── mesh.py                  # Disponible: mesh animado core
├── sequence.py              # Disponible: agregado y operaciones coordinadas
├── provenance.py            # Objetivo: origen de artefactos y operaciones
├── quality.py               # Objetivo: reportes tipados
├── animation.py             # Objetivo: curvas y mappings de rig
├── cli.py                   # Disponible; puede dividirse sólo cuando crezca
├── io/
│   ├── video.py             # Disponible: PyAV metadata/frames
│   ├── audio.py             # Objetivo: PyAV audio/encode
│   ├── landmarks.py         # Disponible: NPZ v1
│   └── project.py           # Objetivo: proyecto multipista
├── tracking/
│   ├── landmarks.py         # Disponible
│   ├── mediapipe.py         # Disponible
│   ├── faces.py             # Objetivo: detecciones, IDs y colecciones
│   ├── pose.py              # Objetivo: contratos de pose
│   └── gaze.py              # Objetivo: contratos de mirada
├── processing/
│   ├── landmarks.py         # Objetivo: smoothing/interpolación/calidad
│   └── synchronization.py   # Objetivo: offset y alineación
├── speech/
│   ├── tracks.py            # Objetivo: voz/texto/fonemas/visemas
│   └── <backend>.py         # Sólo al implementar un backend concreto
├── models/
│   └── flame.py             # Investigación: adapter opcional
├── rendering/
│   ├── plotly.py            # Disponible
│   ├── overlay.py           # Objetivo: video de diagnóstico
│   └── report.py            # Objetivo: reporte HTML
├── export/
│   └── gltf.py              # Objetivo: mesh/animación portable
├── dataset.py               # Objetivo tardío: manifest iterable
└── pipeline.py              # Objetivo tardío: composición validada
```

No se crean `base.py`, `manager.py`, `factory.py` o `registry.py` genéricos sin consumidores reales.

## Agregado de secuencia

`TalkingFaceSequence` sigue siendo la fachada para una fuente y un intervalo. Dirección propuesta:

```text
TalkingFaceSequence
├── source: VideoSource
├── [start_seconds, end_seconds)
├── duration_seconds del intervalo declarado
├── landmark_tracks: name -> FaceLandmarkTrack
├── face_tracks: name -> FaceTrackSet
├── audio_tracks: name -> AudioTrack
├── speech_tracks: name -> Transcript/Phoneme/Viseme track
├── geometry_tracks: name -> FaceMesh/FLAME track
└── provenance graph
```

`VideoSource` contiene la ruta y la metadata del stream completo y puede compartirse entre varias
secuencias. No contiene intervalos ni resultados. No todos los mappings deben agregarse como
atributos públicos inmediatamente. Un registro unificado de artefactos podría ser mejor después de
probar persistencia multipista. Hasta entonces, mappings tipados por familia mantienen la API clara.

## Operaciones puras y operaciones con efectos

### Core/puras

- validar un track;
- convertir coordenadas con metadata explícita;
- smooth/interpolate sobre arrays;
- calcular métricas;
- mapear fonemas a visemas;
- retargetear curvas;
- construir mesh desde landmarks.

### Integración/efectos

- abrir media;
- decodificar/encodear;
- cargar modelos;
- ejecutar frameworks;
- acceder a red o cámara;
- leer/escribir artefactos;
- renderizar con librerías externas.

Una función pura no recibe handles externos ni escribe archivos. Un boundary documenta recursos,
excepciones y ciclo de vida.

## Protocolos

Un `Protocol` es apropiado cuando:

- existen o están implementándose dos backends reemplazables;
- un fake de test representa una frontera real, no una abstracción inventada;
- inputs y outputs pueden expresarse completamente con tipos core;
- la configuración específica permanece en el constructor del adapter.

No hace falta un protocolo universal de `Processor`. Tracking, ASR, fitting y rendering tienen
semánticas de error y recursos diferentes.

## Tiempo y sincronización

La fuente mantiene el timeline maestro, pero cada señal conserva su muestreo:

```text
source time (seconds)
├── video frames: irregular PTS t_v[F]
├── audio samples: start + sample_index / sample_rate
├── audio features: window timestamps t_a[K]
├── speech: intervals [start, end)
├── animation: curve timestamps t_c[M]
└── events: sparse intervals/points
```

Alinear no significa reescribir todos los tracks sobre video. Se crea una vista o resultado nuevo
con política explícita. Offset y drift son transformaciones de timeline con provenance.

## Coordenadas y transformaciones

Dirección propuesta:

```text
source image pixels
        ^  crop transform (invertible)
normalized image coordinates
        |
        +--> display coordinates (relative, aspect-corrected)
        |
        +--> camera coordinates (requires camera model)
        |
        +--> model coordinates (requires fit/schema)
        |
        +--> rig/world coordinates (requires retarget transform)
```

Cada flecha es una operación nombrada. Un string de `coordinate_system` sigue siendo suficiente hoy;
si las conversiones crecen, se evaluará un schema estructurado y versionado.

## Persistencia actual

El NPZ de landmarks v1 guarda scalars de schema/provenance mínima y arrays completos. El writer:

- valida extensión y destino;
- escribe un temporal en el mismo directorio;
- reemplaza al finalizar;
- exige `overwrite=True` si ya existe;
- limpia el temporal al fallar.

Este patrón debe conservarse en nuevos writers.

## Proyecto nativo multipista — objetivo

Formato candidato para validar en una entrega real:

```text
portrait.tfk/
├── manifest.json
├── arrays/
│   ├── landmarks-mediapipe-frame-indices.npy
│   ├── landmarks-mediapipe-timestamps.npy
│   ├── landmarks-mediapipe-values-00000.npy
│   ├── audio-source-00000.npy
│   └── ...
├── tables/
│   ├── transcript.jsonl
│   └── events.jsonl
└── checksums.json
```

Primera propuesta: un directorio con extensión `.tfk`, manifest JSON y shards NPY/JSONL. Razones:

- lectura lazy sin descomprimir un ZIP completo;
- inspección con herramientas comunes;
- chunks sin dependencia nueva;
- checksums y migración por archivo;
- posibilidad de empaquetado posterior para intercambio.

Decisiones aún abiertas:

- directorio versus archivo único;
- chunk size y compresión;
- locking y escritura concurrente;
- enlaces a media relativa/absoluta/fingerprint-only;
- estrategia de garbage collection al reemplazar tracks;
- compatibilidad con object storage;
- si conviene adoptar un formato existente en lugar de uno propio.

### Manifest candidato

```json
{
  "schema": {"name": "talkingfacekit-project", "version": 1},
  "talkingfacekit_version": "0.x",
  "source": {
    "reference": "portrait.webm",
    "fingerprint": {"algorithm": "sha256", "value": "..."},
    "embedded": false
  },
  "interval": {"start_seconds": 0.0, "end_seconds": 2.719},
  "artifacts": {
    "landmarks/mediapipe": {
      "schema": {"name": "face-landmarks", "version": 1},
      "storage": "arrays/landmarks-mediapipe-*.npy",
      "provenance_id": "operation-0001"
    }
  },
  "provenance": {}
}
```

JSON final deberá definir orden, encoding, números no finitos y validación. Los `NaN` viven en NPY,
no en JSON no estándar.

## Cache — objetivo tardío

Cache y proyecto del usuario son distintos:

| Proyecto | Cache |
| --- | --- |
| nombrado y conservado por el usuario | detalle de ejecución descartable |
| overwrite explícito | eviction permitida |
| forma parte del output | acelera recomputación |
| provenance completo | clave derivada de provenance relevante |
| migración soportada | puede invalidarse por versión |

Clave conceptual:

```text
hash(
  source_fingerprint,
  interval,
  operation,
  normalized_config,
  backend_version,
  model_hash,
  output_schema_version,
)
```

## Pipeline — objetivo tardío

Un pipeline será un DAG validado sólo si aparecen dependencias no lineales. Antes alcanza una lista
de pasos. Cada step declara:

- inputs requeridos por nombre/schema;
- output producido;
- configuración tipada y normalizada;
- recursos/dispositivo;
- cacheability;
- progreso/cancelación;
- si tiene efectos externos.

No se ejecuta código arbitrario desde TOML/JSON. Los steps desconocidos fallan antes de procesar.

## Batch y paralelismo — objetivo tardío

La paralelización vive en application, no dentro de dataclasses:

- aislamiento transaccional por item;
- límite de workers;
- backends recreados o compartidos según thread/process safety;
- un worker por GPU o scheduler explícito;
- backpressure para decode y escritura;
- orden de reportes determinista;
- cancelación y resume seguros.

El primer batch puede ser secuencial. La concurrencia llega con perfiles reales, no por default.

## Logging, progreso y diagnósticos — objetivo

Las funciones de biblioteca no imprimen. Pueden aceptar callbacks/protocolos pequeños o emitir logs
estándar sin configurar handlers globales. La CLI elige presentación.

Todo trabajo largo debería poder reportar:

- etapa;
- frames/samples procesados y total si se conoce;
- tiempo transcurrido;
- real-time factor;
- dispositivo;
- warnings acumulados;
- cancelación cooperativa.

No se incluyen paths sensibles o transcripciones en logs por defecto.

## Seguridad y privacidad

- local-first y sin telemetría;
- no deserializar pickle de artefactos no confiables;
- `np.load(..., allow_pickle=False)` como en el loader actual;
- validar paths, extensiones, schemas, shapes y tamaños antes de reservar memoria grande;
- no extraer archives con traversal;
- no incluir credenciales o rutas absolutas privadas en provenance por defecto;
- documentar riesgos de biometría, datasets y licencias;
- remote backends siempre explícitos.

Los modelos externos pueden usar formatos inseguros. El adapter debe advertirlo y preferir formatos
de pesos seguros cuando existan.

## Rendimiento

Principios:

- streaming para pixels/audio;
- arrays contiguos sólo cuando un backend los requiere;
- cero copias como optimización documentada, no a costa de mutabilidad ambigua;
- chunks para datos densos;
- medir memoria pico, throughput y real-time factor;
- conservar precisión del contrato al cruzar frameworks;
- no agregar aceleración nativa antes de perfilar.

Los tests unitarios miden invariantes. Benchmarks separados evalúan performance sin convertirse en
límites frágiles de CI.

## Error model

Las fronteras rechazan temprano inputs inválidos y los mensajes incluyen invariante y valor
observado. La jerarquía de excepciones de dominio sólo se agrega cuando los callers necesitan
reacción diferenciada. Un pipeline registra error por item, pero no lo transforma en resultado
válido.

## Fuentes en vivo

Una sesión en vivo no es una secuencia de archivo con `end_seconds=None`. Requiere estados:

```text
created -> opened -> running -> draining -> closed
                    |             |
                    +--> failed <-+
```

Backpressure, frames tardíos, clock monotónico, reconexión y latencia necesitan contratos propios.
Sólo debería compartir `DecodedVideoFrame` o tracks cuando sus invariantes coincidan exactamente.
