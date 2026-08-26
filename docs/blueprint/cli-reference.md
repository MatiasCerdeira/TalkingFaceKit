# Referencia de CLI propuesta

La CLI es una capa fina sobre las mismas operaciones tipadas de Python. Debe ser scriptable,
reproducible y segura frente a overwrite.

## Convenciones

- Entrada y salida explícitas mediante argumentos.
- Paths con `pathlib.Path` dentro de Python.
- Exit code `0` para éxito, `2` para uso o input inválido y códigos documentados para futuros
  fallos de procesamiento parcial.
- Errores concisos en stderr; resultados y summaries en stdout.
- `--overwrite` requerido para reemplazar archivos.
- Ningún comando descarga modelos salvo un futuro comando de assets explícito y opt-in.
- Los intervalos usan `[start_seconds, end_seconds)`.
- Configuración efectiva y versions se conservan en el artefacto de salida.

## Comandos disponibles

### `extract-landmarks`

```text
uv run python -m talkingfacekit extract-landmarks VIDEO \
  --model MODEL.task \
  --output LANDMARKS.npz \
  [--overwrite]
```

Procesa un rostro con MediaPipe y guarda un NPZ versionado. Actualmente recorre el intervalo
completo expuesto por `TalkingFaceSequence.from_video`.

### `inspect-landmarks`

```text
uv run python -m talkingfacekit inspect-landmarks LANDMARKS.npz
```

Valida y resume tracker, topología, cantidad de frames/puntos, detecciones, faltantes, rango de
timestamps y shape.

### `render-mesh`

```text
uv run python -m talkingfacekit render-mesh LANDMARKS.npz \
  --video SOURCE.webm \
  --output MESH.html \
  [--title TITLE] \
  [--depth-scale SCALE] \
  [--overwrite]
```

Usa el video sólo para ancho y alto, convierte landmarks MediaPipe a mesh y escribe HTML offline.

## Árbol de comandos objetivo

```text
talkingfacekit
├── inspect media|artifact|model
├── extract landmarks|audio|speech|features
├── track faces|landmarks
├── process smooth|interpolate|sync
├── fit flame
├── animate visemes|retarget
├── render mesh|overlay|report
├── export gltf|csv|audio
├── pipeline run
├── batch run|resume|report
└── artifact validate|migrate|list
```

No se implementará el árbol de una vez. Cada subcomando llega junto con su API, contrato y tests.

## `inspect media` — objetivo próximo

```text
talkingfacekit inspect media INPUT [--json] [--video-stream INDEX] [--audio-stream INDEX]
```

La salida humana muestra streams, codecs, dimensiones, rates, duración, time base, rotación, layout
de canales y metadata de color disponible. `--json` emite un schema versionado para automatización,
sin texto decorativo.

## `extract audio` — objetivo

```text
talkingfacekit extract audio INPUT \
  --output OUTPUT.wav \
  [--audio-stream INDEX] \
  [--start-seconds S] \
  [--end-seconds S] \
  [--sample-rate HZ] \
  [--channels source|mono|stereo] \
  [--overwrite]
```

Omitir `--sample-rate` o `--channels` conserva los valores fuente. Toda conversión se resume antes
de escribir y queda en provenance cuando el formato lo permite.

## `track faces` — objetivo

```text
talkingfacekit track faces INPUT \
  --backend BACKEND \
  --model MODEL \
  --max-faces 4 \
  --output FACES.tfk \
  [--start-seconds S] \
  [--end-seconds S] \
  [--device cpu|cuda:0|mps] \
  [--overwrite]
```

La CLI valida que backend y dispositivo sean compatibles antes de procesar. Un fallo en un frame no
implica necesariamente fallo del comando; el artefacto conserva gaps. Un fallo de backend sí evita
publicar una salida parcial.

## `process smooth` e `interpolate` — objetivo

```text
talkingfacekit process smooth INPUT.tfk \
  --track landmarks/mediapipe \
  --method one-euro \
  --config one-euro.toml \
  --name mediapipe-smoothed \
  --output OUTPUT.tfk

talkingfacekit process interpolate INPUT.tfk \
  --track landmarks/mediapipe-smoothed \
  --method linear \
  --max-gap-seconds 0.12 \
  --name mediapipe-filled \
  --output OUTPUT.tfk
```

Los parámetros específicos se agrupan en config validada cuando serían demasiados flags. La CLI
imprime la configuración efectiva y no permite nombres ambiguos.

## `extract speech` — objetivo

```text
talkingfacekit extract speech INPUT \
  --backend BACKEND \
  --language es \
  --word-timestamps \
  --output TRANSCRIPT.tfk
```

El modelo es local por defecto. Si el backend usa un servicio remoto, el nombre del subcomando o
backend y el prompt de configuración deben dejarlo evidente; credenciales nunca se escriben en el
artefacto.

## `fit flame` — investigación

```text
talkingfacekit fit flame INPUT.tfk \
  --landmarks landmarks/mediapipe-smoothed \
  --model MODEL.pkl \
  --device cuda:0 \
  --config flame-fit.toml \
  --name flame \
  --output OUTPUT.tfk
```

El summary muestra frames convergidos, loss, tiempo y dispositivo. La CLI no descarga assets FLAME.

## `render overlay` — objetivo

```text
talkingfacekit render overlay PROJECT.tfk \
  --source SOURCE.webm \
  --layers face-boxes,landmarks,head-pose \
  --output DEBUG.mp4 \
  [--audio copy|reencode|omit] \
  [--overwrite]
```

El comando debe validar que la fuente corresponde al fingerprint del proyecto o exigir
`--allow-source-mismatch` con warning claro.

## `artifact validate` — objetivo

```text
talkingfacekit artifact validate RESULT.tfk [--deep] [--json]
```

Validación normal: manifest, schemas, shapes, dtypes, referencias y hashes disponibles. `--deep`
también recorre chunks y recalcula checksums, por lo que puede ser costoso.

## `pipeline run` — objetivo tardío

```text
talkingfacekit pipeline run pipeline.toml \
  --input SOURCE.webm \
  --output RESULT.tfk \
  [--cache-dir CACHE] \
  [--device cuda:0] \
  [--overwrite]
```

El archivo de pipeline tiene schema versionado, no ejecuta Python arbitrario y contiene operaciones
conocidas y configuración tipada.

## `batch run` — objetivo tardío

```text
talkingfacekit batch run manifest.jsonl \
  --pipeline pipeline.toml \
  --output-dir RESULTS \
  --workers 4 \
  --on-error continue \
  --report REPORT.json
```

Cada item publica su salida de forma transaccional. El proceso puede continuar, pero el exit code y
el reporte reflejan fallos. `resume` sólo salta un item si source fingerprint, pipeline y versiones
coinciden.

## Configuración

Prioridad futura propuesta, de menor a mayor:

1. defaults documentados;
2. archivo TOML versionado;
3. variables de entorno sólo para credenciales, cache o dispositivo global claramente nombrado;
4. flags de CLI.

El summary o artefacto debe guardar la configuración efectiva sin secretos. No se leen archivos de
config ocultos globales en una primera implementación.

## Output machine-readable

Los comandos de inspección y batch deberían aceptar `--json`. Reglas:

- un único documento JSON en stdout;
- warnings y errores en stderr;
- schema y versión incluidos;
- timestamps como segundos numéricos y, cuando importa, time base original separada;
- paths serializados sin asumir separadores de Windows o POSIX;
- sin colores ANSI cuando stdout no es TTY.
