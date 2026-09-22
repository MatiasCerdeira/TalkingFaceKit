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
- `analyze-video` puede adquirir los modelos DeepTalk faltantes en su primera ejecución; los demás
  comandos no descargan modelos.
- Los intervalos usan `[start_seconds, end_seconds)`.
- Configuración efectiva y versions se conservan en el artefacto de salida.

## Comandos disponibles

### `analyze-video`

```text
uv run --extra active-speaker-deeptalk python -m talkingfacekit analyze-video VIDEO \
  [--start-seconds S] \
  [--end-seconds S] \
  [--report OUTPUT.html] \
  [--overwrite]
```

Ejecuta el adapter experimental DeepTalk-ASD sobre el video completo o un intervalo y muestra un
resumen humano de face IDs, observaciones, intervalos VAD, `raw_score` por ventana, decisiones
preliminares con razones, provenance e issues. También enumera gaps rellenados y overlaps recortados
durante la normalización temporal del audio. Los scores se preservan sin presentarlos como
probabilidades ni decisiones finales. `--report` escribe una página offline que referencia el video
original y sincroniza boxes, pose/calidad crudas, scores, decisiones preliminares, VAD, reparaciones
y timeline con el reproductor. El reporte es demo-only y `--overwrite` es obligatorio para
reemplazar uno existente.

### `extract-landmarks`

```text
uv run python -m talkingfacekit extract-landmarks VIDEO \
  --model MODEL.task \
  --output LANDMARKS.npz \
  [--start-seconds S] \
  [--end-seconds S] \
  [--overwrite]
```

Procesa un rostro con MediaPipe y guarda un NPZ versionado. Sin límites recorre hasta EOF. Los
límites opcionales seleccionan `[start_seconds, end_seconds)` sobre el timeline fuente.

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
├── analyze-video
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

## `analyze-video` — expansión prioritaria

```text
talkingfacekit analyze-video INPUT \
  --backend deeptalk \
  --model-dir MODELS \
  --output REPORT.json \
  [--overlay OUTPUT.mp4] \
  [--start-seconds S] \
  [--end-seconds S] \
  [--config POLICY.toml] \
  [--overwrite]
```

La versión diagnóstica disponible muestra speech intervals, face IDs, pose/calidad crudas,
`raw_score` y segmentos preliminares `candidate`, `rejected` o `uncertain` con razones. Un candidato
preliminar se combina con sus vecinos contiguos de la misma identidad. La salida también muestra
cada run resultante, su duración, cobertura de detecciones, mayor hueco visual y score
mínimo/medio/máximo. La política inicial (2 s, 90% de cobertura y 200 ms de hueco máximo) es
conservadora pero todavía no calibrada. Incluso un run que la pasa no es un segmento aceptado:
`camera_facing`, calidad visual, score calibrado y `av_sync` siguen sin evaluarse. El comando no debe
presentar scores LR-ASD como probabilities ni elegir siempre un rostro.

El JSON es la salida canónica. El overlay es diagnóstico y debe representar exactamente ese mismo
reporte. Los nombres y flags finales se fijarán junto con la API Python, no antes.

## `inspect media` — objetivo posterior

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
