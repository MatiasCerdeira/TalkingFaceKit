# Recolección y anotación de videos

Queremos comparar los segmentos que usaríamos para entrenar un modelo de rostro tipo FLAME con los
que elige TalkingFaceKit. Cada persona mira el video y anota únicamente los tramos que conservaría.

## Paso a paso para quienes anotan

1. **Repartir unos 20 videos fuente distintos.** Priorizar material del tipo que realmente vamos a
   procesar: entrevistas, videopodcasts, charlas y videos con gente hablando. Los seis videos
   actuales donde se conservaría todo quedan como controles adicionales.
2. **Elegir una ventana de 45 a 90 segundos de cada video.** Puede durar hasta dos minutos si hace
   falta para incluir un cambio interesante. No hace falta recortar ni recodificar el archivo:
   anotamos el inicio y el fin de la parte que vamos a mirar en el video original.
3. **Incluir variedad en el conjunto.** Aproximadamente la mitad puede ser material habitual elegido
   sin buscar errores. En el resto, buscar voces fuera de cámara, varios rostros, cambios de plano
   o de persona, otras imágenes mientras continúa la voz, perfiles, caras tapadas o pequeñas y
   personas visibles que no están hablando. Conviene incluir uno o dos casos donde no conservaríamos
   nada. Un video puede cubrir varias situaciones; no hace falta asignarle categorías.
4. **Subir los archivos definitivos a la carpeta compartida `videos/`.** Usar nombres únicos, por
   ejemplo `video_01.mp4`, y mantenerlos después de anotar. La persona que coordina reparte los
   nombres para evitar duplicados. Todos usarán estos mismos archivos. Registrar también el enlace
   de origen en la lista compartida de recolección si viene de otra fuente.
5. **Mirar toda la ventana sin consultar la predicción de la librería.** Conservar los tramos que
   uno realmente usaría para entrenar: una persona hablando visiblemente a cámara, con el rostro
   suficientemente visible y audio/imagen utilizables. Marcar el inicio y el fin de cada tramo. Si
   cambia la persona o hay un corte, cerrar el tramo y evaluar el siguiente por separado.
6. **Completar el JSON de abajo.** Todos los tiempos son segundos del archivo que se subió. Por
   ejemplo, `01:15.5` se escribe `75.5`. Mirar los bordes pausando el reproductor; usar decimales
   cuando haga falta, sin buscar precisión de un frame. Anotar los tramos en orden, sin superponerlos.
   Todo lo que no se marca cuenta como material que descartaríamos. `selected: []` significa
   «miré la ventana completa y no conservaría nada»; no se usa para un video pendiente.
7. **Revisar juntos los primeros dos o tres casos.** Acordar qué conservaríamos antes de repartir el
   resto. Si aparece un tramo difícil de decidir, revisarlo con otra persona antes de cerrar la
   anotación. Una persona reúne las entradas de todos en `annotations.json` para evitar que varias
   personas editen el mismo archivo a la vez.

## El JSON que completa el equipo

Hay solo cuatro campos por caso. Se puede copiar la
[plantilla](annotations.example.json) y reemplazar los ejemplos por anotaciones reales.

```json
[
  {
    "file": "video_01.mp4",
    "start_seconds": 30,
    "end_seconds": 90,
    "selected": [[34.2, 46.8], [58.1, 71.5]]
  },
  {
    "file": "video_02.mp4",
    "start_seconds": 0,
    "end_seconds": 60,
    "selected": []
  }
]
```

El primer caso evalúa del segundo 30 al 90 y conserva dos tramos. El segundo evalúa del 0 al 60 y
descarta todo. `selected` usa los mismos tiempos del archivo original: el primer tramo comienza en
`34.2`, no en `4.2`. Si se usa un clip ya recortado, se anotan los tiempos de ese clip definitivo.
Los ejemplos son ficticios; el fin de la ventana debe existir en el video real.

No hay que ingresar hashes, identificadores, tags, cantidad de caras ni cortes individuales.

## Organización de la carpeta compartida

Usar el almacenamiento privado que ya tenga el equipo. Todos descargan o sincronizan la misma
carpeta, por ejemplo a `data/eval/` dentro de su checkout:

```text
data/eval/
  annotations.json
  manifest.json          (lo genera el programa)
  videos/
    video_01.mp4
    video_02.mp4
```

Videos, anotaciones y resultados quedan fuera de Git, según `AGENTS.md`. `data/` y `reports/` ya
están ignorados. El repositorio contiene las instrucciones, la plantilla ficticia y el código.
Compartir archivos de terceros solo cuando su licencia o permiso lo permita.

## Cuando las anotaciones estén listas: para quien corre la evaluación

**1. Generar el manifiesto una vez.** El programa valida las anotaciones y calcula automáticamente
los identificadores y los hashes de los videos. No ejecuta modelos ni modifica `annotations.json`.

```bash
uv run python -m talkingfacekit prepare-benchmark data/eval/annotations.json \
  --videos-dir data/eval/videos --output data/eval/manifest.json
```

**2. Compartir ese `manifest.json` junto con los videos.** Se genera una vez para una versión
aprobada del conjunto. No regenerarlo en cada equipo o antes de cada corrida: sus hashes permiten
detectar si alguien tiene videos distintos. Para reemplazarlo tras un cambio acordado de las
anotaciones, usar `--overwrite` y tratar el cambio como una nueva versión del conjunto.

Opcionalmente, guardar en Git un archivo `benchmark/manifest.sha256` con el SHA-256 del manifiesto
aprobado, generado con el siguiente comando. Así una revisión de código puede referenciar una
versión exacta de las anotaciones sin guardar el dataset en Git.

```bash
uv run python -c "import hashlib, pathlib; p = pathlib.Path('data/eval/manifest.json'); print(hashlib.sha256(p.read_bytes()).hexdigest())" > benchmark/manifest.sha256
```

**3. Correr la evaluación.** Requiere el entorno y los modelos del adapter DeepTalk. Comprueba los
archivos y luego compara los tramos elegidos por la librería con las anotaciones humanas.

```bash
uv run --extra active-speaker-deeptalk python -m talkingfacekit evaluate-benchmark \
  data/eval/manifest.json --videos-dir data/eval/videos \
  --output reports/benchmark-v1-run01.json
```

Si se creó el archivo de hash del manifiesto, agregar `--manifest-lock benchmark/manifest.sha256`.
`--verify-only` es una opción auxiliar para comprobar archivos sin ejecutar los modelos; la corrida
completa ya hace esa comprobación automáticamente.

El resultado muestra, por video y en total, precisión, recall, segundos conservados incorrectamente
y segundos útiles perdidos. El reporte guarda también las predicciones y la versión de código.
Usar un nombre de salida distinto por corrida para conservar las comparaciones.

Las anotaciones se mantienen fijas mientras cambia la librería. Si hay una diferencia, inspeccionar
ese instante con el visualizador para entender su causa. No editar la selección humana solo para
hacerla coincidir con una predicción. Los tests automáticos de código usan ejemplos sintéticos y
no descargan estos videos ni los modelos.
