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
   anotación. Cada persona edita el JSON de sus videos en `benchmark/annotations/` y comparte sus
   cambios mediante Git. El [reparto](assignments.md) registra responsables y pendientes.

## El JSON que completa el equipo

Hay solo cuatro campos por caso. Copiar la [plantilla](annotations.example.json) a un archivo por
video, por ejemplo `benchmark/annotations/video_01.mp4.json`. Cada archivo contiene una lista de
ventanas terminadas para ese video. Los dos objetos siguientes ilustran dos videos distintos;
en el trabajo del equipo van en sus respectivos archivos.

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

## Organización compartida

Las etiquetas y el manifiesto aprobado se comparten mediante Git:

```text
benchmark/
  assignments.md
  annotations/
    video_01.mp4.json
    video_02.mp4.json
  manifest.json          (lo genera el programa; no editar a mano)
```

Cada persona actualiza el repositorio y descarga de Drive los mismos videos a `data/eval/videos/`.
Conservar exactamente los nombres y archivos; sus hashes detectan versiones diferentes. Videos,
modelos y resultados de las corridas siguen fuera de Git; `data/` y `reports/` permanecen ignorados.

Los videos pendientes van en el [reparto](assignments.md), sin un JSON de selección terminado.
`selected: []` significa «revisado y descartado completo». La plantilla ficticia queda fuera de
`annotations/`. Se admiten subcarpetas: el programa lee recursivamente todos los `.json` en orden
de ruta, mantiene el orden de las ventanas de cada archivo y rechaza entradas inválidas o duplicadas.

## Cuando las anotaciones estén listas: para quien corre la evaluación

**1. Comprobar la versión compartida.** Actualizar el repositorio y descargar los videos. El
manifiesto aprobado ya viene en Git. Este comando verifica que coincida con todas las anotaciones
y videos locales, sin escribir ni ejecutar modelos:

```bash
uv run python -m talkingfacekit prepare-benchmark benchmark/annotations \
  --videos-dir data/eval/videos --output benchmark/manifest.json --check
```

**2. Actualizar el manifiesto cuando cambien las anotaciones.** Después de revisar un nuevo video
o una corrección, una persona regenera el manifiesto con los videos definitivos de Drive. Revisar
su diff y subirlo junto con los JSON modificados. No regenerarlo para ocultar un cambio accidental
de video: primero corregir la copia local. No editar hashes ni IDs a mano.

```bash
uv run python -m talkingfacekit prepare-benchmark benchmark/annotations \
  --videos-dir data/eval/videos --output benchmark/manifest.json --overwrite
```

El manifiesto debe quedar fuera de `benchmark/annotations/`, para que no se lea como otra anotación.
El flujo anterior con una lista JSON única sigue funcionando; la carpeta versionada es la fuente
de trabajo del equipo. Las copias anteriores bajo `data/eval/` son snapshots locales.

**3. Correr la evaluación.** Requiere el entorno y los modelos del adapter DeepTalk. Comprueba los
archivos y luego compara los tramos elegidos por la librería con las anotaciones humanas.

```bash
uv run --extra active-speaker-deeptalk python -m talkingfacekit evaluate-benchmark \
  benchmark/manifest.json --videos-dir data/eval/videos \
  --output reports/benchmark-v1-run01.json
```

`--verify-only` es una opción auxiliar para comprobar hashes y etiquetas del manifiesto sin ejecutar
modelos; la corrida completa ya hace esa comprobación. `prepare-benchmark --check` también comprueba
su correspondencia con la carpeta de anotaciones. El resultado guarda el hash exacto del manifiesto.

El resultado muestra, por video y en total, precisión, recall, segundos conservados incorrectamente
y segundos útiles perdidos. El reporte guarda también las predicciones y la versión de código.
Usar un nombre de salida distinto por corrida para conservar las comparaciones.

**4. Generar el comparador visual de esa corrida.** No vuelve a analizar el video ni necesita los
modelos instalados. Comprueba los archivos contra los hashes guardados:

```bash
uv run python -m talkingfacekit render-benchmark reports/benchmark-v1-run01.json \
  --videos-dir data/eval/videos --output reports/benchmark-v1-run01.html
```

Abrir el HTML como archivo local en un navegador que reproduzca el formato de los videos. Elegir un
caso y comparar las bandas **Humano**, **Librería** y **Comparación**. Las diferencias se separan en
**Conservado de más** y **Útil perdido**; se pueden filtrar y recorrer con **Anterior/Siguiente**.
Seleccionar una amplía su entorno y mueve el reproductor al comienzo. El botón de reproducción con
contexto incluye 1,5 segundos a cada lado y se detiene al terminar ese contexto o la ventana.

El panel del instante actual muestra las dos selecciones y, para corridas nuevas, los motivos de las
reglas, IDs detectados, score crudo, duración y cobertura. También identifica interrupciones
conservadas por continuidad. Las corridas antiguas sin diagnóstico permiten comparar los intervalos
y muestran explícitamente que no tienen motivos registrados. Las regiones `ignore` no cuentan como
errores. El reporte no modifica anotaciones, no envía datos a servicios externos y no copia videos:
hay que mantenerlos en las rutas usadas al generar el HTML. Usar `--overwrite` para reemplazarlo.

La comparación es por tiempo, no una evaluación de todas las fronteras: intervalos contiguos pueden
coincidir temporalmente aunque se hayan dividido de otra manera. Sus bordes siguen visibles en las
bandas para poder inspeccionarlos.

Las anotaciones se mantienen fijas mientras cambia la librería. Si hay una diferencia, inspeccionar
ese instante con el visualizador para entender su causa. No editar la selección humana solo para
hacerla coincidir con una predicción. Los tests automáticos de código usan ejemplos sintéticos y
no descargan estos videos ni los modelos.
