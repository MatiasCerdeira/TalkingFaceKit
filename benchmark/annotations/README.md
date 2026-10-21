# Anotaciones humanas

Esta carpeta es la fuente de las selecciones humanas versionadas. Cada video tiene su propio
archivo, por ejemplo `video_01.mp4.json`, con una lista de ventanas ya anotadas para ese video.
El campo `file` identifica el video relativo a `data/eval/videos/`; los tiempos siempre son del
archivo original. Se conservan los cuatro campos de la [plantilla](../annotations.example.json).

Cada persona edita los archivos de los videos que tiene asignados en el [reparto](../assignments.md).
Revisar los cambios mediante Git antes de incorporarlos. `selected: []` significa que se revisó
toda la ventana y se descartó todo. Los videos pendientes se registran en el reparto; todavía no
se agrega un JSON para ellos. La plantilla ficticia queda fuera de esta carpeta.

Se pueden agregar subcarpetas. `prepare-benchmark` reúne recursivamente todos los archivos `.json`
en orden de ruta; conserva el orden de las ventanas dentro de cada archivo. Rechaza archivos JSON
vacíos, entradas inválidas y el mismo video/ventana repetido en varios archivos.

Después de revisar cambios, actualizar `benchmark/manifest.json` desde la raíz del proyecto:

```bash
uv run python -m talkingfacekit prepare-benchmark benchmark/annotations \
  --videos-dir data/eval/videos --output benchmark/manifest.json --overwrite
```

Subir las anotaciones y el manifiesto juntos. El manifiesto se genera automáticamente y permite
verificar que todos tienen los mismos videos. La [guía del equipo](../README.md) explica cómo
comprobarlo y ejecutar la evaluación. Los videos y los resultados de las corridas siguen fuera de Git.
