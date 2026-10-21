# Demo del viernes 23 de octubre de 2026

## Mensaje principal

TalkingFaceKit recibe un video y propone intervalos candidatos de una persona hablando. Combina
modelos existentes con reglas propias de selección y continuidad, y muestra la evidencia detrás de
cada decisión. El benchmark permite comparar esa selección con el criterio humano.

Los candidatos actuales pasan reglas de habla, rostro único, score de hablante activo, cortes,
duración y cobertura de detecciones. La aceptación final para entrenamiento todavía requiere
criterios de orientación, calidad y sincronización audiovisual.

## Recorrido propuesto: 7 a 10 minutos

| Tiempo aproximado | Qué mostrar | Qué explicar |
| --- | --- | --- |
| 1 minuto | Un video original con material aprovechable y descartable | La entrada puede contener entrevistadores, otras imágenes y cambios de plano. Queremos obtener intervalos útiles. |
| 2 minutos | El HTML de análisis: video, caras, voz y scores | InspireFace sigue rostros, Silero detecta habla y LR-ASD aporta evidencia de asociación entre cara y voz. DeepTalk integra estos componentes. |
| 2 minutos | Dos decisiones y sus motivos | Nuestro código combina esa evidencia. Un corte detectado divide segmentos; una pausa breve puede mantenerse dentro de un mismo plano. PySceneDetect aporta los cortes. |
| 1 a 2 minutos | Resultados del benchmark por video y en total | Comparar las predicciones con anotaciones humanas: precisión, recall, segundos mal conservados y segundos útiles perdidos. |
| 1 minuto | Una diferencia concreta respecto del humano | Mostrar el instante, explicar la causa observada y distinguirla de una hipótesis pendiente. |
| 1 minuto | Alcance actual y siguiente paso | Orientación y calidad ya se muestran como datos; faltan sus reglas de aceptación. La sincronización exacta todavía no se mide. |

Los tiempos son una guía de ensayo, no una obligación de mostrar todos los paneles.

## Ejemplos para elegir cuando lleguen las anotaciones

Elegir tres momentos breves entre los videos realmente procesados:

1. Un tramo que el humano y la librería conservarían.
2. Un tramo que ambos descartarían, por ejemplo voz sin un rostro visible adecuado.
3. Una discrepancia que podamos inspeccionar y explicar.

Buscar también un corte y una pausa breve para ilustrar la política temporal. Si no aparecen en
esos tres ejemplos, mostrar otro instante. No es necesario fabricar un caso por cada componente.
Que un video tenga varios rostros detectados es un rechazo deliberado de nuestra política actual,
aunque el modelo devuelva scores para cada uno.

## Primera corrida con datos humanos

Ejecutar desde la raíz de TalkingFaceKit, con los videos definitivos en `data/eval/videos/` y las
selecciones humanas versionadas en `benchmark/annotations/` y el manifiesto en
`benchmark/manifest.json`. Los tiempos deben corresponder a esos archivos.
No consultar la predicción antes de cerrar las selecciones humanas.

```bash
mkdir -p reports/demo

uv run python -m talkingfacekit prepare-benchmark benchmark/annotations \
  --videos-dir data/eval/videos --output benchmark/manifest.json --check

uv run --extra active-speaker-deeptalk python -m talkingfacekit evaluate-benchmark \
  benchmark/manifest.json --videos-dir data/eval/videos \
  --output reports/demo/benchmark-run01.json
```

Reutilizar el manifiesto aprobado que viene con el repositorio. Si cambia el conjunto, revisar las
anotaciones y regenerarlo con `--overwrite`, subiendo ambos cambios juntos a Git. Usar un nombre
nuevo para cada corrida. `--check` detecta diferencias sin modificar archivos ni cargar modelos.

Para generar el HTML diagnóstico de una de las ventanas elegidas, reemplazar el nombre y los
límites del siguiente ejemplo por los reales. Los valores del ejemplo no son anotaciones:

```bash
uv run --extra active-speaker-deeptalk python -m talkingfacekit analyze-video \
  data/eval/videos/video_01.mp4 --start-seconds 30 --end-seconds 90 \
  --report reports/demo/video-01.html
```

El benchmark actual guarda métricas e intervalos en JSON. El HTML diagnóstico se genera por separado
y vuelve a analizar la ventana. El comparador humano/librería, en cambio, se genera directamente con
los resultados guardados, sin repetir el análisis:

```bash
uv run python -m talkingfacekit render-benchmark reports/demo/benchmark-run01.json \
  --videos-dir data/eval/videos --output reports/demo/comparison.html
```

Para la demo, elegir un caso en el comparador, seleccionar una diferencia y usar **Reproducir
diferencia con contexto**. Mostrar las dos bandas y el panel de evidencia, que identifica la regla
registrada en ese instante. El HTML diagnóstico sigue disponible para inspeccionar boxes y pose.

## Qué informar al mostrar los números

Indicar cantidad de videos fuente, ventanas y duración evaluada; versión de código; y criterio de
selección humana. Si el conjunto es pequeño o tiene muchos casos difíciles elegidos a propósito,
describirlo como una primera línea base y una suite de regresión.

Explicar precisión como «de los segundos que elegimos, cuántos eran útiles» y recall como «de los
segundos útiles, cuántos encontramos». Mostrar también los segundos de error para que la magnitud
sea interpretable. No presentar un porcentaje como exactitud general ni un score ASD como una
probabilidad. Si una métrica aparece como `n/a`, explicar que su denominador es cero.

Los intervalos de selección humana miden la utilidad temporal. No etiquetan todos los cortes ni
miden por sí solos la calidad de cada frontera de segmento; esa parte se inspecciona en los ejemplos.

## Preparación y ensayo

- Guardar los reportes y comprobar que abran y reproduzcan el video en la computadora de la demo.
  El HTML referencia el archivo local: mantener los videos en sus rutas.
- Ejecutar el flujo antes del viernes con los modelos ya descargados. Anotar cuánto tarda cada
  ejemplo en ese equipo si se quiere mostrar rendimiento.
- Tener resultados precalculados. Elegir una ventana corta para ejecutar en vivo si el tiempo lo
  permite.
- Anotar los timestamps de los momentos a mostrar y ensayar el recorrido completo.
- Mantener las anotaciones humanas fijas al investigar errores. Registrar cualquier ajuste de
  umbrales como otra corrida; reservar material sin usar para ajustes cuando el conjunto lo permita.

## Comparador disponible

`render-benchmark` genera un HTML a partir del JSON guardado por `evaluate-benchmark`, sin repetir
inferencia. Permite elegir un caso, reproducir su video, comparar las bandas humano/librería,
distinguir material mal conservado de material útil perdido y saltar a las diferencias. Los
intervalos `ignore` quedan excluidos de los errores. Incluye métricas por caso y globales.

Incorporar las anotaciones del equipo requiere correr el benchmark y generar el reporte. Los filtros
de orientación y calidad se pueden priorizar después de inspeccionar
qué errores aparecen en esa primera línea base.
