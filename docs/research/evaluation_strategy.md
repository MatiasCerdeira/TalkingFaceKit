# Evaluación simple de selección de segmentos

## Objetivo

Comparar, sobre los mismos videos, **los tramos que una persona usaría para entrenar un modelo de
rostro tipo FLAME** con los tramos que TalkingFaceKit conserva hoy. Repetir la comparación después de
cada cambio. Los resultados actuales son provisionales: más adelante, pose y otros filtros podrán
mejorarlos. Eso no impide medir el progreso desde ahora.

## Gold set mínimo

1. Elegir unos 20 fragmentos de 45 a 90 segundos, de preferencia de videos fuente distintos, con
   algunos tramos que sí se conservarían y otros que no. Si un fragmento tiene varios cambios
   interesantes, puede durar hasta dos minutos. Los seis archivos en `tmp/test_videos` donde todo
   se conservaría quedan como controles positivos separados; no cuentan entre los 20 principales.
2. Mirar cada video **sin ver primero la predicción** y marcar únicamente los intervalos que uno
   efectivamente usaría para entrenar. Cada intervalo guarda `start_seconds` y `end_seconds` sobre el
   video original. Todo lo no marcado cuenta como «no lo usaría». Si un tramo es imposible de juzgar,
   se puede marcar como `ignore` y excluirlo de la comparación.
3. Explicar en una frase el criterio humano para mantenerlo consistente: «Usaría este tramo como
   ejemplo de una persona hablando visiblemente a cámara y con imagen/audio suficientes para el
   entrenamiento». No hace falta anotar cortes, cantidad de personas, cajas ni motivos en esta etapa.
4. Guardar la selección una vez y no editarla para hacer coincidir el resultado con la librería.
   Si después cambia lo que se considera útil para el entrenamiento, crear una nueva versión de las
   selecciones.

### De dónde obtenerlos

La primera fuente debe ser la **colección real que la librería tendrá que procesar**: elegir la mitad
de los 20 fragmentos de videos diversos de esa colección, antes de ver qué predice el sistema. La
otra mitad se elige deliberadamente para cubrir situaciones difíciles. Evitar que muchos fragmentos
salgan de una sola entrevista; dos por video fuente es un máximo razonable para empezar.

Si faltan casos en la colección, usar fragmentos de [AVA ActiveSpeaker](https://github.com/cvdfoundation/ava-dataset/blob/master/README.md)
(videos de películas con cambios de plano y rostros), [WASD](https://github.com/Tiago-Roxo/WASD)
(habla solapada, oclusión, audio retrasado y voces de fondo) o videos de entrevistas y discursos en
[Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:Interview_videos). Revisar la
licencia de cada archivo. En todos los casos, anotar **nuestra selección humana**: las etiquetas
originales de esos datasets responden otras preguntas.

La lista concreta de 20 puede apuntar aproximadamente a: 8–10 fragmentos representativos tomados
sin buscar errores; 2–3 con entrevistador o narrador fuera de cámara; 2–3 con dos o más rostros; 2
con B-roll mientras continúa el habla; 2 con cortes o cambios de persona; 2 con rostro pequeño,
perfil u oclusión; y 1 casi todo negativo (silencio, música o cara visible sin habla). Las categorías
pueden superponerse. Lo esencial es que el conjunto no sea casi todo positivo ni casi todo negativo.
Una copia con audio desplazado se puede agregar como prueba específica cuando se incorpore sync;
no se cuenta como otro video fuente independiente.

La CLI ya admite analizar una ventana de un video largo con `--start-seconds` y `--end-seconds`, así
que no es necesario recodificar los archivos para crear estos fragmentos.

Formato que completa el equipo (ver la [guía paso a paso](../../benchmark/README.md)):

```json
[
  {
    "file": "entrevista_01.mp4",
    "start_seconds": 0,
    "end_seconds": 60,
    "selected": [[3.2, 8.7], [15.1, 21.4]]
  }
]
```

`prepare-benchmark` genera automáticamente los identificadores y los hashes de los archivos. La
corrida registra la versión de la librería. El anotador no necesita ingresar esos datos.

### ¿Anotar algo más?

No agregar tags, cantidad de caras segundo a segundo ni todos los cortes. Para empezar alcanza con
la selección final de tramos útiles. Cuando aparezca una diferencia, mirar ese instante para
entender su causa. Si una causa se repite, recién entonces considerar una anotación específica para
medir ese componente. Revisar juntos los primeros dos o tres videos para acordar el criterio antes
de repartir el resto del trabajo.

## Comparación después de cada cambio

Tomar los `candidate_runs` con estado `candidate` como «seleccionados por la versión actual» y
compararlos con `selected` del humano sobre la misma línea temporal. Informar cuatro números:

- **Precisión por tiempo:** de todos los segundos elegidos por la librería, qué proporción cae en
  tramos que el humano usaría.
- **Recall por tiempo:** de todos los segundos que el humano usaría, qué proporción recuperó la
  librería.
- **Segundos mal conservados** y **segundos útiles perdidos**, para que la magnitud del error sea
  fácil de interpretar.

Calcular también estos números por video. Un reporte visual con dos bandas sobre la línea temporal
—«humano» y «librería»— debería permitir saltar directamente a las diferencias. Un pequeño desfase
en los bordes se puede mostrar aparte; no debería ocultar un tramo entero mal clasificado. No
hace falta empezar con métricas de tracking, VAD, pose o cortes. Si un video falla, se lo inspecciona
entonces para averiguar la causa.

Los veinte videos dan una línea base útil para detectar regresiones, aunque no una estimación precisa
de cómo funcionará la librería en cualquier colección. Si se ajustan umbrales repetidamente sobre
ellos, reservar algunos videos que no se usen para esos ajustes.

## Datasets públicos

**No importar las etiquetas de datasets públicos como si fueran nuestra decisión final.** Sí se
pueden usar sus videos como fuente y marcar desde cero los tramos que usaríamos. La decisión «yo
usaría este intervalo para entrenar FLAME» depende del caso de uso y requiere ejemplos negativos del
video original. Nuestras selecciones humanas responden directamente la pregunta importante.

- [AVA ActiveSpeaker](https://sites.research.google/gr/ava/download/) y
  [WASD](https://github.com/Tiago-Roxo/WASD) sirven después si un error parece venir de detectar
  quién habla. Evalúan ese componente, no la selección final para FLAME.
- [YouTube_Talking](https://github.com/Juzezhang/ViBES/blob/main/docs/1-data/youtube_talking.md)
  y [Shofo Talking Head](https://huggingface.co/datasets/Shofo/shofo-talking-head-en/blob/main/README.md)
  aportan videos ya filtrados, mayormente positivos. No reemplazan el conjunto propio porque no
  indican qué partes de videos arbitrarios se deben rechazar.
- Un dataset de cortes como [ClipShots](https://github.com/Tangshitao/ClipShots) solo vale la pena
  cuando se implemente o investigue específicamente la detección de cortes.

**Estado técnico:** `evaluate-benchmark` ya lee un [manifiesto versionado](../../benchmark/README.md),
verifica hashes y compara `candidate_runs` con `selected`/`ignore`. Guarda métricas y predicciones en
JSON, sin extraer datos del HTML. Falta elegir, compartir y etiquetar los videos reales; no hay aún
una línea base medida.
