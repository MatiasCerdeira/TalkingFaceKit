# Conceptos fundamentales

## La secuencia como agregado

**Disponible.** `VideoSource` representa de forma inmutable la ruta y la metadata del stream
completo. `TalkingFaceSequence` referencia una fuente, delimita un intervalo temporal y coordina
los resultados asociados. Su construcción directa es barata y no toca el filesystem. Los
constructores alternativos pueden inspeccionar fuentes mediante integraciones.

La fuente y el intervalo de una secuencia no se reasignan después de validarse. El mapping de
resultados nombrados es su estado mutable; los resultados completos deberían ser valores
inmutables en la práctica. Una operación costosa:

1. recibe la fuente y el intervalo;
2. calcula un resultado completo en el backend;
3. valida el contrato independiente del framework;
4. adjunta el resultado a la secuencia sólo si todo terminó correctamente.

Este patrón evita estados parciales y ya se usa en `track_landmarks`.

La secuencia no significa que el video ya sea un talking head válido. Es una fuente o vista temporal
que puede ser analizada. El resultado nuevo será un `VideoAnalysisReport` (nombre propuesto) con
observaciones y decisiones. Una futura colección agrupa fuentes y reportes; no debería contener sólo
secuencias previamente aceptadas.

## Fuente, intervalo y timeline

**Disponible parcialmente.** Una fuente local se identifica con `VideoSource`; el primer stream de
video define metadata y frames. Un intervalo público usa segundos y la forma semiabierta
`[start_seconds, end_seconds)`. `end_seconds=None` significa continuar hasta el final disponible.
La duración reportada permanece en `source.metadata`; `sequence.duration_seconds` describe sólo un
intervalo cerrado y devuelve `None` cuando está abierto.

Los timestamps son datos, no una consecuencia de los FPS. TalkingFaceKit debería conservar el
presentation timestamp del stream y sólo cambiar de base temporal mediante una operación explícita.

Una versión futura deberá distinguir:

- tiempo de presentación de la fuente;
- índice de decode de la fuente;
- índice de posición dentro de un track;
- tiempo de audio expresado por muestra y por segundo;
- tiempo editado de una secuencia recortada, concatenada o retimada;
- latencia y tiempo de captura en fuentes en vivo.

No todos esos tiempos deben entrar en el primer contrato. Sí deben tener nombres distintos cuando
existan.

## Streaming y materialización

**Disponible para video y audio; objetivo para otros resultados grandes.** Las unidades de
procesamiento son `DecodedVideoFrame` y `DecodedAudioChunk`, entregadas una por vez. Esto limita la
memoria del productor, aunque un consumidor puede retener los arrays.

Materializar un track tiene sentido cuando su tamaño es moderado y se reutiliza, como landmarks.
Para máscaras densas, frames, texturas o datasets grandes, el diseño futuro debería ofrecer chunks
y almacenamiento lazy. La elección debe ser visible en el tipo o en el método; una misma propiedad
no debe alternar silenciosamente entre un array y un iterador.

## Tracks

Un track es una señal ordenada sobre el tiempo. Puede representar puntos, geometría, valores
escalares, categorías o intervalos. Los tracks comparten invariantes:

- timeline explícito y estrictamente creciente cuando está basado en frames;
- dtype y shape fijos dentro de un mismo resultado;
- una máscara o estructura explícita para datos faltantes;
- sistema de coordenadas, unidades y semántica documentados;
- provenance del backend y de su configuración;
- arrays tratados como inmutables después de validar.

**Disponible:** `FaceLandmarkTrack` y `FaceMeshTrack`.

**Objetivo inmediato:** face tracks, speech intervals, observaciones de hablante activo y decisiones
de segmentos. `DecodedAudioChunk` ya está disponible como primitiva de streaming.

**Objetivo posterior:** pose, gaze, máscaras, blendshapes, fonemas, visemas, prosodia, cámara y
parámetros de modelo 3D.

No hace falta que todos hereden de una superclase. Una abstracción común sólo debería aparecer
cuando operaciones reales —por ejemplo recorte, resample o persistencia— demuestren que comparten
el mismo comportamiento.

## Resultados nombrados

**Disponible para landmarks.** Una secuencia puede guardar varios resultados bajo nombres como
`"mediapipe"`, `"mediapipe-smoothed"` o `"experiment-2026-08"`. Los nombres permiten comparar
backends y configuraciones sin incorporar sus detalles al agregado.

Reglas propuestas para todas las familias de resultados:

- el nombre no puede estar vacío;
- no hay overwrite implícito;
- el reemplazo ocurre después de completar y validar el resultado nuevo;
- un fallo no elimina ni modifica el resultado anterior;
- los nombres son identificadores del usuario, no IDs globales ni rutas de cache.

## Coordenadas y topologías

Un array `(x, y, z)` no alcanza para describir geometría. Todo track espacial debe declarar:

- origen;
- orientación positiva de cada eje;
- unidades o normalización;
- relación con el tamaño de imagen;
- convención de cámara;
- profundidad relativa o métrica;
- orden y versión de sus puntos o vértices.

**Disponible.** MediaPipe usa 478 landmarks en coordenadas normalizadas de imagen. La conversión a
mesh toma los primeros 468 puntos, aplica corrección de aspect ratio y utiliza 852 triángulos. Su
profundidad es relativa; el resultado no representa una reconstrucción métrica.

**Objetivo.** Las transformaciones entre sistemas deberían ser funciones explícitas que reciban la
metadata necesaria y produzcan un track nuevo con provenance de la transformación.

## Datos faltantes, confianza e interpolación

**Disponible parcialmente.** En landmarks y meshes, un frame no detectado permanece en la línea
temporal, usa `detected=False` y contiene sólo `NaN`. Eliminarlo rompería la alineación.

La dirección futura distingue tres conceptos:

- `observed`: el backend produjo una medición;
- `valid`: la medición pasó las validaciones o umbrales elegidos;
- `interpolated`: el valor fue estimado desde otros instantes.

La interpolación nunca debería convertir un valor estimado en observado. Su política debe incluir
duración máxima del hueco, método y tratamiento de límites. Para confidencias por punto se requiere
un array separado, no codificarlas dentro de una coordenada.

## Rostro único, múltiples rostros e identidad

**Disponible:** un rostro por frame con MediaPipe.

**Objetivo:** `FaceTrackSet` agrupa tracks con IDs estables dentro de una secuencia. La asociación
temporal puede usar movimiento, apariencia o embeddings, pero esos detalles quedan en la
integración. El core conserva resultados, intervalos de presencia, confianza y eventos de
ambigüedad.

Un `face_id` no identifica a una persona fuera del archivo. El reconocimiento biométrico entre
fuentes queda fuera del núcleo y, si se investiga, exige una API separada, consentimiento, manejo de
privacidad y documentación de riesgos.

## Sincronización multimodal

**Objetivo.** Video, audio, fonemas y animación pueden tener frecuencias distintas. La secuencia no
debería forzarlos a una grilla única. Una operación de alineación recibe un timeline destino y una
política adecuada al tipo de señal:

- nearest/hold para categorías;
- interpolación lineal o cúbica para curvas continuas;
- integración por ventanas para energía o probabilidad;
- solapamiento exacto para segmentos;
- nunca interpolación automática de topologías o identidades diferentes.

El offset aplicado, el drift corregido y la calidad estimada deben quedar registrados.

Active speaker detection y sincronización no son sinónimos. LR-ASD responde qué rostro parece
producir el audio en una ventana; SyncNet se evaluará aparte para estimar desplazamiento temporal y
decidir si la relación A/V es aceptable o no medible.

## Observación, política y decisión

Un backend produce observaciones: boxes, VAD y scores crudos. TalkingFaceKit aplica una política
temporal configurable y produce decisiones con razones. Separarlas permite:

- cambiar DeepTalk por LR-ASD directo o TalkNet sin cambiar el significado del reporte;
- recalibrar thresholds sin repetir necesariamente toda la inferencia;
- conservar ambigüedad, ausencia y capacidades `not_evaluated`;
- explicar qué evidencia causó cada segmento.

Un `raw_score` no es confidence salvo que el backend documente y valide esa calibración. Durante el
primer milestone, un intervalo puede ser candidato, rechazado o incierto; no se marca aceptado por
completo hasta medir los criterios de pose, calidad y sync requeridos.

## Provenance y reproducibilidad

**Disponible parcialmente.** Los archivos NPZ de landmarks conservan nombre y versión del tracker,
topología y sistema de coordenadas.

**Objetivo.** Todo artefacto persistido debería poder registrar:

- versión de TalkingFaceKit y versión del schema;
- operación y configuración normalizada;
- backend, versión y dispositivo;
- nombre, versión, hash y licencia declarada del modelo;
- fingerprint de la fuente sin incluir la media completa;
- intervalo procesado;
- dependencias relevantes y seed cuando corresponda;
- transformaciones aplicadas y artefactos de origen.

Provenance no reemplaza una firma criptográfica ni garantiza reproducibilidad bit a bit entre
hardware distinto. Sí permite explicar cómo se obtuvo un resultado.

## Local-first y límites de efectos

El comportamiento por defecto debería ser local:

- sin uploads;
- sin descarga automática de pesos;
- sin telemetría;
- sin escritura al construir tipos del core;
- sin overwrite implícito;
- con archivos temporales en el directorio de destino y reemplazo atómico cuando sea posible.

Acceso a red, cámara, modelo remoto o servicio administrado puede existir como integración explícita
y opcional. Debe documentar credenciales, privacidad, costos, reintentos y semántica de errores.
