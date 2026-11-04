# Integración FLAME: investigación y propuesta

Actualizado: **2 de noviembre de 2026**. Estado: **preparación y encoder SMIRK implementados,
con ausencias explícitas por frame y persistencia de resultados**.
`iter_smirk_inputs` prepara los frames y `SmirkEncoder.track` estima parámetros FLAME por frame.
`FlameParameters` y `FlameResult` mantienen los datos faciales independientes del framework.
La geometría FLAME2020 ya se evalúa desde esos resultados. Validación de calidad para entrenamiento
y proyección sobre el video siguen pendientes. Las firmas futuras de
reconstrucción son ilustrativas. La dirección aceptada sigue en
[`PROJECT_DIRECTION.md`](../../PROJECT_DIRECTION.md).

## Punto de partida y recomendación

El objetivo es transformar segmentos de video en parámetros faciales con correspondencia temporal,
generar sus meshes e inspeccionar y guardar resultados reutilizables para entrenamiento.
El modelo consumidor y su formato de parámetros todavía no están confirmados.

**Propuesta:** probar primero SMIRK, obtener un recorrido completo y después incorporar PEAR mediante
otra integración que entregue el mismo contrato de datos donde las semánticas sean compatibles.
La recomendación responde al alcance facial de la primera entrega; todavía no hay una comparación
local de calidad o velocidad que permita elegir un ganador definitivo.

### Primer paso implementado y verificado

`integrations.smirk.iter_smirk_inputs` prepara arrays RGB float32 `(1, 3, 224, 224)` y conserva
índices, PTS y transformación fuente-recorte. Las pruebas cubren streaming, configuración de
MediaPipe, orden RGB, normalización, geometría y cierre de recursos. La comparación con warping real
está preparada pero se omite si scikit-image no está instalado.

scikit-image 0.22.0 ya está instalado y la comparación numérica del recorte pasa. El extra
`tracking-smirk` ahora incluye MediaPipe, scikit-image, PyTorch, torchvision y timm, con incorporación
de dependencias y descarga de pesos autorizadas por Matias.

El 1 de noviembre se probó la detección real sobre los frames 600–602 de
`Piers_Sellers_JSC_2016_interview.mp4`, con PTS 20.0, 20.033333 y 20.066667 segundos y el asset local
`models/face_landmarker.task`. MediaPipe 1.0.0 produjo 478 landmarks por frame con las opciones del
helper SMIRK. La ejecución necesitó acceso gráfico fuera del sandbox en macOS. Esto valida esa
frontera de detección, no la equivalencia con MediaPipe 0.10.10 del entorno upstream, el warping
completo ni una reconstrucción FLAME. Esa primera comprobación precedió a la incorporación de las
dependencias del encoder descrita a continuación.

### Encoder: evidencia de la prueba local

- Fuente fijada: `georgeretsi/smirk`, revisión `c7de404c4389f073906a6db1adabf62efcea3f35`.
- Checkpoint oficial descargado desde el enlace de `quick_install.sh`, guardado fuera de Git en
  `models/smirk/SMIRK_em1.pt` (140495895 bytes).
- SHA-256: `26b234e3cc31a5de226bcba4321bb5d7343a2ad48234c028b57a1c2f8c2d22d0`.
- Entorno comprobado: macOS Apple Silicon, CPU, PyTorch 2.14.1, torchvision 0.29.2, timm 0.9.16,
  MediaPipe 1.0.0 y NumPy 1.26.4. MPS y CUDA no están validados por esta prueba.
- Estado cargado con `weights_only=True`, 762 claves del encoder verificadas y carga estricta.
  Los pesos del generador incluidos en el archivo no se usan.
- Comparación con la clase upstream original sobre el mismo recorte real del frame 600 y el mismo
  estado: diferencia absoluta máxima **0.0** en pose, cámara, forma, expresión, párpados y mandíbula.
  Ambos modelos se construyeron sin adquirir pesos ImageNet antes de cargar el estado completo.
- La prueba opt-in de decoder → MediaPipe → crop → encoder pasó sobre el intervalo 20–20.1 s del
  video Piers, con parámetros finitos y timestamps dentro de la ventana fuente.
- En la primera versión, al ejecutar el ejemplo sobre 20–25 s produjo 103 predicciones para los frames 600–702
  (20.0–23.4 s) y se detuvo en el frame 703, PTS 23.433333 s, por una detección facial fallida.
  Era el comportamiento estricto de parada de la preparación oficial. El log y resumen históricos
  son `reports/smirk/piers-20-25.log` y `piers-20-25-summary.json`.

### Continuidad con ausencias explícitas

`iter_smirk_inputs` y `SmirkEncoder.track` ahora conservan una salida por frame. Si MediaPipe no
devuelve landmarks, emiten `SmirkMissingFrame` con índice fuente, PTS y motivo
`face_landmarks_unavailable`. `track` no ejecuta el encoder para esos frames y retoma en el siguiente.
La ausencia no distingue qué etapa interna de MediaPipe falló. Los errores de configuración,
decodificación, ejecución del detector, recorte o encoder siguen propagándose.

Las pruebas cubren huecos consecutivos, al inicio/final y en todo el intervalo, cierre temprano y
propagación de errores. El ejemplo imprime ausencias y cantidades estimadas/faltantes. No se borran
filas, inventan parámetros, interpolan huecos ni modifican los segmentos seleccionados. La política
de ventanas de entrenamiento queda separada: podrá conservar una máscara o requerir ventanas
continuas, según el consumidor, sin alterar la evidencia original.

El 2 de noviembre se repitió el recorrido real sobre Piers 20–25 s: **150 salidas, 148 estimaciones
y dos ausencias**, frames 703 (23.433333 s) y 704 (23.466667 s). Retomó en el frame 705 y terminó en
el 749 (24.966667 s). Todos los índices y PTS coincidieron exactamente con el decoder; todas las
estimaciones tuvieron parámetros finitos. El diagnóstico local está en
`reports/smirk/piers-20-25-continuity.json`; es evidencia de esta prueba, no un formato público de
persistencia de parámetros ni una validación de calidad geométrica.

### Resultado completo y persistencia

`SmirkEncoder.extract` ahora reúne el intervalo en un `FlameResult`, con arrays NumPy alineados,
máscara de estimaciones, motivos de ausencia y datos adicionales con convenciones explícitas.
Conserva fuente y metadata, hash del video, hashes del checkpoint y Face Landmarker, versiones de
paquetes, revisión upstream, dispositivo y configuración. La fuente se hashea al extraer, no al
guardar: ese costo adicional de lectura debe contarse al medir rendimiento.

`result.save(...)` y `FlameResult.load(...)` delegan en `io/flame.py`, que usa NPZ comprimido,
metadata JSON y esquema 1 sin pickle. La carga valida los contratos sin volver a ejecutar modelos
ni requerir los assets originales. Las ausencias usan NaN en todas sus filas numéricas, nunca ceros
o parámetros fabricados. Las pruebas verifican igualdad byte por byte, casos con todas/ninguna
estimación, carga en un proceso sin frameworks, archivos corruptos y publicación atómica.

La prueba real Piers 20–25 s generó `reports/smirk/piers-20-25.npz` (206552 bytes): 150 frames,
148 estimaciones y ausencias en 703/704. Tras cargarlo, todos los arrays principales y adicionales
coincidieron byte por byte, junto con motivos, convenciones, fuente y provenance. Los índices y PTS
coincidieron con una decodificación independiente. El resumen de verificación está en
`reports/smirk/piers-20-25-persistence.json`.

### Decoder geométrico FLAME2020

`integrations.flame.FlameDecoder` carga el modelo genérico oficial local y genera un `FaceMeshTrack`
desde `FlameResult`, sin volver a ejecutar SMIRK ni abrir el video. El cálculo implementa las
ecuaciones de blend shapes y articulación/skinning en NumPy. Los archivos fuente FLAME/LBS de
Max Planck no se incluyen ni se importan durante la ejecución de la biblioteca.

El asset descargado contiene una matriz Chumpy `Ch.x` y un regresor articular CSC. La frontera de
lectura reconoce esos contenedores como datos numéricos, valida las matrices y no necesita ejecutar
Chumpy/SciPy, modificar globals de NumPy ni instalar dependencias. No admite grafos Chumpy generales.
La lectura limitada de pickle sigue requiriendo assets confiables.

Se conserva forma por frame, expresión, pose global y mandibular. Cuello y rotación ocular son
neutrales explícitos. Los offsets de párpados SMIRK se suman después del skinning en el orden de
referencia; sus dos assets son obligatorios cuando se decodifica esa extensión. La salida mantiene
coordenadas/unidades nativas, PTS, índices y ausencias. La cámara del recorte no deforma la geometría.

Assets locales comprobados, fuera de Git:

- Modelo: `models/flame2020/generic_model.pkl`, SHA-256
  `efcd14cc4a69f3a3d9af8ded80146b5b6b50df3bd74cf69108213b144eba725b`.
- `models/smirk/l_eyelid.npy`, SHA-256
  `fa50997166c2f884fbfc577d99fe8707966763fc3d83a1870bb786df6cc9410d`.
- `models/smirk/r_eyelid.npy`, SHA-256
  `dd78776fed765be4468889dd7d5b54b4bf61c8efbb2869b5b91409ce798aad8d`.

La comparación local ejecutó el `forward` de la clase FLAME original del SMIRK fijado sobre las
148 estimaciones Piers guardadas, con las matrices originales y los mismos offsets de párpados.
Para cargar la referencia se desenvolvió únicamente la matriz `Ch.x`, sin cambiar sus valores; su
constructor y dependencias se ejecutaron en un proceso de diagnóstico separado. Diferencia máxima
por componente: **4.470348358154297e-08** en unidades nativas; RMS: **4.775617309399931e-09**.
Los triángulos coincidieron exactamente: **5023 vértices, 9976 triángulos**. El resultado conserva
los 150 frames, con los frames 703/704 enteramente NaN. Reporte local:
`reports/smirk/piers-flame-parity.json`.

`examples/flame_mesh.py` usa el renderer Plotly existente para generar
`reports/smirk/flame-mesh.html`. La animación permite inspección geométrica, con las limitaciones de
materialización y timing mediano del visor actual. La comparación numérica establece equivalencia
con la referencia, no precisión geométrica respecto de la persona ni elegibilidad para entrenar.

El renderer también acepta `source_video` para mostrar el original y el mesh en una sola página,
sin inferencia adicional. El video controla el tiempo: se seleccionan los meshes por sus PTS y se
ocultan las ausencias. Los controles permiten pausar y avanzar por frame. La cámara 3D sigue siendo
independiente de la cámara del recorte; esta vista no constituye una proyección sobre la imagen.
El ejemplo con `--with-video` genera `reports/smirk/flame-comparison.html`. Su exportación compacta
almacena una conectividad y vértices float32 binarios, conservando todos los frames originales.
Sobre Piers 20–25 s, el HTML generado ocupa 17377423 bytes frente a los 72938770 bytes de la
animación independiente. Se verificó que los vértices embebidos coinciden byte por byte, junto con
los 150 índices/PTS y los huecos 703/704. La lógica de reproducción se prueba con un reloj de video
simulado en Node; la reproducción visual real quedó pendiente porque el navegador integrado bloquea
URLs de archivos locales. El archivo puede abrirse localmente en un navegador y permite elegir el
video original si su carga directa falla. Ver `reports/smirk/flame-comparison-check.json`.

El diagnóstico de comparación está en el archivo local ignorado `reports/smirk/encoder-parity.json`.
Esta evidencia verifica compatibilidad y equivalencia del forward en esa entrada; no establece
calidad geométrica, estabilidad temporal o precisión para entrenamiento.

## Tres operaciones distintas

1. **Estimación:** un extractor recibe imágenes y predice parámetros.
2. **Evaluación del modelo FLAME:** los parámetros y un asset FLAME compatible producen vértices.
3. **Inspección:** un visor o renderer muestra el mesh y permite compararlo con la fuente.

FLAME representa forma, expresión y articulaciones de una cabeza. La estimación desde imágenes es
una capacidad del extractor elegido. Fuente: [modelo FLAME](https://flame.is.tue.mpg.de/).

Separar estas operaciones permite guardar parámetros sin generar todos los meshes ni renderizar
durante la inferencia. La vista 3D puede generarse después usando el modelo y las convenciones
registrados en el resultado.

## Evidencia sobre los candidatos

| Aspecto | SMIRK | PEAR |
| --- | --- | --- |
| Alcance publicado | Reconstrucción de expresiones faciales desde imágenes | Recuperación de cuerpo y cabeza mediante EHM-s, que combina SMPL-X y FLAME escalado |
| Salida facial observada | Forma, expresión, cabeza, mandíbula, párpados y cámara | Diccionario facial más resultados de cuerpo, escala y cámara |
| Integración inicial | Encoder separado del renderer | La ruta publicada de inferencia mezcla modelo, assets corporales y rendering |
| Hardware | CPU comprobado en esta Mac; MPS y CUDA por validar | La ruta inspeccionada contiene asignaciones explícitas a CUDA; portabilidad por comprobar |
| Papel propuesto | Primera prueba facial | Segunda integración y comparación con los mismos clips |

Fuentes: [SMIRK](https://github.com/georgeretsi/smirk),
[paper de PEAR](https://arxiv.org/abs/2601.22693),
[inferencia de PEAR](https://github.com/Pixel-Talk/PEAR/blob/main/inference_wo_detect.py).
PEAR también contempla recortes de cabeza o torso; su alcance corporal no implica que requiera una
persona de cuerpo entero. Fuente: [método PEAR](https://arxiv.org/html/2601.22693v2#S1).

### Hallazgos de implementación

- El encoder SMIRK inspeccionado tiene por defecto 300 componentes de forma y 50 de expresión,
  además de pose de cabeza, mandíbula, dos controles de párpados y cámara. Sus imports directos son
  PyTorch y timm. El constructor pide backbones preentrenados: hay que comprobar carga completa del
  checkpoint y evitar adquisición implícita de assets. Fuente:
  [encoder SMIRK](https://github.com/georgeretsi/smirk/blob/main/src/smirk_encoder.py).
- El renderer SMIRK usa PyTorch3D por separado. **Hipótesis a probar:** la estimación de parámetros
  puede ejecutarse sin el renderer upstream. Fuente:
  [renderer SMIRK](https://github.com/georgeretsi/smirk/blob/main/src/renderer/renderer.py).
- El evaluador geométrico SMIRK usa FLAME2020 por defecto y agrega deformaciones de párpados; fija
  cuello y ojos si no se suministran. La reconstrucción fiel requiere registrar esas extensiones y
  distinguir parámetros estimados de valores fijados. Fuente:
  [evaluador SMIRK](https://github.com/georgeretsi/smirk/blob/main/src/FLAME/FLAME.py).
- PEAR expone un bloque facial con forma, expresión, cabeza, mandíbula, ojos y párpados. La cámara y
  la escala de cabeza también forman parte del recorrido de reconstrucción. Una forma de array
  coincidente no demuestra equivalencia semántica con SMIRK. Fuente:
  [cabezal PEAR](https://github.com/Pixel-Talk/PEAR/blob/main/models/smplx/smplx_head.py).
- Los requisitos publicados de SMIRK fijan versiones distintas de nuestro entorno. Se necesita una
  prueba de compatibilidad con Python 3.11 y las dependencias del proyecto antes de decidir entre
  un extra opcional y una ejecución aislada administrada con uv. Fuente:
  [dependencias SMIRK](https://github.com/georgeretsi/smirk/blob/main/requirements.txt).
- SMIRK fue entrenado con imágenes individuales y declara limitaciones temporales, con oclusiones,
  poses extremas e iluminación difícil. Ejecutarlo frame a frame no garantiza estabilidad. Fuente:
  [limitaciones SMIRK](https://arxiv.org/html/2404.04104v2#S4.SS5).
- Los tiempos publicados de PEAR distinguen inferencia y animación y corresponden a GPUs NVIDIA
  concretas. Deben medirse de nuevo para nuestro recorrido completo y el hardware que se elija.
  Fuente: [experimentos PEAR, tabla 5](https://arxiv.org/html/2601.22693v2#S4.SS2).

Los repositorios se consultaron en `main`; antes de cualquier ejecución reproducible se fijarán
revisiones de código, versiones de dependencias y hashes de checkpoints y assets.

## Arquitectura propuesta

### Flujo de ejecución

```text
Fuente + intervalos + rostro objetivo
    → frames RGB con PTS fuente
    → preparación e inferencia específicas del extractor
    → parámetros y diagnóstico en contratos TalkingFaceKit
        → guardado y carga
        → evaluación FLAME compatible → FaceMeshTrack → inspección
```

El preprocesamiento es una operación separada. Una fuente ya recortada puede reconstruirse sin
repetir DeepTalk y SyncNet. Si se parte de una selección existente, deben reutilizarse el rostro
objetivo y la evidencia de tracking de ese análisis.

### Responsabilidades

| Pieza | Responsabilidad | Límites |
| --- | --- | --- |
| Contratos faciales core | Arrays NumPy, tiempos, validez, modelo y provenance | Sin PyTorch, OpenCV, lectura de assets ni ejecución de modelos |
| Operación de reconstrucción | Ordenar intervalos, asociar rostro objetivo y coordinar ejecución | Sin lógica particular de redes ni rendering obligatorio |
| Integración SMIRK | Preparar entrada, cargar modelo explícitamente, inferir y traducir resultados | Sin cambiar tiempos fuente ni exponer tensores upstream |
| Integración PEAR futura | Traducir su entrada y salidas al contrato compartido | Conservar diferencias de cámara, escala y capacidades |
| Integración geométrica FLAME | Cargar el modelo compatible y evaluar parámetros | No inferir parámetros desde imágenes |
| Persistencia | Guardar y cargar el resultado versionado | Sin inferencia ni descargas |
| Inspección | Visualizar parámetros o geometría existentes | Sin repetir extracción |

Están implementados `flame.py` para datos faciales, `integrations/smirk.py` para preparación y sesión,
y `integrations/_smirk_model.py` para la arquitectura del encoder. `SmirkParameterFrame` conserva
cámara, párpados y transformación del recorte como evidencia específica de la integración.
`FlameResult` es el resultado común; `io/flame.py` guarda y carga sus arrays y metadata. La coordinación
genérica y el evaluador geométrico se crean cuando su entrega los necesite. `integrations/pear.py`
se incorpora con su implementación real.

Se reutilizan `VideoSource`, `DecodedVideoFrame`, `FaceMeshTrack` y las fronteras de video existentes.
El visor Plotly actual admite topologías genéricas, pero materializa los frames y usa una duración
mediana de reproducción: sirve como candidato para una inspección breve, no demuestra sincronía A/V
exacta ni escalabilidad para videos largos.

Los contratos describen datos comunes desde el primer extractor. Un `Protocol` de backend se extrae
cuando existan implementaciones que lo requieran, siguiendo `AGENTS.md`; la prueba inicial no crea un
registro de plugins, factorías generales ni clases vacías para backends futuros.

### Contrato de datos a validar con la prueba

- Fuente, intervalo, ámbito del track y rostro objetivo. Los IDs siguen siendo locales al análisis.
- Índices de frames y PTS en segundos, manteniendo frames fallidos dentro de la línea temporal.
- Forma, expresión y articulaciones con shapes, dtype, unidades y representación de rotaciones
  explícitos. Las dimensiones dependen del layout y del modelo, no de constantes universales.
- Máscaras de validez y razones de fallo. Una predicción finita no certifica exactitud.
- Campos estimados, fijados o no disponibles: ausencia de estimación de ojos/cuello no equivale a
  medir que su pose sea cero.
- Cámara y transformaciones entre imagen fuente y entrada del extractor. Modelos de cámara distintos
  conservan contratos distintos; no se fuerza una conversión sin validación.
- Variante y versión FLAME, base de parámetros, hash del modelo y extensiones como párpados.
- Extractor, revisión, checkpoint, configuración, dispositivo y transformaciones aplicadas.

La identidad debería mantenerse estable dentro de un track continuo. Primero se conservan las
estimaciones originales; fijar o agregar forma entre frames es una política posterior explícita.
El suavizado también debe preservar los resultados originales y medirse por su efecto en labios,
expresión y alineación. No se aplica entre cortes o identidades diferentes.

## Experiencia pública deseada

Ejemplo ilustrativo para una fuente conocida, con modelos preparados explícitamente:

```python
clip = TalkingFaceSequence.from_video("video.mp4").clip(20.0, 25.0)
result = clip.reconstruct(backend="smirk")
result.save("outputs/face.npz")
```

`reconstruct` todavía no existe. Se evalúa como nombre de una única operación pública, con un
resultado común para ambos extractores. La entrada de un clip sin tracking debe tener una regla
explícita de rostro único u objetivo seleccionado.

Para el recorrido con selección previa, una operación sobre `VideoAnalysisResult` podría reconstruir
sus segmentos retenidos conservando los targets. Ambas entradas deben delegar en la misma
coordinación. La configuración avanzada de modelos y dispositivo permanece explícita, sin exigir
al usuario instanciar todas las piezas internas.

La elección de backend cambia la integración; la lectura de tiempos, parámetros, validez y el
guardado conserva la misma estructura. La cámara, capacidades o extensiones particulares siguen
identificadas. El soporte de ambos no promete resultados numéricamente equivalentes.

## Rendimiento e identidad: dos riesgos concretos del código actual

1. `stream_video_frames` decodifica desde el principio y descarta frames anteriores al intervalo.
   Repetirlo para cada segmento volvería a recorrer prefijos del archivo. Para varios segmentos,
   comenzar con un recorrido ordenado de la fuente; evaluar seek después con sus garantías de PTS e
   índices. Un segundo pase para FLAME puede ser razonable para conservar memoria acotada.
2. `TalkingFaceSegment.to_sequence()` conserva fuente e intervalo, pero no lleva `face_id` ni las
   observaciones del análisis. Esa conversión por sí sola es insuficiente para preservar el rostro
   objetivo en una integración que reutilice tracking.

La preparación de entrada pertenece al backend: el recorte facial SMIRK y la preparación PEAR no
tienen por qué coincidir. La primera prueba reproduce el preprocesamiento esperado por SMIRK;
reutilizar bounding boxes existentes para ahorrar detección requiere una comparación posterior.

Se cargan modelos una vez por sesión de reconstrucción, se usan lotes acotados donde sea posible y
se mide por separado estimación, geometría, render, I/O y carga inicial. No se hereda automáticamente
la grilla interna de 25 Hz de DeepTalk como frecuencia de los parámetros faciales.

## Primer experimento y criterio de avance

1. Elegir un clip fuente de unos cinco segundos con un rostro claro, conservar sus PTS y fijar las
   revisiones de código y modelos. El hardware se decide al preparar la ejecución.
2. Reproducir la preparación de entrada y extraer parámetros SMIRK en una ejecución acotada. Comprobar
   compatibilidad, carga completa de pesos, operación sin descargas implícitas y dimensiones reales.
3. Guardar parámetros, tiempos, validez y provenance; recargar y verificar igualdad de los datos.
4. Evaluarlos con el modelo FLAME compatible, incluidas las extensiones usadas, y comparar el mesh
   animado con el video original. Registrar costos por etapa.
5. Repetir sobre vista de tres cuartos, boca cerrada/abierta y un fallo visible de detección u
   oclusión. Registrar diferencias y estabilidad temporal sin ocultarlas con suavizado inicial.
6. Ajustar el contrato con las salidas reales y probar PEAR sobre las mismas fuentes antes de
   estabilizar la interfaz compartida.

El experimento debe entregar parámetros recargables, geometría inspeccionable, correspondencia
temporal y mediciones. Los tests normales usarán arrays y fronteras simuladas para validar contratos,
identidad, fallos y persistencia; las pruebas con modelos reales serán opt-in.

## Decisiones abiertas

- Plataformas que debe soportar la entrega y validación en hardware adicional al CPU local.
- Modelo consumidor, variante FLAME y layout que espera para entrenar.
- Compatibilidad del extra opcional en otras plataformas y futuras versiones de dependencias.
- Reglas de identidad estable, fallos y eventual suavizado, evaluadas con video.
- Proyección de la geometría con cámara del recorte y comparación visual sincronizada con el video.

El siguiente paso es proyectar la geometría sobre el video y evaluar calidad/estabilidad temporal
en ejemplos variados. La primera integración usa assets locales explícitos; no instala una demo ni
descarga modelos durante inferencia.
