# Integraciones, backends y dependencias opcionales

## Regla de frontera

El core contiene dataclasses, protocolos necesarios y operaciones NumPy. Una integración puede
depender del core; el core no importa objetos de PyAV, OpenCV, MediaPipe, PyTorch, un recognizer de
voz o un renderer.

Un adapter es responsable de:

- validar su configuración y assets;
- traducir inputs core al formato del framework;
- ejecutar side effects o inferencia;
- traducir outputs a un contrato core completo;
- cerrar recursos aun cuando falle;
- exponer nombre/versión/configuración para provenance;
- no adjuntar resultados a la secuencia por su cuenta.

## Matriz de dependencias

| Capacidad | Estado | Dependencia candidata | Extra propuesto |
| --- | --- | --- | --- |
| inspección/decode de video | disponible | PyAV | core |
| landmarks 478 | disponible | MediaPipe | `tracking-mediapipe` |
| mesh HTML | disponible | Plotly | `rendering` |
| audio decode/resample | objetivo | primero PyAV; adapter especializado si hace falta | core o `audio` por decidir |
| VAD | objetivo | backend local liviano a evaluar | `speech` |
| ASR/aligner | objetivo | uno o más backends opcionales | `speech-*` |
| FLAME fitting | investigación | PyTorch + implementación FLAME | `flame` |
| overlay de video | objetivo | PyAV primero; FFmpeg CLI sólo si aporta valor | `rendering-video` |
| glTF | objetivo | writer liviano o implementación propia acotada | `export-gltf` |
| ejecución remota | investigación | cliente por servicio | extra por servicio |

No se agrega una dependencia runtime sin discutirla con el equipo. En especial PyTorch, CUDA,
modelos, video y audio pueden tener diferencias de plataforma importantes.

## PyAV — disponible

PyAV vive bajo `talkingfacekit.io`. Es dueño de:

- abrir el contenedor;
- seleccionar el primer stream actual;
- inspeccionar metadata;
- decodificar frames;
- convertir explícitamente a RGB;
- conservar índices y timestamps;
- traducir errores FFmpeg a mensajes de boundary.

Trackers no deben abrir contenedores por separado. Audio debería extender esta misma frontera o un
módulo hermano para compartir selección de fuente y semántica temporal.

## MediaPipe — disponible

`MediaPipeFaceTracker` importa MediaPipe en runtime para mantenerlo opcional. El usuario proporciona
el `.task`. El adapter:

- configura modo video y un rostro;
- transforma segundos a milisegundos estrictamente crecientes;
- devuelve 478 landmarks `float32`;
- mantiene frames faltantes;
- no conserva pixels;
- cierra el landmarker mediante context manager.

La topología para el mesh también se obtiene de MediaPipe. Si en el futuro se desea construir mesh
sin instalar MediaPipe, la conectividad versionada debería convertirse en un asset pequeño propio,
con revisión de licencia y tests de equivalencia.

## Plotly — disponible

El renderer recibe sólo `FaceMeshTrack` y escribe HTML autocontenido. Plotly no aparece en los tipos
del core. La duración visual usa la mediana de intervalos porque Plotly aplica una duración global;
los timestamps originales siguen visibles.

## OpenCV — objetivo sólo si hay un caso concreto

OpenCV puede aportar estimación de pose, transformaciones, drawing o algunos codecs, pero no debe
convertirse en la frontera de video paralela. Si se agrega:

- recibe RGB core y convierte a BGR explícitamente dentro del adapter;
- no abre el video si el frame stream ya lo resuelve;
- devuelve NumPy y tipos core;
- permanece extra opcional si su peso no justifica hacerlo runtime.

## PyTorch y dispositivos — investigación

Los adapters de modelos deben aceptar dispositivo y precisión explícitos:

```python
BackendConfig(device="cpu", dtype="float32")
BackendConfig(device="cuda:0", dtype="float32")
BackendConfig(device="mps", dtype="float32")
```

Reglas:

- no seleccionar GPU silenciosamente;
- validar disponibilidad antes de iniciar un trabajo largo;
- no almacenar `torch.Tensor` en resultados core;
- transferir a CPU y NumPy en el boundary;
- documentar determinismo y diferencias de precisión;
- liberar recursos según el ciclo de vida del adapter;
- no asumir CUDA en tests unitarios.

## Backends de habla — objetivo

La primera entrega debería evaluar un caso pequeño y local. Criterios:

- Python 3.11 y plataformas del equipo;
- tamaño de la dependencia y del modelo;
- disponibilidad de timestamps y confidence;
- idiomas soportados;
- licencia del código y pesos;
- ejecución CPU/GPU;
- posibilidad de tests sin descargar assets.

ASR, forced alignment y VAD son capacidades distintas. Un único paquete puede implementarlas, pero
los contratos no deben mezclarse.

## Backends remotos — investigación

Un servicio remoto es una integración explícita con riesgos propios:

- el usuario sabe que la media sale del equipo;
- las credenciales vienen de una fuente segura y no se persisten;
- timeout, retry y rate limit son configurables;
- los costos y límites se documentan;
- se registra endpoint lógico y versión, no secretos;
- los errores remotos no dejan artefactos parciales;
- una alternativa local sigue siendo posible cuando el producto lo requiera.

## Assets de modelos

TalkingFaceKit no incluye ni descarga modelos automáticamente. Política objetivo:

1. el caller entrega `Path` o una referencia de asset explícita;
2. el adapter valida existencia, tipo y compatibilidad básica;
3. provenance puede guardar SHA-256, nombre y licencia declarada;
4. los pesos, checkpoints y modelos permanecen fuera de Git;
5. un futuro comando `assets fetch` debe requerir acción explícita, mostrar origen/licencia y
   verificar checksum;
6. los tests reemplazan backends o usan fixtures mínimos autorizados; nunca bajan pesos.

## Descubrimiento de backends

No se propone un plugin system general. Mientras haya pocas integraciones, imports explícitos y una
tabla CLI mantenida son más simples. Un registro dinámico sólo se justifica si terceros realmente
necesitan distribuir adapters independientes y existen contratos estables.

Antes de un registro público deben resolverse:

- identificación y versionado;
- configuración tipada;
- declaración de capacidades y dispositivos;
- errores de carga;
- seguridad de entry points;
- compatibilidad de schemas;
- tests de conformidad.

## Ciclo de vida y concurrencia

Backends con recursos caros deberían soportar context manager o un método `close` claro. Un objeto
puede reutilizar un modelo entre secuencias sólo si documenta thread safety y estado temporal.

- No asumir que un backend es thread-safe.
- No compartir un tracker temporal entre videos sin reset explícito.
- Los workers de batch crean o reciben backends según una política documentada.
- GPU memory, procesos y handles se liberan ante éxito o fallo.
- Cancelación no publica resultados incompletos como válidos.

## Errores de integración

Un adapter debería distinguir en sus mensajes:

- dependencia no instalada;
- modelo ausente o incompatible;
- dispositivo no disponible;
- fuente inválida;
- output del backend que viola el contrato;
- fallo de inferencia;
- recurso agotado.

No debe capturar `BaseException` salvo para limpiar temporales y volver a elevar, como hacen los
writers transaccionales actuales.
