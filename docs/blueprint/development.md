# Guía de desarrollo orientada por el blueprint

Las reglas normativas del repositorio están en [`AGENTS.md`](../../AGENTS.md). Esta página explica
cómo usar la documentación futura para elegir y cerrar una implementación.

## Antes de empezar

1. Leer `README.md`, `AGENTS.md` y `docs/architecture.md` completos.
2. Identificar el slice del [roadmap](roadmap.md), no sólo una clase del diagrama futuro.
3. Inspeccionar código y tests relacionados.
4. Revisar cambios locales y preservar trabajo no relacionado.
5. Acordar cualquier dependencia runtime nueva con el equipo.
6. Escribir el resultado de usuario y los datos que cruzan cada boundary.

No crear branches, commits, pushes ni cambios remotos salvo pedido explícito.

## De blueprint a feature

### 1. Elegir una historia vertical

Buena historia:

> Dado un WebM con audio mono, entregar chunks `float32` con sample rate, canales y timestamp de
> origen, sin materializar el archivo completo.

Historia demasiado horizontal:

> Crear `audio/`, `speech/`, `pipeline/`, protocolos y factories para uso futuro.

La primera produce valor, tests y decisiones reales. La segunda genera capas vacías.

### 2. Fijar el contrato antes del adapter

Para cada array responder:

- ¿qué significa cada eje?;
- ¿shape exacto y dimensiones vacías permitidas?;
- ¿dtype?;
- ¿unidad y rango?;
- ¿orden de canales o coordenadas?;
- ¿timeline y time base?;
- ¿cómo se marca ausencia, padding o interpolación?;
- ¿se copia o se retiene?;
- ¿quién puede mutarlo?;
- ¿qué provenance necesita?;
- ¿qué excepciones distinguen inputs inválidos de fallos externos?

Actualizar `docs/architecture.md` en la misma entrega que acepta el contrato.

### 3. Escribir un test core mínimo

Construir arrays sintéticos diminutos. Testear comportamiento público e invariantes importantes:

- caso válido;
- dtype incorrecto;
- shape incorrecta;
- timeline no creciente;
- valor no finito;
- máscara inconsistente;
- límite vacío/positivo;
- no mutación o no overwrite cuando corresponda.

### 4. Aislar la integración

El adapter traduce el framework hacia el contrato. Si la dependencia no está tipada:

- usar `Protocol` privado para la pequeña superficie consumida;
- usar `cast` en el boundary;
- evitar `Any` en el resto del paquete;
- mantener un ignore estrecho y explicado sólo si no hay alternativa.

El core nunca recibe objetos externos.

### 5. Cerrar el flujo

La feature no está completa con el dataclass. Agregar el camino mínimo:

```text
fuente -> integración -> resultado validado -> API Python -> persistencia/CLI si aplica
```

Un error antes del final no deja estado adjunto o archivo final parcial.

### 6. Documentar realidad, no intención

Al implementar:

- mover la capacidad correspondiente de objetivo a disponible en el blueprint;
- agregar uso real y limitaciones al README;
- actualizar arquitectura con las decisiones aceptadas;
- escribir docstrings NumPy para API pública;
- no documentar flags o backends que no existen en la referencia actual.

El blueprint puede seguir mostrando el futuro, siempre con estado claro.

## Plantilla de diseño de feature

```markdown
# Nombre de la entrega

## Resultado de usuario
Qué puede hacer al terminar.

## En alcance
- flujo vertical mínimo;
- API Python;
- CLI/persistencia si corresponde.

## Fuera de alcance
- extensiones deliberadamente postergadas.

## Contratos
- tipos, shapes, dtypes, unidades, rangos, tiempo y faltantes.

## Boundaries
- filesystem, decoder, framework, modelo, red o renderer.

## Failure modes
- errores de input, backend y output; semántica transaccional.

## Tests
- core, integración fake/real marcada y regresiones.

## Documentación
- README, arquitectura, blueprint y ejemplos afectados.

## Dependencias
- ninguna, o decisión explícita del equipo.
```

## Testing por capa

### Core

- rápido y determinista;
- arrays sintéticos pequeños;
- sin filesystem salvo validaciones puntuales;
- sin red, modelos o hardware especial;
- prueba contratos, no implementación privada.

### Boundary de archivos

- `tmp_path`;
- extensión, parent ausente, directorio, archivo irregular;
- overwrite requerido;
- limpieza de temporales;
- corrupción/schema futuro;
- round-trip sin pérdida.

### Adapter opcional

- fake de la superficie externa;
- import ausente con mensaje de instalación;
- configuración inválida antes de cargar un modelo caro;
- cierre de recursos al éxito y al fallar;
- traducción exacta de dtype/shape/timestamp;
- integration test real marcado y sin descarga automática.

### Secuencia/application

- delegación de path e intervalo;
- attachment sólo después de éxito;
- nombres y overwrite;
- no cambio del resultado anterior ante fallo;
- múltiples configuraciones coexistentes.

### CLI

- parseo y exit code;
- output estable necesario para humanos o `--json`;
- stderr para errores;
- archivo final y summary;
- ningún stack trace para errores esperables de usuario.

## Fixtures

- Preferir media sintética breve y documentada.
- Mantener fixtures suficientemente chicos para Git y CI.
- No incluir caras o voces reales sin licencia/consentimiento claro.
- No incluir modelos o checkpoints.
- No descargar datasets en tests.
- Separar benchmarks externos de pytest.

## Dependencias

El repositorio usa `uv` exclusivamente:

```bash
uv add package-name
uv add --optional extra-name package-name
uv remove package-name
```

Pero primero hay que pedir acuerdo al equipo para cualquier runtime dependency. Al evaluar una:

- licencia;
- soporte Python 3.11;
- macOS/Windows y arquitecturas del equipo;
- tamaño de wheel/modelo;
- CPU/GPU/CUDA/MPS;
- tipos publicados;
- actividad/mantenimiento;
- vulnerabilidades y formatos inseguros;
- si una implementación pequeña con dependencias actuales alcanza.

Si cambia una dependencia, `pyproject.toml` y `uv.lock` cambian juntos; nunca editar el lockfile a
mano.

## Diseño de API

Antes de hacer público un símbolo:

- ¿representa un concepto backend-independent?;
- ¿el nombre seguirá teniendo sentido con un segundo backend?;
- ¿los defaults evitan conversión o selección silenciosa?;
- ¿Path, tiempo y unidades están explícitos?;
- ¿se puede validar completamente al construir?;
- ¿la operación costosa parece operación y no propiedad?;
- ¿el resultado puede persistirse sin objetos externos?;
- ¿el error permite al caller actuar?;
- ¿la firma deja lugar a compatibilidad sin aceptar `**kwargs` sin tipar?;

Preferir funciones y dataclasses cohesionados. No agregar base classes, registries o factories sólo
para anticipar el roadmap.

## Docstrings y documentación

Toda API pública usa docstring NumPy e incluye, según corresponda:

- significado, no repetición de firma;
- units y coordinate system;
- array shapes/dtypes/ranges/channel order/time axis;
- lazy versus eager;
- copies y mutabilidad;
- side effects y recursos;
- semántica de intervalo;
- valores faltantes;
- raises;
- limitaciones del modelo.

Los ejemplos deben ser ejecutables con la versión que dicen documentar. Los ejemplos objetivo viven
en `docs/blueprint` y conservan la etiqueta correspondiente.

## Persistencia y compatibilidad

Al crear o cambiar un schema:

1. versionarlo independientemente del paquete;
2. documentar required/optional fields;
3. desactivar deserialización insegura;
4. validar antes de construir el resultado;
5. copiar arrays fuera del handle si el contrato lo requiere;
6. rechazar versiones futuras desconocidas;
7. escribir temporal y publicar al final;
8. requerir overwrite;
9. agregar fixture/round-trip y casos corruptos;
10. definir migración o declarar incompatibilidad.

## Performance

No optimizar sólo por intuición. Registrar:

- duración de fuente;
- resolución/sample rate;
- frames o samples procesados;
- real-time factor;
- memoria pico;
- backend/device/dtype;
- tamaño de artefacto;
- costo de startup del modelo.

Una optimización no puede cambiar timestamps, dtype, orden de canales o faltantes sin ser una API
distinta.

## Revisión de privacidad y modelos

Para features de identidad, voz, transcripción, emoción o remote processing:

- describir qué datos se procesan y dónde;
- conservar consentimiento/licencia fuera del core si pertenece al dataset;
- no loguear contenido sensible por defecto;
- no persistir embeddings biométricos por accidente;
- documentar incertidumbre y usos no apropiados;
- hacer explícito cualquier upload;
- revisar licencia y riesgos de los pesos.

## Checks obligatorios

Después de todo cambio significativo:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest
```

Un cambio no está completo si uno falla. En un cambio sólo Markdown, los mismos checks verifican que
no se haya entregado sobre una base rota o interferido con trabajo simultáneo.

## Checklist de entrega

- [ ] El scope corresponde a un slice del roadmap.
- [ ] No hay cambios ajenos o destructivos.
- [ ] No se agregó dependencia sin acuerdo.
- [ ] Core y adapter dependen en la dirección correcta.
- [ ] Contratos completos y validados.
- [ ] Timestamps/unidades/conversiones explícitos.
- [ ] Fallos no publican estado parcial.
- [ ] Tests deterministas y regresiones agregadas.
- [ ] Docstrings, README y arquitectura actualizados.
- [ ] Blueprint actualizado de objetivo a disponible.
- [ ] Ruff format/check, mypy y pytest pasan.
- [ ] Limitaciones y trabajo futuro quedan visibles.
