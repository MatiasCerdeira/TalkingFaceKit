# TalkingFaceKit

TalkingFaceKit es una biblioteca modular en Python para procesar videos de personas hablando. El objetivo es representar y procesar de forma reutilizable video, audio, tracking facial y metadata.

## Regla principal

El proyecto usa **Python 3.11** y **uv**. No usamos `pip`, Conda ni entornos creados manualmente para este repositorio.

`uv` crea una `.venv` local y se asegura de que todos usemos las mismas versiones de Python y de las librerías.

## Preparación inicial

Esto se hace una sola vez por computadora.

### 1. Instalar Git

Descargar Git desde [git-scm.com](https://git-scm.com/downloads) si todavía no está instalado.

### 2. Instalar uv

macOS y Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Después de instalarlo, cerrar y volver a abrir la terminal.

### 3. Clonar y preparar el proyecto

```bash
git clone https://github.com/MatiasCerdeira/TalkingFaceKit.git
cd TalkingFaceKit
uv sync
uv run pytest
uv run mypy
```

`uv sync` crea `.venv`, obtiene Python 3.11 si hace falta e instala las dependencias del proyecto.

## Cómo empezar a trabajar cada día

Antes de modificar archivos:

```bash
git switch main
git pull --ff-only
uv sync
```

Esto trae los últimos cambios del equipo y actualiza las librerías locales.

## Cómo ejecutar Python

No hace falta activar `.venv`. Usar `uv run` delante del comando:

```bash
uv run python mi_script.py
uv run python
uv run pytest
uv run ruff check .
```

Por ejemplo, en lugar de:

```bash
python main.py
```

usar:

```bash
uv run python main.py
```

## Cómo agregar o eliminar librerías

Agregar una librería necesaria para el proyecto:

```bash
uv add numpy
```

Agregar una herramienta usada solamente para desarrollar:

```bash
uv add --dev nombre-de-la-herramienta
```

Eliminar una librería:

```bash
uv remove numpy
```

`uv add` y `uv remove` actualizan `pyproject.toml` y `uv.lock`. Estos dos archivos deben subirse a Git.

No usar:

```bash
pip install nombre-de-la-libreria
conda install nombre-de-la-libreria
```

Antes de agregar dependencias grandes como PyTorch, modelos de tracking o librerías con CUDA, hablarlo con el equipo. Pueden necesitar configuraciones diferentes en macOS y Windows.

## Estilo, tipado y arquitectura

El código, los identificadores, comentarios, docstrings y mensajes de error se escriben en inglés.

El proyecto usa:

- **Ruff** para formato, imports y errores comunes.
- **mypy en modo estricto** para verificar type hints.
- **pytest** para verificar comportamiento.
- Docstrings estilo **NumPy** para la API pública.

Las reglas completas están en [`AGENTS.md`](AGENTS.md). Las decisiones de diseño están en
[`docs/architecture.md`](docs/architecture.md).

Codex lee `AGENTS.md` directamente. Claude Code lee [`CLAUDE.md`](CLAUDE.md), que importa las
mismas reglas. Cuando el equipo cambia una regla o decisión arquitectónica, debe actualizar esos
documentos en el mismo commit que el código afectado.

Reglas básicas de tipado:

- Tipar parámetros y retornos de todas las funciones de producción.
- Usar sintaxis de Python 3.11 como `list[str]` y `Value | None`.
- Evitar `Any` y no desactivar mypy globalmente por una librería problemática.
- Documentar y validar shape, dtype, unidades, rangos y ejes de los arrays NumPy.
- Usar `pathlib.Path` para rutas públicas del filesystem.

## Cómo subir cambios

Primero revisar y probar:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest
git status
```

Si todo funciona:

```bash
git add .
git commit -m "descripción corta del cambio"
git pull --rebase
git push
```

Ejemplos de mensajes de commit:

```text
Add video loading function
Fix audio duration validation
Add numpy dependency
Update setup instructions
```

## Cómo recibir cambios de otra persona

```bash
git pull --ff-only
uv sync
```

Siempre ejecutar `uv sync` después de un pull que haya modificado `pyproject.toml` o `uv.lock`.

## Reglas para no romper el proyecto

- Avisar al equipo qué archivos o funcionalidad está tocando cada uno.
- Hacer `git pull --ff-only` antes de empezar a trabajar.
- Hacer commits chicos y con mensajes claros.
- Ejecutar Ruff, mypy y pytest antes de cada push.
- No usar `git push --force` sobre `main`.
- No editar manualmente `uv.lock`.
- No subir `.venv`, datasets, videos generados, modelos, outputs ni archivos `.env`.
- No guardar contraseñas, tokens o credenciales en el repositorio.
- No instalar dependencias del proyecto con `pip` o Conda.
- Si aparece un conflicto de Git y no está claro cómo resolverlo, no adivinar: hablar con la persona que modificó ese archivo.

## Archivos importantes

```text
.python-version    Versión de Python usada por el proyecto
pyproject.toml     Configuración y lista de dependencias
uv.lock            Versiones exactas resueltas por uv
AGENTS.md           Reglas compartidas para humanos y agentes
CLAUDE.md           Importa AGENTS.md para Claude Code
docs/               Decisiones de arquitectura
src/               Código de TalkingFaceKit
tests/             Tests automáticos
.venv/             Entorno local; nunca se sube a Git
```

## Comprobación rápida

Si hay dudas sobre el entorno:

```bash
uv run python --version
uv run ruff check .
uv run mypy
uv run pytest
```

Python debe mostrar una versión `3.11.x` y los tests deben terminar correctamente.
