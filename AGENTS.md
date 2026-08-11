# TalkingFaceKit Agent Instructions

These instructions apply to every coding agent working in this repository. Keep this file concise,
concrete, and current. Human-facing setup instructions live in `README.md`; longer design decisions
live in `docs/architecture.md`.

## Before changing code

- Read `README.md`, this file, and `docs/architecture.md`.
- Inspect the relevant code and tests before proposing a change.
- Preserve unrelated user changes and keep the implementation scoped to the request.
- Do not create branches, commit, push, or modify remote state unless explicitly requested.
- Ask before adding or removing a runtime dependency, especially PyTorch, CUDA, model, video, or
  audio dependencies.

## Environment and dependencies

- The project targets Python 3.11 and uses `uv` exclusively.
- Use `uv sync`, `uv add`, and `uv remove`; do not use `pip` or Conda for project dependencies.
- Commit both `pyproject.toml` and `uv.lock` when dependencies change.
- Never commit `.venv`, credentials, datasets, generated media, model weights, checkpoints, or
  local outputs.

## Required checks

Run these after every meaningful code change:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest
```

Do not claim completion when a required check fails. Report the failure and its cause.

## Language and style

- Write code identifiers, comments, docstrings, exceptions, and commit messages in English.
- Follow Ruff formatting with a 100-character line length.
- Use `snake_case` for modules, functions, methods, and variables.
- Use `PascalCase` for classes and `UPPER_SNAKE_CASE` for constants.
- Prefer `pathlib.Path` over raw path strings at filesystem boundaries.
- Prefer small cohesive functions and explicit data flow over global state or hidden mutation.
- Avoid speculative abstractions, empty architectural layers, and placeholder implementations.
- Use absolute imports from `talkingfacekit` except where a local relative import is clearer inside a
  package initializer.

## Typing

- Type all production function and method parameters and return values.
- Type public attributes and non-obvious local variables.
- Use Python 3.11 syntax such as `list[str]` and `Value | None`.
- Avoid `Any`. Isolate unavoidable untyped third-party behavior in adapter modules.
- Use a narrow `# type: ignore[error-code]` only when no typed alternative exists, and explain why.
- Do not silence mypy errors globally to accommodate one library.
- Use `Protocol` for replaceable backends when multiple implementations actually exist.
- Use `numpy.typing.NDArray` aliases for arrays. Document and validate shape, dtype, units, value
  range, channel order, and time axis at boundaries; type hints alone cannot express all of them.

## Documentation

- Public modules, classes, functions, and methods require NumPy-style docstrings.
- Docstrings describe meaning, units, shapes, ranges, side effects, and raised exceptions. Do not
  merely repeat the signature.
- Private helpers need docstrings only when their behavior is not obvious.
- Update README usage when public behavior changes.
- Update `docs/architecture.md` in the same change as a new architectural decision or changed
  invariant.

## Architecture

- Keep the core data model independent from OpenCV, FFmpeg wrappers, PyTorch, trackers, and speech
  models.
- Put file access, subprocesses, model loading, network access, and framework-specific conversions
  at integration boundaries.
- Depend inward: integrations may depend on core types; core types must not depend on integrations.
- Prefer a small end-to-end feature over a generalized plugin system designed in advance.
- Do not silently convert FPS, sample rate, color order, dtype, shape, units, or timestamps.
- Keep public APIs backend-agnostic unless the API explicitly represents a backend.

## Testing

- Add or update tests for every behavior change. Add a regression test before or with a bug fix.
- Name test files `test_*.py` and test functions `test_*`.
- Keep unit tests deterministic, fast, and independent of network access and large external assets.
- Use tiny synthetic arrays or explicitly documented fixtures instead of real datasets.
- Test public behavior and important invariants rather than private implementation details.
- Mark integration tests clearly when they require external programs, models, or optional packages.

## Updating these instructions

- Treat this file as versioned project policy, not personal preference.
- Change a rule when the team changes its practice, and explain the reason in the commit message.
- Keep `CLAUDE.md` as a thin import of this file so Codex and Claude Code receive the same policy.
- Use nested instruction files only if a future subpackage genuinely needs different rules.
