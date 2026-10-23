"""Read source-code provenance without requiring a Git checkout at runtime."""

import subprocess
from pathlib import Path


def code_revision() -> dict[str, str | bool | None]:
    """Read the installed checkout's commit and dirty state when Git metadata is available.

    Returns
    -------
    dict
        ``commit`` and ``dirty`` values, or ``None`` values for installed distributions without
        accessible Git metadata. No repository state is changed.
    """
    repository = Path(__file__).resolve().parents[3]
    if not (repository / ".git").exists():
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {"commit": None, "dirty": None}
    if commit.returncode != 0 or status.returncode != 0:
        return {"commit": None, "dirty": None}
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}
