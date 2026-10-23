"""Filesystem discovery for lightweight video collections."""

from pathlib import Path

_VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v", ".mpeg", ".mpg", ".mts", ".m2ts"}
)


def discover_videos(folder: str | Path, *, recursive: bool = False) -> tuple[Path, ...]:
    """Find regular video files without inspecting their contents.

    Parameters
    ----------
    folder
        Existing directory; directory symlinks are not traversed recursively.
    recursive
        Include nested files when true; suffix matching is case-insensitive.

    Returns
    -------
    tuple of pathlib.Path
        Absolute paths ordered by relative POSIX path, including a stable case-sensitive tie break.

    Raises
    ------
    FileNotFoundError, NotADirectoryError
        If the requested directory does not exist or is not a directory.
    """
    root = Path(folder).absolute()
    if not root.exists():
        raise FileNotFoundError(f"video folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"video folder is not a directory: {root}")
    candidates = root.rglob("*") if recursive else root.iterdir()
    return tuple(
        sorted(
            (
                path
                for path in candidates
                if path.suffix.lower() in _VIDEO_EXTENSIONS and path.is_file()
            ),
            key=lambda path: (
                path.relative_to(root).as_posix().casefold(),
                path.relative_to(root).as_posix(),
            ),
        )
    )
