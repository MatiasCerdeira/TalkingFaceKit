import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_flame_result import make_result

from talkingfacekit.flame import FlameResult


@pytest.mark.parametrize("mode", ["mixed", "all_missing", "all_estimated"])
def test_round_trip_preserves_every_array_reason_and_metadata(tmp_path: Path, mode: str) -> None:
    expected = make_result(all_missing=mode == "all_missing", all_estimated=mode == "all_estimated")
    expected = replace(
        expected, end_seconds=None, source=replace(expected.source, path=Path("/不存在/video.mp4"))
    )
    archive = expected.save(tmp_path / "created" / "parameters.npz")
    loaded = FlameResult.load(archive)
    assert loaded.source == expected.source
    assert loaded.source_sha256 == expected.source_sha256
    assert loaded.start_seconds == expected.start_seconds
    assert loaded.end_seconds is None
    assert loaded.model_variant == expected.model_variant
    assert loaded.provenance == expected.provenance
    assert loaded.extra_conventions == expected.extra_conventions
    assert loaded.missing_reasons == expected.missing_reasons
    for name in (
        "frame_indices",
        "timestamps_seconds",
        "estimated",
        "shape",
        "expression",
        "global_pose",
        "jaw_pose",
    ):
        original, restored = getattr(expected, name), getattr(loaded, name)
        assert restored.dtype == original.dtype
        assert restored.tobytes() == original.tobytes()
        assert not np.shares_memory(restored, original)
    for name, original in expected.extras.items():
        assert loaded.extras[name].dtype == original.dtype
        assert loaded.extras[name].tobytes() == original.tobytes()
    archive.unlink()
    assert loaded.frame_count == 4


def test_overwrite_is_explicit_and_failed_writes_keep_existing_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_result().save(tmp_path / "result.npz")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        make_result(all_missing=True).save(path)
    assert path.read_bytes() == original

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("disk write failed")

    with monkeypatch.context() as patch:
        patch.setattr(np, "savez_compressed", fail)
        with pytest.raises(OSError, match="disk write failed"):
            make_result().save(path, overwrite=True)
    assert path.read_bytes() == original
    assert sorted(tmp_path.iterdir()) == [path]
    make_result(all_missing=True).save(path, overwrite=True)
    assert FlameResult.load(path).estimated_count == 0


def test_concurrent_destination_creation_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result.npz"
    real_link = os.link

    def concurrent_link(source: Path, destination: Path) -> None:
        destination.write_bytes(b"created concurrently")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", concurrent_link)
    with pytest.raises(FileExistsError):
        make_result().save(path)
    assert path.read_bytes() == b"created concurrently"
    assert sorted(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "problem",
    [
        "schema",
        "boolean_schema",
        "kind",
        "missing_field",
        "unexpected_array",
        "wrong_dtype",
        "object_array",
        "timestamp_range",
        "nan_estimate",
        "fake_missing",
        "reason_type",
        "provenance",
        "extra_conventions",
        "metadata_type",
    ],
)
def test_loading_rejects_corrupt_or_unsupported_archives(tmp_path: Path, problem: str) -> None:
    path = make_result().save(tmp_path / "result.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata: dict[str, Any] = json.loads(str(arrays["metadata"].item()))
    if problem == "schema":
        metadata["schema_version"] = 99
    elif problem == "boolean_schema":
        metadata["schema_version"] = True
    elif problem == "kind":
        metadata["kind"] = "unrelated"
    elif problem == "missing_field":
        del arrays["expression"]
    elif problem == "unexpected_array":
        arrays["unexpected"] = np.array([1])
    elif problem == "wrong_dtype":
        arrays["expression"] = arrays["expression"].astype(np.float64)
    elif problem == "object_array":
        arrays["expression"] = np.array([object()], dtype=object)
    elif problem == "timestamp_range":
        metadata["end_seconds"] = 23.5
    elif problem == "nan_estimate":
        arrays["expression"][0, 0] = np.nan
    elif problem == "fake_missing":
        arrays["expression"][1, 0] = 0
    elif problem == "reason_type":
        arrays["missing_reasons"] = np.arange(4)
    elif problem == "provenance":
        metadata["provenance"]["checkpoint_sha256"] = "invalid"
    elif problem == "extra_conventions":
        metadata["extra_conventions"] = {}
    arrays["metadata"] = np.array(json.dumps(metadata))
    if problem == "metadata_type":
        arrays["metadata"] = np.array([json.dumps(metadata)])
    np.savez_compressed(path, **arrays)
    with pytest.raises((TypeError, ValueError)):
        FlameResult.load(path)


def test_refuses_wrong_extension_and_source_overwrite(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".npz"):
        make_result().save(tmp_path / "result.json")
    path = tmp_path / "source.npz"
    path.write_bytes(b"original source")
    result = make_result()
    result = replace(result, source=replace(result.source, path=path))
    with pytest.raises(ValueError, match="source"):
        result.save(path, overwrite=True)
    assert path.read_bytes() == b"original source"


def test_loading_in_fresh_process_needs_neither_models_nor_original_video(tmp_path: Path) -> None:
    path = make_result().save(tmp_path / "result.npz")
    script = """
import sys
from talkingfacekit.flame import FlameResult
result = FlameResult.load(sys.argv[1])
assert result.frame_count == 4 and result.estimated_count == 2
assert not any(name in sys.modules for name in ('torch', 'timm', 'torchvision', 'mediapipe', 'skimage'))
assert 'talkingfacekit.integrations.smirk' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
