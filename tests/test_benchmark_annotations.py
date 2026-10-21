import json
from pathlib import Path

import pytest

from talkingfacekit import cli
from talkingfacekit.evaluation import prepare_evaluation_manifest


def write_annotations(path: Path, *, start: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "file": "sample.mp4",
                    "start_seconds": start,
                    "end_seconds": start + 10,
                    "selected": [],
                }
            ]
        ),
        encoding="utf-8",
    )


def test_directory_collection_is_stable_and_matches_legacy_lists(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "sample.mp4").write_bytes(b"synthetic shared media")
    annotations = tmp_path / "annotations"
    # Create in reverse order; nested paths and row ordering must also be deterministic.
    write_annotations(annotations / "z.json", start=30)
    first = annotations / "a" / "á.json"
    write_annotations(first)
    rows = json.loads(first.read_text())
    rows.append(
        {"file": "sample.mp4", "start_seconds": 15, "end_seconds": 20, "selected": [[16, 18]]}
    )
    first.write_text(json.dumps(rows), encoding="utf-8")
    (annotations / "notes.md").write_text("Pending videos go in the work list.")
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps([*rows, *json.loads((annotations / "z.json").read_text())]))

    manifest = tmp_path / "manifest.json"
    command = [
        "prepare-benchmark",
        str(annotations),
        "--videos-dir",
        str(videos),
        "--output",
        str(manifest),
    ]
    assert cli.main(command) == 0
    assert json.loads(manifest.read_text()) == prepare_evaluation_manifest(legacy, videos)
    original = manifest.read_bytes()
    assert cli.main([*command, "--overwrite"]) == 0
    assert manifest.read_bytes() == original
    assert (
        cli.main(
            ["evaluate-benchmark", str(manifest), "--videos-dir", str(videos), "--verify-only"]
        )
        == 0
    )


@pytest.mark.parametrize("contents", ["[]", "{", "{}", '[{"file":"sample.mp4","selected":null}]'])
def test_invalid_directory_member_reports_source_and_preserves_manifest(
    tmp_path: Path, contents: str, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "sample.mp4").write_bytes(b"synthetic media")
    annotations = tmp_path / "annotations"
    write_annotations(annotations / "a.json")
    broken = annotations / "broken.json"
    broken.write_text(contents)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("previous approved manifest")

    with pytest.raises(SystemExit):
        cli.main(
            [
                "prepare-benchmark",
                str(annotations),
                "--videos-dir",
                str(tmp_path),
                "--output",
                str(manifest),
                "--overwrite",
            ]
        )
    assert str(broken) in capsys.readouterr().err
    assert manifest.read_text() == "previous approved manifest"


def test_duplicate_windows_across_files_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "sample.mp4").write_bytes(b"synthetic media")
    annotations = tmp_path / "annotations"
    write_annotations(annotations / "first.json")
    duplicate = annotations / "second.json"
    write_annotations(duplicate)

    with pytest.raises(ValueError, match="duplicate benchmark case") as error:
        prepare_evaluation_manifest(annotations, tmp_path)
    assert str(duplicate) in str(error.value)


def test_empty_directory_is_not_a_negative_annotation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contains no JSON files"):
        prepare_evaluation_manifest(tmp_path, tmp_path)


def test_manifest_check_detects_label_and_media_drift_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"synthetic shared media")
    annotations = tmp_path / "annotations"
    source = annotations / "sample.mp4.json"
    write_annotations(source)
    manifest = tmp_path / "manifest.json"
    command = [
        "prepare-benchmark",
        str(annotations),
        "--videos-dir",
        str(tmp_path),
        "--output",
        str(manifest),
    ]
    assert cli.main(command) == 0
    original = manifest.read_bytes()
    modified_ns = manifest.stat().st_mtime_ns
    assert cli.main([*command, "--check"]) == 0
    assert manifest.read_bytes() == original
    assert manifest.stat().st_mtime_ns == modified_ns

    rows = json.loads(source.read_text())
    rows[0]["selected"] = [[2, 4]]
    source.write_text(json.dumps(rows))
    with pytest.raises(SystemExit):
        cli.main([*command, "--check"])
    assert "out of date" in capsys.readouterr().err
    assert manifest.read_bytes() == original
    assert cli.main([*command, "--overwrite"]) == 0
    assert cli.main([*command, "--check"]) == 0

    original = manifest.read_bytes()
    video.write_bytes(b"a different source file")
    with pytest.raises(SystemExit):
        cli.main([*command, "--check"])
    assert "out of date" in capsys.readouterr().err
    assert manifest.read_bytes() == original
    with pytest.raises(SystemExit):
        cli.main([*command, "--check", "--overwrite"])


@pytest.mark.parametrize("mode", [[], ["--overwrite"], ["--check"]])
def test_manifest_cannot_be_written_inside_annotation_directory(
    tmp_path: Path, mode: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    annotations = tmp_path / "annotations"
    source = annotations / "sample.mp4.json"
    write_annotations(source)
    before = source.read_bytes()
    output = annotations / "generated" / "manifest.json"
    with pytest.raises(SystemExit):
        cli.main(
            [
                "prepare-benchmark",
                str(annotations),
                "--videos-dir",
                str(tmp_path),
                "--output",
                str(output),
                *mode,
            ]
        )
    assert "outside the annotations directory" in capsys.readouterr().err
    assert not output.exists()
    assert source.read_bytes() == before
