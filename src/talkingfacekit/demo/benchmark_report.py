"""Offline comparison of human selections and saved benchmark predictions."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

from talkingfacekit.evaluation import (
    EvaluationCase,
    EvaluationInterval,
    combine_evaluation_metrics,
    parse_evaluation_cases,
    score_evaluation_case,
    verify_evaluation_videos,
)
from talkingfacekit.visual import (
    VisualDecision,
    VisualMetrics,
    VisualPolicy,
    VisualRegion,
)


def render_benchmark_report(
    result_path: str | Path,
    videos_dir: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Render a saved benchmark as a standalone, interactive HTML comparison.

    Parameters
    ----------
    result_path
        UTF-8 version-1 JSON written by ``evaluate-benchmark``. Stored metrics are recomputed
        from its labels and predictions. Optional diagnostics explain recorded policy decisions.
    videos_dir
        Local source directory. Paths and complete-file SHA-256 hashes are verified before export.
    output_path
        Destination ending in ``.html``. Its parent must exist. Videos are referenced by absolute
        file URLs; no media is copied, decoded, or sent to a model or remote service.
    overwrite
        Allow replacing an existing HTML file after the complete report has been built.

    Returns
    -------
    pathlib.Path
        Path of the generated HTML. All intervals retain original source seconds and half-open
        bounds. Ignored time is excluded from differences and metrics.

    Raises
    ------
    FileNotFoundError, FileExistsError, IsADirectoryError
        If an input, parent directory, or destination is unavailable.
    TypeError, ValueError
        If labels, predictions, diagnostics, schema, video hashes, or paths are invalid.
    """
    source = Path(result_path).expanduser().resolve()
    destination = Path(output_path).expanduser()
    if destination.suffix.lower() != ".html":
        raise ValueError("benchmark report output must end in .html")
    if destination.is_dir():
        raise IsADirectoryError(destination)
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"report directory not found: {destination.parent}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"benchmark report already exists: {destination}")
    try:
        document = _object(json.loads(source.read_text(encoding="utf-8")), "result")
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid benchmark result JSON: {error}") from error
    payload = _build_payload(document, Path(videos_dir))
    if destination.resolve() == source or any(
        destination.resolve().as_uri() == case["video_url"]
        for case in cast(list[dict[str, object]], payload["cases"])
    ):
        raise ValueError("report destination must differ from inputs")
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    for character, escaped in (
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("&", "\\u0026"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        serialized = serialized.replace(character, escaped)
    template = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    rendered = template.replace("__BENCHMARK_DATA__", serialized)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, suffix=".html", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(rendered)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _build_payload(document: dict[str, object], videos_dir: Path) -> dict[str, object]:
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("benchmark result requires schema_version 1")
    rows = [_object(row, "case") for row in _array(document.get("cases"), "cases")]
    manifest_cases: list[dict[str, object]] = []
    for row in rows:
        window = _array(row.get("window"), "window")
        if len(window) != 2:
            raise ValueError("case window must contain [start, end]")
        manifest_cases.append(
            {
                "id": row.get("id"),
                "file": row.get("file"),
                "sha256": row.get("sha256"),
                "start_seconds": window[0],
                "end_seconds": window[1],
                "selected": row.get("selected"),
                "ignore": row.get("ignore", []),
            }
        )
    cases = parse_evaluation_cases({"schema_version": 1, "cases": manifest_cases})
    paths = verify_evaluation_videos(cases, videos_dir)
    prepared: list[dict[str, object]] = []
    metrics = []
    before_metrics = []
    for case, row in zip(cases, rows, strict=True):
        predicted = _intervals(row.get("predicted"), "predicted", case.window)
        before = (
            None
            if row.get("predicted_before_visual_filter") is None
            else _intervals(
                row["predicted_before_visual_filter"], "predicted_before_visual_filter", case.window
            )
        )
        before_scored = None if before is None else score_evaluation_case(case, before)
        if before_scored is not None:
            before_metrics.append(before_scored)
        visual = _visual_filter(row.get("visual_filter"), case.window)
        if visual is not None:
            decisions = cast(list[dict[str, object]], visual["decisions"])
            selected = tuple(
                EvaluationInterval(
                    cast(float, item["source_start_seconds"]),
                    cast(float, item["source_end_seconds"]),
                )
                for item in decisions
                if item["status"] == "candidate"
            )
            if selected != predicted or before is None:
                raise ValueError(
                    "visual decisions must match predictions and include pre-filter intervals"
                )
            if any(
                not any(
                    base.start_seconds <= item.start_seconds
                    and item.end_seconds <= base.end_seconds
                    for base in before
                )
                for item in predicted
            ):
                raise ValueError("visual filtering must not add selected time")
        scored = score_evaluation_case(case, predicted)
        metrics.append(scored)
        prepared.append(
            {
                "id": case.case_id,
                "file": case.file.as_posix(),
                "video_url": paths[case.case_id].as_uri(),
                "window": [case.window.start_seconds, case.window.end_seconds],
                "selected": _pairs(case.selected),
                "predicted": _pairs(predicted),
                "predicted_before_visual_filter": None if before is None else _pairs(before),
                "metrics_before_visual_filter": None
                if before_scored is None
                else asdict(before_scored),
                "visual_filter": visual,
                "ignore": _pairs(case.ignored),
                "regions": _comparison_regions(case, predicted),
                "metrics": asdict(scored),
                "diagnostics": _diagnostics(row.get("diagnostics"), case.window),
                "synchronization": _synchronization(row.get("synchronization"), case.window),
                "issues": _strings(row.get("issues", []), "issues"),
                "backend_version": _optional_text(row.get("backend_version")),
                "candidate_policy": _policy(row.get("candidate_policy")),
            }
        )
    return {
        "cases": prepared,
        "overall": asdict(combine_evaluation_metrics(tuple(metrics))),
        "overall_before_visual_filter": asdict(combine_evaluation_metrics(tuple(before_metrics)))
        if len(before_metrics) == len(cases)
        else None,
        "code_revision": _revision_text(document.get("code_revision")),
        "package_version": _optional_text(document.get("package_version")),
        "manifest_sha256": _optional_text(document.get("manifest_sha256")),
    }


def _comparison_regions(
    case: EvaluationCase, predicted: tuple[EvaluationInterval, ...]
) -> list[dict[str, object]]:
    boundaries = {case.window.start_seconds, case.window.end_seconds}
    for interval in (*case.selected, *predicted, *case.ignored):
        boundaries.update((interval.start_seconds, interval.end_seconds))
    ordered = sorted(boundaries)
    regions: list[dict[str, object]] = []
    for start, end in pairwise(ordered):
        ignored = _contains(case.ignored, start)
        human = _contains(case.selected, start)
        library = _contains(predicted, start)
        status = (
            "ignored"
            if ignored
            else "both_selected"
            if human and library
            else "false_negative"
            if human
            else "false_positive"
            if library
            else "both_rejected"
        )
        if regions and regions[-1]["status"] == status:
            regions[-1]["end"] = end
        else:
            regions.append({"start": start, "end": end, "status": status})
    return regions


def _contains(intervals: tuple[EvaluationInterval, ...], time: float) -> bool:
    return any(interval.start_seconds <= time < interval.end_seconds for interval in intervals)


def _diagnostics(value: object, window: EvaluationInterval) -> dict[str, object] | None:
    if value is None:
        return None
    raw = _object(value, "diagnostics")
    normalized: dict[str, object] = {}
    for key in ("preliminary_segments", "candidate_runs"):
        records: list[dict[str, object]] = []
        for entry in _array(raw.get(key, []), key):
            item = _object(entry, key)
            bounds = _intervals(
                [[item.get("source_start_seconds"), item.get("source_end_seconds")]], key, window
            )[0]
            status = item.get("status")
            allowed = (
                {"candidate", "rejected", "uncertain"}
                if key == "preliminary_segments"
                else {"candidate", "rejected"}
            )
            if not isinstance(status, str) or status not in allowed:
                raise ValueError(f"invalid {key} status")
            record: dict[str, object] = {
                "start": bounds.start_seconds,
                "end": bounds.end_seconds,
                "status": status,
                "face_id": _face_id(item.get("face_id")),
                "reasons": _strings(item.get("reason_codes"), "reason_codes"),
            }
            if key == "preliminary_segments":
                record["visible_face_ids"] = [
                    _face_id(face_id) for face_id in _array(item.get("visible_face_ids"), key)
                ]
                record["raw_score"] = (
                    None if item.get("raw_score") is None else _number(item["raw_score"], key)
                )
            else:
                for field in (
                    "face_visibility_fraction",
                    "maximum_face_gap_seconds",
                    "raw_score_mean",
                ):
                    record[field] = _number(item.get(field), field)
                coverage = cast(float, record["face_visibility_fraction"])
                if not 0 <= coverage <= 1 or cast(float, record["maximum_face_gap_seconds"]) < 0:
                    raise ValueError("invalid candidate visibility metrics")
                record["bridged_gaps"] = _pairs(
                    _intervals(item.get("bridged_gap_intervals", []), "bridged gaps", bounds)
                )
            records.append(record)
        normalized[key] = records
    cuts = [
        _number(item, "shot boundary") for item in _array(raw.get("shot_boundaries", []), "cuts")
    ]
    if any(not window.start_seconds <= cut <= window.end_seconds for cut in cuts):
        raise ValueError("shot boundaries must be inside the case window")
    normalized["shot_boundaries"] = sorted(cuts)
    return normalized


def _synchronization(value: object, window: EvaluationInterval) -> dict[str, object] | None:
    if value is None:
        return None
    raw = _object(value, "synchronization")
    result: dict[str, object] = {}
    for key in ("windows", "decisions"):
        records = []
        for entry in _array(raw.get(key), f"synchronization.{key}"):
            item = _object(entry, key)
            _intervals(
                [[item.get("source_start_seconds"), item.get("source_end_seconds")]], key, window
            )
            _face_id(item.get("face_id"))
            allowed = (
                {"synchronized", "offset_detected", "uncertain"}
                if key == "windows"
                else {"candidate", "rejected", "uncertain"}
            )
            if item.get("status") not in allowed:
                raise ValueError("invalid synchronization status")
            _strings(item.get("reason_codes"), "synchronization reason_codes")
            for field in (
                "estimated_offset_seconds",
                "confidence",
                "minimum_distance",
                "search_limit_seconds",
            ):
                if item.get(field) is not None:
                    _number(item[field], f"synchronization.{field}")
            records.append(item)
        result[key] = records
    result["elapsed_seconds"] = _number(
        raw.get("elapsed_seconds"), "synchronization elapsed_seconds"
    )
    return result


def _visual_filter(value: object, window: EvaluationInterval) -> dict[str, object] | None:
    if value is None:
        return None
    raw = _object(value, "visual_filter")
    policy = VisualPolicy(
        **{
            key: _number(item, key)
            for key, item in _object(raw.get("policy"), "visual policy").items()
        }
    )
    normalized: dict[str, object] = {"policy": asdict(policy)}
    for dimension in ("image_width", "image_height"):
        val = raw.get(dimension)
        if type(val) is not int or val <= 0:
            raise ValueError("visual source dimensions must be positive integers")
        normalized[dimension] = val
    rate = _number(raw.get("video_sample_rate_hz"), "visual sample rate")
    if rate <= 0:
        raise ValueError("visual sample rate must be positive")
    normalized["video_sample_rate_hz"] = rate
    for key in ("windows", "decisions"):
        entries = [_object(item, key) for item in _array(raw.get(key), key)]
        bounds = _intervals(
            [
                [item.get("source_start_seconds"), item.get("source_end_seconds")]
                for item in entries
            ],
            key,
            window,
        )
        records: list[dict[str, object]] = []
        for item, bound in zip(entries, bounds, strict=True):
            face_id = _face_id(item.get("face_id"))
            if face_id is None:
                raise ValueError("visual regions require a face ID")
            status = item.get("status")
            if status not in {"candidate", "rejected", "uncertain"}:
                raise ValueError("invalid visual status")
            region = VisualRegion(
                bound.start_seconds,
                bound.end_seconds,
                face_id,
                status,
                tuple(_strings(item.get("reason_codes"), "visual reasons")),
            )
            if key == "windows":
                records.append(asdict(region))
                continue
            metrics = None
            if item.get("metrics") is not None:
                m = _object(item["metrics"], "visual metrics")
                angles = m.get("maximum_absolute_yaw_pitch_roll_degrees")
                pose = None
                if angles is not None:
                    angle_list = _array(angles, "visual angles")
                    if len(angle_list) != 3:
                        raise ValueError("visual angles require three axes")
                    pose = (
                        _number(angle_list[0], "yaw"),
                        _number(angle_list[1], "pitch"),
                        _number(angle_list[2], "roll"),
                    )
                metrics = VisualMetrics(
                    _number(m.get("observed_fraction"), "observed fraction"),
                    _number(m.get("passing_fraction"), "passing fraction"),
                    pose,
                    *[
                        None if m.get(field) is None else _number(m[field], field)
                        for field in (
                            "minimum_face_size_pixels",
                            "minimum_face_width_fraction",
                            "maximum_clipped_fraction",
                        )
                    ],
                )
            gaps = tuple(
                (gap.start_seconds, gap.end_seconds)
                for gap in _intervals(item.get("tolerated_gap_intervals", []), "visual gaps", bound)
            )
            records.append(
                asdict(
                    VisualDecision(
                        region.source_start_seconds,
                        region.source_end_seconds,
                        face_id,
                        region.status,
                        region.reason_codes,
                        metrics,
                        gaps,
                    )
                )
            )
        normalized[key] = records
    return normalized


def _policy(value: object) -> dict[str, float]:
    if value is None:
        return {}
    return {
        key: _number(item, f"candidate_policy.{key}")
        for key, item in _object(value, "policy").items()
    }


def _face_id(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError("face_id must be a non-negative integer or null")
    return value


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return cast(list[object], value)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _strings(value: object, name: str) -> list[str]:
    items = _array(value, name)
    if not all(isinstance(item, str) for item in items):
        raise ValueError(f"{name} must contain strings")
    return cast(list[str], items)


def _optional_text(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError("report metadata must be a string or null")
    return value


def _revision_text(value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    revision = _object(value, "code_revision")
    commit = _optional_text(revision.get("commit"))
    dirty = revision.get("dirty")
    if dirty is not None and not isinstance(dirty, bool):
        raise ValueError("code_revision.dirty must be a boolean or null")
    return f"{commit or 'sin commit registrado'}{' (con cambios locales)' if dirty else ''}"


def _intervals(
    value: object, name: str, window: EvaluationInterval
) -> tuple[EvaluationInterval, ...]:
    result: list[EvaluationInterval] = []
    previous_end = window.start_seconds
    for raw in _array(value, name):
        pair = _array(raw, name)
        if len(pair) != 2:
            raise ValueError(f"{name} requires [start, end] pairs")
        interval = EvaluationInterval(_number(pair[0], name), _number(pair[1], name))
        if interval.start_seconds < previous_end or interval.end_seconds > window.end_seconds:
            raise ValueError(f"{name} must be ordered, non-overlapping, and inside the window")
        result.append(interval)
        previous_end = interval.end_seconds
    return tuple(result)


def _pairs(intervals: tuple[EvaluationInterval, ...]) -> list[list[float]]:
    return [[item.start_seconds, item.end_seconds] for item in intervals]
