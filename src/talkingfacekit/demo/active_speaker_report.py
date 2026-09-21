"""Standalone HTML report for inspecting experimental active-speaker results."""

from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from talkingfacekit.integrations.deeptalk import DeepTalkAnalysisResult
from talkingfacekit.sequence import TalkingFaceSequence

_REPORT_TEMPLATE = r"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__REPORT_TITLE__</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101216;
      --panel: #1a1e24;
      --panel-2: #22272f;
      --text: #f3f5f7;
      --muted: #a7b0bc;
      --line: #353c46;
      --candidate: #f4ba43;
      --speech: #42c68a;
      --repair: #d474f2;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background: var(--bg);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1, h2 { margin: 0; font-weight: 600; }
    h1 { font-size: 18px; }
    h2 { font-size: 14px; }
    .source-name { color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 290px;
      gap: 14px;
      max-width: 1400px;
      margin: 0 auto;
      padding: 14px;
    }
    .panel {
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
    }
    .video-wrap { position: relative; line-height: 0; background: #000; }
    video { display: block; width: 100%; height: auto; background: #000; }
    #overlay { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
    .speech-pill {
      position: absolute;
      top: 12px;
      left: 12px;
      padding: 7px 9px;
      border-radius: 6px;
      color: #09140f;
      background: var(--speech);
      font-size: 12px;
      font-weight: 600;
      line-height: 1;
    }
    .speech-pill.inactive { color: var(--text); background: rgb(34 39 47 / 90%); }
    .side { padding: 14px; }
    .time {
      margin: 5px 0 14px;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      font-size: 13px;
    }
    .candidate {
      margin-bottom: 16px;
      padding: 11px;
      border-left: 3px solid var(--candidate);
      border-radius: 5px;
      background: rgb(244 186 67 / 10%);
    }
    .candidate strong { display: block; margin-bottom: 4px; font-size: 14px; }
    .candidate span { color: var(--muted); font-size: 12px; line-height: 1.4; }
    .score-row {
      display: grid;
      grid-template-columns: 10px 1fr auto;
      align-items: center;
      gap: 8px;
      padding: 9px 0;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
    }
    .swatch { width: 9px; height: 9px; border-radius: 50%; }
    .score { font-variant-numeric: tabular-nums; font-weight: 600; }
    .empty { padding: 10px 0; color: var(--muted); font-size: 13px; }
    .note, .issues { color: var(--muted); font-size: 12px; line-height: 1.45; }
    .note { margin: 14px 0 0; }
    .issues { margin: 14px 0 0; padding-top: 12px; border-top: 1px solid var(--line); }
    .timeline { grid-column: 1 / -1; padding: 14px; }
    .timeline-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }
    .legend { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 11px; }
    .legend-item { display: flex; align-items: center; gap: 5px; color: var(--muted); font-size: 12px; }
    .legend-line { width: 15px; height: 3px; border-radius: 2px; }
    #timeline { display: block; width: 100%; height: 190px; touch-action: manipulation; cursor: crosshair; }
    .timeline-help { margin-top: 7px; color: var(--muted); font-size: 12px; }
    @media (max-width: 820px) {
      header { align-items: flex-start; flex-direction: column; }
      main { grid-template-columns: 1fr; }
      .timeline { grid-column: 1; }
      .timeline-head { align-items: flex-start; flex-direction: column; }
      .legend { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <header>
    <h1>TalkingFaceKit · Active speaker report</h1>
    <div class="source-name" id="source-name"></div>
  </header>
  <main>
    <section class="panel" aria-label="Video con detecciones superpuestas">
      <div class="video-wrap">
        <video id="video" controls preload="metadata"></video>
        <canvas id="overlay" aria-label="Bounding boxes de rostros"></canvas>
        <div class="speech-pill inactive" id="speech-pill">Fuera del intervalo analizado</div>
      </div>
    </section>
    <aside class="panel side" aria-label="Evidencia del instante actual">
      <h2>Instante actual</h2>
      <div class="time" id="current-time">—</div>
      <div class="candidate">
        <strong id="candidate-title">Sin candidato</strong>
        <span id="candidate-detail">Reproducí el video para inspeccionar el resultado.</span>
      </div>
      <h2>Raw scores</h2>
      <div id="score-list" aria-live="polite"></div>
      <p class="note">“Top candidate” significa únicamente que ese rostro tiene el raw score más alto durante una ventana con voz. No es una probabilidad ni una decisión final.</p>
      <p class="issues" id="issues"></p>
    </aside>
    <section class="panel timeline" aria-label="Timeline de scores y voz">
      <div class="timeline-head">
        <h2>Timeline sincronizada</h2>
        <div class="legend" id="legend"></div>
      </div>
      <canvas id="timeline" role="img" aria-label="Raw scores por rostro, intervalos de voz, reparaciones de audio y posición actual"></canvas>
      <div class="timeline-help">Hacé click en la timeline para saltar a ese instante. El fondo verde marca voz; las líneas verticales violetas marcan reparaciones del audio.</div>
    </section>
  </main>
  <script>
    "use strict";
    const REPORT = __REPORT_DATA__;
    const COLORS = ["#66a6ff", "#f4ba43", "#e378b5", "#7ed7c4", "#ad8cff", "#ff8c69"];
    const video = document.getElementById("video");
    const overlay = document.getElementById("overlay");
    const timeline = document.getElementById("timeline");
    const overlayContext = overlay.getContext("2d");
    const timelineContext = timeline.getContext("2d");
    const speechPill = document.getElementById("speech-pill");
    const scoreList = document.getElementById("score-list");
    const candidateTitle = document.getElementById("candidate-title");
    const candidateDetail = document.getElementById("candidate-detail");
    const currentTimeLabel = document.getElementById("current-time");
    let lastSideSignature = "";

    document.getElementById("source-name").textContent =
      `${REPORT.video.name} · ${REPORT.backend.name} ${REPORT.backend.version}`;
    document.getElementById("issues").textContent = REPORT.issues.length
      ? `Issues: ${REPORT.issues.join(", ")}`
      : "Issues: none";
    video.src = REPORT.video.url;

    function colorForFace(faceId) {
      const index = REPORT.faceIds.indexOf(faceId);
      return COLORS[(index < 0 ? 0 : index) % COLORS.length];
    }

    function formatTime(seconds) {
      if (!Number.isFinite(seconds)) return "—";
      const minutes = Math.floor(seconds / 60);
      const remaining = seconds - minutes * 60;
      return `${String(minutes).padStart(2, "0")}:${remaining.toFixed(2).padStart(5, "0")}`;
    }

    function formatScore(score) {
      return `${score >= 0 ? "+" : ""}${score.toFixed(3)}`;
    }

    function containingInterval(intervals, sourceTime) {
      let low = 0;
      let high = intervals.length - 1;
      while (low <= high) {
        const middle = Math.floor((low + high) / 2);
        const interval = intervals[middle];
        if (sourceTime < interval[0]) high = middle - 1;
        else if (sourceTime >= interval[1]) low = middle + 1;
        else return interval;
      }
      return null;
    }

    function nearestFaceFrame(sourceTime) {
      const frames = REPORT.faceFrames;
      if (!frames.length) return null;
      let low = 0;
      let high = frames.length - 1;
      while (low <= high) {
        const middle = Math.floor((low + high) / 2);
        if (frames[middle][0] < sourceTime) low = middle + 1;
        else high = middle - 1;
      }
      const candidates = [];
      if (low < frames.length) candidates.push(frames[low]);
      if (low > 0) candidates.push(frames[low - 1]);
      if (!candidates.length) return null;
      const closest = candidates.reduce((best, frame) =>
        Math.abs(frame[0] - sourceTime) < Math.abs(best[0] - sourceTime) ? frame : best
      );
      return Math.abs(closest[0] - sourceTime) <= REPORT.faceFrameTolerance ? closest : null;
    }

    function currentEvidence(sourceTime) {
      const scoreWindow = containingInterval(REPORT.scoreWindows, sourceTime);
      const speech = containingInterval(REPORT.speechIntervals, sourceTime);
      const scores = scoreWindow ? scoreWindow[2] : [];
      const speechActive = speech !== null && speech[2] !== "rejected";
      let candidate = null;
      if (speechActive && scores.length) {
        candidate = scores.reduce((best, item) => item[1] > best[1] ? item : best);
      }
      return { scoreWindow, speech, scores, speechActive, candidate };
    }

    function sizeOverlay() {
      const width = video.videoWidth || REPORT.video.width;
      const height = video.videoHeight || REPORT.video.height;
      if (overlay.width !== width || overlay.height !== height) {
        overlay.width = width;
        overlay.height = height;
      }
    }

    function drawOverlay(sourceTime, evidence) {
      sizeOverlay();
      overlayContext.clearRect(0, 0, overlay.width, overlay.height);
      const frame = nearestFaceFrame(sourceTime);
      if (!frame) return;
      const lineWidth = Math.max(2, overlay.width / 480);
      const fontSize = Math.max(14, Math.round(overlay.width / 55));
      overlayContext.font = `600 ${fontSize}px ui-sans-serif, sans-serif`;
      overlayContext.textBaseline = "top";
      for (const face of frame[1]) {
        const [faceId, x, y, width, height] = face;
        const color = colorForFace(faceId);
        const isCandidate = evidence.candidate !== null && evidence.candidate[0] === faceId;
        overlayContext.strokeStyle = isCandidate ? "#f4ba43" : color;
        overlayContext.lineWidth = isCandidate ? lineWidth * 2 : lineWidth;
        overlayContext.strokeRect(x, y, width, height);
        const scoreItem = evidence.scores.find(item => item[0] === faceId);
        const label = scoreItem
          ? `Face ${faceId} · ${formatScore(scoreItem[1])}${isCandidate ? " · top" : ""}`
          : `Face ${faceId} · no score`;
        const metrics = overlayContext.measureText(label);
        const padding = Math.max(4, Math.round(fontSize * 0.32));
        const labelHeight = fontSize + padding * 2;
        const labelY = Math.max(0, y - labelHeight);
        overlayContext.fillStyle = isCandidate ? "#f4ba43" : color;
        overlayContext.fillRect(x, labelY, metrics.width + padding * 2, labelHeight);
        overlayContext.fillStyle = "#101216";
        overlayContext.fillText(label, x + padding, labelY + padding);
      }
    }

    function updateSide(sourceTime, evidence) {
      const signature = JSON.stringify([Math.floor(sourceTime * 10), evidence.speech, evidence.scores]);
      if (signature === lastSideSignature) return;
      lastSideSignature = signature;
      currentTimeLabel.textContent = `${formatTime(sourceTime)} · source time`;
      const inside = sourceTime >= REPORT.interval.start && sourceTime < REPORT.interval.end;
      if (!inside) {
        speechPill.textContent = "Fuera del intervalo analizado";
        speechPill.classList.add("inactive");
        candidateTitle.textContent = "Sin candidato";
        candidateDetail.textContent = "Este instante no forma parte del análisis.";
      } else if (evidence.speechActive) {
        speechPill.textContent = `Speech ${evidence.speech[2]}`;
        speechPill.classList.remove("inactive");
        if (evidence.candidate) {
          candidateTitle.textContent = `Face ${evidence.candidate[0]} · top candidate`;
          candidateDetail.textContent = "Hay voz y este rostro tiene el raw score más alto.";
        } else {
          candidateTitle.textContent = "Voz sin candidato visible";
          candidateDetail.textContent = "Hay voz, pero esta ventana no tiene scores de rostros.";
        }
      } else {
        speechPill.textContent = evidence.speech ? `Speech ${evidence.speech[2]}` : "No speech";
        speechPill.classList.add("inactive");
        candidateTitle.textContent = "Sin candidato";
        candidateDetail.textContent = "No hay un intervalo de voz activo.";
      }

      scoreList.replaceChildren();
      if (!evidence.scores.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No hay scores para esta ventana.";
        scoreList.appendChild(empty);
        return;
      }
      for (const [faceId, score] of evidence.scores) {
        const row = document.createElement("div");
        row.className = "score-row";
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = colorForFace(faceId);
        const name = document.createElement("span");
        name.textContent = `Face ${faceId}`;
        const value = document.createElement("span");
        value.className = "score";
        value.textContent = formatScore(score);
        row.append(swatch, name, value);
        scoreList.appendChild(row);
      }
    }

    function timelineGeometry() {
      const rect = timeline.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(320, rect.width);
      const height = 190;
      const pixelWidth = Math.round(width * ratio);
      const pixelHeight = Math.round(height * ratio);
      if (timeline.width !== pixelWidth || timeline.height !== pixelHeight) {
        timeline.width = pixelWidth;
        timeline.height = pixelHeight;
      }
      timelineContext.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { width, height, left: 45, right: 12, top: 12, bottom: 28 };
    }

    function drawTimeline(sourceTime) {
      const geometry = timelineGeometry();
      const { width, height, left, right, top, bottom } = geometry;
      const plotWidth = width - left - right;
      const plotHeight = height - top - bottom;
      const duration = REPORT.interval.end - REPORT.interval.start;
      const x = value => left + ((value - REPORT.interval.start) / duration) * plotWidth;
      let minimum = 0;
      let maximum = 0;
      let hasScores = false;
      for (const scoreWindow of REPORT.scoreWindows) {
        for (const item of scoreWindow[2]) {
          minimum = Math.min(minimum, item[1]);
          maximum = Math.max(maximum, item[1]);
          hasScores = true;
        }
      }
      if (!hasScores) { minimum = -1; maximum = 1; }
      if (maximum - minimum < 0.1) { minimum -= 0.5; maximum += 0.5; }
      const padding = (maximum - minimum) * 0.08;
      minimum -= padding;
      maximum += padding;
      const y = value => top + ((maximum - value) / (maximum - minimum)) * plotHeight;

      timelineContext.clearRect(0, 0, width, height);
      for (const interval of REPORT.speechIntervals) {
        if (interval[2] === "rejected") continue;
        timelineContext.fillStyle = "rgb(66 198 138 / 14%)";
        timelineContext.fillRect(x(interval[0]), top, x(interval[1]) - x(interval[0]), plotHeight);
      }
      timelineContext.strokeStyle = "#353c46";
      timelineContext.lineWidth = 1;
      timelineContext.fillStyle = "#a7b0bc";
      timelineContext.font = "12px ui-sans-serif, sans-serif";
      timelineContext.textAlign = "right";
      timelineContext.textBaseline = "middle";
      for (let index = 0; index < 4; index += 1) {
        const value = maximum - ((maximum - minimum) * index) / 3;
        const lineY = y(value);
        timelineContext.beginPath();
        timelineContext.moveTo(left, lineY);
        timelineContext.lineTo(width - right, lineY);
        timelineContext.stroke();
        timelineContext.fillText(value.toFixed(1), left - 7, lineY);
      }
      for (const repair of REPORT.repairs) {
        timelineContext.fillStyle = "#d474f2";
        timelineContext.fillRect(x(repair[1]) - 1, top, 3, plotHeight);
      }
      for (const faceId of REPORT.faceIds) {
        timelineContext.strokeStyle = colorForFace(faceId);
        timelineContext.lineWidth = 2;
        timelineContext.beginPath();
        let drawing = false;
        for (const scoreWindow of REPORT.scoreWindows) {
          const item = scoreWindow[2].find(entry => entry[0] === faceId);
          if (!item) { drawing = false; continue; }
          const startX = x(scoreWindow[0]);
          const endX = x(scoreWindow[1]);
          const scoreY = y(item[1]);
          if (!drawing) timelineContext.moveTo(startX, scoreY);
          else timelineContext.lineTo(startX, scoreY);
          timelineContext.lineTo(endX, scoreY);
          drawing = true;
        }
        timelineContext.stroke();
      }
      const clampedTime = Math.max(REPORT.interval.start, Math.min(REPORT.interval.end, sourceTime));
      timelineContext.strokeStyle = "#f3f5f7";
      timelineContext.lineWidth = 1.5;
      timelineContext.beginPath();
      timelineContext.moveTo(x(clampedTime), top);
      timelineContext.lineTo(x(clampedTime), top + plotHeight);
      timelineContext.stroke();
      timelineContext.fillStyle = "#a7b0bc";
      timelineContext.textAlign = "left";
      timelineContext.textBaseline = "top";
      timelineContext.fillText(formatTime(REPORT.interval.start), left, height - bottom + 8);
      timelineContext.textAlign = "right";
      timelineContext.fillText(formatTime(REPORT.interval.end), width - right, height - bottom + 8);
    }

    function render() {
      const sourceTime = Number.isFinite(video.currentTime) ? video.currentTime : REPORT.interval.start;
      const evidence = currentEvidence(sourceTime);
      drawOverlay(sourceTime, evidence);
      updateSide(sourceTime, evidence);
      drawTimeline(sourceTime);
      if (!video.paused && !video.ended) window.requestAnimationFrame(render);
    }

    function buildLegend() {
      const legend = document.getElementById("legend");
      for (const faceId of REPORT.faceIds) {
        const item = document.createElement("span");
        item.className = "legend-item";
        const line = document.createElement("span");
        line.className = "legend-line";
        line.style.background = colorForFace(faceId);
        item.append(line, `Face ${faceId}`);
        legend.appendChild(item);
      }
      if (REPORT.repairs.length) {
        const item = document.createElement("span");
        item.className = "legend-item";
        const line = document.createElement("span");
        line.className = "legend-line";
        line.style.background = "#d474f2";
        item.append(line, "Audio repair");
        legend.appendChild(item);
      }
    }

    video.addEventListener("loadedmetadata", () => {
      if (video.currentTime < REPORT.interval.start || video.currentTime >= REPORT.interval.end) {
        video.currentTime = REPORT.interval.start;
      }
      render();
    });
    video.addEventListener("play", () => window.requestAnimationFrame(render));
    video.addEventListener("pause", render);
    video.addEventListener("seeked", render);
    video.addEventListener("timeupdate", () => {
      if (video.currentTime >= REPORT.interval.end && !video.paused) video.pause();
      render();
    });
    timeline.addEventListener("click", event => {
      const rect = timeline.getBoundingClientRect();
      const left = 45;
      const right = 12;
      const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left - left) / (rect.width - left - right)));
      video.currentTime = REPORT.interval.start + fraction * (REPORT.interval.end - REPORT.interval.start);
    });
    window.addEventListener("resize", render);
    buildLegend();
    render();
  </script>
</body>
</html>
"""


def render_active_speaker_report(
    sequence: TalkingFaceSequence,
    result: DeepTalkAnalysisResult,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Render a browser-based diagnostic report beside an unchanged source video.

    The report contains the analysis data and UI code in one HTML file. It references the original
    local video through a file URL instead of copying or re-encoding media. Playback time drives
    face boxes, raw-score windows, VAD state, audio-repair markers, and the chart cursor. The largest
    raw score during a non-rejected speech interval is labelled only as a ``top candidate``.

    Parameters
    ----------
    sequence
        Analyzed source and source-timeline interval.
    result
        Immutable DeepTalk observations produced for ``sequence``.
    output_path
        Destination ending in ``.html``. Its parent directory must already exist.
    overwrite
        Whether an existing regular file may be replaced after the complete report is built.

    Returns
    -------
    pathlib.Path
        The validated destination path.

    Raises
    ------
    FileNotFoundError
        If the source video or destination parent does not exist.
    FileExistsError
        If the destination exists and ``overwrite`` is false.
    IsADirectoryError
        If either file path refers to a directory where a regular file is required.
    TypeError
        If ``sequence`` or ``result`` has the wrong type.
    ValueError
        If the destination is not HTML or no positive report interval can be derived.
    """
    if not isinstance(sequence, TalkingFaceSequence):
        raise TypeError(f"sequence must be a TalkingFaceSequence, got {type(sequence).__name__}")
    if not isinstance(result, DeepTalkAnalysisResult):
        raise TypeError(f"result must be a DeepTalkAnalysisResult, got {type(result).__name__}")

    source_path = _validate_source_path(sequence.source.path)
    path = _validate_output_path(output_path, overwrite=overwrite)
    payload = _build_report_payload(sequence, result, source_path=source_path)
    report_title = f"TalkingFaceKit active speaker report · {source_path.name}"
    report_html = _REPORT_TEMPLATE.replace("__REPORT_TITLE__", html.escape(report_title)).replace(
        "__REPORT_DATA__",
        _json_for_script(payload),
    )

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp.html",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(report_html)
        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return path


def _build_report_payload(
    sequence: TalkingFaceSequence,
    result: DeepTalkAnalysisResult,
    *,
    source_path: Path,
) -> dict[str, object]:
    interval_end_seconds = _resolve_interval_end(sequence, result)
    grouped_faces: dict[float, list[list[int | float]]] = defaultdict(list)
    face_ids: set[int] = set()
    for observation in result.face_observations:
        face_ids.add(observation.face_id)
        grouped_faces[observation.source_timestamp_seconds].append(
            [
                observation.face_id,
                *(_rounded(value) for value in observation.bounding_box_xywh),
            ]
        )
    for window in result.score_windows:
        face_ids.update(window.raw_scores)

    face_frames = [
        [_rounded(timestamp), sorted(faces, key=lambda face: int(face[0]))]
        for timestamp, faces in sorted(grouped_faces.items())
    ]
    score_windows = [
        [
            _rounded(window.source_start_seconds),
            _rounded(window.source_end_seconds),
            [[face_id, _rounded(score)] for face_id, score in sorted(window.raw_scores.items())],
        ]
        for window in sorted(result.score_windows, key=lambda item: item.source_start_seconds)
    ]
    speech_intervals = [
        [
            _rounded(interval.source_start_seconds),
            _rounded(interval.source_end_seconds),
            interval.status,
        ]
        for interval in sorted(
            result.speech_intervals,
            key=lambda item: item.source_start_seconds,
        )
    ]
    repairs = [
        [
            repair.kind,
            _rounded(repair.source_start_seconds),
            _rounded(repair.source_end_seconds),
            repair.adjusted_sample_count,
            repair.sample_rate_hz,
        ]
        for repair in sorted(
            result.audio_timeline_repairs,
            key=lambda item: item.source_start_seconds,
        )
    ]
    return {
        "video": {
            "name": source_path.name,
            "url": source_path.as_uri(),
            "width": sequence.source.metadata.width,
            "height": sequence.source.metadata.height,
        },
        "interval": {
            "start": _rounded(sequence.start_seconds),
            "end": _rounded(interval_end_seconds),
        },
        "faceFrameTolerance": max(0.025, 0.75 / result.provenance.video_sample_rate_hz),
        "faceIds": sorted(face_ids),
        "faceFrames": face_frames,
        "scoreWindows": score_windows,
        "speechIntervals": speech_intervals,
        "repairs": repairs,
        "issues": list(result.issues),
        "backend": {
            "name": "DeepTalk-ASD",
            "version": result.provenance.backend_version,
        },
    }


def _resolve_interval_end(
    sequence: TalkingFaceSequence,
    result: DeepTalkAnalysisResult,
) -> float:
    if sequence.end_seconds is not None:
        return sequence.end_seconds

    observed_ends = [window.source_end_seconds for window in result.score_windows]
    observed_ends.extend(interval.source_end_seconds for interval in result.speech_intervals)
    observed_ends.extend(repair.source_end_seconds for repair in result.audio_timeline_repairs)
    observed_ends.extend(
        observation.source_timestamp_seconds for observation in result.face_observations
    )
    if sequence.source.metadata.stream_duration_seconds is not None:
        observed_ends.append(sequence.source.metadata.stream_duration_seconds)
    interval_end_seconds = max(observed_ends, default=sequence.start_seconds)
    if interval_end_seconds <= sequence.start_seconds:
        raise ValueError("active-speaker report requires a positive analyzed interval")
    return interval_end_seconds


def _rounded(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"active-speaker report received a non-finite value: {value}")
    return round(value, 6)


def _json_for_script(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        serialized.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _validate_source_path(source_path: str | Path) -> Path:
    path = Path(source_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Active-speaker report source video not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Active-speaker report source video is a directory: {path}")
    if not path.is_file():
        raise ValueError(f"Active-speaker report source is not a regular file: {path}")
    return path


def _validate_output_path(output_path: str | Path, *, overwrite: bool) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".html":
        raise ValueError(f"Active-speaker report output path must end in .html: {path}")
    if not path.parent.exists():
        raise FileNotFoundError(f"Active-speaker report output directory not found: {path.parent}")
    if not path.parent.is_dir():
        raise ValueError(f"Active-speaker report output parent is not a directory: {path.parent}")
    if path.is_dir():
        raise IsADirectoryError(f"Active-speaker report output path is a directory: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Active-speaker report output already exists: {path}")
    return path
