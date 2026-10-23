"""Reusable tools for processing talking-face video sequences."""

from talkingfacekit.active_speaker import (
    ActiveSpeakerProvenance,
    ActiveSpeakerResult,
    ActiveSpeakerScoreWindow,
    AudioTimelineRepair,
    CandidatePolicy,
    CandidateRun,
    FaceObservation,
    PreliminarySegment,
    ShotBoundary,
    SpeechInterval,
)
from talkingfacekit.analysis import (
    AnalysisConfig,
    AnalysisTimings,
    TalkingFaceSegment,
    VideoAnalysisResult,
)
from talkingfacekit.audio import DecodedAudioChunk
from talkingfacekit.collection import (
    AnalysisFailure,
    CollectionAnalysisResult,
    CollectionProgress,
    TalkingFaceCollection,
    VideoAnalysisOutcome,
)
from talkingfacekit.evaluation import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationInterval,
    EvaluationMetrics,
    EvaluationResult,
)
from talkingfacekit.io.audio import stream_audio_chunks
from talkingfacekit.io.landmarks import load_landmark_track, save_landmark_track
from talkingfacekit.io.video import stream_video_frames
from talkingfacekit.mesh import FaceMeshTrack
from talkingfacekit.metadata import VideoMetadata
from talkingfacekit.processing import VideoAnalyzer, evaluate_benchmark
from talkingfacekit.rendering import render_face_mesh_html
from talkingfacekit.sequence import TalkingFaceSequence
from talkingfacekit.synchronization import (
    SynchronizationDecision,
    SynchronizationPolicy,
    SynchronizationResult,
    SynchronizationWindow,
)
from talkingfacekit.tracking import FaceLandmarkTrack, LandmarkTracker
from talkingfacekit.tracking.mediapipe_mesh import build_mediapipe_face_mesh
from talkingfacekit.video import DecodedVideoFrame, VideoSource
from talkingfacekit.visual import (
    VisualDecision,
    VisualMetrics,
    VisualObservation,
    VisualPolicy,
    VisualRegion,
    VisualResult,
    assess_visual_suitability,
)

__all__ = [
    "ActiveSpeakerProvenance",
    "ActiveSpeakerResult",
    "ActiveSpeakerScoreWindow",
    "AnalysisConfig",
    "AnalysisFailure",
    "AnalysisTimings",
    "AudioTimelineRepair",
    "CandidatePolicy",
    "CandidateRun",
    "CollectionAnalysisResult",
    "CollectionProgress",
    "DecodedAudioChunk",
    "DecodedVideoFrame",
    "EvaluationCase",
    "EvaluationCaseResult",
    "EvaluationInterval",
    "EvaluationMetrics",
    "EvaluationResult",
    "FaceLandmarkTrack",
    "FaceMeshTrack",
    "FaceObservation",
    "LandmarkTracker",
    "PreliminarySegment",
    "ShotBoundary",
    "SpeechInterval",
    "SynchronizationDecision",
    "SynchronizationPolicy",
    "SynchronizationResult",
    "SynchronizationWindow",
    "TalkingFaceCollection",
    "TalkingFaceSegment",
    "TalkingFaceSequence",
    "VideoAnalysisOutcome",
    "VideoAnalysisResult",
    "VideoAnalyzer",
    "VideoMetadata",
    "VideoSource",
    "VisualDecision",
    "VisualMetrics",
    "VisualObservation",
    "VisualPolicy",
    "VisualRegion",
    "VisualResult",
    "assess_visual_suitability",
    "build_mediapipe_face_mesh",
    "evaluate_benchmark",
    "load_landmark_track",
    "render_face_mesh_html",
    "save_landmark_track",
    "stream_audio_chunks",
    "stream_video_frames",
]
