"""Run from the repository root with the analysis extras and local models prepared."""

from pathlib import Path

from talkingfacekit import AnalysisConfig, TalkingFaceCollection, VisualPolicy
from talkingfacekit.demo import render_benchmark_report

videos_dir = Path("data/eval/videos")
videos = TalkingFaceCollection.from_folder(videos_dir)
config = AnalysisConfig(visual_policy=VisualPolicy(), syncnet_model=Path("models/syncnet_v2.model"))
evaluation = videos.evaluate(
    "benchmark/annotations",
    videos_dir=videos_dir,
    config=config,
    video_aliases={"tfk_iphone_1,5min.mp4": "iphone.mp4"},
)
clips = evaluation.selected_segments
result_path = evaluation.save("reports/demo/demo.json", overwrite=True)
render_benchmark_report(result_path, videos_dir, "reports/demo/demo.html", overwrite=True)
print(evaluation.metrics)
