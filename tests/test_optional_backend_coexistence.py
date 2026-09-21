import pytest


def test_deeptalk_and_mediapipe_can_import_in_one_environment() -> None:
    deeptalk = pytest.importorskip("deeptalk_asd")
    mediapipe = pytest.importorskip("mediapipe")
    opencv = pytest.importorskip("cv2")

    assert deeptalk.ASDDetectorFactory is not None
    assert mediapipe.Image is not None
    assert opencv.__version__ == "4.11.0"
