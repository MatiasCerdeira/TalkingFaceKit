def test_package_can_be_imported() -> None:
    import talkingfacekit
    import talkingfacekit.io

    assert talkingfacekit.DecodedVideoFrame is not None
    assert talkingfacekit.FaceMeshTrack is not None
    assert talkingfacekit.TalkingFaceSequence is not None
    assert talkingfacekit.VideoMetadata is not None
    assert talkingfacekit.build_mediapipe_face_mesh is not None
    assert talkingfacekit.load_landmark_track is not None
    assert talkingfacekit.render_face_mesh_html is not None
    assert talkingfacekit.save_landmark_track is not None
    assert talkingfacekit.io.inspect_video_metadata is not None
    assert talkingfacekit.stream_video_frames is talkingfacekit.io.stream_video_frames
