def test_package_can_be_imported() -> None:
    import talkingfacekit
    import talkingfacekit.io

    assert talkingfacekit.TalkingFaceSequence is not None
    assert talkingfacekit.VideoMetadata is not None
    assert talkingfacekit.io.inspect_video_metadata is not None
