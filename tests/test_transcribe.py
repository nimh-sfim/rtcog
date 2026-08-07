from rtcog.matching.transcribe import transcript_path


def test_transcript_path_uses_requested_output_directory(tmp_path):
    input_path = tmp_path / "audio" / "sub-001.hit001.wav"
    output_dir = tmp_path / "transcripts"

    result = transcript_path(str(input_path), str(output_dir))

    assert result == str(output_dir / "sub-001.hit001.transcript.txt")


def test_transcript_path_discards_input_subdirectories():
    result = transcript_path("session/audio/run.hit042.wav", "results")

    assert result == "results/run.hit042.transcript.txt"
