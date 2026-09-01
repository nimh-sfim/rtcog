import sys

import pytest

from rtcog.matching.transcribe import main, transcript_path


def test_transcript_path_uses_requested_output_directory(tmp_path):
    input_path = tmp_path / "audio" / "sub-001.hit001.wav"
    output_dir = tmp_path / "transcripts"

    result = transcript_path(str(input_path), str(output_dir))

    assert result == str(output_dir / "sub-001.hit001.transcript.txt")


def test_transcript_path_discards_input_subdirectories():
    result = transcript_path("session/audio/run.hit042.wav", "results")

    assert result == "results/run.hit042.transcript.txt"


def test_help_does_not_require_whisper(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["transcribe.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert "Transcribe audio using OpenAI Whisper" in capsys.readouterr().out


def test_no_matches_does_not_require_whisper_or_create_output(
    tmp_path, monkeypatch, capsys
):
    input_dir = tmp_path / "audio"
    output_dir = tmp_path / "transcripts"
    input_dir.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "transcribe.py",
            "--in_dir",
            str(input_dir),
            "--out_dir",
            str(output_dir),
            "--prefix",
            "sub-001",
        ],
    )

    main()

    assert not output_dir.exists()
    assert "No files found matching" in capsys.readouterr().out
