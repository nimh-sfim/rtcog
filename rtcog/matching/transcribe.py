import glob
import os
import os.path as osp
import argparse

def transcribe(file, model):
    result = model.transcribe(file)
    return result["text"]

def transcript_path(file_path, out_dir):
    """Return the transcript path for an input audio file."""
    audio_name = osp.splitext(osp.basename(file_path))[0]
    return osp.join(out_dir, f"{audio_name}.transcript.txt")

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using OpenAI Whisper.")
    parser.add_argument("-i", "--in_dir", required=True, help="The input directory where the audio files are located.", dest="in_dir")
    parser.add_argument("-o", "--out_dir", required=True, help="The output directory where the transcripts will be saved.", dest="out_dir")
    parser.add_argument("-p", "--prefix", required=True, help="The prefix for the audio files.", dest="prefix")
    parser.add_argument("-m", "--model", default="turbo", help="Whisper model to use (default: turbo).")

    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    prefix = args.prefix
    model_name = args.model

    pattern = osp.join(in_dir, f"{prefix}.hit???.wav")
    files = glob.glob(pattern)

    if not files:
        print(f"No files found matching: {pattern}")
        return

    import whisper

    os.makedirs(out_dir, exist_ok=True)
    model = whisper.load_model(model_name)

    for file_path in sorted(files):
        print(f"++ Transcribing {file_path}")
        transcript = transcribe(file_path, model)

        out_path = transcript_path(file_path, out_dir)
        with open(out_path, "w") as f:
            f.write(transcript)
        print(f"Transcript saved to {out_path}")

if __name__ == "__main__":
    main()
