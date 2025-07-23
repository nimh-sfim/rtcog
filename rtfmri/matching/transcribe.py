import glob
import os.path as osp
import os
import argparse
import whisper

# os.environ['KMP_DUPLICATE_LIB_OK']='True'


def transcribe(file, model):
    result = model.transcribe(file)
    return result["text"]

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using OpenAI Whisper.")
    parser.add_argument("-i", "--in_dir", required=True, help="The input directory where the audio files are located.", dest="in_dir")
    parser.add_argument("-o", "--out_dir", required=True, help="The output directory where the transcripts will be saved.", dest="out_dir")
    parser.add_argument("-p", "--prefix", required=True, help="The prefix for the audio files.", dest="prefix")
    parser.add_argument("--model", default="turbo", help="Whisper model to use (default: turbo).")

    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    prefix = args.prefix
    model_name = args.model

    model = whisper.load_model(model_name)

    pattern = osp.join(in_dir, f"{prefix}.hit???.wav")
    files = glob.glob(pattern)

    if not files:
        print(f"No files found matching: {pattern}")
        return

    for file_path in sorted(files):
        print(f"++ Transcribing {file_path}")
        transcript = transcribe(file_path, model)

        # Save transcript to a .txt file with the same base name
        base = osp.splitext(file_path)[0]
        out_path = osp.join(out_dir, f"{base}.transcript.txt")
        with open(out_path, "w") as f:
            f.write(transcript)
        print(f"Transcript saved to {out_path}")

if __name__ == "__main__":
    main()