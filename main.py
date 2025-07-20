import argparse
from src.synthesizer import ChatterboxSynthesizer
from src.utils import detect_device


def main():
    parser = argparse.ArgumentParser(description="ChatterTwin: Voice Cloning with Chatterbox")
    parser.add_argument("command", choices=["synthesize"], help="Only 'synthesize' is supported.")
    parser.add_argument("--audio_prompt", type=str, required=True,
                        help="Path to your reference voice sample (WAV or MP3)")
    parser.add_argument("--text_dir", default="texts", help="Directory containing .txt files to synthesize")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save generated WAV files")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion exaggeration (default 0.5)")
    parser.add_argument("--cfg", type=float, default=0.5, help="CFG scale (default 0.5)")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: cuda / mps / cpu. Default: auto-detect.")
    args = parser.parse_args()

    if args.command == "synthesize":
        device = args.device or detect_device()
        synth = ChatterboxSynthesizer(device=device)
        synth.synthesize(
            text_dir=args.text_dir,
            output_dir=args.output_dir,
            audio_prompt_path=args.audio_prompt,
            exaggeration=args.exaggeration,
            cfg=args.cfg,
        )


if __name__ == "__main__":
    main()
