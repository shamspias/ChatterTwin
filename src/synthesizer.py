from pathlib import Path
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from src.utils import detect_device


class ChatterboxSynthesizer:
    """
    Synthesize speech using Chatterbox with a reference audio prompt.
    """

    def __init__(self, device=None):
        self.device = device or detect_device()
        self.model = ChatterboxTTS.from_pretrained(device=self.device)

    def synthesize(self, text_dir, output_dir, audio_prompt_path, exaggeration=0.5, cfg=0.5):
        text_dir = Path(text_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        txt_files = [p for p in text_dir.rglob("*.txt") if p.is_file()]
        if not txt_files:
            print(f"No .txt files found in {text_dir}")
            return

        if not audio_prompt_path or not Path(audio_prompt_path).is_file():
            print(f"Audio prompt not found at {audio_prompt_path}")
            return

        for tf in txt_files:
            text = tf.read_text(encoding="utf-8").strip()
            if not text:
                print(f"{tf.name} is empty, skipping.")
                continue

            output_path = output_dir / (tf.stem + ".wav")
            wav = self.model.generate(
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg=cfg,
            )
            ta.save(str(output_path), wav, self.model.sr)
            print(f"Saved: {output_path}")
