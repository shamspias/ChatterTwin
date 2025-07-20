# ChatterTwin

**Personalized voice cloning & text-to-speech using [Chatterbox TTS](https://huggingface.co/ResembleAI/chatterbox).**

## Features

- Generate speech in your voice (via prompt audio)
- Easy batch synthesis: Put text files in `texts/` and reference audio in `audio_prompts/`
- Supports emotion exaggeration and CFG control
- Automatic device detection (CUDA GPU, Apple Silicon, or CPU)

## Quick Start

1. Put a clean WAV/MP3 of your voice in `audio_prompts/` (e.g. `me.wav`)
2. Put `.txt` files you want to synthesize in `texts/`
3. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
4. Run:
    ```bash
    python main.py synthesize --audio_prompt audio_prompts/me.wav
    ```
    (WAVs will appear in `outputs/`)

## Advanced

- Set emotion: `--exaggeration 0.8`
- Change CFG: `--cfg 0.3`
- Force device: `--device cuda` or `--device mps`

## Project Structure

```

ChatterTwin/
├── audio_prompts/
├── texts/
├── outputs/
├── src/
│   ├── synthesizer.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md

```

---

**Made with Pain using ResembleAI/Chatterbox**
