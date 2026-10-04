"""
Voice-memo recipe book
----------------------
1. Put audio files (mp3, m4a, wav, ogg) inside a folder called "recordings"
2. Run:  python make_recipe_book.py
3. Open "recipe_book.md" to see your recipes

Everything runs on your own laptop:
- Whisper (faster-whisper) turns speech into text
- Gemma (through Ollama) turns the messy text into a clean recipe
"""

import os
import requests
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

# ---------- Settings you can change ----------
RECORDINGS_FOLDER = "recordings"
OUTPUT_FILE = "recipe_book.md"
OLLAMA_MODEL = "gemma3:1b"   # the open model we download with Ollama
WHISPER_SIZE = "base"        # "base" uses less memory, "small" is more accurate
CHUNK_SECONDS = 30           # audio is processed in pieces this long
MAX_MINUTES = 3              # only use the first 3 minutes of each file (None = all)
LANGUAGE = None              # None = auto detect. For Hindi use "hi", English "en"
# ---------------------------------------------

AUDIO_TYPES = (".mp3", ".m4a", ".wav", ".ogg", ".mp4", ".aac")

PROMPT = """You are helping a family save a loved one's recipe.
Below is a rough transcript of them explaining a dish out loud.
Turn it into a clean recipe using this format:

# Dish name
**Told by:** (leave blank if unknown)
## Ingredients
- item (with quantity if they said it)
## Steps
1. ...
## Tips from the cook
- keep their personal advice and sayings as close to their own words as possible

Rules:
- Write in the same language the transcript is in.
- Do NOT invent ingredients or quantities. If they did not say a quantity, write "as needed" or "to taste".
- Keep the warm, personal voice of the speaker.

Transcript:
"""


def transcribe(model, path):
    """Speech -> text using Whisper, in small chunks so it uses little memory."""
    audio = decode_audio(path, sampling_rate=16000)
    chunk_size = CHUNK_SECONDS * 16000
    if MAX_MINUTES:
        audio = audio[: MAX_MINUTES * 60 * 16000]

    pieces = []
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
        segments, info = model.transcribe(chunk, language=LANGUAGE)
        pieces.append(" ".join(seg.text.strip() for seg in segments))
        print(f"  ...done {min(start + chunk_size, len(audio)) // 16000} seconds")
    return " ".join(pieces)


def make_recipe(transcript):
    """Messy text -> clean recipe using Gemma running in Ollama."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": PROMPT + transcript, "stream": False},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def main():
    if not os.path.isdir(RECORDINGS_FOLDER):
        print(f'Please create a folder named "{RECORDINGS_FOLDER}" and put your audio files in it.')
        return

    files = sorted(f for f in os.listdir(RECORDINGS_FOLDER) if f.lower().endswith(AUDIO_TYPES))
    if not files:
        print("No audio files found in the recordings folder.")
        return

    print("Loading Whisper (first time it downloads, please wait)...")
    whisper = WhisperModel(WHISPER_SIZE, compute_type="int8")

    book = ["# Our Family Recipe Book\n"]

    for name in files:
        print(f"\nListening to: {name}")
        transcript = transcribe(whisper, os.path.join(RECORDINGS_FOLDER, name))
        print("Transcript preview:", transcript[:150], "...")

        print("Writing the recipe with Gemma...")
        try:
            recipe = make_recipe(transcript)
        except requests.exceptions.ConnectionError:
            print("Could not reach Ollama. Is it running? Open the Ollama app and try again.")
            return

        book.append(recipe)
        book.append("\n---\n")

        # Save the raw transcript too, so you can check Gemma did not change anything
        with open(os.path.join(RECORDINGS_FOLDER, name + ".txt"), "w", encoding="utf-8") as f:
            f.write(transcript)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(book))

    print(f"\nDone! Open {OUTPUT_FILE} to see your recipe book.")


if __name__ == "__main__":
    main()