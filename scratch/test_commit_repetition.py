import sys
import wave
import numpy as np
import torch
import whisper

def main():
    print("Loading base.en model...")
    model = whisper.load_model("base.en", device="cuda")

    with wave.open("tests/fixtures/test_speech.wav", "rb") as wf:
        raw = wf.readframes(wf.getnframes())

    # Full transcription first
    audio_full = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    res_full = model.transcribe(audio_full, word_timestamps=True, language="en")
    print("FULL TRANSCRIPT:", res_full["text"])
    print("\nWORDS:")
    for seg in res_full["segments"]:
        for w in seg.get("words", []):
            print(f"  {w['start']:.2f}s - {w['end']:.2f}s: '{w['word']}'")

    # Now let's simulate what Segment.commit() does:
    # Say we commit up to 'Sales' (around 2.5s)
    # Cut audio on a 250ms chunk boundary (e.g. 2.5s = 2500ms -> 10 chunks of 250ms = 2500ms)
    cut_sec = 2.5
    cut_samples = int(cut_sec * 16000)
    audio_tail = audio_full[cut_samples:]

    print(f"\n--- Testing Tail audio from {cut_sec}s ({len(audio_tail)/16000:.2f}s) ---")
    
    # 1. Tail WITHOUT initial_prompt
    res_no_prompt = model.transcribe(audio_tail, language="en", temperature=0.1)
    print("Tail without initial_prompt:", f"'{res_no_prompt['text']}'")

    # 2. Tail WITH initial_prompt="Hello and welcome to the Realtime AI Sales"
    prompt = "Hello and welcome to the Realtime AI Sales"
    res_with_prompt = model.transcribe(audio_tail, language="en", initial_prompt=prompt, temperature=0.1)
    print("Tail WITH initial_prompt:   ", f"'{res_with_prompt['text']}'")

    # 3. What if cut is slightly inside 'Sales' (e.g. 2.25s)?
    cut_sec_early = 2.25
    audio_tail_early = audio_full[int(cut_sec_early * 16000):]
    res_early_prompt = model.transcribe(audio_tail_early, language="en", initial_prompt=prompt, temperature=0.1)
    print(f"Tail from {cut_sec_early}s WITH prompt:", f"'{res_early_prompt['text']}'")

    # What if cut is at 2.0s?
    cut_sec_2 = 2.0
    audio_tail_2 = audio_full[int(cut_sec_2 * 16000):]
    res_2_prompt = model.transcribe(audio_tail_2, language="en", initial_prompt=prompt, temperature=0.1)
    print(f"Tail from {cut_sec_2}s WITH prompt:", f"'{res_2_prompt['text']}'")

if __name__ == "__main__":
    main()
