import wave
import numpy as np
import whisper

model = whisper.load_model("base.en", device="cuda")
with wave.open("tests/fixtures/test_speech.wav", "rb") as wf:
    raw = wf.readframes(wf.getnframes())
audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

prompt = "Hello and welcome to the Realtime AI sales"
print("Prompt:", prompt)
for cut in [1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]:
    tail = audio[int(cut * 16000):]
    res = model.transcribe(tail, initial_prompt=prompt, language="en", temperature=0.1)
    text = res["text"].strip()
    print(f"Cut at {cut:.2f}s -> '{text}'")
    # Check if text repeats words from prompt
    combined = prompt + " " + text
    print(f"  Combined: '{combined}'")
