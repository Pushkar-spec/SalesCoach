import asyncio
import os
import sys
import wave

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stt_service import STTService

async def main():
    stt = STTService(
        model_name="base.en",
        device="cuda",
        silence_threshold=400,
        silence_ms=600.0,
        commit_prefix=True,
        commit_after_ms=4000.0,
        commit_margin_ms=1000.0,
    )
    stt.initialize()

    events = []
    async def on_transcript(evt):
        events.append(evt)
        status = "FINAL  " if evt.get("is_final") else "PARTIAL"
        seg = evt.get("segment_id")
        text = evt.get("text")
        lat = evt.get("latency_ms")
        print(f"[{status}] seg_id={seg} | lat={lat}ms | text: '{text}'")

    session_id = "test_repeat_01"
    session = stt.create_session(session_id, on_transcript)

    wav_path = "tests/fixtures/test_speech.wav"
    with wave.open(wav_path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())

    # Concatenate 2 sentences: raw + 0.3s silence + raw
    silence_03s = b"\x00" * int(16000 * 2 * 0.3)
    long_raw = raw + silence_03s + raw

    chunk_size = 8000 # 250ms
    print(f"Feeding {len(long_raw)} bytes ({len(long_raw)//chunk_size} chunks)...")

    for i in range(0, len(long_raw), chunk_size):
        chunk = long_raw[i:i + chunk_size]
        stt.feed_chunk(session_id, chunk)
        await asyncio.sleep(0.12)

    print("Flushing...")
    await asyncio.sleep(1.0)
    await stt.flush_and_stop(session_id)
    print("\n--- SUMMARY OF ALL FINAL SEGMENTS ---")
    finals = [e for e in events if e.get("is_final")]
    for f in finals:
        print(f"Segment #{f['segment_id']}: '{f['text']}'")

if __name__ == "__main__":
    asyncio.run(main())
