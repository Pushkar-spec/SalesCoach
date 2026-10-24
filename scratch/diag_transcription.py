import asyncio
import os
import sys
import time
import wave

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stt_service import STTService

async def main():
    print("Initializing STTService...")
    stt = STTService(
        model_name="base.en",
        device="cuda",
        silence_threshold=400,
        silence_ms=600.0,
        commit_prefix=True,
    )
    stt.initialize()

    events = []
    async def on_transcript(evt):
        events.append(evt)
        status = "FINAL  " if evt.get("is_final") else "PARTIAL"
        seg = evt.get("segment_id")
        text = evt.get("text")
        ts = evt.get("timestamp")
        lat = evt.get("latency_ms")
        print(f"[{status}] seg_id={seg} | time={ts} | lat={lat}ms | text: '{text}'")

    session_id = "test_diag_01"
    session = stt.create_session(session_id, on_transcript)

    wav_path = "tests/fixtures/test_speech.wav"
    with wave.open(wav_path, "rb") as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    chunk_size = 8000 # 250ms at 16kHz 16-bit mono
    print(f"Feeding {len(raw)} bytes ({len(raw)//chunk_size} chunks)...")

    # Feed chunks with 150ms sleep between chunks to simulate streaming
    for i in range(0, len(raw), chunk_size):
        chunk = raw[i:i + chunk_size]
        stt.feed_chunk(session_id, chunk)
        await asyncio.sleep(0.15)

    print("Finished feeding chunks. Flushing...")
    await asyncio.sleep(1.0)
    await stt.flush_and_stop(session_id)
    print("Done. Total events emitted:", len(events))

if __name__ == "__main__":
    asyncio.run(main())
