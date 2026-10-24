import asyncio
import os
import sys
import wave

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import stt_service
from stt_service import STTService, Segment, normalize

def merge_committed_and_tail(committed: str, tail: str) -> str:
    committed = committed.rstrip()
    tail = tail.lstrip()
    if not committed:
        return tail
    if not tail:
        return committed

    committed_words = committed.split()
    tail_words = tail.split()

    max_overlap = min(len(committed_words), len(tail_words), 15)
    best_overlap = 0

    for k in range(max_overlap, 0, -1):
        c_slice = [normalize(w) for w in committed_words[-k:]]
        t_slice = [normalize(w) for w in tail_words[:k]]
        if c_slice == t_slice:
            best_overlap = k
            break

    if best_overlap > 0:
        tail_words_clean = tail_words[best_overlap:]
        tail_clean = " ".join(tail_words_clean)
        return f"{committed} {tail_clean}".strip()

    return f"{committed} {tail}".strip()

# Monkey patch full_text
def full_text(self, data):
    tail = (data or {}).get("text") or ""
    return merge_committed_and_tail(self.committed, tail)

Segment.full_text = full_text

# Update _sync_transcribe options
orig_sync_transcribe = STTService._sync_transcribe
def patched_sync_transcribe(self, chunks, context=None, word_timestamps=False):
    import numpy as np
    raw_pcm = b"".join(chunks)
    if len(raw_pcm) < 2:
        return {"text": "", "segments": []}
    audio_arr = np.frombuffer(raw_pcm[: len(raw_pcm) // 2 * 2], dtype=np.int16).flatten().astype(np.float32) / 32768.0
    use_fp16 = self.device.startswith("cuda")
    with self.model_pool.acquire() as replica:
        return replica.transcribe(
            audio_arr,
            fp16=use_fp16,
            language="en",
            initial_prompt=context,
            word_timestamps=word_timestamps,
            temperature=0.0, # Greedy, deterministic, zero hallucination
            condition_on_previous_text=False,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

STTService._sync_transcribe = patched_sync_transcribe

# Patch worker loop to not close segments prematurely on 1 cycle of same_text
orig_worker_loop = stt_service.TranscribeSession._worker_loop
async def patched_worker_loop(self):
    segment = self._new_segment()
    import time
    from stt_service import FlushRequest

    while not self.should_stop[0]:
        try:
            await asyncio.sleep(0.02)
            drained = []
            while not self.queue.empty():
                drained.append(self.queue.get_nowait())

            if not drained and not segment.has_speech:
                continue

            for item in drained:
                if item is None:
                    continue
                if isinstance(item, FlushRequest):
                    segment = await self._close_segment(segment)
                    if not item.done.done():
                        item.done.set_result(True)
                    continue

                segment.add(item)
                if segment.is_full():
                    segment = await self._close_segment(segment)

            if segment.ended_by_silence():
                segment = await self._close_segment(segment)
                continue

            if not segment.has_speech:
                continue

            # Run partial transcription
            start = time.perf_counter()
            options = {}
            if segment.context():
                options["context"] = segment.context()
            if segment.wants_words():
                options["word_timestamps"] = True

            data = await self._safe_transcribe(segment.window, **options)
            if data is None:
                continue

            full_text_str = segment.full_text(data)
            if self.commit_prefix:
                segment.commit(data)

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if full_text_str:
                await self._emit_transcript(full_text_str, is_final=False, latency_ms=elapsed_ms)

        except asyncio.CancelledError:
            break
        except Exception as exc:
            pass

    try:
        if segment.has_speech or segment.committed:
            await self._close_segment(segment)
    except Exception:
        pass
    self._stopped_event.set()

stt_service.TranscribeSession._worker_loop = patched_worker_loop

async def main():
    print("Initializing STTService with targeted improvements...")
    stt = STTService(
        model_name="base.en",
        device="cuda",
        silence_threshold=400,
        silence_ms=700.0,
        commit_prefix=True,
        commit_after_ms=3000.0,
        commit_margin_ms=1000.0,
    )
    stt.initialize()

    with wave.open("tests/fixtures/test_speech.wav", "rb") as wf:
        raw = wf.readframes(wf.getnframes())

    # Two distinct spoken sentences separated by 0.8s of silence
    silence_08s = b"\x00" * int(16000 * 2 * 0.8)
    speech_data = raw + silence_08s + raw

    events = []
    async def cb(e):
        events.append(e)
        status = "FINAL  " if e.get("is_final") else "PARTIAL"
        seg = e.get("segment_id")
        text = e.get("text")
        lat = e.get("latency_ms")
        print(f"[{status}] seg_id={seg} | lat={lat:5.1f}ms | text: '{text}'")

    session_id = "test_fixed_01"
    stt.create_session(session_id, cb)

    chunk_size = 8000 # 250ms
    print(f"Streaming {len(speech_data)} bytes...")
    for i in range(0, len(speech_data), chunk_size):
        chunk = speech_data[i:i + chunk_size]
        stt.feed_chunk(session_id, chunk)
        await asyncio.sleep(0.12)

    print("Flushing...")
    await asyncio.sleep(0.5)
    await stt.flush_and_stop(session_id)

    print("\n" + "=" * 70)
    print("SUMMARY OF ALL FINAL SEGMENTS:")
    finals = [e for e in events if e.get("is_final")]
    for f in finals:
        print(f"Segment #{f['segment_id']}: '{f['text']}'")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
