import asyncio
import os
import sys
import wave

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import stt_service
from stt_service import STTService, FlushRequest

# Test the chunk-level ended_by_silence check in worker loop
orig_worker_loop = stt_service.TranscribeSession._worker_loop
async def patched_worker_loop(self):
    segment = self._new_segment()
    import time

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
                if segment.ended_by_silence():
                    segment = await self._close_segment(segment)
                elif segment.is_full():
                    segment = await self._close_segment(segment)

            if not segment.has_speech:
                continue

            window_duration_ms = sum(stt_service.duration_ms(c, self.sample_rate) for c in segment.window)
            if not segment.committed and window_duration_ms < self.min_partial_ms:
                continue

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
        except Exception:
            pass

    try:
        if segment.has_speech or segment.committed:
            await self._close_segment(segment)
    except Exception:
        pass
    self._stopped_event.set()

stt_service.TranscribeSession._worker_loop = patched_worker_loop

async def main():
    print("Testing pause endpointing with chunk-level silence detection...")
    stt = STTService(
        model_name="base.en",
        device="cuda",
        silence_threshold=400,
        silence_ms=1500.0,
    )
    # Ensure silence_ms is 1500.0
    stt.silence_ms = 1500.0
    stt.initialize()

    with wave.open("tests/fixtures/test_speech.wav", "rb") as wf:
        raw = wf.readframes(wf.getnframes())

    p1 = raw[: int(3.0 * 16000 * 2)]
    p2 = raw[int(3.86 * 16000 * 2):]

    chunk_sz = 8000 # 250ms
    silence_chunk = b"\x00" * chunk_sz

    # --- TEST 1: Sentence split by 1.0s pause (< 1.5s) ---
    print("\n--- TEST 1: Sentence split by 1.0s pause (< 1.5s) ---")
    events_t1 = []
    async def cb_t1(e):
        events_t1.append(e)
        status = "FINAL  " if e.get("is_final") else "PARTIAL"
        print(f"[{status}] seg={e['segment_id']}: '{e['text']}'")

    s1 = stt.create_session("sess_t1", cb_t1)
    s1.silence_ms = 1500.0
    for i in range(0, len(p1), chunk_sz):
        stt.feed_chunk("sess_t1", p1[i:i + chunk_sz])
        await asyncio.sleep(0.05)

    # 1.0s pause = 4 chunks
    for _ in range(4):
        stt.feed_chunk("sess_t1", silence_chunk)
        await asyncio.sleep(0.05)

    for i in range(0, len(p2), chunk_sz):
        stt.feed_chunk("sess_t1", p2[i:i + chunk_sz])
        await asyncio.sleep(0.05)

    await stt.flush_and_stop("sess_t1")
    finals_t1 = [e for e in events_t1 if e.get("is_final")]
    print(f"Total finals in Test 1: {len(finals_t1)}")
    for f in finals_t1:
        print(f"  Final Seg #{f['segment_id']}: '{f['text']}'")

    # --- TEST 2: Two distinct sentences separated by 2.0s pause (> 1.5s) ---
    print("\n--- TEST 2: Two distinct sentences separated by 2.0s pause (> 1.5s) ---")
    events_t2 = []
    async def cb_t2(e):
        events_t2.append(e)
        status = "FINAL  " if e.get("is_final") else "PARTIAL"
        print(f"[{status}] seg={e['segment_id']}: '{e['text']}'")

    s2 = stt.create_session("sess_t2", cb_t2)
    s2.silence_ms = 1500.0
    for i in range(0, len(p1), chunk_sz):
        stt.feed_chunk("sess_t2", p1[i:i + chunk_sz])
        await asyncio.sleep(0.05)

    # 2.0s pause = 8 chunks
    for _ in range(8):
        stt.feed_chunk("sess_t2", silence_chunk)
        await asyncio.sleep(0.05)

    for i in range(0, len(p2), chunk_sz):
        stt.feed_chunk("sess_t2", p2[i:i + chunk_sz])
        await asyncio.sleep(0.05)

    await stt.flush_and_stop("sess_t2")
    finals_t2 = [e for e in events_t2 if e.get("is_final")]
    print(f"Total finals in Test 2: {len(finals_t2)}")
    for f in finals_t2:
        print(f"  Final Seg #{f['segment_id']}: '{f['text']}'")

if __name__ == "__main__":
    asyncio.run(main())
