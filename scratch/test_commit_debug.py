import asyncio
import os
import sys
import wave

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stt_service import STTService, Segment

orig_commit = Segment.commit
def debug_commit(self, data):
    w_before = len(self.window)
    c_before = self.committed
    orig_commit(self, data)
    if self.committed != c_before:
        print(f'>>> COMMITTED! New committed: "{self.committed}", dropped {w_before - len(self.window)} chunks')

Segment.commit = debug_commit

async def test():
    stt = STTService(
        model_name="base.en",
        device="cuda",
        commit_prefix=True,
        commit_after_ms=3000.0,
        commit_margin_ms=1000.0,
    )
    stt.initialize()

    with wave.open("tests/fixtures/test_speech.wav", "rb") as wf:
        raw = wf.readframes(wf.getnframes())

    long_raw = raw + raw + raw

    events = []
    async def cb(e):
        events.append(e)
        if e.get("is_final"):
            print(f"[FINAL] seg={e['segment_id']}: '{e['text']}'")
        else:
            print(f"[PARTIAL] seg={e['segment_id']}: '{e['text']}'")

    stt.create_session("s1", cb)
    for i in range(0, len(long_raw), 8000):
        stt.feed_chunk("s1", long_raw[i:i + 8000])
        await asyncio.sleep(0.12)

    await stt.flush_and_stop("s1")

if __name__ == "__main__":
    asyncio.run(test())
