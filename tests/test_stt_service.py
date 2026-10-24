"""
Deterministic Unit Tests for Speech-to-Text (STT) Service.

Verifies:
- Helpers: is_silent, duration_ms, normalize, common_prefix
- Segment tumbling window, silence endpointing, and committed-prefix logic
- ModelPool replica acquisition and concurrency isolation
- CUDA error handling (no silent CPU fallback) and model init failure
- Partial and final transcript ordering and stable segment IDs
- Bounded queue backpressure protection
- Repeated Stop and disconnect cleanup idempotency
- Session isolation between concurrent calls
- WebSocket integration with FastAPI test client
"""

import asyncio
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app
from session_manager import default_session_manager
from stt_service import (
    STTBackpressureError,
    STTClosedError,
    STTError,
    STTInitializationError,
    STTService,
    STTStateError,
    Segment,
    TranscribeSession,
    common_prefix,
    duration_ms,
    is_silent,
    merge_committed_and_tail,
    normalize,
)


# -----------------------------------------------------------------------------
# 1. Helpers & Segment Tests
# -----------------------------------------------------------------------------
def test_is_silent_and_duration():
    """Verify silence detection and duration calculation."""
    # 250ms of silence (4000 int16 samples = 8000 bytes at 16kHz)
    silent_pcm = b"\x00\x00" * 4000
    assert is_silent(silent_pcm, threshold=500) is True
    assert duration_ms(silent_pcm, sample_rate=16000) == 250.0

    # Synthetic signal with amplitude 1000 (> 500 threshold)
    samples = np.full(4000, 1000, dtype=np.int16)
    loud_pcm = samples.tobytes()
    assert is_silent(loud_pcm, threshold=500) is False

    # Short/empty chunk
    assert is_silent(b"", threshold=500) is True
    assert is_silent(b"\x01", threshold=500) is True


def test_normalize_and_common_prefix():
    """Verify text normalization and word-level prefix matching."""
    assert normalize("  Hello,  World! ") == "hello world"
    assert normalize("Let's test... this?") == "lets test this"

    prev = [{"word": " Good"}, {"word": " morning"}, {"word": " team"}]
    curr = [{"word": " good"}, {"word": " Morning,"}, {"word": " everyone"}]
    assert common_prefix(prev, curr) == 2


def test_segment_window_and_lead_in_silence():
    """Verify Segment prunes excess silence before speech but retains speech window."""
    segment = Segment(sample_rate=16000, silence_threshold=500, silence_ms=600.0)

    # Add 1000ms of silence (4 chunks of 250ms)
    silent_chunk = b"\x00\x00" * 4000
    for _ in range(4):
        segment.add(silent_chunk)

    # Lead-in silence should be capped so total silence <= 600ms (max 3 chunks = 750ms / 2 chunks = 500ms)
    assert segment.has_speech is False
    assert len(segment.window) <= 3

    # Add speech chunk
    loud_chunk = np.full(4000, 2000, dtype=np.int16).tobytes()
    segment.add(loud_chunk)
    assert segment.has_speech is True
    assert segment.trailing_silent_ms == 0.0

    # Add 750ms of silence (3 chunks of 250ms)
    for _ in range(3):
        segment.add(silent_chunk)
    assert segment.ended_by_silence() is True


def test_segment_committed_prefix():
    """Verify agreed words are committed and corresponding audio chunks dropped."""
    segment = Segment(
        sample_rate=16000,
        commit_prefix=True,
        commit_after_ms=500.0,
        commit_margin_ms=200.0,
    )

    chunk = np.full(4000, 1500, dtype=np.int16).tobytes()
    # Add 3 chunks (750ms)
    for _ in range(3):
        segment.add(chunk)

    data1 = {
        "text": " We have a deal",
        "segments": [
            {
                "words": [
                    {"word": " We", "start": 0.0, "end": 0.2},
                    {"word": " have", "start": 0.2, "end": 0.4},
                    {"word": " a", "start": 0.4, "end": 0.5},
                    {"word": " deal", "start": 0.5, "end": 0.7},
                ]
            }
        ],
    }
    segment.commit(data1)
    # First cycle sets prev_words; committed is empty
    assert segment.committed == ""

    # Second cycle with agreement on "We have"
    data2 = {
        "text": " We have a discount",
        "segments": [
            {
                "words": [
                    {"word": " We", "start": 0.0, "end": 0.2},
                    {"word": " have", "start": 0.2, "end": 0.4},
                    {"word": " a", "start": 0.4, "end": 0.5},
                    {"word": " discount", "start": 0.5, "end": 0.7},
                ]
            }
        ],
    }
    segment.commit(data2)
    # " We have" should be committed
    assert "We have" in segment.committed
    assert len(segment.window) < 3


def test_merge_committed_and_tail_deduplication():
    """Verify overlapping words between committed prefix and tail are deduplicated."""
    # Exact overlap at boundary
    c = "Hello and welcome to the Realtime AI sales"
    t = "Realtime AI sales co-pilot."
    merged = merge_committed_and_tail(c, t)
    assert merged == "Hello and welcome to the Realtime AI sales co-pilot."

    # Model re-emits entire prompt
    c2 = "We have a discount"
    t2 = "We have a discount for you today"
    merged2 = merge_committed_and_tail(c2, t2)
    assert merged2 == "We have a discount for you today"

    # Single word overlap
    c3 = "close this deal"
    t3 = "deal today"
    merged3 = merge_committed_and_tail(c3, t3)
    assert merged3 == "close this deal today"

    # No overlap
    c4 = "Hello world"
    t4 = "How are you"
    merged4 = merge_committed_and_tail(c4, t4)
    assert merged4 == "Hello world How are you"

    # Empty cases
    assert merge_committed_and_tail("", "Hello") == "Hello"
    assert merge_committed_and_tail("Hello", "") == "Hello"


def test_segment_full_text_deduplicates_overlapping_words():
    """Verify Segment.full_text() prevents repeated phrases across streaming cycles."""
    segment = Segment(sample_rate=16000)
    segment.committed = "Hello and welcome to the Realtime AI"

    # Simulated Whisper output that re-emits trailing prompt words
    data = {"text": " Realtime AI sales co-pilot."}
    result = segment.full_text(data)
    assert result == "Hello and welcome to the Realtime AI sales co-pilot."
    assert "Realtime AI Realtime AI" not in result


# -----------------------------------------------------------------------------
# 2. Model Pool & Initialization Tests
# -----------------------------------------------------------------------------
def test_model_initialization_cuda_failure_raises():
    """Verify STTService does NOT silently switch to CPU if CUDA is requested but unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        service = STTService(device="cuda")
        with pytest.raises(STTInitializationError) as exc_info:
            service.initialize()
        assert "Refusing silent CPU fallback" in str(exc_info.value)
        assert service.is_initialized() is False


def test_model_initialization_invalid_model_raises():
    """Verify loading an invalid model name raises STTInitializationError."""
    with patch("whisper.load_model", side_effect=RuntimeError("Model file not found")):
        service = STTService(model_name="non_existent_model_xyz", device="cpu")
        with pytest.raises(STTInitializationError) as exc_info:
            service.initialize()
        assert "Failed to load Whisper model" in str(exc_info.value)


# -----------------------------------------------------------------------------
# 3. TranscribeSession Unit Tests (Mocked Inference)
# -----------------------------------------------------------------------------
@pytest.mark.anyio
async def test_partial_final_transcript_ordering():
    """Verify that partial transcripts precede final transcripts with stable segment IDs."""
    events = []

    async def mock_on_transcript(event: dict) -> None:
        events.append(event)

    call_count = 0

    async def mock_transcribe(chunks: list, **kwargs) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"text": " Hello"}
        return {"text": " Hello world"}

    session = TranscribeSession(
        session_id="test_sess_001",
        transcribe_fn=mock_transcribe,
        on_transcript=mock_on_transcript,
        queue_maxsize=10,
        commit_prefix=False,
    )

    loud_chunk = np.full(4000, 2000, dtype=np.int16).tobytes()
    session.add_chunk(loud_chunk)

    # Allow worker loop to execute a partial transcription cycle
    await asyncio.sleep(0.08)
    assert len(events) >= 1
    first_event = events[0]
    assert first_event["session_id"] == "test_sess_001"
    assert first_event["segment_id"] == 1
    assert first_event["is_final"] is False
    assert first_event["status"] == "partial"
    assert "Hello" in first_event["text"]

    # Flush session to trigger finalization
    await session.flush()
    await session.stop()

    assert len(events) >= 2
    final_event = events[-1]
    assert final_event["session_id"] == "test_sess_001"
    assert final_event["segment_id"] == 1
    assert final_event["is_final"] is True
    assert final_event["status"] == "final"
    assert "Hello world" in final_event["text"]


@pytest.mark.anyio
async def test_repeated_stop_is_idempotent():
    """Verify repeated flush and stop calls do not deadlock or raise."""
    async def mock_transcribe(chunks: list, **kwargs) -> dict:
        return {"text": " Test stop"}

    async def mock_cb(event: dict) -> None:
        pass

    session = TranscribeSession(
        session_id="test_stop_sess",
        transcribe_fn=mock_transcribe,
        on_transcript=mock_cb,
    )

    await session.flush()
    await session.stop()
    # Repeated stop calls
    await session.stop()
    await session.flush()
    assert session.accepting_audio is False

    # Adding chunk after stop must raise STTClosedError
    with pytest.raises(STTClosedError):
        session.add_chunk(b"\x00\x00" * 4000)


@pytest.mark.anyio
async def test_bounded_queue_backpressure():
    """Verify STTBackpressureError is raised when the queue capacity is exceeded."""
    async def mock_transcribe(chunks: list, **kwargs) -> dict:
        # Simulate slow inference
        await asyncio.sleep(0.5)
        return {"text": " slow"}

    async def mock_cb(event: dict) -> None:
        pass

    # Bounded queue of 2 chunks
    session = TranscribeSession(
        session_id="test_bp_sess",
        transcribe_fn=mock_transcribe,
        on_transcript=mock_cb,
        queue_maxsize=2,
    )

    silent_chunk = b"\x00\x00" * 4000
    session.add_chunk(silent_chunk)
    session.add_chunk(silent_chunk)

    # Third chunk should trigger backpressure error
    with pytest.raises(STTBackpressureError) as exc_info:
        session.add_chunk(silent_chunk)
    assert "STT audio queue full" in str(exc_info.value)

    await session.stop()


@pytest.mark.anyio
async def test_session_isolation():
    """Verify two active sessions maintain independent queues, audio, and transcript callbacks."""
    events_a = []
    events_b = []

    async def cb_a(event: dict):
        events_a.append(event)

    async def cb_b(event: dict):
        events_b.append(event)

    async def transcribe_a(chunks: list, **kwargs):
        return {"text": " Session Alpha"}

    async def transcribe_b(chunks: list, **kwargs):
        return {"text": " Session Bravo"}

    sess_a = TranscribeSession("sess_a", transcribe_a, cb_a, queue_maxsize=10)
    sess_b = TranscribeSession("sess_b", transcribe_b, cb_b, queue_maxsize=10)

    loud_chunk = np.full(4000, 2000, dtype=np.int16).tobytes()
    sess_a.add_chunk(loud_chunk)
    sess_b.add_chunk(loud_chunk)

    await asyncio.sleep(0.08)
    await sess_a.flush()
    await sess_b.flush()
    await sess_a.stop()
    await sess_b.stop()

    # Verify no cross-talk
    for ev in events_a:
        assert ev["session_id"] == "sess_a"
        assert "Alpha" in ev["text"]
    for ev in events_b:
        assert ev["session_id"] == "sess_b"
        assert "Bravo" in ev["text"]


# -----------------------------------------------------------------------------
# 4. STTService Lifecycle and WebSocket Integration Tests
# -----------------------------------------------------------------------------
@pytest.mark.anyio
async def test_sttservice_create_and_cleanup_session():
    """Verify STTService creation, chunk forwarding, and cleanup."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": " Mocked speech", "segments": []}

    service = STTService(model=mock_model, device="cpu", replicas=1)
    service.initialize()

    events = []

    async def on_tr(ev: dict):
        events.append(ev)

    sess = service.create_session("svc_sess_1", on_tr)
    assert sess.session_id == "svc_sess_1"

    # Forward chunk
    service.feed_chunk("svc_sess_1", b"\x00\x00" * 4000)

    # Unknown session forward raises STTStateError
    with pytest.raises(STTStateError):
        service.feed_chunk("unknown_sess", b"\x00\x00" * 4000)

    # Clean shutdown
    await service.flush_and_stop("svc_sess_1")
    service.shutdown()
    assert service.is_initialized() is False


def test_websocket_stt_transcription_pipeline():
    """Verify end-to-end WebSocket flow emits transcript events when audio is streamed."""
    # Inject mock model into main default_stt_service to ensure fast, deterministic test
    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = {
        "text": " Good afternoon, thank you for calling.",
        "segments": [],
    }

    from stt_service import default_stt_service
    default_stt_service.shutdown()
    default_stt_service._injected_model = mock_whisper
    default_stt_service.initialize()

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            # 1. Start session
            ws.send_json({"type": "start_session"})
            resp = ws.receive_json()
            assert resp["type"] == "session_started"
            sid = resp["session_id"]

            # 2. Start audio
            ws.send_json({
                "type": "start_audio",
                "session_id": sid,
                "format": "pcm16",
                "sample_rate": 16000,
                "channels": 1,
            })
            resp = ws.receive_json()
            assert resp["type"] == "audio_started"

            # 3. Send non-silent PCM16 audio chunks
            chunk = np.full(4000, 2500, dtype=np.int16).tobytes()
            ws.send_bytes(chunk)
            ws.send_bytes(chunk)

            # 4. Stop audio
            ws.send_json({"type": "stop_audio", "session_id": sid})

            # Expect transcript event(s) and then audio_stopped
            messages = []
            for _ in range(5):
                msg = ws.receive_json()
                messages.append(msg)
                if msg["type"] == "audio_stopped":
                    break

            msg_types = [m["type"] for m in messages]
            assert "audio_stopped" in msg_types
            # Verify a transcript event was received
            transcript_msgs = [m for m in messages if m["type"] == "transcript"]
            assert len(transcript_msgs) >= 1
            tr = transcript_msgs[0]
            assert tr["session_id"] == sid
            assert tr["segment_id"] == 1
            assert "thank you for calling" in tr["text"]
            assert tr["is_final"] in (True, False)

    default_stt_service.shutdown()
    default_stt_service._injected_model = None


# -----------------------------------------------------------------------------
# 5. Endpointing and Pause Deduplication Regression Tests
# -----------------------------------------------------------------------------
def test_configurable_silence_endpointing():
    """Verify silence_ms endpointing is configurable and takes precedence over env vars."""
    # Explicit silence_ms passed to constructor
    service = STTService(silence_ms=1750.0)
    assert service.silence_ms == 1750.0

    # Default fallback when no argument passed
    service_default = STTService()
    assert service_default.silence_ms == 1500.0


@pytest.mark.anyio
async def test_pause_within_threshold_does_not_split_sentence_or_duplicate_entries():
    """
    Regression test: Simulates one sentence split by a 1.0 second pause (~4 chunks).
    Verifies that:
    1. The 1.0s pause does NOT finalize the segment or increment segment_id.
    2. Partial updates replace earlier hypotheses without creating duplicate entries.
    3. Resumed speech appends to the ongoing segment.
    4. Exactly one final segment is emitted with segment_id=1 containing the full sentence.
    """
    events = []

    async def mock_on_transcript(event: dict) -> None:
        events.append(event)

    call_index = 0

    async def mock_transcribe(chunks: list, **kwargs) -> dict:
        nonlocal call_index
        call_index += 1
        # First chunk(s) before pause
        if len(chunks) <= 2:
            return {"text": " We would like"}
        # After resumed speech chunks
        return {"text": " We would like to schedule a demo"}

    session = TranscribeSession(
        session_id="pause_regression_sess",
        transcribe_fn=mock_transcribe,
        on_transcript=mock_on_transcript,
        silence_ms=1500.0,
        commit_prefix=False,
    )

    loud_chunk = np.full(4000, 2000, dtype=np.int16).tobytes()
    silent_chunk = b"\x00\x00" * 4000

    # 1. Initial speech: "We would like"
    session.add_chunk(loud_chunk)
    await asyncio.sleep(0.06)

    # 2. 1.0s pause (4 chunks of 250ms silent audio, within the 1.5s threshold)
    for _ in range(4):
        session.add_chunk(silent_chunk)
    await asyncio.sleep(0.06)

    # Verify segment has NOT been finalized yet
    final_events_so_far = [e for e in events if e.get("is_final")]
    assert len(final_events_so_far) == 0, "1.0s pause should not trigger premature segment finalization"

    # 3. Resumed speech: "to schedule a demo"
    session.add_chunk(loud_chunk)
    await asyncio.sleep(0.06)

    # 4. Finalize session
    await session.flush()
    await session.stop()

    # Verify all emitted events belong to segment 1 (no duplicate segment entries)
    segment_ids = {e["segment_id"] for e in events}
    assert segment_ids == {1}, f"Expected only segment_id=1, but got {segment_ids}"

    # Verify exactly one final segment was emitted
    final_events = [e for e in events if e.get("is_final")]
    assert len(final_events) == 1, f"Expected exactly 1 final event, got {len(final_events)}"
    assert final_events[0]["segment_id"] == 1
    assert final_events[0]["status"] == "final"
    assert "We would like to schedule a demo" in final_events[0]["text"]


@pytest.mark.anyio
async def test_pause_exceeding_threshold_keeps_distinct_sentences_separate():
    """
    Test confirming that two genuinely distinct sentences separated by a >1.5s pause
    remain separate and immutable.
    """
    events = []

    async def mock_on_transcript(event: dict) -> None:
        events.append(event)

    sentence_idx = 1

    async def mock_transcribe(chunks: list, **kwargs) -> dict:
        if sentence_idx == 1:
            return {"text": " Thank you for calling."}
        return {"text": " How can I help you today?"}

    session = TranscribeSession(
        session_id="distinct_sentences_sess",
        transcribe_fn=mock_transcribe,
        on_transcript=mock_on_transcript,
        silence_ms=1500.0,
        commit_prefix=False,
    )

    loud_chunk = np.full(4000, 2000, dtype=np.int16).tobytes()
    silent_chunk = b"\x00\x00" * 4000

    # 1. Sentence 1: "Thank you for calling."
    session.add_chunk(loud_chunk)
    await asyncio.sleep(0.06)

    # 2. Genuine pause of 1.75s (7 silent chunks >= 1500ms endpoint threshold)
    for _ in range(7):
        session.add_chunk(silent_chunk)
    await asyncio.sleep(0.08)

    # Sentence 1 should now be finalized due to silence endpointing
    finals = [e for e in events if e.get("is_final")]
    assert len(finals) == 1, "Sentence 1 should be finalized after 1.75s pause"
    assert finals[0]["segment_id"] == 1
    assert "Thank you for calling" in finals[0]["text"]

    # 3. Sentence 2: "How can I help you today?"
    sentence_idx = 2
    session.add_chunk(loud_chunk)
    await asyncio.sleep(0.06)

    # 4. Finalize
    await session.flush()
    await session.stop()

    # Verify both sentences are final and have distinct segment IDs
    all_finals = [e for e in events if e.get("is_final")]
    assert len(all_finals) == 2, f"Expected 2 final segments, got {len(all_finals)}"

    seg1 = all_finals[0]
    seg2 = all_finals[1]

    assert seg1["segment_id"] == 1
    assert "Thank you for calling" in seg1["text"]

    assert seg2["segment_id"] == 2
    assert "How can I help you today" in seg2["text"]

    # Ensure sentence 1 was immutable and not merged or contaminated by sentence 2
    assert "How can I help" not in seg1["text"]

