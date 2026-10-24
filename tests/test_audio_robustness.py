"""
tests/test_audio_robustness.py
Step 2.6: Robustness, noise tolerance, timeouts, stall detection,
and graceful cleanup tests for the audio streaming pipeline.
"""

import math
import struct
import time
import pytest
from fastapi.testclient import TestClient
from main import app
from session_manager import (
    SessionManager,
    default_session_manager,
    InvalidSessionError,
    AudioStateError,
    InvalidAudioDataError,
)

client = TestClient(app)


def generate_pcm16_sine_wave(frequency_hz: float = 440.0, sample_rate: int = 16000, duration_sec: float = 0.25) -> bytes:
    """Generate deterministic synthetic PCM16 mono sine wave chunk (8000 bytes for 0.25s at 16kHz)."""
    num_samples = int(sample_rate * duration_sec)
    samples = []
    for i in range(num_samples):
        # Sine wave oscillating between -0.8 and 0.8 to avoid clipping
        t = i / sample_rate
        val = 0.8 * math.sin(2.0 * math.pi * frequency_hz * t)
        int_val = int(val * 32767.0)
        samples.append(int_val)
    return struct.pack(f"<{len(samples)}h", *samples)


def generate_pcm16_noise(num_samples: int = 4000) -> bytes:
    """Generate deterministic pseudo-random PCM16 noise using an LCG formula."""
    samples = []
    seed = 42
    for _ in range(num_samples):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        # Map to signed 16-bit range [-16000, 16000]
        val = (seed % 32001) - 16000
        samples.append(val)
    return struct.pack(f"<{len(samples)}h", *samples)


# --- 1. Synthetic Audio Signal Tests ---

def test_synthetic_silence_transport():
    """Verify silence frames (all zero bytes) transport without alteration."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_session"})
        sid = ws.receive_json()["session_id"]

        ws.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        assert ws.receive_json()["type"] == "audio_started"

        # 4000 zero samples = 8000 zero bytes
        silence_chunk = b"\x00" * 8000
        ws.send_bytes(silence_chunk)

        ws.send_json({"type": "stop_audio", "session_id": sid})
        stop_resp = ws.receive_json()
        assert stop_resp["type"] == "audio_stopped"
        assert stop_resp["audio_chunk_count"] == 1
        assert stop_resp["audio_bytes_received"] == 8000

        ws.send_json({"type": "stop_session", "session_id": sid})
        assert ws.receive_json()["type"] == "session_stopped"


def test_synthetic_sine_wave_transport():
    """Verify mathematically generated sine wave PCM16 frames transport without byte loss."""
    sine_chunk = generate_pcm16_sine_wave(frequency_hz=440.0, sample_rate=16000, duration_sec=0.25)
    assert len(sine_chunk) == 8000

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_session"})
        sid = ws.receive_json()["session_id"]

        ws.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        assert ws.receive_json()["type"] == "audio_started"

        for _ in range(4):
            ws.send_bytes(sine_chunk)

        ws.send_json({"type": "stop_audio", "session_id": sid})
        stop_resp = ws.receive_json()
        assert stop_resp["type"] == "audio_stopped"
        assert stop_resp["audio_chunk_count"] == 4
        assert stop_resp["audio_bytes_received"] == 32000

        ws.send_json({"type": "stop_session", "session_id": sid})
        assert ws.receive_json()["type"] == "session_stopped"


def test_synthetic_noise_transport():
    """Verify high-entropy synthetic noisy audio frames transport without byte corruption."""
    noise_chunk = generate_pcm16_noise(num_samples=4000)
    assert len(noise_chunk) == 8000

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_session"})
        sid = ws.receive_json()["session_id"]

        ws.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        assert ws.receive_json()["type"] == "audio_started"

        for _ in range(3):
            ws.send_bytes(noise_chunk)

        ws.send_json({"type": "stop_audio", "session_id": sid})
        stop_resp = ws.receive_json()
        assert stop_resp["type"] == "audio_stopped"
        assert stop_resp["audio_chunk_count"] == 3
        assert stop_resp["audio_bytes_received"] == 24000

        ws.send_json({"type": "stop_session", "session_id": sid})
        assert ws.receive_json()["type"] == "session_stopped"


# --- 2. Interruption and Disconnection Tests ---

def test_websocket_disconnect_during_streaming_cleans_up_session():
    """Verify that when a client drops connection mid-stream, session & audio are cleaned up."""
    mgr = SessionManager()
    cid = mgr.generate_connection_id()
    session = mgr.create_session(cid)
    sid = session["session_id"]

    mgr.start_audio(cid, sid, "pcm16", 16000, 1)
    chunk = b"\x00\x01" * 4000
    mgr.record_audio_chunk(cid, len(chunk))
    assert mgr.get_active_session(cid)["audio_state"] == "streaming"
    assert mgr.get_active_session(cid)["audio_chunk_count"] == 1

    # Simulate abrupt disconnect
    cleaned = mgr.cleanup_connection(cid)
    assert cleaned is not None
    assert cleaned["status"] == "stopped"
    assert cleaned["audio_state"] == "idle"
    assert cleaned["audio_chunk_count"] == 1
    assert cleaned["audio_bytes_received"] == 8000
    assert cleaned["audio_duration_seconds"] >= 0.0
    assert cleaned["expected_audio_bytes"] >= 0

    # Connection mapping must be removed
    assert mgr.get_active_session(cid) is None


def test_stop_session_while_audio_actively_streaming():
    """Verify that calling stop_session while audio is streaming automatically stops audio cleanly."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_session"})
        sid = ws.receive_json()["session_id"]

        ws.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        assert ws.receive_json()["type"] == "audio_started"
        ws.send_bytes(b"\x00\x02" * 4000)

        # Skip stop_audio and directly call stop_session
        ws.send_json({"type": "stop_session", "session_id": sid})
        stop_resp = ws.receive_json()
        assert stop_resp["type"] == "session_stopped"
        assert stop_resp["session_id"] == sid
        assert stop_resp["status"] == "stopped"

        # Further audio chunks should be rejected
        ws.send_bytes(b"\x00\x02" * 4000)
        err = ws.receive_json()
        assert err["type"] == "error"


# --- 3. Robustness Against Malformed and Out-of-Order Messages ---

def test_audio_chunk_rejected_when_audio_idle():
    """Verify binary frame is rejected if sent before start_audio."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_session"})
        sid = ws.receive_json()["session_id"]

        # Send binary chunk without start_audio
        ws.send_bytes(b"\x00\x01" * 4000)
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "Audio is not streaming" in err["message"]

        ws.send_json({"type": "stop_session", "session_id": sid})
        assert ws.receive_json()["type"] == "session_stopped"


def test_audio_chunk_rejected_after_audio_stopped():
    """Verify binary frame is rejected if sent after stop_audio."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_session"})
        sid = ws.receive_json()["session_id"]

        ws.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        assert ws.receive_json()["type"] == "audio_started"

        ws.send_json({"type": "stop_audio", "session_id": sid})
        assert ws.receive_json()["type"] == "audio_stopped"

        # Send binary chunk after audio stopped
        ws.send_bytes(b"\x00\x01" * 4000)
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "Audio is not streaming" in err["message"]

        ws.send_json({"type": "stop_session", "session_id": sid})
        assert ws.receive_json()["type"] == "session_stopped"


def test_empty_audio_chunk_rejected():
    """Verify empty binary frame is rejected with an error."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_session"})
        sid = ws.receive_json()["session_id"]

        ws.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        assert ws.receive_json()["type"] == "audio_started"

        # Send 0-byte frame
        ws.send_bytes(b"")
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "Empty audio frame received" in err["message"]

        ws.send_json({"type": "stop_audio", "session_id": sid})
        assert ws.receive_json()["type"] == "audio_stopped"

        ws.send_json({"type": "stop_session", "session_id": sid})
        assert ws.receive_json()["type"] == "session_stopped"


def test_odd_byte_length_audio_chunk_rejected():
    """Verify odd byte length binary frame is rejected because 16-bit samples require 2 bytes."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_session"})
        sid = ws.receive_json()["session_id"]

        ws.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        assert ws.receive_json()["type"] == "audio_started"

        # Send 5 bytes (odd length)
        ws.send_bytes(b"\x01\x02\x03\x04\x05")
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "even number of bytes" in err["message"]

        ws.send_json({"type": "stop_audio", "session_id": sid})
        assert ws.receive_json()["type"] == "audio_stopped"

        ws.send_json({"type": "stop_session", "session_id": sid})
        assert ws.receive_json()["type"] == "session_stopped"


# --- 4. Stall Detection and Timing Diagnostics ---

def test_audio_staleness_detector():
    """Verify check_audio_staleness detects when streaming stalls past threshold."""
    mgr = SessionManager()
    cid = mgr.generate_connection_id()
    session = mgr.create_session(cid)
    sid = session["session_id"]

    # Before audio starts, staleness is False
    assert mgr.check_audio_staleness(cid, max_idle_seconds=0.1) is False

    mgr.start_audio(cid, sid, "pcm16", 16000, 1)

    # Immediately after start, not yet stale with large threshold
    assert mgr.check_audio_staleness(cid, max_idle_seconds=5.0) is False

    # Simulate stall with tiny threshold
    time.sleep(0.05)
    assert mgr.check_audio_staleness(cid, max_idle_seconds=0.02) is True

    # Record a chunk: staleness resets
    mgr.record_audio_chunk(cid, len(b"\x00\x01" * 100))
    assert mgr.check_audio_staleness(cid, max_idle_seconds=5.0) is False

    # Sleep past tiny threshold again: becomes stale
    time.sleep(0.05)
    assert mgr.check_audio_staleness(cid, max_idle_seconds=0.02) is True

    # Stop audio: staleness is False when idle
    mgr.stop_audio(cid, sid)
    assert mgr.check_audio_staleness(cid, max_idle_seconds=0.01) is False


def test_audio_metrics_control_query():
    """Verify client can query active audio streaming metrics via audio_metrics message."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_session"})
        sid = ws.receive_json()["session_id"]

        ws.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        assert ws.receive_json()["type"] == "audio_started"

        # Stream 2 chunks
        ws.send_bytes(b"\x00\x01" * 4000)
        ws.send_bytes(b"\x00\x02" * 4000)

        # Query metrics
        ws.send_json({"type": "audio_metrics", "session_id": sid})
        metrics = ws.receive_json()
        assert metrics["type"] == "audio_metrics"
        assert metrics["session_id"] == sid
        assert metrics["audio_state"] == "streaming"
        assert metrics["audio_chunk_count"] == 2
        assert metrics["audio_bytes_received"] == 16000
        assert metrics["audio_duration_seconds"] >= 0.0
        assert metrics["expected_audio_bytes"] >= 0
        assert metrics["time_since_last_chunk_seconds"] is not None

        ws.send_json({"type": "stop_audio", "session_id": sid})
        assert ws.receive_json()["type"] == "audio_stopped"

        ws.send_json({"type": "stop_session", "session_id": sid})
        assert ws.receive_json()["type"] == "session_stopped"


def test_repeated_stop_calls_handled_safely():
    """Verify repeated stop calls on already stopped sessions/audio return handled errors rather than crashing."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_session"})
        sid = ws.receive_json()["session_id"]

        ws.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        assert ws.receive_json()["type"] == "audio_started"

        ws.send_json({"type": "stop_audio", "session_id": sid})
        assert ws.receive_json()["type"] == "audio_stopped"

        # Second stop_audio call
        ws.send_json({"type": "stop_audio", "session_id": sid})
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "Audio is not streaming" in err["message"]

        # Stop session
        ws.send_json({"type": "stop_session", "session_id": sid})
        assert ws.receive_json()["type"] == "session_stopped"

        # Second stop_session call
        ws.send_json({"type": "stop_session", "session_id": sid})
        err2 = ws.receive_json()
        assert err2["type"] == "error"
        assert "Session is not active" in err2["message"]
