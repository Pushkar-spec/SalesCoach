"""
tests/test_audio_durations.py
Step 2.6: Deterministic tests for varying audio recording durations,
chunk counts, partial final chunks, immediate stops, and repeated start/stop cycles.
"""

import time
import pytest
from fastapi.testclient import TestClient
from main import app
from session_manager import SessionManager, default_session_manager

client = TestClient(app)


def test_short_recording_single_chunk():
    """Verify a short recording with a single 250ms chunk (8000 bytes)."""
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

        # 1 chunk of 250ms (4000 samples @ 2 bytes = 8000 bytes)
        chunk = b"\x00\x01" * 4000
        ws.send_bytes(chunk)

        ws.send_json({"type": "stop_audio", "session_id": sid})
        stop_resp = ws.receive_json()
        assert stop_resp["type"] == "audio_stopped"
        assert stop_resp["session_id"] == sid
        assert stop_resp["audio_chunk_count"] == 1
        assert stop_resp["audio_bytes_received"] == 8000
        assert stop_resp["audio_duration_seconds"] >= 0.0
        assert stop_resp["expected_audio_bytes"] >= 0

        ws.send_json({"type": "stop_session", "session_id": sid})
        assert ws.receive_json()["type"] == "session_stopped"


def test_longer_recording_multiple_chunks():
    """Verify a longer recording with 20 consecutive chunks (5.0 seconds = 160,000 bytes)."""
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

        chunk_size = 8000  # 4000 samples of 16-bit PCM
        chunk = b"\x12\x34" * 4000
        num_chunks = 20

        for _ in range(num_chunks):
            ws.send_bytes(chunk)

        ws.send_json({"type": "stop_audio", "session_id": sid})
        stop_resp = ws.receive_json()
        assert stop_resp["type"] == "audio_stopped"
        assert stop_resp["audio_chunk_count"] == num_chunks
        assert stop_resp["audio_bytes_received"] == num_chunks * chunk_size
        assert stop_resp["audio_bytes_received"] == 160000

        ws.send_json({"type": "stop_session", "session_id": sid})
        assert ws.receive_json()["type"] == "session_stopped"


def test_recording_with_partial_final_chunk():
    """Verify recording with full chunks followed by a partial final chunk (e.g. 1500 samples = 3000 bytes)."""
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

        # Full chunk 1: 8000 bytes
        ws.send_bytes(b"\x00\x00" * 4000)
        # Full chunk 2: 8000 bytes
        ws.send_bytes(b"\x01\x01" * 4000)
        # Partial final chunk: 1500 samples = 3000 bytes
        ws.send_bytes(b"\x02\x02" * 1500)

        ws.send_json({"type": "stop_audio", "session_id": sid})
        stop_resp = ws.receive_json()
        assert stop_resp["type"] == "audio_stopped"
        assert stop_resp["audio_chunk_count"] == 3
        assert stop_resp["audio_bytes_received"] == 19000  # 8000 + 8000 + 3000

        ws.send_json({"type": "stop_session", "session_id": sid})
        assert ws.receive_json()["type"] == "session_stopped"


def test_stop_immediately_after_start_zero_chunks():
    """Verify stopping audio immediately after starting (0 chunks streamed)."""
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

        # Stop immediately without streaming any audio chunks
        ws.send_json({"type": "stop_audio", "session_id": sid})
        stop_resp = ws.receive_json()
        assert stop_resp["type"] == "audio_stopped"
        assert stop_resp["session_id"] == sid
        assert stop_resp["audio_chunk_count"] == 0
        assert stop_resp["audio_bytes_received"] == 0
        assert stop_resp["audio_duration_seconds"] >= 0.0

        ws.send_json({"type": "stop_session", "session_id": sid})
        assert ws.receive_json()["type"] == "session_stopped"


def test_repeated_start_stop_cycles_same_connection():
    """Verify multiple consecutive start_audio -> stream -> stop_audio cycles within a session."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_session"})
        sid = ws.receive_json()["session_id"]

        for cycle in range(3):
            # Start audio
            ws.send_json({
                "type": "start_audio",
                "session_id": sid,
                "format": "pcm16",
                "sample_rate": 16000,
                "channels": 1,
            })
            assert ws.receive_json()["type"] == "audio_started"

            # Stream 2 chunks (16,000 bytes)
            ws.send_bytes(b"\x00\x00" * 4000)
            ws.send_bytes(b"\x00\x00" * 4000)

            # Stop audio
            ws.send_json({"type": "stop_audio", "session_id": sid})
            stop_resp = ws.receive_json()
            assert stop_resp["type"] == "audio_stopped"
            assert stop_resp["audio_chunk_count"] == 2
            assert stop_resp["audio_bytes_received"] == 16000

        ws.send_json({"type": "stop_session", "session_id": sid})
        assert ws.receive_json()["type"] == "session_stopped"


def test_repeated_full_session_cycles():
    """Verify repeated complete call lifecycles (session + audio + teardown) on successive connections."""
    for i in range(3):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "start_session"})
            s_resp = ws.receive_json()
            sid = s_resp["session_id"]

            ws.send_json({
                "type": "start_audio",
                "session_id": sid,
                "format": "pcm16",
                "sample_rate": 16000,
                "channels": 1,
            })
            assert ws.receive_json()["type"] == "audio_started"

            ws.send_bytes(b"\xaa\x55" * 2000)  # 4000 bytes

            ws.send_json({"type": "stop_audio", "session_id": sid})
            assert ws.receive_json()["audio_chunk_count"] == 1

            ws.send_json({"type": "stop_session", "session_id": sid})
            assert ws.receive_json()["type"] == "session_stopped"
