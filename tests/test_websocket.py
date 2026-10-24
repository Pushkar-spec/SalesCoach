from fastapi.testclient import TestClient
from main import app
from session_manager import default_session_manager

client = TestClient(app)


# Its name begins with test_, so pytest can discover it automatically.
def test_websocket_connection_establishment():
    """Verify that a client can establish a WebSocket connection at /ws."""
    with client.websocket_connect("/ws") as websocket:
        # If the connection handshake succeeds, websocket object is active
        assert websocket is not None


def test_websocket_ping_pong():
    """Verify that a valid ping receives the expected pong response."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "ping", "message": "Hello from client"})
        data = websocket.receive_json()
        assert data == {"type": "pong", "message": "Hello from client"}


def test_websocket_multiple_messages():
    """Verify that multiple messages can be exchanged on the same connection."""
    with client.websocket_connect("/ws") as websocket:
        # Message 1
        websocket.send_json({"type": "ping", "message": "Message 1"})
        res1 = websocket.receive_json()
        assert res1 == {"type": "pong", "message": "Message 1"}

        # Message 2
        websocket.send_json({"type": "ping", "message": "Message 2"})
        res2 = websocket.receive_json()
        assert res2 == {"type": "pong", "message": "Message 2"}

        # Message 3
        websocket.send_json({"type": "ping", "message": "Message 3"})
        res3 = websocket.receive_json()
        assert res3 == {"type": "pong", "message": "Message 3"}


def test_websocket_invalid_json():
    """Verify that invalid JSON text returns an error response without crashing."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text("this is not valid json {{{")
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert "Invalid JSON" in data["message"]

        # Ensure the connection is still alive and handles subsequent valid messages
        websocket.send_json({"type": "ping", "message": "still alive"})
        recovery = websocket.receive_json()
        assert recovery == {"type": "pong", "message": "still alive"}


def test_websocket_unsupported_message_type():
    """Verify that unsupported message types receive an error response."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "unknown_action", "payload": "sample"})
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["message"] == "Unsupported message type"


def test_websocket_non_object_json_array():
    """Verify that a JSON array receives an error response rather than crashing."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text('["item1", "item2"]')
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert "expected a JSON object" in data["message"]


def test_websocket_non_object_json_string():
    """Verify that a raw JSON string receives an error response rather than crashing."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text('"just a string"')
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert "expected a JSON object" in data["message"]


def test_websocket_clean_disconnect():
    """Verify that the client can disconnect cleanly without unhandled server exceptions."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "ping", "message": "disconnect test"})
        data = websocket.receive_json()
        assert data["type"] == "pong"
        # Context manager exit calls websocket.close(), triggering WebSocketDisconnect on the server


def test_websocket_start_session_success():
    """Verify that start_session creates a new active session with a generated UUID."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        resp = websocket.receive_json()
        assert resp["type"] == "session_started"
        assert resp["status"] == "active"
        assert resp["session_id"] is not None
        assert len(resp["session_id"]) > 0


def test_websocket_start_session_unique_ids():
    """Verify that separate sessions receive unique session IDs."""
    with client.websocket_connect("/ws") as ws1, client.websocket_connect("/ws") as ws2:
        ws1.send_json({"type": "start_session"})
        r1 = ws1.receive_json()

        ws2.send_json({"type": "start_session"})
        r2 = ws2.receive_json()

        assert r1["session_id"] != r2["session_id"]


def test_websocket_start_second_session_rejected():
    """Verify that starting a second session on the same connection is rejected."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        r1 = websocket.receive_json()
        assert r1["type"] == "session_started"

        # Second start request
        websocket.send_json({"type": "start_session"})
        r2 = websocket.receive_json()
        assert r2["type"] == "error"
        assert "already exists" in r2["message"]


def test_websocket_status_query_idle_and_active():
    """Verify querying session status when idle and when active."""
    with client.websocket_connect("/ws") as websocket:
        # Idle status before starting session
        websocket.send_json({"type": "status"})
        r_idle = websocket.receive_json()
        assert r_idle == {"type": "status", "session_id": None, "status": "idle"}

        # Start session
        websocket.send_json({"type": "start_session"})
        r_start = websocket.receive_json()
        sid = r_start["session_id"]

        # Active status after starting session
        websocket.send_json({"type": "status"})
        r_active = websocket.receive_json()
        assert r_active == {"type": "status", "session_id": sid, "status": "active"}


def test_websocket_stop_session_success():
    """Verify stopping an active session confirms termination."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        r_start = websocket.receive_json()
        sid = r_start["session_id"]

        websocket.send_json({"type": "stop_session", "session_id": sid})
        r_stop = websocket.receive_json()
        assert r_stop["type"] == "session_stopped"
        assert r_stop["session_id"] == sid
        assert r_stop["status"] == "stopped"

        # Verify status is now idle
        websocket.send_json({"type": "status"})
        r_status = websocket.receive_json()
        assert r_status["status"] == "idle"
        assert r_status["session_id"] is None


def test_websocket_stop_session_invalid_requests():
    """Verify rejection of missing or unauthorized session IDs."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "stop_session", "session_id": ""})
        r1 = websocket.receive_json()
        assert r1["type"] == "error"

        websocket.send_json({"type": "stop_session", "session_id": "unknown-uuid-123"})
        r2 = websocket.receive_json()
        assert r2["type"] == "error"
        assert "Session not found or unauthorized" in r2["message"]


def test_websocket_stop_already_stopped_session():
    """Verify stopping an already stopped session is rejected."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        r_start = websocket.receive_json()
        sid = r_start["session_id"]

        websocket.send_json({"type": "stop_session", "session_id": sid})
        websocket.receive_json()  # Confirmed stopped

        # Try stopping again
        websocket.send_json({"type": "stop_session", "session_id": sid})
        r_err = websocket.receive_json()
        assert r_err["type"] == "error"
        assert "Session is not active" in r_err["message"]


def test_websocket_session_isolation_across_connections():
    """Verify that Connection B cannot stop Connection A's session."""
    with client.websocket_connect("/ws") as ws_a, client.websocket_connect("/ws") as ws_b:
        ws_a.send_json({"type": "start_session"})
        session_a = ws_a.receive_json()
        sid_a = session_a["session_id"]

        # Connection B tries to stop Session A
        ws_b.send_json({"type": "stop_session", "session_id": sid_a})
        r_b = ws_b.receive_json()
        assert r_b["type"] == "error"
        assert "Session not found or unauthorized" in r_b["message"]

        # Verify Session A is still active on Connection A
        ws_a.send_json({"type": "status"})
        status_a = ws_a.receive_json()
        assert status_a["session_id"] == sid_a
        assert status_a["status"] == "active"


def test_websocket_disconnect_cleans_up_active_session():
    """Verify that client disconnect automatically marks its active session as stopped."""
    sid = None
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        resp = websocket.receive_json()
        sid = resp["session_id"]
        assert default_session_manager._sessions[sid]["status"] == "active"

    # Context manager exited, socket closed and cleanup executed
    assert sid is not None
    assert default_session_manager._sessions[sid]["status"] == "stopped"


def test_http_endpoints_remain_functional():
    """Verify that existing HTTP endpoints (/ and /health) continue to work properly."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json() == {"message": "Hello World"}

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"


def test_websocket_start_audio_success():
    """Verify starting audio with valid parameters succeeds."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        start_res = websocket.receive_json()
        sid = start_res["session_id"]

        websocket.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        resp = websocket.receive_json()
        assert resp == {
            "type": "audio_started",
            "session_id": sid,
        }


def test_websocket_start_audio_invalid_config():
    """Verify rejecting unsupported audio format, sample rate, or channels."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        start_res = websocket.receive_json()
        sid = start_res["session_id"]

        # Unsupported format
        websocket.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "mp3",
            "sample_rate": 16000,
            "channels": 1,
        })
        r1 = websocket.receive_json()
        assert r1["type"] == "error"
        assert "pcm16" in r1["message"]

        # Unsupported sample rate
        websocket.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 44100,
            "channels": 1,
        })
        r2 = websocket.receive_json()
        assert r2["type"] == "error"
        assert "16000" in r2["message"]

        # Unsupported channels
        websocket.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 2,
        })
        r3 = websocket.receive_json()
        assert r3["type"] == "error"
        assert "channel" in r3["message"].lower()


def test_websocket_start_audio_without_active_session():
    """Verify rejecting start_audio when no active session exists."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({
            "type": "start_audio",
            "session_id": "non-existent-session-id",
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        resp = websocket.receive_json()
        assert resp["type"] == "error"
        assert "Session not found or unauthorized" in resp["message"]


def test_websocket_start_audio_duplicate_rejected():
    """Verify duplicate start_audio request is rejected."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        start_res = websocket.receive_json()
        sid = start_res["session_id"]

        # First start succeeds
        websocket.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        resp1 = websocket.receive_json()
        assert resp1["type"] == "audio_started"

        # Duplicate start fails
        websocket.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        resp2 = websocket.receive_json()
        assert resp2["type"] == "error"
        assert "already streaming" in resp2["message"]


def test_websocket_binary_audio_frames_and_stop_audio():
    """Verify receiving valid binary audio frames and stopping audio returns correct counters."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        sid = websocket.receive_json()["session_id"]

        websocket.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        assert websocket.receive_json()["type"] == "audio_started"

        # Send 3 binary chunks (each 3200 bytes = 1600 16-bit samples = 100ms of audio)
        chunk = b"\x00\x00" * 1600
        for _ in range(3):
            websocket.send_bytes(chunk)

        # Stop audio
        websocket.send_json({"type": "stop_audio", "session_id": sid})
        stop_resp = websocket.receive_json()
        assert stop_resp["type"] == "audio_stopped"
        assert stop_resp["session_id"] == sid
        assert stop_resp["audio_chunk_count"] == 3
        assert stop_resp["audio_bytes_received"] == 9600
        assert "audio_duration_seconds" in stop_resp
        assert "expected_audio_bytes" in stop_resp


def test_websocket_binary_audio_rejected_empty_and_odd():
    """Verify rejection of empty and odd-length binary frames."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        sid = websocket.receive_json()["session_id"]

        websocket.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        assert websocket.receive_json()["type"] == "audio_started"

        # Empty binary frame
        websocket.send_bytes(b"")
        r_empty = websocket.receive_json()
        assert r_empty["type"] == "error"
        assert "Empty audio frame" in r_empty["message"]

        # Odd-length binary frame (3 bytes)
        websocket.send_bytes(b"\x01\x02\x03")
        r_odd = websocket.receive_json()
        assert r_odd["type"] == "error"
        assert "even number of bytes" in r_odd["message"]

        # Connection should still be functional after errors
        websocket.send_bytes(b"\x00\x00" * 100)
        websocket.send_json({"type": "stop_audio", "session_id": sid})
        r_stop = websocket.receive_json()
        assert r_stop["audio_chunk_count"] == 1
        assert r_stop["audio_bytes_received"] == 200


def test_websocket_binary_audio_before_start_and_after_stop():
    """Verify rejecting binary frames before audio starts and after audio stops."""
    with client.websocket_connect("/ws") as websocket:
        # Before starting a session
        websocket.send_bytes(b"\x00\x00" * 10)
        r1 = websocket.receive_json()
        assert r1["type"] == "error"
        assert "active session" in r1["message"]

        # After starting session, but before start_audio
        websocket.send_json({"type": "start_session"})
        sid = websocket.receive_json()["session_id"]

        websocket.send_bytes(b"\x00\x00" * 10)
        r2 = websocket.receive_json()
        assert r2["type"] == "error"
        assert "not streaming" in r2["message"]

        # Start and stop audio
        websocket.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        assert websocket.receive_json()["type"] == "audio_started"

        websocket.send_json({"type": "stop_audio", "session_id": sid})
        assert websocket.receive_json()["type"] == "audio_stopped"

        # Binary frame after stop_audio
        websocket.send_bytes(b"\x00\x00" * 10)
        r3 = websocket.receive_json()
        assert r3["type"] == "error"
        assert "not streaming" in r3["message"]


def test_websocket_stop_audio_when_not_streaming():
    """Verify rejecting stop_audio when audio is not streaming."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        sid = websocket.receive_json()["session_id"]

        websocket.send_json({"type": "stop_audio", "session_id": sid})
        resp = websocket.receive_json()
        assert resp["type"] == "error"
        assert "not streaming" in resp["message"]


def test_websocket_stop_session_stops_streaming_audio():
    """Verify that stop_session stops audio streaming and resets state."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        sid = websocket.receive_json()["session_id"]

        websocket.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        assert websocket.receive_json()["type"] == "audio_started"

        # Stream one chunk
        websocket.send_bytes(b"\x00\x00" * 100)

        # Stop the session directly
        websocket.send_json({"type": "stop_session", "session_id": sid})
        stop_res = websocket.receive_json()
        assert stop_res["type"] == "session_stopped"
        assert stop_res["status"] == "stopped"

        # Subsequent binary frame must be rejected
        websocket.send_bytes(b"\x00\x00" * 10)
        r_bin = websocket.receive_json()
        assert r_bin["type"] == "error"
        assert "active session" in r_bin["message"]


def test_websocket_audio_cross_connection_isolation():
    """Verify Connection B cannot start, stop, or stream audio for Connection A's session."""
    with client.websocket_connect("/ws") as ws_a, client.websocket_connect("/ws") as ws_b:
        ws_a.send_json({"type": "start_session"})
        sid_a = ws_a.receive_json()["session_id"]

        # Connection B tries to start audio for Session A
        ws_b.send_json({
            "type": "start_audio",
            "session_id": sid_a,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        r_b_start = ws_b.receive_json()
        assert r_b_start["type"] == "error"
        assert "Session not found or unauthorized" in r_b_start["message"]

        # Connection A starts audio legitimately
        ws_a.send_json({
            "type": "start_audio",
            "session_id": sid_a,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        assert ws_a.receive_json()["type"] == "audio_started"

        # Connection B tries to stop audio for Session A
        ws_b.send_json({"type": "stop_audio", "session_id": sid_a})
        r_b_stop = ws_b.receive_json()
        assert r_b_stop["type"] == "error"
        assert "Session not found or unauthorized" in r_b_stop["message"]

        # Connection B sends a binary frame without having its own active session
        ws_b.send_bytes(b"\x00\x00" * 50)
        r_b_bin = ws_b.receive_json()
        assert r_b_bin["type"] == "error"
        assert "active session" in r_b_bin["message"]

        # Connection A sends binary frame and stops cleanly
        ws_a.send_bytes(b"\x00\x00" * 50)
        ws_a.send_json({"type": "stop_audio", "session_id": sid_a})
        r_a_stop = ws_a.receive_json()
        assert r_a_stop["audio_chunk_count"] == 1
        assert r_a_stop["audio_bytes_received"] == 100


def test_websocket_disconnect_cleans_up_audio_state():
    """Verify disconnecting a WebSocket while audio is streaming stops both session and audio."""
    sid = None
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        sid = websocket.receive_json()["session_id"]

        websocket.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })
        assert websocket.receive_json()["type"] == "audio_started"
        assert default_session_manager._sessions[sid]["audio_state"] == "streaming"

    # After disconnect, verify both session and audio state are cleaned up
    assert sid is not None
    assert default_session_manager._sessions[sid]["status"] == "stopped"
    assert default_session_manager._sessions[sid]["audio_state"] == "idle"


def test_websocket_browser_recorder_lifecycle_simulation():
    """Simulate and validate the complete 11-step WebSocket audio lifecycle executed by recorder.js:

    1. Browser opens one WebSocket connection.
    2. Backend creates one session.
    3. Browser receives the session ID.
    4. Browser starts audio using PCM16, 16000 Hz, mono configuration.
    5. Backend acknowledges audio_started.
    6. Binary audio frames arrive while recording.
    7. Backend chunk and byte counters increase correctly.
    8. Stop Call stops audio production and releases microphone resources.
    9. Backend acknowledges stop_audio and returns final counters.
    10. Browser ends the session and closes the WebSocket.
    11. Backend session and audio state are cleaned up.
    """
    sid = None
    # Step 1: The browser opens one WebSocket connection
    with client.websocket_connect("/ws") as ws:
        # Step 2: The backend creates one session
        ws.send_json({"type": "start_session"})
        res_session = ws.receive_json()

        # Step 3: The browser receives the session ID
        assert res_session["type"] == "session_started"
        assert res_session["status"] == "active"
        sid = res_session["session_id"]
        assert len(sid) > 0

        # Step 4: The browser starts audio using supported PCM16, 16000 Hz, mono configuration
        ws.send_json({
            "type": "start_audio",
            "session_id": sid,
            "format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        })

        # Step 5: The backend acknowledges audio_started
        res_audio = ws.receive_json()
        assert res_audio == {"type": "audio_started", "session_id": sid}

        # Step 6 & 7: Binary audio frames arrive while recording; backend counters increase
        # 4 chunks of 250ms (4000 samples @ 2 bytes = 8000 bytes each)
        chunk_250ms = b"\x00\x00" * 4000
        for _ in range(4):
            ws.send_bytes(chunk_250ms)

        # Step 8: Stop Call stops audio production and releases microphone resources
        # Step 9: The backend acknowledges stop_audio and returns final counters
        ws.send_json({"type": "stop_audio", "session_id": sid})
        res_stop_audio = ws.receive_json()
        assert res_stop_audio["type"] == "audio_stopped"
        assert res_stop_audio["session_id"] == sid
        assert res_stop_audio["audio_chunk_count"] == 4
        assert res_stop_audio["audio_bytes_received"] == 32000
        assert "audio_duration_seconds" in res_stop_audio
        assert "expected_audio_bytes" in res_stop_audio

        # Step 10: The browser ends the session
        ws.send_json({"type": "stop_session", "session_id": sid})
        res_stop_session = ws.receive_json()
        assert res_stop_session == {
            "type": "session_stopped",
            "session_id": sid,
            "status": "stopped",
        }

        # Verify status is now idle
        ws.send_json({"type": "status"})
        res_status = ws.receive_json()
        assert res_status == {
            "type": "status",
            "session_id": None,
            "status": "idle",
        }

    # Step 10 (cont) & 11: The browser closes the WebSocket and backend cleans up
    assert sid is not None
    session_record = default_session_manager._sessions[sid]
    assert session_record["status"] == "stopped"
    assert session_record["audio_state"] == "idle"
    assert session_record["audio_chunk_count"] == 4
    assert session_record["audio_bytes_received"] == 32000



