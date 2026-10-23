import pytest
from session_manager import (
    ActiveSessionExistsError,
    AudioStateError,
    InvalidAudioConfigError,
    InvalidAudioDataError,
    InvalidSessionError,
    SessionManager,
)


def test_session_manager_create_session():
    """Verify session creation with unique IDs, initial active status, and idle audio state."""
    sm = SessionManager()
    conn_id = sm.generate_connection_id()
    assert conn_id

    session = sm.create_session(conn_id)
    assert session["session_id"]
    assert session["connection_id"] == conn_id
    assert session["status"] == "active"
    assert "created_at" in session
    assert session["audio_state"] == "idle"
    assert session["audio_chunk_count"] == 0
    assert session["audio_bytes_received"] == 0

    # Check active session retrieval
    active = sm.get_active_session(conn_id)
    assert active is not None
    assert active["session_id"] == session["session_id"]


def test_session_manager_unique_session_ids():
    """Verify that every created session receives a distinct UUID."""
    sm = SessionManager()
    conn1 = sm.generate_connection_id()
    conn2 = sm.generate_connection_id()

    s1 = sm.create_session(conn1)
    s2 = sm.create_session(conn2)
    assert s1["session_id"] != s2["session_id"]


def test_session_manager_reject_duplicate_active_session():
    """Verify that a single connection cannot start a second session while active."""
    sm = SessionManager()
    conn = sm.generate_connection_id()
    sm.create_session(conn)

    with pytest.raises(ActiveSessionExistsError, match="already exists"):
        sm.create_session(conn)


def test_session_manager_stop_session():
    """Verify that stopping an active session marks it as stopped."""
    sm = SessionManager()
    conn = sm.generate_connection_id()
    s = sm.create_session(conn)
    sid = s["session_id"]

    stopped = sm.stop_session(conn, sid)
    assert stopped["status"] == "stopped"
    assert "stopped_at" in stopped

    # Active session should now be None
    assert sm.get_active_session(conn) is None


def test_session_manager_stop_invalid_session_ids():
    """Verify rejection of empty, whitespace, or non-existent session IDs."""
    sm = SessionManager()
    conn = sm.generate_connection_id()

    with pytest.raises(InvalidSessionError, match="non-empty string"):
        sm.stop_session(conn, "")

    with pytest.raises(InvalidSessionError, match="Session not found or unauthorized"):
        sm.stop_session(conn, "non-existent-uuid")


def test_session_manager_ownership_isolation():
    """Verify that connection A cannot stop connection B's session."""
    sm = SessionManager()
    conn_a = sm.generate_connection_id()
    conn_b = sm.generate_connection_id()

    session_a = sm.create_session(conn_a)

    # Connection B attempts to stop Connection A's session
    with pytest.raises(InvalidSessionError, match="Session not found or unauthorized"):
        sm.stop_session(conn_b, session_a["session_id"])

    # Session A must still remain active
    assert sm.get_active_session(conn_a) is not None
    assert sm.get_active_session(conn_a)["status"] == "active"


def test_session_manager_reject_stop_already_stopped():
    """Verify that stopping an already stopped session raises an error."""
    sm = SessionManager()
    conn = sm.generate_connection_id()
    s = sm.create_session(conn)
    sm.stop_session(conn, s["session_id"])

    with pytest.raises(InvalidSessionError, match="Session is not active"):
        sm.stop_session(conn, s["session_id"])


def test_session_manager_cleanup_on_disconnect():
    """Verify that disconnecting a connection marks its active session as stopped."""
    sm = SessionManager()
    conn = sm.generate_connection_id()
    s = sm.create_session(conn)

    cleaned = sm.cleanup_connection(conn)
    assert cleaned is not None
    assert cleaned["status"] == "stopped"
    assert sm.get_active_session(conn) is None

    # Cleanup when no active session exists should return None
    assert sm.cleanup_connection(conn) is None


# -----------------------------------------------------------------------------
# Audio State & Streaming Tests (Step 2.2)
# -----------------------------------------------------------------------------

def test_session_manager_start_audio_valid():
    """Verify starting audio with valid parameters transitions audio_state to streaming."""
    sm = SessionManager()
    conn = sm.generate_connection_id()
    s = sm.create_session(conn)
    sid = s["session_id"]

    session = sm.start_audio(conn, sid, audio_format="pcm16", sample_rate=16000, channels=1)
    assert session["audio_state"] == "streaming"


def test_session_manager_start_audio_invalid_configs():
    """Verify rejection of unsupported audio format, sample rate, or channels."""
    sm = SessionManager()
    conn = sm.generate_connection_id()
    s = sm.create_session(conn)
    sid = s["session_id"]

    # Unsupported format
    with pytest.raises(InvalidAudioConfigError, match="Only 'pcm16' is supported"):
        sm.start_audio(conn, sid, audio_format="mp3", sample_rate=16000, channels=1)

    # Unsupported sample rate
    with pytest.raises(InvalidAudioConfigError, match="Only 16000 Hz is supported"):
        sm.start_audio(conn, sid, audio_format="pcm16", sample_rate=44100, channels=1)

    # Unsupported channels
    with pytest.raises(InvalidAudioConfigError, match="Only 1"):
        sm.start_audio(conn, sid, audio_format="pcm16", sample_rate=16000, channels=2)


def test_session_manager_start_audio_duplicate_rejected():
    """Verify that starting audio when already streaming raises an error."""
    sm = SessionManager()
    conn = sm.generate_connection_id()
    s = sm.create_session(conn)
    sid = s["session_id"]

    sm.start_audio(conn, sid, "pcm16", 16000, 1)

    with pytest.raises(AudioStateError, match="already streaming"):
        sm.start_audio(conn, sid, "pcm16", 16000, 1)


def test_session_manager_start_audio_unauthorized():
    """Verify connection B cannot start audio for connection A's session."""
    sm = SessionManager()
    conn_a = sm.generate_connection_id()
    conn_b = sm.generate_connection_id()
    s_a = sm.create_session(conn_a)

    with pytest.raises(InvalidSessionError, match="Session not found or unauthorized"):
        sm.start_audio(conn_b, s_a["session_id"], "pcm16", 16000, 1)


def test_session_manager_stop_audio_success():
    """Verify stopping audio returns correct counters and sets audio_state to idle."""
    sm = SessionManager()
    conn = sm.generate_connection_id()
    s = sm.create_session(conn)
    sid = s["session_id"]

    sm.start_audio(conn, sid, "pcm16", 16000, 1)
    sm.record_audio_chunk(conn, 3200)
    sm.record_audio_chunk(conn, 3200)

    result = sm.stop_audio(conn, sid)
    assert result["session_id"] == sid
    assert result["audio_chunk_count"] == 2
    assert result["audio_bytes_received"] == 6400
    assert s["audio_state"] == "idle"


def test_session_manager_stop_audio_when_not_streaming():
    """Verify stopping audio when not streaming raises AudioStateError."""
    sm = SessionManager()
    conn = sm.generate_connection_id()
    s = sm.create_session(conn)

    with pytest.raises(AudioStateError, match="not streaming"):
        sm.stop_audio(conn, s["session_id"])


def test_session_manager_record_audio_chunk_valid():
    """Verify recording valid chunks increments chunk and byte counters."""
    sm = SessionManager()
    conn = sm.generate_connection_id()
    s = sm.create_session(conn)
    sm.start_audio(conn, s["session_id"], "pcm16", 16000, 1)

    res1 = sm.record_audio_chunk(conn, 8000)
    assert res1["audio_chunk_count"] == 1
    assert res1["audio_bytes_received"] == 8000

    res2 = sm.record_audio_chunk(conn, 8000)
    assert res2["audio_chunk_count"] == 2
    assert res2["audio_bytes_received"] == 16000


def test_session_manager_record_audio_chunk_empty_and_odd():
    """Verify rejection of empty chunks and odd-length chunks."""
    sm = SessionManager()
    conn = sm.generate_connection_id()
    s = sm.create_session(conn)
    sm.start_audio(conn, s["session_id"], "pcm16", 16000, 1)

    with pytest.raises(InvalidAudioDataError, match="Empty audio frame"):
        sm.record_audio_chunk(conn, 0)

    with pytest.raises(InvalidAudioDataError, match="even number of bytes"):
        sm.record_audio_chunk(conn, 3001)


def test_session_manager_record_chunk_when_not_streaming():
    """Verify recording chunks when audio is idle raises AudioStateError."""
    sm = SessionManager()
    conn = sm.generate_connection_id()
    sm.create_session(conn)

    with pytest.raises(AudioStateError, match="not streaming"):
        sm.record_audio_chunk(conn, 3200)


def test_session_manager_stop_session_stops_streaming_audio():
    """Verify that stopping a session automatically stops audio streaming."""
    sm = SessionManager()
    conn = sm.generate_connection_id()
    s = sm.create_session(conn)
    sm.start_audio(conn, s["session_id"], "pcm16", 16000, 1)
    assert s["audio_state"] == "streaming"

    sm.stop_session(conn, s["session_id"])
    assert s["status"] == "stopped"
    assert s["audio_state"] == "idle"


def test_session_manager_disconnect_stops_streaming_audio():
    """Verify that connection disconnect stops audio streaming and marks session stopped."""
    sm = SessionManager()
    conn = sm.generate_connection_id()
    s = sm.create_session(conn)
    sm.start_audio(conn, s["session_id"], "pcm16", 16000, 1)

    cleaned = sm.cleanup_connection(conn)
    assert cleaned["status"] == "stopped"
    assert cleaned["audio_state"] == "idle"
