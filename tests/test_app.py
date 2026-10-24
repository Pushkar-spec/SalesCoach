"""
test_app.py — Streamlit UI Integration Tests (Step 2.4)

Tests the unified Call Controls workflow:
- Start Call, Stop Call, and Reset Session driving the browser audio recorder
- One authoritative session lifecycle per call
- Idempotent repeated clicks (button disabling and state protection)
- Handling component state events (recording, stopped, error)
- Full cleanup on Reset
"""

import os
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))


def test_app_initial_state():
    """Verify default initial session state upon loading app.py."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    assert at.session_state["session_id"] is None
    assert at.session_state["session_status"] == "idle"
    assert at.session_state["ws_status"] == "disconnected"
    assert at.session_state["recorder_command"] is None
    assert at.session_state["audio_metrics"] == {"chunks_sent": 0, "bytes_sent": 0}
    assert at.session_state["transcript_entries"] == []
    assert at.session_state["coaching_suggestions"] == []


def test_app_start_call_triggers_unified_command():
    """Verify Start Call sets active state, connecting status, and issues recorder start command."""
    at = AppTest.from_file(APP_PATH).run()
    btn_start = [b for b in at.button if "Start Call" in b.label][0]
    assert btn_start.disabled is False
    btn_start.click().run()

    assert not at.exception
    assert at.session_state["session_status"] == "active"
    assert at.session_state["ws_status"] == "connecting"

    cmd = at.session_state["recorder_command"]
    assert cmd is not None
    assert cmd["action"] == "start"
    assert len(cmd["id"]) > 0
    assert "Starting call" in at.session_state["status_message"]

    # Verify Start Call button is now disabled to prevent duplicate starts
    btn_start_after = [b for b in at.button if "Start Call" in b.label][0]
    assert btn_start_after.disabled is True


def test_app_start_call_repeated_click_idempotent():
    """Verify Start Call button cannot be clicked repeatedly when active."""
    at = AppTest.from_file(APP_PATH).run()
    btn_start = [b for b in at.button if "Start Call" in b.label][0]
    btn_start.click().run()

    first_cmd_id = at.session_state["recorder_command"]["id"]

    # Verify Start Call is disabled
    btn_start_disabled = [b for b in at.button if "Start Call" in b.label][0]
    assert btn_start_disabled.disabled is True

    # Stop Call button is now enabled
    btn_stop = [b for b in at.button if "Stop Call" in b.label][0]
    assert btn_stop.disabled is False
    assert at.session_state["recorder_command"]["id"] == first_cmd_id


def test_app_stop_call_triggers_stop_command():
    """Verify Stop Call sets stopped state, disconnected status, and issues recorder stop command."""
    at = AppTest.from_file(APP_PATH).run()

    # 1. Start Call
    btn_start = [b for b in at.button if "Start Call" in b.label][0]
    btn_start.click().run()

    # 2. Stop Call
    btn_stop = [b for b in at.button if "Stop Call" in b.label][0]
    btn_stop.click().run()

    assert not at.exception
    assert at.session_state["session_status"] == "stopped"
    assert at.session_state["ws_status"] == "disconnected"

    cmd = at.session_state["recorder_command"]
    assert cmd is not None
    assert cmd["action"] == "stop"
    assert len(cmd["id"]) > 0
    assert "Stopping call" in at.session_state["status_message"]

    # Stop Call button is now disabled
    btn_stop_after = [b for b in at.button if "Stop Call" in b.label][0]
    assert btn_stop_after.disabled is True


def test_app_reset_session_cleans_up_all_state():
    """Verify Reset Session issues reset command and cleans up all session and transcript state."""
    at = AppTest.from_file(APP_PATH).run()

    # 1. Start Call
    btn_start = [b for b in at.button if "Start Call" in b.label][0]
    btn_start.click().run()

    # 2. Add sample transcript and simulated metrics
    at.session_state["transcript_entries"].append({
        "speaker": "customer",
        "text": "Hello, interested in your software.",
        "timestamp": "12:00:00",
    })
    at.session_state["audio_metrics"] = {"chunks_sent": 15, "bytes_sent": 120000}
    at.session_state["session_id"] = "test-session-1234"
    at.run()

    # 3. Click Reset Session
    btn_reset = [b for b in at.button if "Reset Session" in b.label][0]
    btn_reset.click().run()

    assert not at.exception
    assert at.session_state["session_status"] == "idle"
    assert at.session_state["ws_status"] == "disconnected"
    assert at.session_state["session_id"] is None
    assert at.session_state["transcript_entries"] == []
    assert at.session_state["coaching_suggestions"] == []
    assert at.session_state["audio_metrics"] == {"chunks_sent": 0, "bytes_sent": 0}

    cmd = at.session_state["recorder_command"]
    assert cmd is not None
    assert cmd["action"] == "reset"


def test_app_handles_component_recording_event():
    """Verify app updates session state when audio recorder component reports 'recording'."""
    mock_event = {
        "state": "recording",
        "session_id": "server-uuid-5678",
        "audio_state": "streaming",
        "chunks_sent": 5,
        "bytes_sent": 40000,
    }

    with patch("components.audio_recorder.audio_recorder", return_value=mock_event):
        at = AppTest.from_file(APP_PATH).run()
        assert not at.exception
        assert at.session_state["session_id"] == "server-uuid-5678"
        assert at.session_state["session_status"] == "active"
        assert at.session_state["ws_status"] == "connected"
        assert "server-uuid-5678" in at.session_state["status_message"]


def test_app_handles_component_stopped_event():
    """Verify app updates counters when audio recorder component reports 'stopped'."""
    mock_event = {
        "state": "stopped",
        "session_id": "server-uuid-5678",
        "audio_state": "idle",
        "chunks_sent": 30,
        "bytes_sent": 240000,
    }

    with patch("components.audio_recorder.audio_recorder", return_value=mock_event):
        at = AppTest.from_file(APP_PATH).run()
        assert not at.exception
        assert at.session_state["session_status"] == "stopped"
        assert at.session_state["ws_status"] == "disconnected"
        assert at.session_state["audio_metrics"]["chunks_sent"] == 30
        assert at.session_state["audio_metrics"]["bytes_sent"] == 240000
        assert "30 chunks transmitted" in at.session_state["status_message"]


def test_app_handles_component_error_event():
    """Verify app gracefully handles error events from audio recorder component."""
    mock_event = {
        "state": "error",
        "session_id": None,
        "error": "Microphone access denied by user.",
    }

    with patch("components.audio_recorder.audio_recorder", return_value=mock_event):
        at = AppTest.from_file(APP_PATH).run()
        assert not at.exception
        assert at.session_state["session_status"] == "idle"
        assert at.session_state["ws_status"] == "error"
        assert at.session_state["session_id"] is None
        assert "Microphone access denied" in at.session_state["status_message"]


def test_app_displays_updated_empty_state():
    """Verify obsolete placeholder is replaced with 'Start recording to see the live transcript.'."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    # Check that old placeholder text is absent
    info_texts = [info.value for info in at.info]
    assert not any("Once speech-to-text is integrated in future steps" in t for t in info_texts)
    # Check that new empty state message is present
    assert any("Start recording to see the live transcript." in t for t in info_texts)


def test_app_handles_component_transcript_events():
    """Verify app receives and renders partial and final transcript events from recorder."""
    mock_event_partial = {
        "state": "recording",
        "session_id": "test-transcript-sid",
        "audio_state": "streaming",
        "chunks_sent": 8,
        "bytes_sent": 64000,
        "transcripts": [
            {
                "segment_id": 1,
                "speaker": "Sales Rep",
                "text": "Hello, thank you for calling",
                "is_final": False,
                "status": "partial",
                "timestamp": "2026-10-25T01:30:00Z",
            }
        ],
    }

    with patch("components.audio_recorder.audio_recorder", return_value=mock_event_partial):
        at = AppTest.from_file(APP_PATH).run()
        assert not at.exception
        assert len(at.session_state["transcript_entries"]) == 1
        entry = at.session_state["transcript_entries"][0]
        assert entry["text"] == "Hello, thank you for calling"
        assert entry["is_final"] is False

    mock_event_final = {
        "state": "recording",
        "session_id": "test-transcript-sid",
        "audio_state": "streaming",
        "chunks_sent": 12,
        "bytes_sent": 96000,
        "transcripts": [
            {
                "segment_id": 1,
                "speaker": "Sales Rep",
                "text": "Hello, thank you for calling today.",
                "is_final": True,
                "status": "final",
                "timestamp": "2026-10-25T01:30:02Z",
            }
        ],
    }

    with patch("components.audio_recorder.audio_recorder", return_value=mock_event_final):
        at2 = AppTest.from_file(APP_PATH).run()
        assert not at2.exception
        assert len(at2.session_state["transcript_entries"]) == 1
        entry = at2.session_state["transcript_entries"][0]
        assert entry["text"] == "Hello, thank you for calling today."
        assert entry["is_final"] is True

