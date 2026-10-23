import os
from unittest.mock import MagicMock, patch
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))


def test_app_initial_state():
    """Verify default initial session state upon loading app.py."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    assert at.session_state["session_status"] == "idle"
    assert at.session_state["ws_status"] == "disconnected"
    assert at.session_state["ws_client"] is None


def test_app_start_call_success():
    """Verify Start Call connects successfully when handshake succeeds."""
    mock_conn = MagicMock()
    mock_conn.state.name = "OPEN"
    mock_conn.recv.return_value = '{"type": "pong", "message": "Hello from Streamlit"}'

    with patch("ws_client.connect", return_value=mock_conn):
        at = AppTest.from_file(APP_PATH).run()
        btn_start = [b for b in at.button if "Start Call" in b.label][0]
        btn_start.click().run()

        assert not at.exception
        assert at.session_state["session_status"] == "active"
        assert at.session_state["ws_status"] == "connected"
        assert "Handshake successful" in at.session_state["status_message"]

        # Stop Call
        btn_stop = [b for b in at.button if "Stop Call" in b.label][0]
        btn_stop.click().run()

        assert not at.exception
        assert at.session_state["session_status"] == "stopped"
        assert at.session_state["ws_status"] == "disconnected"
        assert at.session_state["ws_client"] is None
        mock_conn.close.assert_called_once()


def test_app_start_call_failure_handles_error():
    """Verify Start Call handles connection errors truthfully and remains idle."""
    with patch("ws_client.connect", side_effect=ConnectionRefusedError("Connection refused")):
        at = AppTest.from_file(APP_PATH).run()
        btn_start = [b for b in at.button if "Start Call" in b.label][0]
        btn_start.click().run()

        assert not at.exception
        # Must NOT report connected
        assert at.session_state["ws_status"] == "error"
        assert at.session_state["session_status"] == "idle"
        assert at.session_state["ws_client"] is None
        assert "Backend connection failed" in at.session_state["status_message"]


def test_app_reset_session_cleans_up():
    """Verify Reset Session terminates connection and resets state."""
    mock_conn = MagicMock()
    mock_conn.state.name = "OPEN"
    mock_conn.recv.return_value = '{"type": "pong", "message": "Hello from Streamlit"}'

    with patch("ws_client.connect", return_value=mock_conn):
        at = AppTest.from_file(APP_PATH).run()
        # Start Call
        btn_start = [b for b in at.button if "Start Call" in b.label][0]
        btn_start.click().run()

        # Add dummy transcript
        at.session_state["transcript_entries"].append({"speaker": "customer", "text": "hello", "timestamp": "12:00"})
        at.run()

        # Click Reset
        btn_reset = [b for b in at.button if "Reset Session" in b.label][0]
        btn_reset.click().run()

        assert not at.exception
        assert at.session_state["session_status"] == "idle"
        assert at.session_state["ws_status"] == "disconnected"
        assert at.session_state["ws_client"] is None
        assert len(at.session_state["transcript_entries"]) == 0
        mock_conn.close.assert_called()
