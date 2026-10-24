"""
test_audio_component.py
Tests for the browser-side audio capture component (Step 2.3).
"""

import os
import subprocess
import pytest
from streamlit.testing.v1 import AppTest
from components.audio_recorder import _FRONTEND_DIR, audio_recorder


def test_audio_component_files_exist():
    """Verify that all required frontend assets for the audio component exist."""
    assert os.path.isdir(_FRONTEND_DIR), f"Frontend directory not found: {_FRONTEND_DIR}"

    required_files = ["index.html", "audio-utils.js", "recorder.js", "styles.css"]
    for filename in required_files:
        filepath = os.path.join(_FRONTEND_DIR, filename)
        assert os.path.isfile(filepath), f"Missing component file: {filename}"
        assert os.path.getsize(filepath) > 0, f"Component file is empty: {filename}"


def test_audio_component_index_html_structure():
    """Verify that index.html contains necessary UI elements and scripts."""
    index_path = os.path.join(_FRONTEND_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Essential telemetry and visualizer elements
    assert 'id="status-pill"' in html
    assert 'id="status-text"' in html
    assert 'id="meter-fill"' in html
    assert 'id="val-session-id"' in html
    assert 'id="val-audio-state"' in html
    assert 'id="val-chunks"' in html
    assert 'id="val-bytes"' in html
    assert 'id="val-endpoint"' in html

    # Essential script imports
    assert 'src="audio-utils.js"' in html
    assert 'src="recorder.js"' in html


def test_javascript_audio_processor_node_tests():
    """Run Node.js unit tests for audio-utils.js and verify all assertions pass."""
    test_script = os.path.join(os.path.dirname(__file__), "test_audio_processor.js")
    result = subprocess.run(
        ["node", "--test", test_script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Node.js tests failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "fail 0" in result.stdout
    assert "pass 13" in result.stdout or "pass" in result.stdout


def test_app_renders_with_audio_component():
    """Verify that app.py initializes and renders the audio component without exceptions."""
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
    at = AppTest.from_file(app_path).run()
    assert not at.exception
    # Verify app title still exists and UI loaded successfully
    assert any("Real-Time AI Sales Copilot" in t.value for t in at.title)


def test_audio_recorder_accepts_and_forwards_command_keyword(monkeypatch):
    """Verify that audio_recorder() accepts command as a keyword argument and forwards it to _component_func."""
    mock_calls = []

    def fake_component_func(**kwargs):
        mock_calls.append(kwargs)
        return {"state": "recording", "session_id": "test-session-1"}

    import components.audio_recorder as ar_module
    monkeypatch.setattr(ar_module, "_component_func", fake_component_func)

    cmd = {"action": "start", "id": "cmd-uuid-123"}
    result = ar_module.audio_recorder(command=cmd)

    assert len(mock_calls) == 1
    assert mock_calls[0]["command"] == cmd
    assert mock_calls[0]["backend_ws_url"] == "ws://127.0.0.1:8000/ws"
    assert mock_calls[0]["key"] is None
    assert mock_calls[0]["default"] is None
    assert result == {"state": "recording", "session_id": "test-session-1"}


def test_audio_recorder_forwards_backend_ws_url_command_and_key(monkeypatch):
    """Verify that audio_recorder() accepts and forwards backend_ws_url, command, and key together."""
    mock_calls = []

    def fake_component_func(**kwargs):
        mock_calls.append(kwargs)
        return {"state": "stopped", "session_id": "test-session-2", "chunks_sent": 10}

    import components.audio_recorder as ar_module
    monkeypatch.setattr(ar_module, "_component_func", fake_component_func)

    cmd = {"action": "stop", "id": "cmd-uuid-456"}
    result = ar_module.audio_recorder(
        backend_ws_url="ws://127.0.0.1:8000/ws",
        command=cmd,
        key="sales_copilot_audio_recorder",
    )

    assert len(mock_calls) == 1
    assert mock_calls[0]["backend_ws_url"] == "ws://127.0.0.1:8000/ws"
    assert mock_calls[0]["command"] == cmd
    assert mock_calls[0]["key"] == "sales_copilot_audio_recorder"
    assert mock_calls[0]["default"] is None
    assert result["state"] == "stopped"
    assert result["chunks_sent"] == 10


def test_audio_recorder_accepts_positional_command_or_ws_url(monkeypatch):
    """Verify that audio_recorder() gracefully routes positional command dict or ws_url string."""
    mock_calls = []

    def fake_component_func(**kwargs):
        mock_calls.append(kwargs)
        return None

    import components.audio_recorder as ar_module
    monkeypatch.setattr(ar_module, "_component_func", fake_component_func)

    # 1. Positional command dict
    reset_cmd = {"action": "reset", "id": "cmd-uuid-789"}
    ar_module.audio_recorder(reset_cmd)
    assert mock_calls[-1]["command"] == reset_cmd
    assert mock_calls[-1]["backend_ws_url"] == "ws://127.0.0.1:8000/ws"

    # 2. Positional ws_url string
    ar_module.audio_recorder("ws://custom-backend:9000/ws", key="custom_key")
    assert mock_calls[-1]["backend_ws_url"] == "ws://custom-backend:9000/ws"
    assert mock_calls[-1]["command"] is None
    assert mock_calls[-1]["key"] == "custom_key"

