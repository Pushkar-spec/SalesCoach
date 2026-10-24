import json
from unittest.mock import MagicMock, patch
import pytest
from ws_client import WebSocketClient, get_backend_ws_url, DEFAULT_BACKEND_WS_URL


def test_default_ws_url(monkeypatch):
    """Verify that get_backend_ws_url returns the default when no env var is set."""
    monkeypatch.delenv("BACKEND_WS_URL", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    assert get_backend_ws_url() == DEFAULT_BACKEND_WS_URL

    monkeypatch.setenv("PORT", "8001")
    assert get_backend_ws_url() == "ws://127.0.0.1:8001/ws"

    monkeypatch.setenv("BACKEND_WS_URL", "ws://custom-host:9000/ws")
    assert get_backend_ws_url() == "ws://custom-host:9000/ws"


def test_ws_client_connect_and_ping_success():
    """Verify successful connect and ping/pong handshake."""
    mock_conn = MagicMock()
    mock_conn.state.name = "OPEN"
    mock_conn.recv.return_value = json.dumps({
        "type": "pong",
        "message": "Hello from Streamlit"
    })

    with patch("ws_client.connect", return_value=mock_conn) as mock_connect:
        client = WebSocketClient(url="ws://127.0.0.1:8000/ws")
        response = client.connect_and_ping("Hello from Streamlit")

        mock_connect.assert_called_once()
        mock_conn.send.assert_called_once_with(
            json.dumps({"type": "ping", "message": "Hello from Streamlit"})
        )
        assert response["type"] == "pong"
        assert response["message"] == "Hello from Streamlit"
        assert client.is_connected is True

        client.close()
        assert client.is_connected is False
        mock_conn.close.assert_called_once()


def test_ws_client_start_session_success():
    """Verify that start_session sends request and returns confirmed session_id."""
    mock_conn = MagicMock()
    mock_conn.state.name = "OPEN"
    mock_conn.recv.return_value = json.dumps({
        "type": "session_started",
        "session_id": "test-uuid-1234",
        "status": "active"
    })

    client = WebSocketClient()
    client.connection = mock_conn

    sid = client.start_session()
    assert sid == "test-uuid-1234"
    mock_conn.send.assert_called_once_with(json.dumps({"type": "start_session"}))


def test_ws_client_start_session_backend_error():
    """Verify that backend error during start_session raises RuntimeError."""
    mock_conn = MagicMock()
    mock_conn.state.name = "OPEN"
    mock_conn.recv.return_value = json.dumps({
        "type": "error",
        "message": "An active session already exists for this connection"
    })

    client = WebSocketClient()
    client.connection = mock_conn

    with pytest.raises(RuntimeError, match="already exists"):
        client.start_session()


def test_ws_client_start_session_unexpected_response():
    """Verify that unexpected response type raises ValueError."""
    mock_conn = MagicMock()
    mock_conn.state.name = "OPEN"
    mock_conn.recv.return_value = json.dumps({"type": "unexpected_type"})

    client = WebSocketClient()
    client.connection = mock_conn

    with pytest.raises(ValueError, match="Expected 'session_started'"):
        client.start_session()


def test_ws_client_get_status_success():
    """Verify querying session status returns status dictionary."""
    mock_conn = MagicMock()
    mock_conn.state.name = "OPEN"
    mock_conn.recv.return_value = json.dumps({
        "type": "status",
        "session_id": "test-uuid-1234",
        "status": "active"
    })

    client = WebSocketClient()
    client.connection = mock_conn

    status = client.get_status()
    assert status["type"] == "status"
    assert status["session_id"] == "test-uuid-1234"
    assert status["status"] == "active"


def test_ws_client_stop_session_success():
    """Verify stopping a session sends stop_session and validates confirmation."""
    mock_conn = MagicMock()
    mock_conn.state.name = "OPEN"
    mock_conn.recv.return_value = json.dumps({
        "type": "session_stopped",
        "session_id": "test-uuid-1234",
        "status": "stopped"
    })

    client = WebSocketClient()
    client.connection = mock_conn

    resp = client.stop_session("test-uuid-1234")
    assert resp["type"] == "session_stopped"
    assert resp["session_id"] == "test-uuid-1234"
    assert resp["status"] == "stopped"


def test_ws_client_stop_session_invalid_id():
    """Verify stop_session rejects empty session IDs."""
    client = WebSocketClient()
    with pytest.raises(ValueError, match="must be a non-empty string"):
        client.stop_session("")


def test_ws_client_stop_session_backend_error():
    """Verify stop_session raises RuntimeError when backend returns an error."""
    mock_conn = MagicMock()
    mock_conn.state.name = "OPEN"
    mock_conn.recv.return_value = json.dumps({
        "type": "error",
        "message": "Session not found or unauthorized"
    })

    client = WebSocketClient()
    client.connection = mock_conn

    with pytest.raises(RuntimeError, match="Session not found or unauthorized"):
        client.stop_session("invalid-uuid")


def test_ws_client_connection_refused():
    """Verify that connection failure raises an exception and leaves no open connection."""
    with patch("ws_client.connect", side_effect=ConnectionRefusedError("Connection refused")):
        client = WebSocketClient(url="ws://127.0.0.1:8000/ws")
        with pytest.raises(ConnectionRefusedError):
            client.connect_and_ping("Test Ping")

        assert client.is_connected is False
        assert client.connection is None


def test_ws_client_recv_timeout():
    """Verify that a timeout on receiving pong triggers cleanup and raises TimeoutError."""
    mock_conn = MagicMock()
    mock_conn.state.name = "OPEN"
    mock_conn.recv.side_effect = TimeoutError("Recv timed out")

    with patch("ws_client.connect", return_value=mock_conn):
        client = WebSocketClient()
        with pytest.raises(TimeoutError):
            client.connect_and_ping("Test Ping")

        mock_conn.close.assert_called_once()
        assert client.is_connected is False


def test_ws_client_invalid_pong_type():
    """Verify that an unexpected response type raises ValueError and cleans up."""
    mock_conn = MagicMock()
    mock_conn.state.name = "OPEN"
    mock_conn.recv.return_value = json.dumps({"type": "error", "message": "wrong type"})

    with patch("ws_client.connect", return_value=mock_conn):
        client = WebSocketClient()
        with pytest.raises(ValueError, match="Expected 'pong' response"):
            client.connect_and_ping("Hello")

        mock_conn.close.assert_called_once()
        assert client.is_connected is False


def test_ws_client_close_idempotent():
    """Verify that calling close multiple times or on an uninitialized client is safe."""
    client = WebSocketClient()
    client.close()  # No error on None
    assert client.connection is None
