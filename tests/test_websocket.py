from fastapi.testclient import TestClient
from main import app

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


def test_http_endpoints_remain_functional():
    """Verify that existing HTTP endpoints (/ and /health) continue to work properly."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json() == {"message": "Hello World"}

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"
