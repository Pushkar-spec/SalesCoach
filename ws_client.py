"""
WebSocket Client Helper for Streamlit Frontend (Step 1.5)

Provides a robust, synchronous WebSocket client wrapper using `websockets.sync.client`
to manage the connection lifecycle, ping/pong handshake, and graceful closure.
"""

import json
import logging
import os
from typing import Any, Optional
from websockets.sync.client import ClientConnection, connect

logger = logging.getLogger(__name__)

# Default development WebSocket endpoint
DEFAULT_BACKEND_WS_URL = "ws://127.0.0.1:8000/ws"


def get_backend_ws_url() -> str:
    """Retrieve the configured backend WebSocket URL from environment or default."""
    return os.getenv("BACKEND_WS_URL", DEFAULT_BACKEND_WS_URL)


class WebSocketClient:
    """
    Manages a single synchronous WebSocket connection lifecycle.

    Designed for Streamlit's script-rerun execution model:
    - Synchronous connection avoids complex async loops or background threads.
    - Connection state can be safely inspected across script reruns.
    - Handshake failure cleans up immediately, leaving no stale sockets.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        open_timeout: float = 3.0,
        recv_timeout: float = 3.0,
    ):
        self.url = url or get_backend_ws_url()
        self.open_timeout = open_timeout
        self.recv_timeout = recv_timeout
        self.connection: Optional[ClientConnection] = None

    @property
    def is_connected(self) -> bool:
        """Check if the underlying socket exists and is in an OPEN state."""
        return (
            self.connection is not None
            and hasattr(self.connection, "state")
            and getattr(self.connection.state, "name", None) == "OPEN"
        )

    def connect_and_ping(
        self, ping_message: str = "Hello from Streamlit"
    ) -> dict[str, Any]:
        """
        Establish connection to backend and perform initial ping/pong handshake.

        Returns:
            dict: The validated pong response from the backend.

        Raises:
            ConnectionError / TimeoutError / ValueError / WebSocketException on failure.
        """
        try:
            # 1. Establish connection with timeout
            self.connection = connect(
                self.url,
                open_timeout=self.open_timeout,
                close_timeout=self.open_timeout,
                legacy=True,
            )

            # 2. Send ping frame
            ping_payload = {"type": "ping", "message": ping_message}
            self.connection.send(json.dumps(ping_payload))

            # 3. Receive pong frame with timeout
            raw_response = self.connection.recv(timeout=self.recv_timeout)

            # 4. Parse JSON
            response = json.loads(raw_response)

            # 5. Validate response structure
            if not isinstance(response, dict):
                raise ValueError("Expected JSON object in response from backend")

            if response.get("type") != "pong":
                raise ValueError(
                    f"Expected 'pong' response, received: {response.get('type')}"
                )

            if response.get("message") != ping_message:
                raise ValueError(
                    f"Pong message mismatch. Expected '{ping_message}', got '{response.get('message')}'"
                )

            return response

        except Exception as exc:
            # If handshake or connection fails, clean up immediately
            self.close()
            raise exc

    def close(self) -> None:
        """Gracefully close the connection and reset socket reference."""
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception as e:
                logger.debug("Error while closing WebSocket: %s", e)
            finally:
                self.connection = None
