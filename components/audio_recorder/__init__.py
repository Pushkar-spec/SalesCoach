"""
Audio Recorder Custom Component for Streamlit

Declares and serves the browser-side microphone capture component for Step 2.3 / 2.4.
Transmits real-time 16kHz PCM16 audio chunks directly to the FastAPI WebSocket backend.
"""

import os
from typing import Any, Optional
import streamlit.components.v1 as components

# Path to the frontend static directory containing index.html, recorder.js, audio-utils.js, styles.css
_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

# Declare the Streamlit component
_component_func = components.declare_component(
    "audio_recorder",
    path=_FRONTEND_DIR,
)


def audio_recorder(
    backend_ws_url: Any = "ws://127.0.0.1:8000/ws",
    command: Optional[Any] = None,
    key: Optional[str] = None,
    **kwargs: Any,
) -> Optional[dict]:
    """
    Renders the browser-side audio capture component.

    Parameters
    ----------
    backend_ws_url : Any
        The WebSocket URL of the FastAPI backend (default: "ws://127.0.0.1:8000/ws").
        If a control command dictionary or string action is passed as the first
        positional argument, it is automatically routed to `command`.
    command : Optional[Any]
        Control command sent from Python to the browser recorder, e.g.
        {"action": "start" | "stop" | "reset", "id": "<unique_id>"}.
    key : Optional[str]
        Optional unique key for the component widget in Streamlit.
    **kwargs : Any
        Additional keyword arguments forwarded to the Streamlit custom component.

    Returns
    -------
    Optional[dict]
        Lifecycle state dictionary returned by the component to Python:
        {"state": str, "session_id": str | None, "audio_state": str,
         "chunks_sent": int, "bytes_sent": int, "error": str | None}
    """
    # If the first positional argument is a command dict or non-ws action string, route to command
    if isinstance(backend_ws_url, dict) or (
        isinstance(backend_ws_url, str)
        and not (backend_ws_url.startswith("ws://") or backend_ws_url.startswith("wss://"))
    ):
        if command is None:
            command = backend_ws_url
            backend_ws_url = kwargs.pop("backend_ws_url", "ws://127.0.0.1:8000/ws")

    return _component_func(
        backend_ws_url=backend_ws_url,
        command=command,
        key=key,
        default=None,
        **kwargs,
    )

