"""
Real-Time AI Sales Copilot - Streamlit Frontend Dashboard (Step 1.5)

This module implements the Streamlit dashboard connected to the FastAPI backend
via a bidirectional WebSocket connection.
In this step (Step 1.5):
- Start Call establishes a WebSocket connection and exchanges a ping/pong handshake.
- Stop Call closes the WebSocket connection gracefully.
- Reset Session resets session collections and terminates any open socket.
- Audio streaming and LLM coaching remain decoupled for future phases.
"""

from datetime import datetime
import os
from typing import Optional
import streamlit as st

from ws_client import WebSocketClient, get_backend_ws_url

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Real-Time AI Sales Copilot",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
# Streamlit re-runs this script from top-to-bottom on every user interaction.
# st.session_state persists state variables across these reruns.

if "session_status" not in st.session_state:
    # Allowed states: "idle", "active", "stopped"
    st.session_state.session_status = "idle"

if "status_message" not in st.session_state:
    st.session_state.status_message = "Ready to start a session."

if "ws_client" not in st.session_state:
    # Holds the active WebSocketClient instance or None
    st.session_state.ws_client = None

if "ws_status" not in st.session_state:
    # Allowed states: "disconnected", "connecting", "connected", "error"
    st.session_state.ws_status = "disconnected"

if "transcript_entries" not in st.session_state:
    # Structured list of transcript items:
    # [{"speaker": "customer" | "salesperson", "text": str, "timestamp": str}]
    st.session_state.transcript_entries = []

if "coaching_suggestions" not in st.session_state:
    # Structured list of AI suggestions:
    # [{
    #     "signal_category": str,
    #     "recommended_response": str,
    #     "follow_up_question": str | None,
    #     "timestamp": str
    # }]
    st.session_state.coaching_suggestions = []


# -----------------------------------------------------------------------------
# 3. Call Control Handlers
# -----------------------------------------------------------------------------
def handle_start_call() -> None:
    """
    Attempts to establish a WebSocket connection with the backend, sends a ping,
    and transitions session to active upon a validated pong response.
    """
    # 1. Check if an active connection already exists
    existing_client: Optional[WebSocketClient] = st.session_state.get("ws_client")
    if existing_client and existing_client.is_connected:
        st.session_state.session_status = "active"
        st.session_state.ws_status = "connected"
        st.session_state.status_message = "Session already active and connected."
        return

    # 2. Instantiate and attempt handshake
    ws_client = WebSocketClient()
    st.session_state.ws_status = "connecting"
    try:
        response = ws_client.connect_and_ping("Hello from Streamlit")
        st.session_state.ws_client = ws_client
        st.session_state.ws_status = "connected"
        st.session_state.session_status = "active"
        st.session_state.status_message = (
            f"Connected to backend ({ws_client.url}). "
            f"Handshake successful (received pong: '{response.get('message')}'). "
            "Note: Audio streaming and AI models are scheduled for subsequent steps."
        )
    except Exception as exc:
        # Connection or handshake failed
        st.session_state.ws_client = None
        st.session_state.ws_status = "error"
        st.session_state.session_status = "idle"
        st.session_state.status_message = (
            f"Backend connection failed ({ws_client.url}): {exc}. "
            "Ensure the FastAPI server is running with 'python -m uvicorn main:app --reload'."
        )


def handle_stop_call() -> None:
    """Closes the active WebSocket connection gracefully and stops the session."""
    existing_client: Optional[WebSocketClient] = st.session_state.get("ws_client")
    if existing_client is not None:
        existing_client.close()
        st.session_state.ws_client = None

    st.session_state.ws_status = "disconnected"
    st.session_state.session_status = "stopped"
    st.session_state.status_message = (
        "Call session stopped. Backend WebSocket connection closed."
    )


def handle_reset_session() -> None:
    """Closes any active connection and resets all collections to initial state."""
    existing_client: Optional[WebSocketClient] = st.session_state.get("ws_client")
    if existing_client is not None:
        existing_client.close()
        st.session_state.ws_client = None

    st.session_state.ws_status = "disconnected"
    st.session_state.session_status = "idle"
    st.session_state.transcript_entries = []
    st.session_state.coaching_suggestions = []
    st.session_state.status_message = (
        "Session reset. All local transcripts, suggestions, and socket connections cleared."
    )


# -----------------------------------------------------------------------------
# 4. Sidebar: Session Info & Architecture Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Session Info")

    status = st.session_state.session_status
    if status == "active":
        st.success("Session: **Active**")
    elif status == "stopped":
        st.warning("Session: **Stopped**")
    else:
        st.info("Session: **Idle**")

    # Backend WebSocket Connection Indicator
    ws_state = st.session_state.ws_status
    if ws_state == "connected":
        st.success("WebSocket: **Connected**")
    elif ws_state == "connecting":
        st.warning("WebSocket: **Connecting...**")
    elif ws_state == "error":
        st.error("WebSocket: **Connection Error**")
    else:
        st.caption("WebSocket: **Disconnected**")

    st.divider()

    st.subheader("Session Counters")
    st.metric(
        label="Transcript Entries",
        value=len(st.session_state.transcript_entries),
    )
    st.metric(
        label="Coaching Suggestions",
        value=len(st.session_state.coaching_suggestions),
    )

    st.divider()

    st.subheader("Integration Pipeline Status")
    st.caption("Current architectural status:")

    ws_label = "Connected" if ws_state == "connected" else "Disconnected"
    st.markdown(
        f"""
        - **FastAPI HTTP API**: `Implemented` *(Routes: `/`, `/health`)*
        - **FastAPI WebSocket Backend**: `Implemented` *(Route: `/ws`)*
        - **Streamlit WebSocket Link**: `{ws_label}`
        - **Audio Streaming**: `Not Connected` *(Planned: Phase 2)*
        - **Speech-to-Text**: `Not Connected` *(Planned: Phase 3)*
        - **LLM Sales Coach**: `Not Connected` *(Planned: Phase 4)*
        - **RAG Knowledge**: `Not Connected` *(Planned: Phase 5)*
        """
    )


# -----------------------------------------------------------------------------
# 5. Header Section
# -----------------------------------------------------------------------------
st.title("🎙️ Real-Time AI Sales Copilot")
st.markdown(
    "A live assistant designed to support sales representatives during customer "
    "conversations by providing real-time objection handling, talking points, "
    "and deal guidance."
)

# Display current feedback banner
if st.session_state.ws_status == "error":
    st.error(st.session_state.status_message, icon="🚨")
elif st.session_state.session_status == "active":
    st.success(st.session_state.status_message, icon="🟢")
elif st.session_state.session_status == "stopped":
    st.warning(st.session_state.status_message, icon="⏹️")
else:
    st.info(st.session_state.status_message, icon="ℹ️")


# -----------------------------------------------------------------------------
# 6. Call Controls Section
# -----------------------------------------------------------------------------
st.subheader("Call Controls")

is_active = st.session_state.session_status == "active"
is_idle = st.session_state.session_status == "idle"

col_start, col_stop, col_reset, _ = st.columns([1.2, 1.2, 1.2, 3.4])

with col_start:
    st.button(
        "▶ Start Call",
        on_click=handle_start_call,
        disabled=is_active,
        use_container_width=True,
        help="Connect to backend WebSocket and initiate session.",
    )

with col_stop:
    st.button(
        "⏹ Stop Call",
        on_click=handle_stop_call,
        disabled=(not is_active),
        use_container_width=True,
        help="Stop session and close the WebSocket connection.",
    )

with col_reset:
    st.button(
        "🔄 Reset Session",
        on_click=handle_reset_session,
        disabled=(
            is_idle
            and len(st.session_state.transcript_entries) == 0
            and st.session_state.ws_status == "disconnected"
        ),
        use_container_width=True,
        help="Reset session status, clear entries, and close socket.",
    )

st.caption(
    f"Backend Endpoint: `{get_backend_ws_url()}` | "
    "Prototype note: Start Call establishes a live WebSocket handshake with FastAPI. "
    "Audio recording and LLM suggestions are scheduled for subsequent phases."
)

st.divider()


# -----------------------------------------------------------------------------
# 7. Main Dashboard Area: Transcript and Coaching Suggestions
# -----------------------------------------------------------------------------
col_transcript, col_coaching = st.columns([1, 1], gap="large")

# --- Column 1: Conversation Transcript ---
with col_transcript:
    st.subheader("💬 Conversation Transcript")

    if not st.session_state.transcript_entries:
        st.info(
            "No transcript entries yet.\n\n"
            "Once speech-to-text is integrated in future steps, live dialogue between "
            "the salesperson and customer will appear here in real time.",
            icon="📝",
        )
    else:
        for item in st.session_state.transcript_entries:
            speaker = item.get("speaker", "unknown").lower()
            text = item.get("text", "")
            timestamp = item.get("timestamp", "")

            if speaker == "customer":
                with st.chat_message("user"):
                    st.markdown(f"**Customer** `{timestamp}`")
                    st.write(text)
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"**Sales Rep** `{timestamp}`")
                    st.write(text)

# --- Column 2: AI Coaching Suggestions ---
with col_coaching:
    st.subheader("💡 AI Coaching Suggestions")

    if not st.session_state.coaching_suggestions:
        st.info(
            "No coaching suggestions available yet.\n\n"
            "Once the LLM coaching pipeline is connected in future steps, "
            "detected sales objections, recommended talking points, and "
            "recommended questions will appear here dynamically.",
            icon="🧠",
        )
    else:
        for idx, suggestion in enumerate(st.session_state.coaching_suggestions, 1):
            category = suggestion.get("signal_category", "General Observation")
            response = suggestion.get("recommended_response", "")
            follow_up = suggestion.get("follow_up_question")
            ts = suggestion.get("timestamp", "")

            with st.container(border=True):
                st.markdown(f"**Signal Detected #{idx}:** `{category}` *({ts})*")
                st.markdown(f"**Recommended Response:**\n> {response}")
                if follow_up:
                    st.markdown(f"**Suggested Follow-up Question:**\n> *{follow_up}*")
