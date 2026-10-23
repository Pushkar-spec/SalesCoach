"""
Real-Time AI Sales Copilot - Streamlit Frontend Dashboard (Step 1.6)

This module implements the Streamlit dashboard connected to the FastAPI backend
via a bidirectional WebSocket connection with session management.
In this step (Step 1.6):
- Start Call establishes WebSocket connection, performs ping/pong, and sends start_session.
- Stop Call sends stop_session with the active session_id and verifies backend confirmation.
- Reset Session safely stops any active session and resets all collections.
- Displays session ID, connection status, and session lifecycle state.
"""

from datetime import datetime
import os
from typing import Optional
import uuid
import streamlit as st

from ws_client import WebSocketClient, get_backend_ws_url
from components.audio_recorder import audio_recorder

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

if "session_id" not in st.session_state:
    # Stores the unique UUID assigned by the backend for the active session
    st.session_state.session_id = None

if "session_status" not in st.session_state:
    # Allowed states: "idle", "active", "stopped"
    st.session_state.session_status = "idle"

if "status_message" not in st.session_state:
    st.session_state.status_message = "Ready to start a session."

if "ws_client" not in st.session_state:
    # Holds optional standalone WebSocketClient or None
    st.session_state.ws_client = None

if "ws_status" not in st.session_state:
    # Allowed states: "disconnected", "connecting", "connected", "error"
    st.session_state.ws_status = "disconnected"

if "recorder_command" not in st.session_state:
    # Latest control command sent to the browser audio recorder component:
    # {"action": "start" | "stop" | "reset", "id": "<unique_uuid>"}
    st.session_state.recorder_command = None

if "audio_metrics" not in st.session_state:
    # Telemetry counters from audio recording
    st.session_state.audio_metrics = {"chunks_sent": 0, "bytes_sent": 0}

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
# 3. Call Control Handlers (Unified Browser Audio Workflow)
# -----------------------------------------------------------------------------
def handle_start_call() -> None:
    """
    Initiates call: commands the browser audio component to start microphone capture
    and establish the live WebSocket audio stream. One authoritative session.
    """
    # Prevent duplicate session start on repeated clicks
    if st.session_state.session_status == "active":
        st.session_state.status_message = "Session is already active."
        return

    cmd_id = str(uuid.uuid4())
    st.session_state.recorder_command = {"action": "start", "id": cmd_id}
    st.session_state.ws_status = "connecting"
    st.session_state.session_status = "active"
    st.session_state.status_message = "Starting call and requesting microphone access..."


def handle_stop_call() -> None:
    """
    Stops call: commands the browser audio component to stop audio and terminate the session.
    """
    if st.session_state.session_status != "active":
        return

    cmd_id = str(uuid.uuid4())
    st.session_state.recorder_command = {"action": "stop", "id": cmd_id}
    st.session_state.status_message = "Stopping call and finalizing audio stream..."
    st.session_state.session_status = "stopped"
    st.session_state.ws_status = "disconnected"


def handle_reset_session() -> None:
    """
    Resets session: commands audio component to tear down live connection,
    and clears all local state, metrics, and transcript entries.
    """
    cmd_id = str(uuid.uuid4())
    st.session_state.recorder_command = {"action": "reset", "id": cmd_id}
    st.session_state.session_id = None
    st.session_state.session_status = "idle"
    st.session_state.ws_status = "disconnected"
    st.session_state.transcript_entries = []
    st.session_state.coaching_suggestions = []
    st.session_state.audio_metrics = {"chunks_sent": 0, "bytes_sent": 0}
    st.session_state.status_message = (
        "Session reset. All local transcripts and audio streams cleared."
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

    # Session ID Display
    active_sid = st.session_state.session_id
    if active_sid:
        st.markdown(f"**Session ID:**\n`{active_sid}`")
    else:
        st.caption("Session ID: None")

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
    st.metric(
        label="Audio Chunks Sent",
        value=st.session_state.audio_metrics.get("chunks_sent", 0),
    )

    st.divider()

    st.subheader("Integration Pipeline Status")
    st.caption("Current architectural status:")

    ws_label = "Connected" if ws_state == "connected" else "Disconnected"
    session_label = f"Active ({active_sid[:8]}...)" if active_sid else "Idle"
    audio_label = "Active (16kHz PCM16)" if (ws_state == "connected" and status == "active") else "Idle"

    st.markdown(
        f"""
        - **FastAPI HTTP API**: `Implemented` *(Routes: `/`, `/health`)*
        - **FastAPI WebSocket Backend**: `Implemented` *(Route: `/ws`)*
        - **Streamlit WebSocket Link**: `{ws_label}`
        - **Session Protocol**: `{session_label}`
        - **Audio Streaming**: `{audio_label}` *(Unified Step 2.4)*
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
# 6. Call Controls Section (Unified Microphone & Session Management)
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
        help="Start a sales session and initiate live microphone audio streaming.",
    )

with col_stop:
    st.button(
        "⏹ Stop Call",
        on_click=handle_stop_call,
        disabled=(not is_active),
        use_container_width=True,
        help="Stop audio streaming and terminate the session.",
    )

with col_reset:
    st.button(
        "🔄 Reset Session",
        on_click=handle_reset_session,
        disabled=(
            is_idle
            and len(st.session_state.transcript_entries) == 0
            and st.session_state.ws_status == "disconnected"
            and st.session_state.session_id is None
        ),
        use_container_width=True,
        help="Stop session on backend, release microphone, and reset local data.",
    )

# Unified Browser Audio Telemetry & Streamer Component
recorder_event = audio_recorder(
    backend_ws_url=get_backend_ws_url(),
    command=st.session_state.get("recorder_command"),
    key="sales_copilot_audio_recorder",
)

# Process lifecycle state events sent back by the browser component
if recorder_event and isinstance(recorder_event, dict):
    ev_state = recorder_event.get("state")
    ev_sid = recorder_event.get("session_id")

    if ev_sid and ev_sid != st.session_state.session_id:
        st.session_state.session_id = ev_sid

    if ev_state == "recording":
        st.session_state.session_status = "active"
        st.session_state.ws_status = "connected"
        if ev_sid:
            st.session_state.status_message = (
                f"Sales session active: `{ev_sid}`. Live microphone streaming."
            )
    elif ev_state == "stopped":
        st.session_state.session_status = "stopped"
        st.session_state.ws_status = "disconnected"
        chunks = recorder_event.get("chunks_sent", 0)
        bytes_cnt = recorder_event.get("bytes_sent", 0)
        st.session_state.audio_metrics["chunks_sent"] = chunks
        st.session_state.audio_metrics["bytes_sent"] = bytes_cnt
        st.session_state.status_message = (
            f"Call stopped. Finalized audio stream ({chunks} chunks transmitted)."
        )
    elif ev_state == "error":
        st.session_state.session_status = "idle"
        st.session_state.ws_status = "error"
        st.session_state.session_id = None
        err_text = recorder_event.get("error") or "Audio capture encountered an error."
        st.session_state.status_message = f"Error: {err_text}"
    elif ev_state == "idle" and st.session_state.session_status == "active":
        st.session_state.session_status = "idle"
        st.session_state.ws_status = "disconnected"

st.caption(
    f"Backend Endpoint: `{get_backend_ws_url()}` | "
    "Single authoritative WebSocket connection established directly from browser to FastAPI."
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

