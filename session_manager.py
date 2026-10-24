"""
In-Memory Session Manager for Real-Time AI Sales Copilot (Step 1.6 & Step 2.2)

Manages sales sessions associated with active WebSocket connections.
Enforces session isolation, ownership rules, lifecycle transitions, and audio streaming state.

Note:
All session records are stored in-memory and will not persist across backend server restarts.
"""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Optional
import uuid

logger = logging.getLogger(__name__)


class SessionError(Exception):
    """Base exception for session management errors."""
    pass


class ActiveSessionExistsError(SessionError):
    """Raised when attempting to start a session while one is already active on the connection."""
    pass


class InvalidSessionError(SessionError):
    """Raised when a session ID is missing, invalid, unauthorized, or not active."""
    pass


class AudioError(SessionError):
    """Base exception for audio streaming errors."""
    pass


class AudioStateError(AudioError):
    """Raised when an audio operation is invalid for the current audio state."""
    pass


class InvalidAudioConfigError(AudioError):
    """Raised when audio configuration parameters are invalid or unsupported."""
    pass


class InvalidAudioDataError(AudioError):
    """Raised when an incoming binary audio frame is malformed (empty or odd byte length)."""
    pass


class SessionManager:
    """
    In-memory session manager tracking sales sessions, connection ownership,
    and audio streaming lifecycle.
    """

    def __init__(self) -> None:
        # Map of session_id -> session record dict
        self._sessions: dict[str, dict[str, Any]] = {}
        # Map of connection_id -> active session_id
        self._active_connections: dict[str, str] = {}

    def generate_connection_id(self) -> str:
        """Generate a unique ID for a newly accepted WebSocket connection."""
        return str(uuid.uuid4())

    def create_session(self, connection_id: str) -> dict[str, Any]:
        """
        Create and activate a new sales session for the given connection.

        Args:
            connection_id: Server-generated unique ID of the WebSocket connection.

        Returns:
            dict: The newly created session record.

        Raises:
            ValueError: If connection_id is missing.
            ActiveSessionExistsError: If this connection already has an active session.
        """
        if not connection_id or not isinstance(connection_id, str):
            raise ValueError("connection_id must be a non-empty string")

        if connection_id in self._active_connections:
            raise ActiveSessionExistsError(
                "An active session already exists for this connection"
            )

        session_id = str(uuid.uuid4())
        session = {
            "session_id": session_id,
            "connection_id": connection_id,
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_at_perf": time.perf_counter(),
            # Audio streaming state & diagnostics (Step 2.2 & 2.6)
            "audio_state": "idle",
            "audio_chunk_count": 0,
            "audio_bytes_received": 0,
            "audio_started_at_perf": None,
            "audio_stopped_at_perf": None,
            "audio_duration_seconds": 0.0,
            "last_chunk_at_perf": None,
            "expected_audio_bytes": 0,
        }

        self._sessions[session_id] = session
        self._active_connections[connection_id] = session_id
        logger.info(
            "Created active session: session_id=%s for connection_id=%s",
            session_id,
            connection_id,
        )
        return session

    def get_active_session(self, connection_id: str) -> Optional[dict[str, Any]]:
        """
        Retrieve the active session associated with the given connection.

        Args:
            connection_id: The connection ID to look up.

        Returns:
            dict or None: Active session record if one exists, otherwise None.
        """
        session_id = self._active_connections.get(connection_id)
        if not session_id:
            return None
        return self._sessions.get(session_id)

    def start_audio(
        self,
        connection_id: str,
        session_id: str,
        audio_format: str,
        sample_rate: int,
        channels: int,
    ) -> dict[str, Any]:
        """
        Start audio streaming for the active session.
        Validates connection ownership, active status, and audio parameters.

        Args:
            connection_id: Server-assigned connection ID.
            session_id: The active session ID.
            audio_format: Must be 'pcm16'.
            sample_rate: Must be 16000 (Hz).
            channels: Must be 1 (mono).

        Returns:
            dict: The updated session record.
        """
        if not session_id or not isinstance(session_id, str):
            raise InvalidSessionError("Session ID must be a non-empty string")

        session = self._sessions.get(session_id)
        if session is None or session.get("connection_id") != connection_id:
            logger.warning(
                "Start audio rejected: session_id=%s not found or not owned by connection_id=%s",
                session_id,
                connection_id,
            )
            raise InvalidSessionError("Session not found or unauthorized")

        if session.get("status") != "active":
            logger.warning(
                "Start audio rejected: session_id=%s is %s",
                session_id,
                session.get("status"),
            )
            raise InvalidSessionError("Session is not active")

        if session.get("audio_state") == "streaming":
            logger.warning(
                "Start audio rejected: session_id=%s audio is already streaming",
                session_id,
            )
            raise AudioStateError("Audio is already streaming for this session")

        # Validate audio format parameters
        if audio_format != "pcm16":
            raise InvalidAudioConfigError(
                f"Unsupported audio format: '{audio_format}'. Only 'pcm16' is supported."
            )

        if sample_rate != 16000:
            raise InvalidAudioConfigError(
                f"Unsupported sample rate: {sample_rate}. Only 16000 Hz is supported."
            )

        if channels != 1:
            raise InvalidAudioConfigError(
                f"Unsupported channel count: {channels}. Only 1 (mono) is supported."
            )

        session["audio_state"] = "streaming"
        session["audio_started_at_perf"] = time.perf_counter()
        session["last_chunk_at_perf"] = None
        session["audio_chunk_count"] = 0
        session["audio_bytes_received"] = 0
        session["audio_duration_seconds"] = 0.0
        session["expected_audio_bytes"] = 0
        logger.info(
            "Audio started for session_id=%s connection_id=%s (format=%s, rate=%d, channels=%d)",
            session_id,
            connection_id,
            audio_format,
            sample_rate,
            channels,
        )
        return session

    def stop_audio(self, connection_id: str, session_id: str) -> dict[str, Any]:
        """
        Stop audio streaming for the active session and return audio metrics.

        Args:
            connection_id: Server-assigned connection ID.
            session_id: The active session ID.

        Returns:
            dict: Summary containing session_id, audio_chunk_count, audio_bytes_received,
                  audio_duration_seconds, and expected_audio_bytes.
        """
        if not session_id or not isinstance(session_id, str):
            raise InvalidSessionError("Session ID must be a non-empty string")

        session = self._sessions.get(session_id)
        if session is None or session.get("connection_id") != connection_id:
            logger.warning(
                "Stop audio rejected: session_id=%s not found or not owned by connection_id=%s",
                session_id,
                connection_id,
            )
            raise InvalidSessionError("Session not found or unauthorized")

        if session.get("status") != "active":
            raise InvalidSessionError("Session is not active")

        if session.get("audio_state") != "streaming":
            logger.warning(
                "Stop audio rejected: session_id=%s audio is not streaming",
                session_id,
            )
            raise AudioStateError("Audio is not streaming for this session")

        now = time.perf_counter()
        session["audio_state"] = "idle"
        session["audio_stopped_at_perf"] = now
        started = session.get("audio_started_at_perf")
        duration = max(0.0, now - started) if started is not None else 0.0
        session["audio_duration_seconds"] = duration
        # 16000 samples/sec * 2 bytes/sample (pcm16) = 32000 bytes/sec
        expected_bytes = int(duration * 32000)
        session["expected_audio_bytes"] = expected_bytes

        logger.info(
            "Audio stopped for session_id=%s connection_id=%s (chunks=%d, bytes=%d, duration=%.3fs, expected_bytes=%d)",
            session_id,
            connection_id,
            session["audio_chunk_count"],
            session["audio_bytes_received"],
            duration,
            expected_bytes,
        )
        return {
            "session_id": session["session_id"],
            "audio_chunk_count": session["audio_chunk_count"],
            "audio_bytes_received": session["audio_bytes_received"],
            "audio_duration_seconds": round(duration, 3),
            "expected_audio_bytes": expected_bytes,
        }

    def record_audio_chunk(self, connection_id: str, byte_count: int) -> dict[str, Any]:
        """
        Validate and record an incoming binary audio chunk for the active session.
        Increments chunk and byte counters without storing raw audio bytes.

        Args:
            connection_id: Server-assigned connection ID sending the frame.
            byte_count: Length in bytes of the incoming binary frame.

        Returns:
            dict: Updated audio metrics.
        """
        active_session = self.get_active_session(connection_id)
        if active_session is None:
            raise InvalidSessionError("Binary audio received without an active session")

        if active_session.get("audio_state") != "streaming":
            raise AudioStateError("Audio is not streaming for this session")

        if byte_count <= 0:
            raise InvalidAudioDataError("Empty audio frame received")

        # PCM16 requires 2 bytes per sample, so frame byte length must be even
        if byte_count % 2 != 0:
            raise InvalidAudioDataError(
                "Invalid audio frame length: PCM16 requires an even number of bytes"
            )

        active_session["last_chunk_at_perf"] = time.perf_counter()
        active_session["audio_chunk_count"] += 1
        active_session["audio_bytes_received"] += byte_count

        return {
            "session_id": active_session["session_id"],
            "audio_chunk_count": active_session["audio_chunk_count"],
            "audio_bytes_received": active_session["audio_bytes_received"],
        }

    def get_audio_metrics(self, connection_id: str, session_id: str) -> dict[str, Any]:
        """
        Get current audio metrics, duration, and staleness information.

        Args:
            connection_id: Server-assigned connection ID.
            session_id: Active or stopped session ID.

        Returns:
            dict: Audio metrics summary.
        """
        if not session_id or not isinstance(session_id, str):
            raise InvalidSessionError("Session ID must be a non-empty string")

        session = self._sessions.get(session_id)
        if session is None or session.get("connection_id") != connection_id:
            raise InvalidSessionError("Session not found or unauthorized")

        now = time.perf_counter()
        started = session.get("audio_started_at_perf")
        if session.get("audio_state") == "streaming" and started is not None:
            duration = max(0.0, now - started)
        else:
            duration = session.get("audio_duration_seconds", 0.0)

        last_chunk = session.get("last_chunk_at_perf")
        time_since_last_chunk = (
            max(0.0, now - last_chunk) if last_chunk is not None else None
        )

        return {
            "session_id": session["session_id"],
            "audio_state": session.get("audio_state", "idle"),
            "audio_chunk_count": session.get("audio_chunk_count", 0),
            "audio_bytes_received": session.get("audio_bytes_received", 0),
            "audio_duration_seconds": round(duration, 3),
            "expected_audio_bytes": int(duration * 32000),
            "time_since_last_chunk_seconds": (
                round(time_since_last_chunk, 3) if time_since_last_chunk is not None else None
            ),
        }

    def check_audio_staleness(self, connection_id: str, max_idle_seconds: float = 5.0) -> bool:
        """
        Check if an active streaming session has stalled without receiving audio chunks.

        Args:
            connection_id: The connection ID to inspect.
            max_idle_seconds: Maximum allowed time without chunks (default 5.0 seconds).

        Returns:
            bool: True if audio is streaming and stalled past threshold, False otherwise.
        """
        active_session = self.get_active_session(connection_id)
        if not active_session or active_session.get("audio_state") != "streaming":
            return False

        now = time.perf_counter()
        last_chunk = active_session.get("last_chunk_at_perf")
        started = active_session.get("audio_started_at_perf")

        if last_chunk is not None:
            return (now - last_chunk) > max_idle_seconds
        elif started is not None:
            # No chunks received since audio started
            return (now - started) > max_idle_seconds

        return False

    def stop_session(self, connection_id: str, session_id: str) -> dict[str, Any]:
        """
        Stop an active session belonging to the given connection.
        If audio was streaming, it is automatically marked as idle.

        Args:
            connection_id: Server-assigned connection ID performing the request.
            session_id: The ID of the session to stop.

        Returns:
            dict: The updated session record with status 'stopped'.

        Raises:
            InvalidSessionError: If session_id is missing, invalid, not active,
                                 or does not belong to connection_id.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise InvalidSessionError("Session ID must be a non-empty string")

        session = self._sessions.get(session_id)
        # Enforce existence and ownership using a generic error message
        if session is None or session.get("connection_id") != connection_id:
            logger.warning(
                "Stop rejected: session_id=%s not found or not owned by connection_id=%s",
                session_id,
                connection_id,
            )
            raise InvalidSessionError("Session not found or unauthorized")

        if session.get("status") != "active":
            logger.warning(
                "Stop rejected: session_id=%s is already %s",
                session_id,
                session.get("status"),
            )
            raise InvalidSessionError("Session is not active")

        now = time.perf_counter()
        # Automatically stop audio streaming if active
        if session.get("audio_state") == "streaming":
            session["audio_state"] = "idle"
            session["audio_stopped_at_perf"] = now
            started = session.get("audio_started_at_perf")
            if started is not None:
                duration = max(0.0, now - started)
                session["audio_duration_seconds"] = duration
                session["expected_audio_bytes"] = int(duration * 32000)
            logger.info(
                "Audio automatically stopped on session stop: session_id=%s",
                session_id,
            )

        # Mark session as stopped
        session["status"] = "stopped"
        session["stopped_at"] = datetime.now(timezone.utc).isoformat()
        session["stopped_at_perf"] = now
        created_perf = session.get("created_at_perf")
        if created_perf is not None:
            session["total_duration_seconds"] = round(max(0.0, now - created_perf), 3)

        # Release connection's active session reference
        if self._active_connections.get(connection_id) == session_id:
            del self._active_connections[connection_id]

        logger.info(
            "Stopped session: session_id=%s for connection_id=%s",
            session_id,
            connection_id,
        )
        return session

    def cleanup_connection(self, connection_id: str) -> Optional[dict[str, Any]]:
        """
        Clean up connection state upon WebSocket disconnection.
        Marks any active session and audio streaming belonging to this connection as stopped.

        Args:
            connection_id: The connection ID that disconnected.

        Returns:
            dict or None: The stopped session record, or None if no session was active.
        """
        session_id = self._active_connections.pop(connection_id, None)
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            now = time.perf_counter()
            if session.get("audio_state") == "streaming":
                session["audio_state"] = "idle"
                session["audio_stopped_at_perf"] = now
                started = session.get("audio_started_at_perf")
                if started is not None:
                    session["audio_duration_seconds"] = round(max(0.0, now - started), 3)
                    session["expected_audio_bytes"] = int(session["audio_duration_seconds"] * 32000)
            if session.get("status") == "active":
                session["status"] = "stopped"
                session["stopped_at"] = datetime.now(timezone.utc).isoformat()
                session["stopped_at_perf"] = now
                logger.info(
                    "Cleaned up session on disconnect: session_id=%s for connection_id=%s",
                    session_id,
                    connection_id,
                )
            return session
        return None


# Module-level default instance for FastAPI application
default_session_manager = SessionManager()
