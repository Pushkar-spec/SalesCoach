from contextlib import asynccontextmanager
import json
import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from session_manager import (
    ActiveSessionExistsError,
    AudioStateError,
    InvalidAudioConfigError,
    InvalidAudioDataError,
    InvalidSessionError,
    default_session_manager,
)
from stt_service import (
    STTBackpressureError,
    STTClosedError,
    STTError,
    STTStateError,
    default_stt_service,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    Loads Whisper model once on startup into ModelPool on configured device (CUDA/CPU),
    and performs graceful cleanup and executor shutdown on server termination.
    """
    if os.getenv("DISABLE_STT_PREWARM") != "1" and not default_stt_service.is_initialized():
        try:
            logger.info("Lifespan: pre-warming STTService...")
            default_stt_service.initialize()
        except Exception as exc:
            logger.error("Failed to pre-warm STTService during startup: %s", exc)

    yield

    logger.info("Lifespan: shutting down STTService...")
    default_stt_service.shutdown()


app = FastAPI(title="Real-Time AI Sales Copilot API", lifespan=lifespan)


@app.get('/')
def read_root():
    return {"message": "Hello World"}


@app.get('/health')
def health_check():
    return {
        "status": "ok",
        "service": "sales-copilot-api",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "stt_initialized": default_stt_service.is_initialized(),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Bidirectional WebSocket endpoint for real-time sales copilot communication.

    Lifecycle:
    1. Accept the incoming client handshake and assign a unique connection ID.
    2. Enter an event loop while the client is connected.
    3. Receive incoming frames (text control messages or binary audio chunks).
    4. Handle control types: 'ping', 'start_session', 'status', 'stop_session',
       'start_audio', 'stop_audio', 'audio_metrics'.
    5. Handle binary audio frames for active streaming sessions and feed STT.
    6. Emit streaming partial and final transcript events asynchronously.
    7. Enforce connection-level session isolation and reject unauthorized operations.
    8. Catch WebSocketDisconnect gracefully and clean up any active session/audio/STT state.
    """
    connection_id = default_session_manager.generate_connection_id()
    await websocket.accept()
    logger.info("WebSocket connection accepted: connection_id=%s", connection_id)

    try:
        while True:
            # Receive ASGI event frame (handles both text and binary frames)
            message = await websocket.receive()

            # Handle disconnect signal
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(code=message.get("code", 1000))

            # Case A: Binary frame received (Raw PCM16 audio data)
            if "bytes" in message and message["bytes"] is not None:
                raw_bytes: bytes = message["bytes"]
                try:
                    default_session_manager.record_audio_chunk(
                        connection_id, len(raw_bytes)
                    )
                    # Safe logging: log chunk size, NEVER log raw audio samples
                    logger.debug(
                        "Received audio chunk: connection_id=%s bytes=%d",
                        connection_id,
                        len(raw_bytes),
                    )

                    # Forward to STT streaming session if audio is active
                    active_sess = default_session_manager.get_active_session(connection_id)
                    if active_sess and (
                        active_sess.get("audio_state") == "streaming"
                        or active_sess.get("is_audio_streaming")
                    ):
                        sid = active_sess["session_id"]
                        try:
                            default_stt_service.feed_chunk(sid, raw_bytes)
                        except STTBackpressureError as exc:
                            logger.warning(
                                "STT backpressure triggered: connection_id=%s session_id=%s reason=%s",
                                connection_id,
                                sid,
                                str(exc),
                            )
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Backpressure: {exc}",
                            })
                        except (STTClosedError, STTStateError):
                            # Session stopping or no STT session registered
                            pass

                except (InvalidSessionError, AudioStateError, InvalidAudioDataError) as exc:
                    logger.warning(
                        "Rejected binary audio frame: connection_id=%s reason=%s",
                        connection_id,
                        str(exc),
                    )
                    await websocket.send_json({
                        "type": "error",
                        "message": str(exc),
                    })
                continue

            # Case B: Text frame received (JSON control message)
            raw_data = message.get("text", "")

            # 1. Parse JSON safely
            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError:
                logger.warning(
                    "Invalid JSON received: connection_id=%s", connection_id
                )
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format",
                })
                continue

            # 2. Validate that payload is a JSON object (dict)
            if not isinstance(payload, dict):
                logger.warning(
                    "Non-object payload received: connection_id=%s", connection_id
                )
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid message: expected a JSON object",
                })
                continue

            # 3. Retrieve and dispatch message type
            message_type = payload.get("type")

            # Heartbeat ping/pong
            if message_type == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "message": payload.get("message", ""),
                })

            # Start a new sales coaching session
            elif message_type == "start_session":
                try:
                    session = default_session_manager.create_session(connection_id)
                    logger.info(
                        "Session started: connection_id=%s session_id=%s",
                        connection_id,
                        session["session_id"],
                    )
                    await websocket.send_json({
                        "type": "session_started",
                        "session_id": session["session_id"],
                        "status": session["status"],
                    })
                except ActiveSessionExistsError as exc:
                    logger.warning(
                        "Start session rejected: connection_id=%s reason=%s",
                        connection_id,
                        str(exc),
                    )
                    await websocket.send_json({
                        "type": "error",
                        "message": str(exc),
                    })

            # Query current session status
            elif message_type == "status":
                active_session = default_session_manager.get_active_session(connection_id)
                if active_session:
                    await websocket.send_json({
                        "type": "status",
                        "session_id": active_session["session_id"],
                        "status": active_session["status"],
                    })
                else:
                    await websocket.send_json({
                        "type": "status",
                        "session_id": None,
                        "status": "idle",
                    })

            # Stop active sales coaching session
            elif message_type == "stop_session":
                req_session_id = payload.get("session_id")
                try:
                    if req_session_id:
                        await default_stt_service.cleanup_session(req_session_id)

                    stopped = default_session_manager.stop_session(
                        connection_id, req_session_id
                    )
                    logger.info(
                        "Session stopped: connection_id=%s session_id=%s",
                        connection_id,
                        stopped["session_id"],
                    )
                    await websocket.send_json({
                        "type": "session_stopped",
                        "session_id": stopped["session_id"],
                        "status": stopped["status"],
                    })
                except InvalidSessionError as exc:
                    logger.warning(
                        "Stop session rejected: connection_id=%s reason=%s",
                        connection_id,
                        str(exc),
                    )
                    await websocket.send_json({
                        "type": "error",
                        "message": str(exc),
                    })

            # Start audio streaming
            elif message_type == "start_audio":
                req_session_id = payload.get("session_id")
                audio_format = payload.get("format")
                sample_rate = payload.get("sample_rate")
                channels = payload.get("channels")
                try:
                    session = default_session_manager.start_audio(
                        connection_id,
                        req_session_id,
                        audio_format,
                        sample_rate,
                        channels,
                    )
                    sid = session["session_id"]
                    logger.info(
                        "Audio started: connection_id=%s session_id=%s",
                        connection_id,
                        sid,
                    )

                    # Wire up STT streaming session if service is initialized
                    if default_stt_service.is_initialized():
                        async def on_transcript(event: dict) -> None:
                            try:
                                await websocket.send_json(event)
                            except Exception as exc:
                                logger.debug("Could not send transcript event: %s", exc)

                        default_stt_service.create_session(sid, on_transcript)

                    await websocket.send_json({
                        "type": "audio_started",
                        "session_id": sid,
                    })
                except (InvalidSessionError, AudioStateError, InvalidAudioConfigError) as exc:
                    logger.warning(
                        "Start audio rejected: connection_id=%s reason=%s",
                        connection_id,
                        str(exc),
                    )
                    await websocket.send_json({
                        "type": "error",
                        "message": str(exc),
                    })

            # Stop audio streaming
            elif message_type == "stop_audio":
                req_session_id = payload.get("session_id")
                try:
                    # 1. Stop accepting and flush trailing STT audio before emitting audio_stopped
                    if req_session_id:
                        default_stt_service.stop_accepting(req_session_id)
                        await default_stt_service.flush_and_stop(req_session_id)

                    # 2. Record metrics and stop audio in session manager
                    metrics = default_session_manager.stop_audio(
                        connection_id, req_session_id
                    )
                    logger.info(
                        "Audio stopped: connection_id=%s session_id=%s chunks=%d bytes=%d duration=%.3fs expected=%d",
                        connection_id,
                        metrics["session_id"],
                        metrics["audio_chunk_count"],
                        metrics["audio_bytes_received"],
                        metrics["audio_duration_seconds"],
                        metrics["expected_audio_bytes"],
                    )
                    await websocket.send_json({
                        "type": "audio_stopped",
                        "session_id": metrics["session_id"],
                        "audio_chunk_count": metrics["audio_chunk_count"],
                        "audio_bytes_received": metrics["audio_bytes_received"],
                        "audio_duration_seconds": metrics["audio_duration_seconds"],
                        "expected_audio_bytes": metrics["expected_audio_bytes"],
                    })
                except (InvalidSessionError, AudioStateError) as exc:
                    logger.warning(
                        "Stop audio rejected: connection_id=%s reason=%s",
                        connection_id,
                        str(exc),
                    )
                    await websocket.send_json({
                        "type": "error",
                        "message": str(exc),
                    })

            # Query real-time audio metrics and diagnostics
            elif message_type == "audio_metrics":
                req_session_id = payload.get("session_id")
                active_session = default_session_manager.get_active_session(connection_id)
                sid = req_session_id or (active_session["session_id"] if active_session else None)
                if sid:
                    try:
                        metrics = default_session_manager.get_audio_metrics(connection_id, sid)
                        await websocket.send_json({
                            "type": "audio_metrics",
                            **metrics,
                        })
                    except InvalidSessionError as exc:
                        await websocket.send_json({"type": "error", "message": str(exc)})
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No active session for audio metrics query",
                    })

            # Unsupported message types
            else:
                logger.warning(
                    "Unsupported message type: connection_id=%s type=%s",
                    connection_id,
                    message_type,
                )
                await websocket.send_json({
                    "type": "error",
                    "message": "Unsupported message type",
                })

    except WebSocketDisconnect:
        # Gracefully handle normal client disconnection
        stopped = default_session_manager.cleanup_connection(connection_id)
        if stopped:
            await default_stt_service.cleanup_session(stopped["session_id"])
            logger.info(
                "Cleaned up active session on disconnect: connection_id=%s session_id=%s",
                connection_id,
                stopped["session_id"],
            )
        logger.info("WebSocket disconnected: connection_id=%s", connection_id)
