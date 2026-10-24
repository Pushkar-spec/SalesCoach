# Real-Time AI Sales Copilot

A real-time AI assistant designed to listen to sales conversations, provide live objection handling, suggest contextual talking points, and deliver instant deal-coaching feedback.

---

## Current Status: Step 2.6 Complete (Measure & Harden the Audio Pipeline)

- [x] **FastAPI Application**: Running server with `GET /`, `GET /health`, and WebSocket `ws://127.0.0.1:8000/ws`.
- [x] **In-Memory Session Manager**: Server-side session tracking (`session_manager.py`) with monotonic clock diagnostics (`time.perf_counter`), UUID generation, connection-level session isolation, stall detection (`check_audio_staleness`), and lifecycle cleanup on disconnect.
- [x] **Structured Message Protocol**: Support for `ping`/`pong`, `start_session`, `status`, `stop_session`, `start_audio`, `stop_audio`, `audio_metrics`, and structured error messages.
- [x] **Harded Audio Pipeline**: Bounded timeouts for WebSocket connection (5s), control messages (5s), and shutdown acknowledgement (3s bounded fallback), plus 5s streaming stall detector. Immediate track and node teardown on stop/reset.
- [x] **Audio Diagnostics & Telemetry**: Lightweight measurement of recording duration, audio chunks/bytes sent vs received, expected vs actual bytes (at 32 kB/s), startup latency, and shutdown latency. Real-time telemetry card with live duration and startup latency display.
- [x] **Unified Streamlit Call Controls**: The main Call Controls (`▶ Start Call`, `⏹ Stop Call`, `🔄 Reset Session`) directly drive the browser microphone capture and its WebSocket stream with one authoritative live session.
- [x] **Automated Testing**: 99 deterministic `pytest` tests and 13 Node.js audio unit tests covering HTTP routes, WebSocket protocols, variable recording durations, synthetic audio signals (silence, sine, noise), audio stalls, and graceful cleanup.
- [x] **Virtual Environment**: Python 3.12 virtual environment established (`venv`).
- [x] **Environment Config**: `.gitignore` and `.env.example` templates created.
- [x] **Version Control**: Git repository initialized with appropriate ignore rules.
- [ ] **Speech-to-Text & Transcription Engine**: *Planned for Phase 3*
- [ ] **LLM Coaching & Objection Handling**: *Planned for Phase 4*
- [ ] **Retrieval-Augmented Generation (RAG)**: *Planned for Phase 5*
- [ ] **ML Objection Classifier**: *Planned for Phase 6*

> **Important Architecture & Deployment Note**: In Step 2.6, the audio transport pipeline is comprehensively measured and hardened. Clicking `▶ Start Call` in the Streamlit UI sends an authoritative start command to the browser audio component, which requests microphone access, connects to the FastAPI backend, and streams 16kHz PCM16 audio frames. Bounded timeouts ensure neither the frontend nor backend can hang during connection or shutdown. Clicking `⏹ Stop Call` stops the stream and releases microphone tracks immediately. When deployed to HTTPS environments, browsers require a secure origin (`https://`) to grant microphone access, and the WebSocket target must use `wss://`.




---

## Planned Technology Stack

| Layer | Component / Technology | Status |
| :--- | :--- | :--- |
| **Backend API** | FastAPI + Uvicorn | Implemented (`/` and `/health` routes) |
| **Real-time Streaming** | WebSockets (Bidirectional) | Implemented (`/ws` endpoint) |
| **Session Management** | In-Memory Session Manager | Implemented (`session_manager.py`) |
| **Frontend UI** | Streamlit | Implemented (Dashboard linked to FastAPI via WebSockets) |
| **Speech Processing** | Speech-to-Text engine | Planned (Phase 3) |
| **Intelligence** | Large Language Model (Prompt-engineered sales coach) | Planned (Phase 4) |
| **Knowledge Retrieval**| RAG (Sales playbook & product documentation) | Planned (Phase 5) |
| **Classification** | scikit-learn (Objection intent classification) | Planned (Phase 6) |

---

## Understanding Sessions vs. WebSocket Connections

It is important to distinguish between the **transport layer** and the **application session**:

1. **WebSocket Connection**:
   - The underlying, full-duplex TCP network channel established between the browser/client and the FastAPI server.
   - Assigned a unique, server-generated `connection_id`.
   - Persists as long as the socket remains open.
2. **Sales Session**:
   - A domain-level conversation instance assigned a unique `session_id` (UUIDv4).
   - Created when the user explicitly clicks **▶ Start Call** and terminated when clicking **⏹ Stop Call**.
   - Enforces **session isolation**: each WebSocket connection may own at most one active session. One connection cannot stop or inspect another connection's session.
   - If the WebSocket disconnects, the server automatically terminates and cleans up any associated active session.

> **Storage Limitation**: Sessions are maintained in-memory for fast prototyping and will not survive a backend server restart.

---

## WebSocket Message Protocol

The `/ws` endpoint exchanges JSON text frames following a structured request/response protocol:

### 1. Heartbeat Ping / Pong
- **Client Request**:
  ```json
  {"type": "ping", "message": "Hello from Streamlit"}
  ```
- **Server Response**:
  ```json
  {"type": "pong", "message": "Hello from Streamlit"}
  ```

### 2. Start Sales Session
- **Client Request**:
  ```json
  {"type": "start_session"}
  ```
- **Server Response (Success)**:
  ```json
  {
    "type": "session_started",
    "session_id": "0f7e0689-f647-46e2-9bec-a041b311a321",
    "status": "active"
  }
  ```
- **Server Response (If session already active)**:
  ```json
  {
    "type": "error",
    "message": "An active session already exists for this connection"
  }
  ```

### 3. Query Session Status
- **Client Request**:
  ```json
  {"type": "status"}
  ```
- **Server Response (Active)**:
  ```json
  {
    "type": "status",
    "session_id": "0f7e0689-f647-46e2-9bec-a041b311a321",
    "status": "active"
  }
  ```
- **Server Response (Idle)**:
  ```json
  {
    "type": "status",
    "session_id": null,
    "status": "idle"
  }
  ```

### 4. Stop Sales Session
- **Client Request**:
  ```json
  {
    "type": "stop_session",
    "session_id": "0f7e0689-f647-46e2-9bec-a041b311a321"
  }
  ```
- **Server Response (Success)**:
  ```json
  {
    "type": "session_stopped",
    "session_id": "0f7e0689-f647-46e2-9bec-a041b311a321",
    "status": "stopped"
  }
  ```
- **Server Response (Unauthorized or Not Found)**:
  ```json
  {
    "type": "error",
    "message": "Session not found or unauthorized"
  }
  ```

### 5. Start Audio Streaming
- **Client Request**:
  ```json
  {
    "type": "start_audio",
    "session_id": "0f7e0689-f647-46e2-9bec-a041b311a321",
    "format": "pcm16",
    "sample_rate": 16000,
    "channels": 1
  }
  ```
- **Server Response (Success)**:
  ```json
  {
    "type": "audio_started",
    "session_id": "0f7e0689-f647-46e2-9bec-a041b311a321"
  }
  ```
- **Server Response (Unsupported Config or Invalid State)**:
  ```json
  {
    "type": "error",
    "message": "Unsupported audio format: 'mp3'. Only 'pcm16' is supported."
  }
  ```

### 6. Binary Audio Frames
- **Payload**: Raw binary bytes over the WebSocket connection (PCM16 mono audio).
- **Validation**:
  - Connection must have an active session owned by the client.
  - Audio state must be `streaming` (`start_audio` previously confirmed).
  - Chunk must be non-empty and have an even number of bytes (2 bytes per sample).
- **Server Action**: Increments `audio_chunk_count` and `audio_bytes_received`. Does not buffer or persist audio.
- **Server Response on Error**: If invalid or sent while idle, returns a structured JSON error frame:
  ```json
  {
    "type": "error",
    "message": "Invalid audio chunk: frame has odd number of bytes (3)"
  }
  ```

### 7. Stop Audio Streaming
- **Client Request**:
  ```json
  {
    "type": "stop_audio",
    "session_id": "0f7e0689-f647-46e2-9bec-a041b311a321"
  }
  ```
- **Server Response (Success)**:
  ```json
  {
    "type": "audio_stopped",
    "session_id": "0f7e0689-f647-46e2-9bec-a041b311a321",
    "audio_chunk_count": 42,
    "audio_bytes_received": 134400,
    "audio_duration_seconds": 4.205,
    "expected_audio_bytes": 134560
  }
  ```
- **Server Response (Not Streaming)**:
  ```json
  {
    "type": "error",
    "message": "Cannot stop audio: audio is not streaming for session 0f7e0689-f647-46e2-9bec-a041b311a321"
  }
  ```

### 8. Query Audio Streaming Metrics
- **Client Request**:
  ```json
  {
    "type": "audio_metrics",
    "session_id": "0f7e0689-f647-46e2-9bec-a041b311a321"
  }
  ```
- **Server Response (Success)**:
  ```json
  {
    "type": "audio_metrics",
    "session_id": "0f7e0689-f647-46e2-9bec-a041b311a321",
    "audio_state": "streaming",
    "audio_chunk_count": 12,
    "audio_bytes_received": 96000,
    "audio_duration_seconds": 3.012,
    "expected_audio_bytes": 96384,
    "time_since_last_chunk_seconds": 0.084
  }
  ```

---

## Getting Started

### 1. Activate the Virtual Environment

Open Windows PowerShell in the project directory (`c:\Users\pushk\Desktop\SalesCoach`) and run:

```powershell
.\venv\Scripts\Activate.ps1
```

*(If you encounter an execution policy restriction in PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in your current terminal session).*

### 2. Environment Configuration

Copy the sample environment file to create your local `.env`:

```powershell
Copy-Item .env.example .env
```

Configurable settings in `.env`:
- `HOST`: Server host (Default: `127.0.0.1`)
- `PORT`: Server port (Default: `8000`)
- `ENVIRONMENT`: Deployment mode (Default: `development`)
- `BACKEND_WS_URL`: Backend WebSocket URL for frontend client (Default: `ws://127.0.0.1:8000/ws`)

### 3. Install Dependencies

Install dependencies into your activated virtual environment:

```powershell
pip install -r requirements.txt
```

---

## Running the Application

Run the backend and frontend in **two separate terminal windows**.

### Terminal 1 — Backend (FastAPI + Uvicorn)

In your first activated PowerShell terminal, start the FastAPI server:

```powershell
python -m uvicorn main:app --reload
```

The server will start listening at:
- HTTP API: `http://127.0.0.1:8000`
- WebSocket: `ws://127.0.0.1:8000/ws`
- Interactive API Docs (Swagger UI): `http://127.0.0.1:8000/docs`

Test HTTP endpoints:
```powershell
# Root route
Invoke-RestMethod -Uri http://127.0.0.1:8000/

# Health check route
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

### Terminal 2 — Frontend (Streamlit Dashboard)

In a second PowerShell terminal (also with `venv` activated):

```powershell
python -m streamlit run app.py
```

The Streamlit dashboard will automatically open in your default browser at:
- Local URL: `http://localhost:8501`

Click **▶ Start Call** to establish the WebSocket handshake and start a new sales session. The active session UUID will appear in the sidebar and header banner.

---

## Running Automated Tests

Run the test suite using `pytest` inside your activated virtual environment:

```powershell
python -m pytest -v
```

Run the JavaScript audio processing unit tests using Node.js:

```powershell
node --test tests/test_audio_processor.js
```

The test suite contains **81 automated Python tests** and **11 JavaScript audio tests** across 7 modules:
- [tests/test_health.py](file:///c:/Users/pushk/Desktop/SalesCoach/tests/test_health.py): Validates HTTP status codes and payloads for `/` and `/health`.
- [tests/test_websocket.py](file:///c:/Users/pushk/Desktop/SalesCoach/tests/test_websocket.py): Validates backend WebSocket handshake, ping/pong, start_session, status query, stop_session, start_audio, binary audio streaming, stop_audio, connection session isolation, and disconnect cleanup.
- [tests/test_session_manager.py](file:///c:/Users/pushk/Desktop/SalesCoach/tests/test_session_manager.py): Unit tests for `SessionManager` verifying unique UUID generation, duplicate active session rejection, audio state management, chunk and byte counting, ownership checks, and cleanup.
- [tests/test_ws_client.py](file:///c:/Users/pushk/Desktop/SalesCoach/tests/test_ws_client.py): Validates frontend client helper URL configuration, handshake validation, session lifecycle calls, error propagation, and idempotent socket closure.
- [tests/test_app.py](file:///c:/Users/pushk/Desktop/SalesCoach/tests/test_app.py): Validates Streamlit UI button clicks (`Start Call`, `Stop Call`, `Reset Session`), state changes, session ID rendering, recorder command propagation, and socket cleanup via Streamlit's `AppTest` framework.
- [tests/test_audio_component.py](file:///c:/Users/pushk/Desktop/SalesCoach/tests/test_audio_component.py): Validates custom Streamlit audio recorder component assets, HTML structure, wrapper argument forwarding (`command`, `backend_ws_url`, `key`), and component rendering.
- [tests/test_audio_processor.js](file:///c:/Users/pushk/Desktop/SalesCoach/tests/test_audio_processor.js): Validates client-side linear-interpolation resampling (48kHz/44.1kHz -> 16kHz), 16-bit little-endian signed PCM conversion, and 250ms chunk accumulation.

---

## Project Structure

```text
SalesCoach/
├── .env.example        # Environment variable template with safe placeholders
├── .gitignore           # Git ignore rules for venv, secrets, caches, and logs
├── app.py              # Streamlit frontend dashboard (Step 1.2, 1.5, 1.6 & 2.4)
├── main.py             # FastAPI entry point ('/', '/health', and '/ws' audio endpoint)
├── README.md           # Project documentation and developer guide
├── requirements.txt    # Project dependencies
├── session_manager.py  # In-memory session manager with audio state tracking (Step 1.6 & 2.2)
├── ws_client.py        # Synchronous WebSocket client helper (Step 1.5 & 1.6)
├── components/         # Streamlit custom components
│   └── audio_recorder/ # Browser microphone capture and audio streaming component (Step 2.3 & 2.4)
│       ├── __init__.py       # Component declaration and Python wrapper
│       └── frontend/         # Frontend static assets
│           ├── index.html    # Telemetry card, visualizer, metrics UI
│           ├── recorder.js   # AudioStreamRecorder, getUserMedia, Web Audio, WS streamer
│           ├── audio-utils.js# Resampling, PCM16 conversion, chunk accumulator
│           └── styles.css    # Dark-themed glassmorphic styling
├── tests/              # Automated test suite (81 Python tests + 11 JS audio tests)
│   ├── test_app.py            # Streamlit AppTest UI state and session tests
│   ├── test_audio_component.py# Streamlit audio component wrapper tests
│   ├── test_audio_processor.js# Node.js audio processing unit tests
│   ├── test_health.py         # HTTP tests for root and health routes
│   ├── test_session_manager.py# Unit tests for in-memory SessionManager
│   ├── test_websocket.py      # Backend WebSocket protocol, session, and audio tests
│   └── test_ws_client.py      # Frontend WebSocketClient lifecycle and validation tests
└── venv/               # Local Python virtual environment (ignored by Git)
```

---

## Common Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Script execution disabled` error | PowerShell execution policy restriction | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` in your PowerShell window. |
| `streamlit` or `uvicorn: command not found` | Virtual environment is not activated | Ensure `(venv)` appears before your terminal prompt by running `.\venv\Scripts\Activate.ps1`. |
| Port 8000 or 8501 already in use | Another process is holding the port | Run `python -m uvicorn main:app --reload --port 8001` or `python -m streamlit run app.py --server.port 8502`. |
| WebSocket connection fails in Streamlit | FastAPI server is not running | Ensure `python -m uvicorn main:app --reload` is running in Terminal 1 on port 8000. |

---

## Project Roadmap

- **Step 1.1**: Development Environment & Foundation *(Complete)*
- **Step 1.2**: Streamlit Frontend Foundation *(Complete)*
- **Step 1.3**: FastAPI Health Check & Backend Readiness *(Complete)*
- **Step 1.4**: First WebSocket Connection & Bidirectional Communication *(Complete)*
- **Step 1.5**: Connect Streamlit Frontend to FastAPI WebSocket Backend *(Complete)*
- **Step 1.6**: Add Session Management & Message Protocol *(Complete)*
- **Phase 2**: Real-Time Audio Streaming Pipeline *(Complete)*
  - **Step 2.1**: Design Real-Time Audio Streaming Pipeline *(Complete)*
  - **Step 2.2**: Backend Audio Protocol Extension *(Complete)*
  - **Step 2.3**: Browser Microphone Capture Component *(Complete)*
  - **Step 2.4**: Integrate Browser Audio with Main Streamlit Call Controls *(Complete)*
  - **Step 2.5**: End-to-End Audio Pipeline Validation *(Complete)*
- **Phase 3**: Speech-to-Text Transcription Engine *(Next)*
- **Phase 4**: LLM-Powered Sales Coaching Engine & Prompt Engineering
- **Phase 5**: Knowledge Base Integration with Retrieval-Augmented Generation (RAG)
- **Phase 6**: Objection Intent Classification with scikit-learn
- **Phase 7**: End-to-End Integration & Real-Time Performance Tuning

