# Real-Time AI Sales Copilot

A real-time AI assistant designed to listen to sales conversations, provide live objection handling, suggest contextual talking points, and deliver instant deal-coaching feedback.

---

## Current Status: Step 1.5 Complete (Streamlit-FastAPI WebSocket Integration)

- [x] **FastAPI Application**: Running server with `GET /`, `GET /health`, and WebSocket `ws://127.0.0.1:8000/ws`.
- [x] **Bidirectional WebSocket Protocol**: Full-duplex JSON ping/pong protocol with structured validation and error handling.
- [x] **Streamlit WebSocket Integration**: Live client connection (`ws_client.py`) connecting the Streamlit dashboard to the FastAPI backend with ping/pong handshake, connection status indicator, and graceful disconnection.
- [x] **Automated Testing**: 24 deterministic `pytest` tests covering HTTP endpoints, backend WebSocket protocols, frontend client lifecycle, and Streamlit UI state transitions.
- [x] **Virtual Environment**: Python 3.12 virtual environment established (`venv`).
- [x] **Environment Config**: `.gitignore` and `.env.example` templates created.
- [x] **Version Control**: Git repository initialized with appropriate ignore rules.
- [ ] **Speech-to-Text & Audio Pipeline**: *Planned for future phase*
- [ ] **LLM Coaching & Objection Handling**: *Planned for future phase*
- [ ] **Retrieval-Augmented Generation (RAG)**: *Planned for future phase*
- [ ] **ML Objection Classifier**: *Planned for future phase*

> **Important Architecture Note**: In this step (Step 1.5), the Streamlit frontend actively connects to the FastAPI backend over WebSockets and performs an initial handshake. Live microphone streaming and AI coaching inference remain decoupled and will be introduced in subsequent phases.

---

## Planned Technology Stack

| Layer | Component / Technology | Status |
| :--- | :--- | :--- |
| **Backend API** | FastAPI + Uvicorn | Implemented (`/` and `/health` routes) |
| **Real-time Streaming** | WebSockets (Bidirectional) | Implemented (`/ws` ping/pong protocol) |
| **Frontend UI** | Streamlit | Implemented (Dashboard linked to FastAPI via WebSockets) |
| **Speech Processing** | Speech-to-Text engine | Planned (Phase 3) |
| **Intelligence** | Large Language Model (Prompt-engineered sales coach) | Planned (Phase 4) |
| **Knowledge Retrieval**| RAG (Sales playbook & product documentation) | Planned (Phase 5) |
| **Classification** | scikit-learn (Objection intent classification) | Planned (Phase 6) |

---

## Prerequisites

- **OS**: Windows (PowerShell recommended)
- **Python**: Version 3.10+ (Current environment uses Python 3.12)
- **Git**: Installed and available in PATH

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

Install the locked foundation dependencies into your activated virtual environment:

```powershell
pip install -r requirements.txt
```

---

## Backend Endpoints & Protocols

### 1. HTTP Root Endpoint (`GET /`)
- **Purpose**: Confirms that the ASGI web server is mounted, routing is active, and base HTTP traffic is served.
- **Response**: `{"message": "Hello World"}`

### 2. HTTP Health Check (`GET /health`)
- **Purpose**: Verifies that the FastAPI process is responsive, returning service status and environment.
- **Response**:
  ```json
  {
    "status": "ok",
    "service": "sales-copilot-api",
    "environment": "development"
  }
  ```

### 3. WebSocket Endpoint (`ws://127.0.0.1:8000/ws`)
- **Purpose**: Full-duplex communication channel for real-time streaming between Streamlit and FastAPI.
- **Protocol Definition (JSON)**:
  - **Client Ping Request**:
    ```json
    {
      "type": "ping",
      "message": "Hello from Streamlit"
    }
    ```
  - **Server Pong Response**:
    ```json
    {
      "type": "pong",
      "message": "Hello from Streamlit"
    }
    ```
  - **Server Error Response**:
    ```json
    {
      "type": "error",
      "message": "Unsupported message type"
    }
    ```

---

## Frontend WebSocket Connection Lifecycle

The Streamlit dashboard interacts with the WebSocket backend through [ws_client.py](file:///c:/Users/pushk/Desktop/SalesCoach/ws_client.py):

1. **Start Call**:
   - Opens a synchronous WebSocket connection to `BACKEND_WS_URL`.
   - Sends the `{"type": "ping", "message": "Hello from Streamlit"}` handshake.
   - Awaits and validates the `pong` response within a 3-second timeout.
   - On success: updates connection indicator to `🟢 Connected` and session status to `Active`.
   - On failure: catches the error, updates indicator to `🔴 Connection Error`, and leaves session as `Idle`.
2. **Stop Call**:
   - Sends a close frame, closes the socket, and clears the socket reference.
   - Updates indicator to `⚪ Disconnected` and session status to `Stopped`.
3. **Reset Session**:
   - Closes any active socket, resets all transcript and suggestion collections, and restores the initial `Idle` state.

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

Click **▶ Start Call** to establish the live WebSocket handshake with the running FastAPI backend.

---

## Running Automated Tests

Run the test suite using `pytest` inside your activated virtual environment:

```powershell
python -m pytest -q
```

The test suite contains **24 automated tests** across 4 modules:
- [tests/test_health.py](file:///c:/Users/pushk/Desktop/SalesCoach/tests/test_health.py): Validates HTTP status codes and payloads for `/` and `/health`.
- [tests/test_websocket.py](file:///c:/Users/pushk/Desktop/SalesCoach/tests/test_websocket.py): Validates backend WebSocket handshake, ping/pong echo, multi-message exchange, invalid JSON resilience, and client disconnection.
- [tests/test_ws_client.py](file:///c:/Users/pushk/Desktop/SalesCoach/tests/test_ws_client.py): Validates frontend client helper URL configuration, handshake validation, timeouts, error recovery, and idempotent closure.
- [tests/test_app.py](file:///c:/Users/pushk/Desktop/SalesCoach/tests/test_app.py): Validates Streamlit UI button clicks (`Start Call`, `Stop Call`, `Reset Session`), state changes, error handling, and socket cleanup via Streamlit's `AppTest` framework.

---

## Project Structure

```text
SalesCoach/
├── .env.example        # Environment variable template with safe placeholders
├── .gitignore           # Git ignore rules for venv, secrets, caches, and logs
├── app.py              # Streamlit frontend dashboard (Step 1.2 & Step 1.5)
├── main.py             # FastAPI entry point ('/', '/health', and '/ws')
├── README.md           # Project documentation and developer guide
├── requirements.txt    # Project dependencies
├── ws_client.py        # Synchronous WebSocket client helper (Step 1.5)
├── tests/              # Automated test suite (24 tests)
│   ├── test_app.py        # Streamlit AppTest UI state and button callback tests
│   ├── test_health.py     # HTTP tests for root and health routes
│   ├── test_websocket.py  # Backend WebSocket protocol and error resilience tests
│   └── test_ws_client.py  # Frontend WebSocketClient lifecycle and validation tests
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
- **Step 1.5**: Connect Streamlit Frontend to FastAPI WebSocket Backend *(Current - Complete)*
- **Phase 2**: Real-Time Audio Streaming Pipeline (Audio Capture & Streaming over WebSockets)
- **Phase 3**: Speech-to-Text Transcription Engine
- **Phase 4**: LLM-Powered Sales Coaching Engine & Prompt Engineering
- **Phase 5**: Knowledge Base Integration with Retrieval-Augmented Generation (RAG)
- **Phase 6**: Objection Intent Classification with scikit-learn
- **Phase 7**: End-to-End Integration & Real-Time Performance Tuning
