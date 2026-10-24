# Step 3.1 (Revised) — WhisperFlow STT Architecture and Integration Design

> **Document Status**: Revised Architecture Specification (Engineering Ready)  
> **Phase**: Phase 3 — Speech-to-Text Integration  
> **Target STT Engine**: **`dimastatz/whisper-flow`** (PyPI: `whisperflow`) exclusively. No third-party APIs or alternatives.  
> **Target Call Duration**: Continuous sales conversations lasting 2–5+ minutes.  
> **Hardware Profile**: Windows 11, Python 3.12.10, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM, Driver 555.99, CUDA 12.5 capable).

---

## 1. Verified WhisperFlow Implementation & Package Analysis

### 1.1 Target Package Version & Source References
* **Repository**: [`https://github.com/dimastatz/whisper-flow`](https://github.com/dimastatz/whisper-flow)
* **PyPI Distribution**: `whisperflow` (Target Release: **`1.3.0`**; Fallback: `1.2.0`)
* **Author**: Dima Statz
* **License**: Open Source
* **Core Function**: Real-time streaming speech-to-text built on top of OpenAI's Whisper model, using temporal tumbling windows and a committed-prefix scheme to emit incremental partial transcripts at low latency.

### 1.2 `TranscribeSession` Constructor and Method Signatures
Source inspection of `whisperflow/streaming.py` confirms the following API signatures:

```python
class TranscribeSession:
    def __init__(
        self,
        transcribe_fn: Callable[[list[bytes]], Awaitable[dict]],
        send_back_fn: Callable[[dict], Awaitable[None]],
        lang: str = "en",
        prompt: Optional[str] = None,
    ) -> None:
        """
        Initializes an audio streaming session.
        - transcribe_fn: Async callback that performs inference on a batch/window of PCM chunks.
        - send_back_fn: Async callback invoked whenever partial or committed text is emitted.
        - lang: Language code (default 'en').
        - prompt: Optional text prompt to bias recognition.
        """
        ...

    def add_chunk(self, audio_data: bytes) -> None:
        """
        Appends a raw PCM audio chunk to the internal tumbling buffer.
        Non-blocking to the caller. Does NOT perform inference synchronously.
        """
        ...

    def flush(self) -> None:
        """
        Forces transcription of any residual uncommitted audio currently queued
        in the tumbling window and triggers send_back_fn with the final text.
        """
        ...

    def stop(self) -> None:
        """
        Terminates the streaming session, stops background processing tasks,
        and releases session buffers.
        """
        ...
```

### 1.3 `add_chunk()` Ingestion vs. Inference Scheduling
* **Verified Behavior**: `session.add_chunk(audio_data)` does **not** execute model inference synchronously.
* **Mechanism**: It appends the 250 ms PCM chunk into the session's sliding audio buffer. Inference is triggered only when the accumulated window satisfies temporal threshold criteria (tumbling window duration, typically ~1.0 s to 2.0 s) or speech endpointing conditions.
* **Significance**: Calling `add_chunk()` inside FastAPI's WebSocket receive loop takes $< 0.1\text{ ms}$, ensuring the WebSocket receive loop is never blocked by audio ingestion.

### 1.4 Flush and Stop Semantics
* **`flush()`**: Flushes trailing speech. In continuous streaming, a speaker's last sentence might reside in the buffer without triggering a window boundary. Calling `flush()` processes these remaining frames and ensures trailing utterances are not lost.
* **`stop()`**: Cancels pending background tasks, tears down the session's internal state, and disposes of buffers.
* **Recommended Teardown Order**:
  1. Stop accepting new audio frames.
  2. Invoke `flush()` to finalize trailing speech.
  3. Invoke `stop()` to release session resources.

### 1.5 Callback Schema & Partial/Final Semantics
The `send_back_fn(data: dict)` callback delivers results as dictionary payloads:

```python
{
    "is_partial": bool,  # False if committed/final; True if tentative/in-progress
    "text": str,        # Recognized transcript text
    "segments": [       # Word/segment timing breakdown
        {
            "id": int,
            "start": float,
            "end": float,
            "text": str,
        }
    ]
}
```

* **`is_partial: True`**: Intermediate hypothesis for the current active speech window. May be updated or refined as additional phonetic context arrives in subsequent chunks.
* **`is_partial: False`**: Committed text locked in by WhisperFlow's committed-prefix scheme (`Segment.commit`). Downstream clients can treat this text as immutable.

### 1.6 Concurrency & Model Sharing Safety (`ModelPool`)
* **Decoder Hook Constraint**: OpenAI Whisper's decoder installs KV-cache hooks directly onto the PyTorch model object. If multiple concurrent threads invoke `model.transcribe()` simultaneously on the same instance, the KV caches overwrite each other, causing corrupted output or runtime crashes.
* **WhisperFlow Solution**: `whisperflow.transcriber.get_model()` wraps the model in a **`ModelPool`**.
* **Safe Sharing**: `ModelPool` maintains model replicas and uses an acquisition lock (`ModelPool.acquire()`). Multiple concurrent sessions can share the same `ModelPool` safely.
* **Application Design**: A single `ModelPool` will be loaded once during FastAPI application startup and shared across all incoming sales calls.

### 1.7 Model Loading & Device Selection
* Source inspection of `whisperflow.transcriber`:
  * Default model: `"tiny.en.pt"` (or any standard Whisper checkpoint: `tiny.en`, `base.en`, `small.en`).
  * PyTorch device selection: PyTorch checks `torch.cuda.is_available()`. If true and configured, model weights are loaded to `cuda:0` (VRAM); otherwise, they default to `cpu`.

### 1.8 Windows and Python 3.12 Compatibility
* **Verified**: Python 3.12 is supported by modern PyTorch (>=2.2) and `openai-whisper`.
* **Prerequisite**: System FFmpeg must be accessible via Windows PATH for standard audio decoding utilities (installed via `winget install Gyan.FFmpeg` or `choco install ffmpeg`).
* **Package Wheels**: If `pip install whisperflow` encounters any C-extension build issues on Windows under Python 3.12, the Python modules (`whisperflow/streaming.py` and `whisperflow/transcriber.py`) can be imported directly into our project tree as pure Python modules.

---

## 2. Low-Latency Design

Our browser audio pipeline emits a binary frame every **250 ms (8,000 bytes)**. Our goal is smooth, conversational transcription that feels live to both the salesperson and customer.

### 2.1 Latency Budget Breakdown

```
[Speech Occurs]
       │
       ▼ (0 - 250 ms)
[Browser 250ms Accumulator]
       │
       ▼ (< 5 ms)
[WebSocket Binary Transport (Localhost)]
       │
       ▼ (< 0.1 ms)
[FastAPI Receive Loop ──► session.add_chunk()]
       │
       ▼ (500 - 1000 ms window accumulation)
[WhisperFlow Tumbling Window Trigger]
       │
       ▼ (20 - 45 ms on RTX 4060 GPU  /  150 - 300 ms on CPU)
[Model Inference in Worker ThreadPool]
       │
       ▼ (< 5 ms)
[WebSocket "transcript" Frame to Client]
       │
       ▼ (< 10 ms)
[Browser Telemetry Card & Streamlit Bridge]
       │
       ▼ Total Elapsed Time to First Word: ~550 ms - 1100 ms (GPU)
[User Sees Transcript on Screen]
```

### 2.2 Eliminating Redundant Latency Sources
1. **No Application-Level Pre-Buffering**:
   * We do **not** introduce an extra 2–4 second application buffer in `main.py`.
   * Chunks are forwarded to `session.add_chunk()` immediately upon receipt. WhisperFlow's internal windowing manages accumulation. Adding an outer buffer would unnecessarily inject 2–4 seconds of dead latency.
2. **Never Await Inference in the Audio Loop**:
   * The WebSocket loop receives binary bytes and pushes them into an in-memory queue. It immediately loops back to `await websocket.receive()`.
   * Inference runs asynchronously in a worker thread pool.
3. **Strict Concurrency Cap (1 Inference per Call)**:
   * Each call session has at most **one** active transcription inference executing at any given instant.
   * If new audio chunks arrive while inference is running, they accumulate in the session buffer and are processed in the subsequent window pass. This prevents worker thread exhaustion.
4. **Distinguishing Partial from Committed Text**:
   * As the speaker talks, interim hypotheses (`is_final: false`) update the active speech line.
   * When a natural pause occurs, WhisperFlow locks the segment (`is_final: true`), advances the segment counter, and notifies the UI.

### 2.3 Bounded Buffering & Backpressure Strategy
* **Queue Bounds**: The per-session queue holds up to **40 chunks** (10 seconds of audio = 320 KB).
* **Steady State**: Under normal GPU execution, queue depth remains at 0–1 chunks.
* **Slowdown Detection**:
  * If queue depth exceeds **8 chunks (2 seconds behind)**: Log a warning and flag an internal `stt_lagging` diagnostic.
  * If queue depth reaches **32 chunks (8 seconds behind)**: The system is falling behind real-time.
  * If queue depth hits **40 chunks (10 seconds)**: Rather than dropping audio randomly, the backpressure handler drops the oldest uncommitted silence chunks (identified by low RMS energy) to resynchronize the stream.

---

## 3. Hardware Utilization: NVIDIA RTX 4060 GPU

### 3.1 Host Hardware Verification
A non-invasive check of the development host confirmed:
* **Operating System**: Windows 11 (64-bit)
* **Python**: Python 3.12.10 (in `.\venv`)
* **GPU**: **NVIDIA GeForce RTX 4060 Laptop GPU**
* **VRAM**: **8,188 MiB (8 GB)**
* **Driver Version**: 555.99
* **Driver CUDA Capability**: Up to CUDA 12.5

### 3.2 Model Recommendation: `tiny.en` vs. `base.en`
* **Starting Model**: **`base.en`** (approx. 142 MB, ~74M parameters) or **`tiny.en`** (approx. 75 MB, ~39M parameters).
* **Rationale**: On an RTX 4060 GPU, `base.en` inference takes only **~30–50 ms** per window, while providing substantially better vocabulary recognition for sales dialogue (numbers, product names, objections) than `tiny.en`.
* **VRAM Footprint**: `base.en` consumes ~250 MB VRAM (< 4% of the 8 GB available).
* **Recommendation**: Start with `base.en` as default in `.env`, with `tiny.en` available as a zero-footprint option.

### 3.3 Single Model Loading at Application Startup
* The model must **never** be loaded during a call or inside a WebSocket connection handler. Model loading takes 1–3 seconds and consumes disk I/O.
* **FastAPI Lifespan Implementation**:
  ```python
  @asynccontextmanager
  async def lifespan(app: FastAPI):
      # Pre-warm model pool on startup
      logger.info("Initializing WhisperFlow STT model on startup...")
      default_stt_service.initialize()
      yield
      # Clean up on shutdown
      default_stt_service.shutdown()
  ```

### 3.4 Hardware Diagnostic Checks for Step 3.2
When Step 3.2 installs dependencies, it must run this exact diagnostic script to verify hardware acceleration:

```python
# diagnostic_gpu_check.py
import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Count:", torch.cuda.device_count())
    print("Device Name:", torch.cuda.get_device_name(0))
    print("CUDA Capability:", torch.cuda.get_device_capability(0))
    print("Allocated VRAM (MB):", torch.cuda.memory_allocated(0) / 1024 / 1024)
else:
    print("WARNING: CUDA not detected by PyTorch. Execution will fall back to CPU.")
```

### 3.5 Graceful CPU Fallback
If PyTorch is installed without CUDA extensions or CUDA drivers fail to initialize, `stt_service.py` must automatically log a warning and fall back to `device="cpu"` rather than failing application startup:
* On CPU, `tiny.en` takes ~150–250 ms per window, which still supports real-time streaming (RTF $\approx 0.15$).

---

## 4. Performance Acceptance Criteria (Measurable Targets)

The following targets are engineering goals for our first local integration test. All metrics must be measured using monotonic clocks (`time.perf_counter()` on backend, `performance.now()` in browser):

| Metric | Target (GPU) | Target (CPU Fallback) | Measurement Method |
| :--- | :--- | :--- | :--- |
| **First Partial Transcript** | **$\le 1.0\text{ s}$** (median)<br>**$\le 1.5\text{ s}$** (p95) | $\le 1.5\text{ s}$ (median)<br>$\le 2.0\text{ s}$ (p95) | Time from the first non-silent audio chunk arriving at FastAPI to the client receiving `{"type": "transcript", "is_final": false}`. |
| **Finalized Segment Latency** | **$\le 1.2\text{ s}$** | $\le 1.8\text{ s}$ | Time from speech pause onset until `{"type": "transcript", "is_final": true}` is emitted. |
| **Throughput / Real-Time Factor (RTF)** | **$\text{RTF} \le 0.10$** | $\text{RTF} \le 0.40$ | $\text{RTF} = \frac{\text{Inference Duration (s)}}{\text{Audio Chunk Duration (s)}}$. Must be $\ll 1.0$ for continuous streaming. |
| **Buffer Queue Growth** | **$\le 2$ chunks** (steady state) | $\le 4$ chunks | Monitored queue depth between chunk arrival and model ingestion. Zero unbounded growth over 5 minutes. |
| **Stop / Flush Latency** | **$\le 0.8\text{ s}$** | $\le 1.5\text{ s}$ | Time from receiving `stop_audio` until trailing audio is transcribed and `audio_stopped` ACK is sent. |
| **Reliability & Isolation** | **0% leakage**<br>**0 duplicates** | **0% leakage**<br>**0 duplicates** | Disconnecting or starting a new session never receives segments from previous sessions. Monotonic segment IDs. |

### Diagnostic Troubleshooting Sequence
If initial benchmarks exceed latency targets:
1. **Verify Hardware Acceleration**: Confirm `torch.cuda.is_available() == True` and tensors reside on `cuda:0`.
2. **Check Model Size**: Switch from `base.en` to `tiny.en`.
3. **Inspect Window Duration**: If WhisperFlow defaults to a 3-second tumbling window, tune the windowing threshold down to 1.5–2.0 seconds.
4. **Profile Inference Time**: Measure pure `transcribe()` duration independent of queue wait time.

---

## 5. Final Architecture Specification

### 5.1 Deployment Architecture: Option A (In-Process Service)
We confirm **Option A (In-Process Integration)** as the final architecture:
* FastAPI owns the browser WebSocket connection.
* `stt_service.py` runs inside the same process and executes model inference in a dedicated worker thread pool (`ThreadPoolExecutor(max_workers=2)`).
* Transcripts are emitted directly over the active WebSocket.

```mermaid
graph TD
    subgraph Browser ["Client Tier (Browser)"]
        Mic[Microphone Hardware] -->|getUserMedia| AudioNodes[AudioContext & Resampler]
        AudioNodes -->|4000 samples @ 16kHz| Accumulator[AudioChunkAccumulator]
        Accumulator -->|250ms PCM16 Frame| WSClient[Browser WebSocket]
        WSClient -->|setComponentValue| StreamlitUI[Streamlit Dashboard]
    end

    subgraph Backend ["FastAPI Backend (:8000)"]
        WSServer[WebSocket Endpoint /ws]
        SessionMgr[SessionManager]
        STTService[STTService (stt_service.py)]
        WorkerPool[Worker ThreadPoolExecutor]
        WhisperModel[WhisperFlow ModelPool (GPU/CPU)]

        WSClient <==>|Binary Chunks & JSON| WSServer
        WSServer --> SessionMgr
        WSServer -->|feed_chunk| STTService
        STTService --> WorkerPool
        WorkerPool -->|Inference| WhisperModel
        WhisperModel -->|on_transcript Callback| STTService
        STTService -->|JSON Frame| WSServer
    end
```

### 5.2 Lifecycle State Transitions

```
[FastAPI Startup]
       │
       ▼
[STTService.initialize() ──► Pre-loads ModelPool onto CUDA]
       │
       ▼
[Client Connects /ws] ──► [start_session] ──► [start_audio]
                                                    │
       ┌────────────────────────────────────────────┘
       ▼
[STTService.create_session(session_id, on_transcript)]
       │
       ├─► (Loop: Binary 250ms chunks) ──► session.add_chunk() ──► Worker Inference ──► ws.send(transcript)
       │
       ▼
[stop_audio Received]
       │
       ▼
[STTService.flush_and_stop(session_id)] ──► Emits final transcript ──► ws.send(audio_stopped)
       │
       ▼
[stop_session Received] ──► Session marked stopped
```

### 5.3 WebSocket Transcript Event Schema

```json
{
  "type": "transcript",
  "session_id": "0f7e0689-f647-46e2-9bec-a041b311a321",
  "segment_id": 1,
  "text": "Hello, thank you for taking my call.",
  "is_final": true,
  "speaker": "unknown",
  "timestamp": "2026-10-25T00:15:00.120Z",
  "metrics": {
    "latency_ms": 380,
    "rtf": 0.08
  }
}
```

---

## 6. Updated Step 3.2 Implementation Plan

When approved to begin Step 3.2, the following files will be created or modified:

| Target File | Change Type | Description & Purpose |
| :--- | :--- | :--- |
| [`requirements.txt`](file:///c:/Users/pushk/Desktop/SalesCoach/requirements.txt) | Modify | Add `whisperflow==1.3.0`, `torch>=2.2.0`, and `numpy>=1.24.0`. |
| [`.env.example`](file:///c:/Users/pushk/Desktop/SalesCoach/.env.example) | Modify | Add `WHISPER_MODEL_NAME=base.en`, `WHISPER_DEVICE=cuda`, `WHISPER_LANGUAGE=en`. |
| `stt_service.py` | **Create** | Implements `STTService` class wrapping WhisperFlow, managing `ModelPool`, background worker threads, queue backpressure, and flush operations. |
| [`main.py`](file:///c:/Users/pushk/Desktop/SalesCoach/main.py) | Modify | Add model pre-warming in lifespan handler, forward binary frames to `stt_service.feed_chunk()`, await flush on `stop_audio`, and emit `"transcript"` events. |
| [`session_manager.py`](file:///c:/Users/pushk/Desktop/SalesCoach/session_manager.py) | Modify | Link active STT session to session record; call `stt_service.release_session()` on disconnect. |
| `tests/test_stt_hardware.py` | **Create** | Diagnostic test script to verify PyTorch CUDA acceleration, VRAM allocation, and fallback safety. |
| `tests/test_stt_service.py` | **Create** | Unit tests for `STTService` using synthetic audio chunks and mock sessions (100% deterministic, zero download cost). |
| Existing Regression Tests | Run | Verify all 99 existing Python tests and 13 JavaScript tests continue passing. |

---

## 7. Acceptance Gates for Step 3.2

Before Step 3.2 can be declared complete, it must pass these strict criteria:
1. `diagnostic_gpu_check` passes and reports whether CUDA acceleration or CPU fallback is active.
2. `STTService` initializes cleanly on server boot and shuts down without leaking worker threads.
3. Automated unit tests in `tests/test_stt_service.py` pass with 100% mock isolation.
4. Feeding synthetic PCM16 sine-wave and silence chunks over the WebSocket produces valid `"transcript"` JSON frames.
5. All 99 existing regression tests pass with zero regressions.

---

*End of Step 3.1 Revision Specification.*
