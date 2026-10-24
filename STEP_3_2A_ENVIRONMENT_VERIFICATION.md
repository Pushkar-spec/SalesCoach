# Step 3.2A — Final Source-Level Verification & Integration Diagnostics

> **Document Status**: Final Pre-Implementation Verification Report  
> **Phase**: Phase 3.2A — WhisperFlow Source-Level Integration Verification  
> **Target Package**: `whisperflow==1.3.0` (Source distribution & Wheel unpacked from PyPI)  
> **Host Environment**: Windows 11 Home (64-bit), Python 3.12.10, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM, Driver 555.99, CUDA 12.5 capable).

---

## 1. Executive Summary & Recommendation

### Recommendation: **GO (via Targeted Dependency Strategy)**

This report represents a source-code-level inspection of the exact `whisperflow-1.3.0-py3-none-any.whl` distribution downloaded from PyPI and unpacked in the environment scratchpad.

* **API Verification**: **CONFIRMED WITH REVISIONS**. The classes `TranscribeSession` and `ModelPool`, and methods `add_chunk()`, `flush()`, `stop()`, `get_model()`, and `transcribe_pcm_chunks_async()` exist in `whisperflow==1.3.0`.
  - **Critical Signature Finding**: `flush()` and `stop()` are **`async def`** methods. `await session.flush()` internally creates a `FlushRequest` and awaits its completion, ensuring all buffered audio is transcribed before resolving.
  - **Critical Model-Loading Finding**: `get_model(file_name)` hardcodes `MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")` and checks `os.path.isfile(...)`. In the 1.3.0 wheel, **only `tiny.en.pt` is bundled**. Calling `get_model("base.en")` raises `ValueError: unknown model: base.en` out-of-the-box unless `base.en.pt` is placed in that directory or loaded directly via `whisper.load_model("base.en")`.
* **Packaging & Dependency Finding**: The `whisperflow==1.3.0` wheel metadata hard-pins `fastapi==0.108.0`, `starlette<0.33.0`, and `PyAudio==0.2.14`. Our project uses `fastapi==0.158.1` and `starlette==1.9.0`. A standard `pip install whisperflow` would force a major downgrade of FastAPI.
* **Audio Format & FFmpeg Finding**: The server-side streaming path (`transcribe_pcm_chunks`) converts raw PCM16 bytes directly into a NumPy `float32` array in memory and passes it to `model.transcribe(arr)`. **FFmpeg is never invoked during live streaming** because audio is decoded in-memory rather than read from audio files.
* **Hardware & CUDA**: The host has an RTX 4060 GPU with Driver 555.99. PyTorch is not yet installed in `venv`. To utilize the GPU, PyTorch must be installed via the official CUDA wheel index (`cu124` or `cu121`), as standard PyPI defaults to CPU-only on Windows.

---

## 2. Source-Level Inspection of `whisperflow==1.3.0`

The exact wheel `whisperflow-1.3.0-py3-none-any.whl` (69.9 MB) was downloaded and extracted to `scratch/whisperflow_unpacked/`. Below are the verified source implementations:

### 2.1 `TranscribeSession` (`whisperflow/streaming.py:284-310`)

```python
class TranscribeSession:  # pylint: disable=too-few-public-methods
    """transcription state"""

    def __init__(self, transcribe_async, send_back_async) -> None:
        """ctor"""
        self.id = uuid.uuid4()  # pylint: disable=invalid-name
        self.queue = Queue()
        self.should_stop = [False]
        self.task = asyncio.create_task(
            transcribe(self.should_stop, self.queue, transcribe_async, send_back_async)
        )

    def add_chunk(self, chunk: bytes):
        """add new chunk"""
        self.queue.put_nowait(chunk)

    async def flush(self):
        """finalize all audio queued so far and send it as a final result"""
        request = FlushRequest()
        self.queue.put_nowait(request)
        await asyncio.wait({request.done, self.task}, return_when=FIRST_COMPLETED)

    async def stop(self):
        """stop session"""
        self.should_stop[0] = True
        await self.task
```

#### Verified Behavioral Facts:
1. **`__init__(self, transcribe_async, send_back_async)`**:
   - Accepts exactly two callbacks: `transcribe_async` and `send_back_async`.
   - Automatically launches an internal background task (`asyncio.create_task`) running the `transcribe()` loop.
2. **`add_chunk(self, chunk: bytes)`**:
   - Calls `self.queue.put_nowait(chunk)`.
   - **Non-blocking and synchronous**. It does **not** run model inference synchronously.
3. **`async def flush(self)`**:
   - Enqueues a `FlushRequest` marker containing an asyncio future (`item.done`).
   - Awaits `asyncio.wait({request.done, self.task}, return_when=FIRST_COMPLETED)`.
   - **Guarantees that trailing audio in the buffer is transcribed and emitted** before resolving.
4. **`async def stop(self)`**:
   - Flags `self.should_stop[0] = True` and awaits `self.task` to ensure the background task exits cleanly.

### 2.2 Callback Schema & Data Delivery (`whisperflow/streaming.py:204-210, 253-270`)

When partial or committed results are ready, `send_back_async(result)` is invoked with the following dictionary structure:

```python
{
    "is_partial": bool,  # True for tentative interim updates; False for committed/final text
    "data": {
        "text": str,     # Committed prefix + current window transcription
        "segments": list # Segment details (id, start, end, text, words)
    },
    "time": float        # Inference duration in milliseconds
}
```

* **Note for Step 3.2B Integration**: The recognized text is located at `result["data"]["text"]`, not `result["text"]`.

### 2.3 `ModelPool` (`whisperflow/transcriber.py:33-56`)

```python
class ModelPool:  # pylint: disable=too-few-public-methods
    """replicas of one model; each replica runs one transcription at a time

    A Whisper model is not safe to use from two threads at once: decoding installs
    kv-cache hooks on the model's layers, so concurrent calls corrupt each other's
    output or crash. Sessions share the pool and wait for a free replica.
    """

    def __init__(self, model: Whisper, replicas: int = 1) -> None:
        """ctor: the model plus replicas - 1 deep copies"""
        self.replicas = [model] + [copy.deepcopy(model) for _ in range(replicas - 1)]
        self.free = queue.Queue()
        for replica in self.replicas:
            self.free.put(replica)

    @contextmanager
    def acquire(self) -> Iterator[Whisper]:
        """borrow a replica, blocking until one is free"""
        replica = self.free.get()
        try:
            yield replica
        finally:
            self.free.put(replica)
```

#### Verified Behavioral Facts:
* OpenAI Whisper's decoder installs KV-cache hooks on model layers, making raw concurrent calls unsafe.
* `ModelPool` resolves this by managing model replicas (default `replicas=1` via `config.MODEL_REPLICAS`) using a thread-safe `queue.Queue`.
* Calls to `transcribe_pcm_chunks` borrow a model via `with model.acquire():`.

### 2.4 Model Resolution and Loading (`whisperflow/transcriber.py:74-95`)

```python
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def resolve_model_path(file_name: str) -> str:
    """validate a model name and return its path inside the models dir"""
    path = os.path.normpath(os.path.join(MODELS_DIR, file_name))
    if os.path.dirname(path) != MODELS_DIR:
        raise ValueError(f"invalid model name: {file_name}")
    if not os.path.isfile(path):
        raise ValueError(f"unknown model: {file_name}")
    return path

def get_model(file_name: Optional[str] = None) -> ModelPool:
    """load a model from disk into a pool of WF_MODEL_REPLICAS replicas (cached)"""
    name = file_name or config.DEFAULT_MODEL
    if name not in models:
        path = resolve_model_path(name)
        with _models_lock:
            if name not in models:
                model = whisper.load_model(path).to(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                models[name] = ModelPool(model, max(1, config.MODEL_REPLICAS))
    return models[name]
```

#### Critical Finding on Model Availability (`base.en`):
1. `resolve_model_path()` strictly checks `if not os.path.isfile(path): raise ValueError(...)`.
2. Inspecting `scratch/whisperflow_unpacked/whisperflow/models/` reveals:
   - **`tiny.en.pt`** is the **only** model bundled inside the wheel package (~69 MB).
3. Therefore:
   - `get_model("tiny.en.pt")` or `get_model()` **works out of the box**.
   - `get_model("base.en")` or `get_model("base.en.pt")` will **raise `ValueError: unknown model: base.en`** unless `base.en.pt` is downloaded into that directory, OR `whisper.load_model("base.en")` is called directly by our application.

### 2.5 In-Memory Audio Conversion (FFmpeg Independence)

Lines 120–131 of `whisperflow/transcriber.py`:

```python
arr = (
    np.frombuffer(b"".join(chunks), np.int16).flatten().astype(np.float32) / 32768.0
)
return model.transcribe(
    arr,
    fp16=False,
    language=lang,
    logprob_threshold=log_prob,
    temperature=temperature,
    initial_prompt=prompt,
    word_timestamps=word_timestamps,
)
```

#### Verified Fact:
* WhisperFlow converts binary PCM16 audio chunks directly into a NumPy `float32` array in-memory (`np.frombuffer(...)`).
* It passes the NumPy array directly to `model.transcribe(arr)`.
* In `openai-whisper`, `ffmpeg` is only invoked when `model.transcribe()` is passed a file path string to load audio from disk. Passing a NumPy array bypasses `whisper.load_audio()` entirely.
* **Result**: **FFmpeg is not required for live streaming transcription in our pipeline.**

---

## 3. Package Dependency Pin Comparison

Inspection of `scratch/whisperflow_unpacked/whisperflow-1.3.0.dist-info/METADATA`:

```text
Requires-Dist: fastapi==0.108.0
Requires-Dist: uvicorn[standard]==0.52.2
Requires-Dist: python-multipart==0.0.32
Requires-Dist: openai-whisper==20250625
Requires-Dist: torch
Requires-Dist: numpy
Requires-Dist: PyAudio==0.2.14
```

### Side-by-Side Comparison

| Dependency | `whisperflow==1.3.0` Specification | Installed in Project (`venv`) | Impact / Conflict Analysis |
| :--- | :--- | :--- | :--- |
| **`fastapi`** | `==0.108.0` (Hard Pin) | `0.158.1` | **Severe Conflict**. Installing without restriction attempts to downgrade FastAPI by 50 minor releases. |
| **`starlette`** | `<0.33.0,>=0.29.0` (via fastapi 0.108) | `1.9.0` | **Severe Conflict**. Would downgrade Starlette to 0.32.0. |
| **`uvicorn[standard]`** | `==0.52.2` | `0.54.0` | Minor downgrade conflict. |
| **`python-multipart`** | `==0.0.32` | `0.0.32` | Compatible (Exact match). |
| **`PyAudio`** | `==0.2.14` | *Not installed* | **Not needed**. Only used by WhisperFlow's desktop CLI (`client.py`). Our client runs in the browser. |
| **`openai-whisper`** | `==20250625` | *Not installed* | Required. Must be installed. |
| **`torch`** | (Unpinned) | *Not installed* | Required. Must be installed with CUDA index. |
| **`numpy`** | (Unpinned) | `2.5.4` | Compatible. |

---

## 4. Minimum Packages Required for Server-Side Transcription

1. **`torch` (with CUDA 12.4/12.1 runtime)**: Required for model execution on the RTX 4060 GPU.
2. **`openai-whisper`**: Required for Whisper model architectures, tokenizer, and mel-spectrogram computations.
3. **`whisperflow` streaming logic**: `TranscribeSession`, `Segment`, and `ModelPool`.
4. **Packages NOT needed**:
   - `PyAudio`: Not needed (browser microphone capture handles recording).
   - `torchvision`: Not needed (no image processing).
   - `torchaudio`: Not needed (Whisper uses NumPy and PyTorch tensors directly).
   - `ffmpeg`: Not strictly required on PATH for PCM16 NumPy array inference.

---

## 5. Evaluation of Installation Strategies

We evaluated three potential strategies for integrating WhisperFlow 1.3.0:

### Option 1: Standard `pip install whisperflow==1.3.0`
* **Mechanism**: Standard pip install.
* **Tradeoffs**:
  - *Major Con*: Forces `fastapi==0.108.0` and `starlette==0.32.0`, which breaks our current modern FastAPI 0.158 setup and risks regressions across our 99 existing tests.
  - *Major Con*: Attempts to install `PyAudio`, which is unneeded on the server.
* **Verdict**: **Rejected** due to breaking dependency conflicts.

### Option 2: `pip install whisperflow==1.3.0 --no-deps` with Explicit Dependencies
* **Mechanism**:
  1. `pip install torch --index-url https://download.pytorch.org/whl/cu124`
  2. `pip install openai-whisper`
  3. `pip install whisperflow==1.3.0 --no-deps`
* **Tradeoffs**:
  - *Pros*: Uses the official wheel package directly from PyPI; preserves `fastapi==0.158.1`; avoids `PyAudio`.
  - *Cons*: `pip check` or dependency scanning tools will emit a warning that `fastapi==0.108.0` requirement is unsatisfied.
  - *Cons*: `base.en` cannot be loaded through `get_model("base.en")` without physically writing the model file into site-packages.
* **Verdict**: **Viable**, but requires documented exception for `pip check`.

### Option 3: Clean In-Tree Service Adapter (`stt_service.py` adopting WhisperFlow Engine Logic)
* **Mechanism**:
  1. Install `torch` (CUDA) and `openai-whisper` via pip.
  2. In `stt_service.py`, directly adopt WhisperFlow's verified core logic (`TranscribeSession`, `Segment`, `ModelPool`).
* **Tradeoffs**:
  - *Pros*: Zero packaging conflicts, 100% clean dependency tree, zero `pip check` warnings, zero unneeded `PyAudio`.
  - *Pros*: Fixes the `base.en` model-loading limitation: allows loading `base.en` natively via `whisper.load_model("base.en")` without hardcoded path restrictions.
  - *Cons*: Requires maintaining ~200 lines of streaming window logic directly in `stt_service.py` rather than importing an external wheel.
* **Verdict**: **Recommended (Least Disruptive & Cleanest)**.

---

## 6. Exact Host Environment Diagnostics

* **Python Executable**: `C:\Users\pushk\Desktop\SalesCoach\venv\Scripts\python.exe`
* **Python Version**: `3.12.10 (tags/v3.12.10:0cc8128, Apr 8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]`
* **GPU**: `NVIDIA GeForce RTX 4060 Laptop GPU` (8,188 MiB VRAM)
* **Driver Version**: `555.99` (Supports CUDA 12.5)
* **Current Torch Status**: Not installed.
* **FFmpeg**: Not on PATH (not required for in-memory PCM16 array transcription).

---

## 7. Concrete Next Steps for Step 3.2B

Upon approval to proceed to Step 3.2B:
1. Install CUDA-enabled PyTorch:
   ```powershell
   .\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
2. Install OpenAI Whisper:
   ```powershell
   .\venv\Scripts\python.exe -m pip install openai-whisper
   ```
3. Create `stt_service.py` encapsulating WhisperFlow's tumbling window (`Segment`), `TranscribeSession`, and `ModelPool`, with direct support for `tiny.en` and `base.en`.
4. Create diagnostic script `tests/test_stt_hardware.py` verifying CUDA acceleration on RTX 4060.
5. Create deterministic unit tests `tests/test_stt_service.py` verifying session lifecycle, chunk ingestion, and flush.

---

*End of Step 3.2A Final Verification Report.*
