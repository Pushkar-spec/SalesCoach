"""
Real-Time Speech-to-Text (STT) Service for Sales Copilot.

Integrates WhisperFlow 1.3.0 streaming transcription architecture:
- ModelPool with thread-safe replica acquisition to isolate KV-cache hooks
- Temporal tumbling windows with silence endpointing
- Committed-prefix scheme for low-latency streaming partials
- Asynchronous executor offloading to prevent blocking the FastAPI event loop
- Explicit bounded queues and backpressure protection
- Per-session isolation and clean shutdown handling
"""

import asyncio
import copy
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
import logging
import os
import queue
import re
import threading
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
import whisper
from whisper import Whisper

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class STTError(Exception):
    """Base exception for STT service errors."""


class STTInitializationError(STTError):
    """Raised when model loading or hardware initialization fails."""


class STTStateError(STTError):
    """Raised on invalid session state or transition."""


class STTBackpressureError(STTError):
    """Raised when the session audio queue capacity is exceeded."""


class STTClosedError(STTError):
    """Raised when feeding audio to a closed or stopping session."""


# -----------------------------------------------------------------------------
# Audio & Text Helpers (WhisperFlow 1.3.0 streaming algorithms)
# -----------------------------------------------------------------------------
def is_silent(chunk: bytes, threshold: int = 500) -> bool:
    """True when all int16 samples in the chunk are below the silence threshold."""
    if len(chunk) < 2:
        return True
    samples = np.frombuffer(chunk[: len(chunk) // 2 * 2], dtype=np.int16).astype(np.int32)
    return not samples.size or int(np.max(np.abs(samples))) < threshold


def duration_ms(chunk: bytes, sample_rate: int = 16000) -> float:
    """Playback duration of an int16 mono chunk in milliseconds."""
    return len(chunk) / 2 / sample_rate * 1000.0


def normalize(text: str) -> str:
    """Lowercase and strip punctuation so cosmetic changes compare equal."""
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def words_of(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract word-level timestamp objects from Whisper result segments."""
    return [
        word
        for segment in (data.get("segments") or [])
        for word in (segment.get("words") or [])
    ]


def common_prefix(prev: List[Dict[str, Any]], curr: List[Dict[str, Any]]) -> int:
    """Number of leading words two hypotheses agree on, ignoring case/punctuation."""
    count = 0
    for old, new in zip(prev, curr):
        if normalize(old.get("word", "")) != normalize(new.get("word", "")):
            break
        count += 1
    return count


def merge_committed_and_tail(committed: str, tail: str) -> str:
    """
    Merge committed prefix and uncommitted tail, removing any overlapping words
    that the model re-emitted from the prompt or audio window boundary.
    """
    committed = committed.rstrip()
    tail = tail.lstrip()
    if not committed:
        return tail
    if not tail:
        return committed

    committed_words = committed.split()
    tail_words = tail.split()

    # Find maximum overlap between end of committed and start of tail (up to 15 words)
    max_overlap = min(len(committed_words), len(tail_words), 15)
    best_overlap = 0

    for k in range(max_overlap, 0, -1):
        c_slice = [normalize(w) for w in committed_words[-k:]]
        t_slice = [normalize(w) for w in tail_words[:k]]
        if c_slice == t_slice:
            best_overlap = k
            break

    if best_overlap > 0:
        tail_words_clean = tail_words[best_overlap:]
        tail_clean = " ".join(tail_words_clean)
        return f"{committed} {tail_clean}".strip()

    return f"{committed} {tail}".strip()


def same_text(result: Dict[str, Any], prev_result: Dict[str, Any]) -> bool:
    """True when two transcription results differ only in case/punctuation/spacing."""
    cur_text = result.get("data", {}).get("text", "")
    prev_text = prev_result.get("data", {}).get("text", "")
    return normalize(cur_text) == normalize(prev_text)


def should_close_segment(
    result: Dict[str, Any], prev_result: Dict[str, Any], cycles: int, max_cycles: int = 1
) -> bool:
    """Returns True if the segment text has stabilized across cycles and should be finalized."""
    return cycles >= max_cycles and same_text(result, prev_result)


# -----------------------------------------------------------------------------
# Streaming Segment Manager
# -----------------------------------------------------------------------------
CONTEXT_CHARS = 200


class Segment:
    """
    Manages the tumbling audio window and silence endpointing state for a stream.

    Supports committed-prefix streaming: words agreed upon by consecutive partials
    are committed to `self.committed` and their audio dropped from `self.window`.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold: int = 500,
        silence_ms: float = 1500.0,
        max_window_chunks: int = 1000,
        commit_prefix: bool = True,
        commit_after_ms: float = 4000.0,
        commit_margin_ms: float = 1000.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_ms = silence_ms
        self.max_window_chunks = max_window_chunks
        self.commit_prefix = commit_prefix
        self.commit_after_ms = commit_after_ms
        self.commit_margin_ms = commit_margin_ms

        self.window: List[bytes] = []
        self.prev_result: Dict[str, Any] = {}
        self.cycles: int = 0
        self.has_speech: bool = False
        self.trailing_silent_ms: float = 0.0
        self.committed: str = ""
        self.prev_words: List[Dict[str, Any]] = []

    def add(self, chunk: bytes) -> None:
        """Append a PCM16 chunk and track silence / speech presence."""
        self.window.append(chunk)
        chunk_dur = duration_ms(chunk, self.sample_rate)

        if is_silent(chunk, self.silence_threshold):
            self.trailing_silent_ms += chunk_dur
        else:
            self.has_speech = True
            self.trailing_silent_ms = 0.0

        # Before any speech starts, discard excess leading silence beyond silence_ms
        while not self.has_speech and len(self.window) > 1:
            head_dur = duration_ms(self.window[0], self.sample_rate)
            if self.trailing_silent_ms - head_dur < self.silence_ms:
                break
            self.trailing_silent_ms -= head_dur
            self.window.pop(0)

    def is_full(self) -> bool:
        """True when the audio window reaches the maximum chunk capacity."""
        return len(self.window) >= self.max_window_chunks

    def ended_by_silence(self) -> bool:
        """True when speech was followed by at least silence_ms of silence."""
        return self.has_speech and self.trailing_silent_ms >= self.silence_ms

    def wants_words(self) -> bool:
        """True when the uncommitted window is long enough for committing to pay off."""
        if not self.commit_prefix:
            return False
        window_ms = sum(duration_ms(c, self.sample_rate) for c in self.window)
        return window_ms >= self.commit_after_ms

    def context(self) -> str:
        """Trailing portion of committed text to preserve context across commits."""
        return self.committed[-CONTEXT_CHARS:].strip()

    def full_text(self, data: Optional[Dict[str, Any]]) -> str:
        """Committed prefix text concatenated with the uncommitted window transcript."""
        tail = (data or {}).get("text") or ""
        return merge_committed_and_tail(self.committed, tail)

    def commit(self, data: Dict[str, Any]) -> None:
        """Commit words that agree with the previous partial result and prune audio."""
        if not self.commit_prefix:
            return

        words = words_of(data)
        agreed = common_prefix(self.prev_words, words)
        window_ms = sum(duration_ms(c, self.sample_rate) for c in self.window)

        # Do not commit words ending near the uncommitted tail
        while agreed and words[agreed - 1].get("end", 0.0) * 1000.0 > window_ms - self.commit_margin_ms:
            agreed -= 1

        self.prev_words = words
        if not agreed or agreed >= len(words):
            return

        cut_ms = words[agreed].get("start", 0.0) * 1000.0
        dropped = 0
        while dropped < len(self.window) - 1 and duration_ms(self.window[dropped], self.sample_rate) <= cut_ms:
            cut_ms -= duration_ms(self.window[dropped], self.sample_rate)
            dropped += 1

        if not dropped:
            return

        self.committed += "".join(word.get("word", "") for word in words[:agreed])
        del self.window[:dropped]
        self.prev_words = words[agreed:]


# -----------------------------------------------------------------------------
# Model Pool (Replicas and Thread Isolation)
# -----------------------------------------------------------------------------
class ModelPool:
    """
    Manages Whisper model replicas to isolate concurrent KV-cache hooks.
    """

    def __init__(self, model: Whisper, replicas: int = 1) -> None:
        self.replicas = [model]
        for _ in range(replicas - 1):
            try:
                self.replicas.append(copy.deepcopy(model))
            except Exception as exc:
                logger.warning("Could not deepcopy model for replica; using shared replica: %s", exc)
                self.replicas.append(model)

        self.free: queue.Queue = queue.Queue()
        for replica in self.replicas:
            self.free.put(replica)

    @contextmanager
    def acquire(self) -> Iterator[Whisper]:
        """Borrow an idle model replica, blocking until one becomes available."""
        replica = self.free.get()
        try:
            yield replica
        finally:
            self.free.put(replica)


# -----------------------------------------------------------------------------
# Flush Request Marker
# -----------------------------------------------------------------------------
class FlushRequest:
    """Marker placed in the session queue to signal a trailing audio flush."""

    def __init__(self) -> None:
        self.done: asyncio.Future = asyncio.get_running_loop().create_future()


# -----------------------------------------------------------------------------
# Active Transcription Session
# -----------------------------------------------------------------------------
class TranscribeSession:
    """
    State and asynchronous worker task for a single active call session.
    """

    def __init__(
        self,
        session_id: str,
        transcribe_fn: Callable[..., Awaitable[Dict[str, Any]]],
        on_transcript: Callable[[Dict[str, Any]], Awaitable[None]],
        queue_maxsize: int = 200,
        sample_rate: int = 16000,
        silence_threshold: int = 500,
        silence_ms: float = 1500.0,
        commit_prefix: bool = True,
        commit_after_ms: float = 4000.0,
        commit_margin_ms: float = 1000.0,
        transcribe_timeout: float = 30.0,
        min_partial_ms: float = 250.0,
    ) -> None:
        self.session_id = session_id
        self.transcribe_fn = transcribe_fn
        self.on_transcript = on_transcript
        self.queue_maxsize = queue_maxsize
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_ms = silence_ms
        self.commit_prefix = commit_prefix
        self.commit_after_ms = commit_after_ms
        self.commit_margin_ms = commit_margin_ms
        self.transcribe_timeout = transcribe_timeout
        self.min_partial_ms = min_partial_ms

        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        self.accepting_audio: bool = True
        self.should_stop: List[bool] = [False]
        self.current_segment_id: int = 1
        self._stopped_event = asyncio.Event()

        # Start asynchronous transcription worker
        self.task = asyncio.create_task(self._worker_loop())

    def add_chunk(self, chunk: bytes) -> None:
        """
        Enqueue a PCM16 audio chunk for processing.

        Raises:
            STTClosedError: If session is stopping or closed.
            STTBackpressureError: If the queue is at capacity.
        """
        if not self.accepting_audio or self.should_stop[0]:
            raise STTClosedError(f"Session {self.session_id} is stopping or closed; rejecting audio.")

        try:
            self.queue.put_nowait(chunk)
        except asyncio.QueueFull:
            raise STTBackpressureError(
                f"STT audio queue full for session {self.session_id} (capacity {self.queue_maxsize}). Inference falling behind."
            )

    async def flush(self) -> None:
        """Flush all pending queued audio and finalize current segment."""
        if self.should_stop[0] and self._stopped_event.is_set():
            return
        request = FlushRequest()
        try:
            self.queue.put_nowait(request)
            await asyncio.wait({request.done, self.task}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.QueueFull:
            logger.warning("Queue full during flush request for session %s", self.session_id)

    async def stop(self) -> None:
        """Signal the worker loop to terminate and await task completion."""
        self.accepting_audio = False
        self.should_stop[0] = True
        if not self.task.done():
            # Wake up worker if idle
            try:
                self.queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
            await self.task
        self._stopped_event.set()

    def _new_segment(self) -> Segment:
        return Segment(
            sample_rate=self.sample_rate,
            silence_threshold=self.silence_threshold,
            silence_ms=self.silence_ms,
            commit_prefix=self.commit_prefix,
            commit_after_ms=self.commit_after_ms,
            commit_margin_ms=self.commit_margin_ms,
        )

    async def _emit_transcript(self, text: str, is_final: bool, latency_ms: float = 0.0) -> None:
        """Format and dispatch structured transcript event."""
        if not text:
            return

        seg_id = self.current_segment_id
        if is_final:
            self.current_segment_id += 1

        event = {
            "type": "transcript",
            "session_id": self.session_id,
            "segment_id": seg_id,
            "text": text,
            "is_final": is_final,
            "status": "final" if is_final else "partial",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": round(latency_ms, 2),
        }

        try:
            await self.on_transcript(event)
        except Exception as exc:
            logger.warning("Error dispatching transcript event for session %s: %s", self.session_id, exc)

    async def _close_segment(self, segment: Segment) -> Segment:
        """Transcribe and finalize the segment window, then emit final event."""
        start = time.perf_counter()
        if segment.has_speech and segment.window:
            options: Dict[str, Any] = {}
            if segment.context():
                options["context"] = segment.context()
            data = await self._safe_transcribe(segment.window, **options)
            full_text = segment.full_text(data)
            if full_text:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                await self._emit_transcript(full_text, is_final=True, latency_ms=elapsed_ms)
        elif segment.committed.strip():
            await self._emit_transcript(segment.committed.strip(), is_final=True)
        return self._new_segment()

    async def _safe_transcribe(self, chunks: List[bytes], **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Execute transcription with timeout guard."""
        try:
            return await asyncio.wait_for(
                self.transcribe_fn(chunks, **kwargs),
                timeout=self.transcribe_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Transcription timed out (%.1fs) for session %s", self.transcribe_timeout, self.session_id)
        except Exception as exc:
            logger.exception("Transcription inference failed for session %s: %s", self.session_id, exc)
        return None

    async def _worker_loop(self) -> None:
        """Asynchronous transcription processing loop."""
        segment = self._new_segment()

        while not self.should_stop[0]:
            try:
                # Sleep briefly to batch incoming chunks arriving in real time
                await asyncio.sleep(0.02)

                # Drain all available items in the queue
                drained: List[Any] = []
                while not self.queue.empty():
                    drained.append(self.queue.get_nowait())

                if not drained and not segment.has_speech:
                    # Nothing to process
                    continue

                for item in drained:
                    if item is None:
                        continue
                    if isinstance(item, FlushRequest):
                        segment = await self._close_segment(segment)
                        if not item.done.done():
                            item.done.set_result(True)
                        continue

                    segment.add(item)
                    if segment.ended_by_silence():
                        segment = await self._close_segment(segment)
                    elif segment.is_full():
                        segment = await self._close_segment(segment)

                if segment.ended_by_silence():
                    segment = await self._close_segment(segment)
                    continue

                if not segment.has_speech:
                    continue

                # Wait until min_partial_ms of speech audio is buffered before first partial
                # to prevent single-chunk phonetic ambiguity and hallucinations
                window_duration_ms = sum(duration_ms(c, self.sample_rate) for c in segment.window)
                if not segment.committed and window_duration_ms < self.min_partial_ms:
                    continue

                # Run partial transcription
                start = time.perf_counter()
                options: Dict[str, Any] = {}
                if segment.context():
                    options["context"] = segment.context()
                if segment.wants_words():
                    options["word_timestamps"] = True

                data = await self._safe_transcribe(segment.window, **options)
                if data is None:
                    continue

                full_text = segment.full_text(data)
                if self.commit_prefix:
                    segment.commit(data)

                elapsed_ms = (time.perf_counter() - start) * 1000.0
                result_payload = {"is_partial": True, "data": {"text": full_text}}

                if same_text(result_payload, segment.prev_result):
                    segment.cycles += 1
                else:
                    segment.cycles = 0
                    segment.prev_result = result_payload

                if full_text:
                    await self._emit_transcript(full_text, is_final=False, latency_ms=elapsed_ms)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Unexpected error in STT worker loop for session %s: %s", self.session_id, exc)

        # On stopping, close any residual segment if speech was recorded
        try:
            if segment.has_speech or segment.committed:
                await self._close_segment(segment)
        except Exception as exc:
            logger.warning("Error finalizing trailing audio on session %s shutdown: %s", self.session_id, exc)
        self._stopped_event.set()


# -----------------------------------------------------------------------------
# Main Speech-to-Text Service
# -----------------------------------------------------------------------------
class STTService:
    """
    Singleton service managing model loading, replica pool, and transcription sessions.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        replicas: Optional[int] = None,
        max_workers: Optional[int] = None,
        queue_maxsize: Optional[int] = None,
        silence_threshold: Optional[int] = None,
        silence_ms: Optional[float] = None,
        commit_prefix: Optional[bool] = None,
        commit_after_ms: Optional[float] = None,
        commit_margin_ms: Optional[float] = None,
        transcribe_timeout: Optional[float] = None,
        min_partial_ms: Optional[float] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name or os.getenv("WHISPER_MODEL_NAME", "base.en")
        self.device = device or os.getenv("WHISPER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.replicas = int(replicas if replicas is not None else os.getenv("WHISPER_MODEL_REPLICAS", "1"))
        self.max_workers = int(max_workers if max_workers is not None else os.getenv("WHISPER_TRANSCRIBE_WORKERS", "4"))
        self.queue_maxsize = int(queue_maxsize if queue_maxsize is not None else os.getenv("WHISPER_QUEUE_MAXSIZE", "200"))
        self.silence_threshold = int(
            silence_threshold if silence_threshold is not None
            else os.getenv("WHISPER_SILENCE_THRESHOLD", "500")
        )
        self.silence_ms = float(
            silence_ms if silence_ms is not None
            else os.getenv("WHISPER_SILENCE_MS", "1500.0")
        )
        if commit_prefix is not None:
            self.commit_prefix = commit_prefix
        else:
            self.commit_prefix = os.getenv("WHISPER_COMMIT_PREFIX", "1") not in ("0", "false", "False")
        self.commit_after_ms = float(
            commit_after_ms if commit_after_ms is not None
            else os.getenv("WHISPER_COMMIT_AFTER_MS", "4000.0")
        )
        self.commit_margin_ms = float(
            commit_margin_ms if commit_margin_ms is not None
            else os.getenv("WHISPER_COMMIT_MARGIN_MS", "1000.0")
        )
        self.transcribe_timeout = float(
            transcribe_timeout if transcribe_timeout is not None
            else os.getenv("WHISPER_TRANSCRIBE_TIMEOUT", "30.0")
        )
        self.min_partial_ms = float(
            min_partial_ms if min_partial_ms is not None
            else os.getenv("WHISPER_MIN_PARTIAL_MS", "250.0")
        )

        self._injected_model = model
        self.model_pool: Optional[ModelPool] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.active_sessions: Dict[str, TranscribeSession] = {}
        self._lock = threading.Lock()
        self._initialized = False

    def initialize(self) -> None:
        """
        Loads the Whisper model onto the configured device and initializes workers.

        Raises:
            STTInitializationError: If CUDA is requested but unavailable, or model loading fails.
        """
        if self._initialized:
            logger.info("STTService already initialized.")
            return

        with self._lock:
            if self._initialized:
                return

            # Explicit CUDA availability verification
            if self.device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise STTInitializationError(
                        f"CUDA device '{self.device}' requested, but torch.cuda.is_available() is False. Refusing silent CPU fallback."
                    )
                gpu_name = torch.cuda.get_device_name(0)
                logger.info("Initializing STTService on CUDA GPU: %s (%s)", gpu_name, self.device)
            else:
                logger.info("Initializing STTService on device: %s", self.device)

            # Load or inject Whisper model
            if self._injected_model is not None:
                base_model = self._injected_model
            else:
                try:
                    logger.info("Loading Whisper model '%s' on %s...", self.model_name, self.device)
                    base_model = whisper.load_model(self.model_name, device=self.device)
                except Exception as exc:
                    raise STTInitializationError(
                        f"Failed to load Whisper model '{self.model_name}': {exc}"
                    ) from exc

            self.model_pool = ModelPool(base_model, replicas=max(1, self.replicas))
            self.executor = ThreadPoolExecutor(
                max_workers=max(1, self.max_workers),
                thread_name_prefix="stt_worker",
            )
            self._initialized = True
            logger.info(
                "STTService successfully initialized with model '%s' (replicas=%d, workers=%d).",
                self.model_name,
                self.replicas,
                self.max_workers,
            )

    def is_initialized(self) -> bool:
        """True if the STT model and worker executor have been initialized."""
        return self._initialized

    def _sync_transcribe(
        self,
        chunks: List[bytes],
        context: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> Dict[str, Any]:
        """Synchronous inference executed inside the ThreadPoolExecutor worker thread."""
        if not chunks:
            return {"text": "", "segments": []}

        if self.model_pool is None:
            raise STTStateError("Model pool not initialized")

        raw_pcm = b"".join(chunks)
        if len(raw_pcm) < 2:
            return {"text": "", "segments": []}

        # Convert PCM16 little-endian bytes to float32 audio array normalized to [-1.0, 1.0]
        audio_arr = (
            np.frombuffer(raw_pcm[: len(raw_pcm) // 2 * 2], dtype=np.int16)
            .flatten()
            .astype(np.float32)
            / 32768.0
        )

        use_fp16 = self.device.startswith("cuda")
        with self.model_pool.acquire() as replica:
            return replica.transcribe(
                audio_arr,
                fp16=use_fp16,
                language="en",
                initial_prompt=context,
                word_timestamps=word_timestamps,
                temperature=0.0,
                condition_on_previous_text=False,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            )

    async def transcribe_chunks_async(
        self,
        chunks: List[bytes],
        context: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> Dict[str, Any]:
        """Runs synchronous inference in the ThreadPoolExecutor, offloading FastAPI loop."""
        if not self._initialized or self.executor is None:
            raise STTStateError("STTService is not initialized")

        loop = asyncio.get_running_loop()
        call = partial(
            self._sync_transcribe,
            chunks,
            context=context,
            word_timestamps=word_timestamps,
        )
        return await loop.run_in_executor(self.executor, call)

    def create_session(
        self,
        session_id: str,
        on_transcript: Callable[[Dict[str, Any]], Awaitable[None]],
        silence_ms: Optional[float] = None,
        silence_threshold: Optional[int] = None,
    ) -> TranscribeSession:
        """Create and register a new isolated transcription session."""
        if not self._initialized:
            raise STTStateError("STTService must be initialized before creating sessions")

        effective_silence_ms = silence_ms if silence_ms is not None else self.silence_ms
        effective_silence_threshold = silence_threshold if silence_threshold is not None else self.silence_threshold

        with self._lock:
            if session_id in self.active_sessions:
                logger.warning("Active STT session exists for %s; replacing it.", session_id)

            session = TranscribeSession(
                session_id=session_id,
                transcribe_fn=self.transcribe_chunks_async,
                on_transcript=on_transcript,
                queue_maxsize=self.queue_maxsize,
                silence_threshold=effective_silence_threshold,
                silence_ms=effective_silence_ms,
                commit_prefix=self.commit_prefix,
                commit_after_ms=self.commit_after_ms,
                commit_margin_ms=self.commit_margin_ms,
                transcribe_timeout=self.transcribe_timeout,
                min_partial_ms=self.min_partial_ms,
            )
            self.active_sessions[session_id] = session
            logger.info("Created STT transcription session: %s", session_id)
            return session

    def feed_chunk(self, session_id: str, chunk: bytes) -> None:
        """
        Forward incoming PCM16 audio chunk to the session queue.

        Raises:
            STTStateError: If session is not found.
            STTBackpressureError: If the session queue is full.
            STTClosedError: If session is stopping or stopped.
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise STTStateError(f"No active STT session found for ID: {session_id}")
        session.add_chunk(chunk)

    def stop_accepting(self, session_id: str) -> None:
        """Mark session as no longer accepting new audio chunks."""
        session = self.active_sessions.get(session_id)
        if session:
            session.accepting_audio = False

    async def flush_and_stop(self, session_id: str) -> None:
        """
        Flush remaining audio in session queue, emit final transcript, and stop worker.
        """
        session = self.active_sessions.pop(session_id, None)
        if not session:
            return

        session.accepting_audio = False
        try:
            await session.flush()
        except Exception as exc:
            logger.warning("Error during flush on session %s: %s", session_id, exc)
        finally:
            await session.stop()
        logger.info("Flushed and stopped STT session: %s", session_id)

    async def cleanup_session(self, session_id: str) -> None:
        """Forcefully stop and clean up a session without waiting for trailing audio."""
        session = self.active_sessions.pop(session_id, None)
        if not session:
            return
        session.accepting_audio = False
        await session.stop()
        logger.info("Cleaned up STT session: %s", session_id)

    def shutdown(self) -> None:
        """Clean shutdown handling: terminates workers and executors."""
        with self._lock:
            if not self._initialized:
                return

            logger.info("Shutting down STTService...")
            # Cancel active session tasks
            for sid, sess in list(self.active_sessions.items()):
                sess.accepting_audio = False
                sess.should_stop[0] = True
                if not sess.task.done():
                    sess.task.cancel()
            self.active_sessions.clear()

            if self.executor is not None:
                self.executor.shutdown(wait=False, cancel_futures=True)
                self.executor = None

            self.model_pool = None
            self._initialized = False
            logger.info("STTService shutdown complete.")


# Global default STT service singleton
default_stt_service = STTService()
