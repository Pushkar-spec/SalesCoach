"""
Real-Hardware Transcription Smoke Test & Latency Benchmark.

Uses real CUDA GPU (NVIDIA GeForce RTX 4060) and Whisper 'base.en' model to transcribe
a 6.95-second 16kHz mono PCM16 spoken audio clip sliced into 250ms chunks.

Measures and reports:
- GPU Device & VRAM Usage
- Model initialization time
- Streaming transcript events (partial and final)
- First partial transcript latency
- Final transcript latency
- Total processing time & Real-Time Factor (RTF)
- Sustained processing speed (x Real-Time)
"""

import asyncio
import os
import sys
import time
import wave
from datetime import datetime, timezone
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stt_service import STTService


async def run_smoke_test():
    print("=" * 70)
    print("WHISPERFLOW 1.3.0 REAL-TIME TRANSCRIPTION SMOKE TEST & BENCHMARK")
    print("=" * 70)

    # 1. GPU Diagnostics
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if not cuda_avail:
        raise RuntimeError("CUDA is not available on this machine. Test requires CUDA GPU.")

    device_name = torch.cuda.get_device_name(0)
    dev_capability = torch.cuda.get_device_capability(0)
    torch.cuda.reset_peak_memory_stats(0)
    mem_before = torch.cuda.memory_allocated(0) / (1024**2)
    print(f"GPU Device: {device_name} (Compute Capability: {dev_capability})")
    print(f"Initial VRAM Allocated: {mem_before:.2f} MB")

    model_name = os.getenv("WHISPER_MODEL_NAME", "base.en")
    t0_init = time.perf_counter()
    stt = STTService(
        model_name=model_name,
        device="cuda",
        replicas=1,
        max_workers=4,
        silence_threshold=400,
        silence_ms=600.0,
        commit_prefix=True,
    )
    stt.initialize()
    init_duration = time.perf_counter() - t0_init
    mem_after_init = torch.cuda.memory_allocated(0) / (1024**2)
    print(f"Model '{model_name}' loaded in: {init_duration:.3f} s")
    print(f"VRAM Allocated after model load: {mem_after_init:.2f} MB")
    print("-" * 70)

    # 3. Load Audio Fixture
    wav_path = "tests/fixtures/test_speech.wav"
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_pcm = wf.readframes(n_frames)

    audio_duration_sec = n_frames / framerate
    print(f"Loaded audio fixture: {wav_path}")
    print(f"Format: {n_channels} ch, {sampwidth * 8} bits, {framerate} Hz")
    print(f"Audio Duration: {audio_duration_sec:.2f} s ({len(raw_pcm)} bytes)")
    print("-" * 70)

    # Slice into 250ms chunks (4000 samples = 8000 bytes at 16kHz 16-bit mono)
    chunk_size_bytes = 8000
    chunks = [raw_pcm[i : i + chunk_size_bytes] for i in range(0, len(raw_pcm), chunk_size_bytes)]
    print(f"Total 250ms chunks to stream: {len(chunks)}")
    print("-" * 70)

    # 4. Stream Audio into STT Session
    events = []
    first_chunk_sent_time = None
    first_partial_event_time = None
    first_partial_latency_ms = None

    async def on_transcript_received(event: dict):
        nonlocal first_partial_event_time, first_partial_latency_ms
        recv_time = time.perf_counter()
        if first_partial_event_time is None and not event.get("is_final"):
            first_partial_event_time = recv_time
            if first_chunk_sent_time is not None:
                first_partial_latency_ms = (recv_time - first_chunk_sent_time) * 1000.0

        events.append((recv_time, event))
        status_label = "FINAL  " if event.get("is_final") else "PARTIAL"
        seg_id = event.get("segment_id")
        text = event.get("text", "")
        latency = event.get("latency_ms", 0.0)
        print(f"[{status_label}] Seg #{seg_id} | Latency: {latency:6.1f}ms | Text: \"{text}\"")

    session_id = "smoke_benchmark_session_01"
    session = stt.create_session(session_id, on_transcript_received)

    print("Streaming audio chunks in 250ms intervals...")
    t_stream_start = time.perf_counter()
    first_chunk_sent_time = t_stream_start

    for i, chunk in enumerate(chunks):
        stt.feed_chunk(session_id, chunk)
        # Simulate real-time 250ms audio chunk cadence (with 100ms pause to test streaming throughput)
        await asyncio.sleep(0.10)

    # 5. Flush and Finalize
    print("Audio stream finished. Flushing trailing audio...")
    t_flush_start = time.perf_counter()
    stt.stop_accepting(session_id)
    await stt.flush_and_stop(session_id)
    t_flush_end = time.perf_counter()
    flush_latency_ms = (t_flush_end - t_flush_start) * 1000.0

    total_processing_wall_time = t_flush_end - t_stream_start
    peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024**2)

    # 6. Report Measured Performance Metrics
    print("=" * 70)
    print("BENCHMARK RESULTS & MEASURED LATENCY")
    print("=" * 70)
    print(f"Audio Duration:              {audio_duration_sec:.2f} s")
    print(f"Total Wall-Clock Time:       {total_processing_wall_time:.2f} s")
    print(f"First Partial Latency:       {first_partial_latency_ms if first_partial_latency_ms else 'N/A':.1f} ms")
    print(f"Trailing Audio Flush Time:   {flush_latency_ms:.1f} ms")
    print(f"Total Transcript Events:     {len(events)} (Partials: {sum(1 for _, e in events if not e.get('is_final'))}, Finals: {sum(1 for _, e in events if e.get('is_final'))})")

    # Compute Real-Time Factor (RTF = processing_time / audio_duration)
    # Using inference execution time
    inference_times = [e.get("latency_ms", 0.0) for _, e in events]
    avg_inference_latency_ms = np.mean(inference_times) if inference_times else 0.0
    p95_inference_latency_ms = np.percentile(inference_times, 95) if inference_times else 0.0
    print(f"Avg Inference Latency/Cycle: {avg_inference_latency_ms:.1f} ms")
    print(f"P95 Inference Latency/Cycle: {p95_inference_latency_ms:.1f} ms")
    
    # Processing speed ratio
    speedup = audio_duration_sec / total_processing_wall_time
    print(f"Real-Time Processing Speed:  {speedup:.2f}x Real-Time (Faster than real-time: {speedup > 1.0})")
    print(f"Peak GPU VRAM Allocated:     {peak_vram_mb:.2f} MB")

    final_transcripts = [e.get("text", "") for _, e in events if e.get("is_final")]
    full_text = " ".join(final_transcripts)
    print("-" * 70)
    print(f"Final Recognized Transcript: \"{full_text}\"")
    print("=" * 70)

    # Clean shutdown
    stt.shutdown()

    # Assertions for smoke test validation
    assert len(events) >= 1, "Expected at least one transcript event"
    assert any(e.get("is_final") for _, e in events), "Expected at least one final transcript event"
    assert "co-pilot" in full_text.lower() or "copilot" in full_text.lower() or "sales" in full_text.lower(), \
        f"Transcript did not match expected speech: {full_text}"
    print("SUCCESS: Real-hardware smoke test and transcription benchmark completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_smoke_test())
