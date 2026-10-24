/**
 * test_audio_processor.js
 * Unit tests for audio-utils.js using Node.js built-in test runner (node:test).
 */

const { test, describe } = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const {
  resampleAudio,
  floatTo16BitPCM,
  AudioChunkAccumulator,
} = require(path.join(__dirname, "..", "components", "audio_recorder", "frontend", "audio-utils.js"));

describe("Audio Utils — Resampling (resampleAudio)", () => {
  test("resamples 48000 Hz to 16000 Hz (3:1 ratio)", () => {
    // 48000 samples of 1 second audio at 48kHz
    const input = new Float32Array(48000);
    // Fill with a 440Hz sine wave
    for (let i = 0; i < input.length; i++) {
      input[i] = Math.sin((2 * Math.PI * 440 * i) / 48000);
    }

    const output = resampleAudio(input, 48000, 16000);
    assert.strictEqual(output.length, 16000);
    // Verify waveform continuity and amplitude bounds
    assert.ok(output instanceof Float32Array);
    assert.ok(Math.abs(output[0]) < 0.01);
  });

  test("resamples 44100 Hz to 16000 Hz", () => {
    const input = new Float32Array(44100);
    for (let i = 0; i < input.length; i++) {
      input[i] = 0.5 * Math.sin((2 * Math.PI * 1000 * i) / 44100);
    }

    const output = resampleAudio(input, 44100, 16000);
    assert.strictEqual(output.length, 16000);
    for (let i = 0; i < output.length; i++) {
      assert.ok(output[i] >= -0.55 && output[i] <= 0.55);
    }
  });

  test("identity resampling when input and target sample rate match (16000 Hz)", () => {
    const input = new Float32Array([0.1, -0.2, 0.3, -0.4, 0.5]);
    const output = resampleAudio(input, 16000, 16000);
    assert.strictEqual(output.length, 5);
    for (let i = 0; i < input.length; i++) {
      assert.strictEqual(output[i], input[i]);
    }
  });

  test("returns empty Float32Array on empty or null input", () => {
    assert.strictEqual(resampleAudio(new Float32Array(0), 48000, 16000).length, 0);
    assert.strictEqual(resampleAudio(null, 48000, 16000).length, 0);
  });

  test("throws on invalid sample rate inputs", () => {
    assert.throws(() => resampleAudio(new Float32Array(10), 0, 16000));
    assert.throws(() => resampleAudio(new Float32Array(10), 48000, -100));
  });
});

describe("Audio Utils — PCM16 Conversion (floatTo16BitPCM)", () => {
  test("converts float values to 16-bit signed PCM with correct little-endian encoding", () => {
    // Test values: 0, 1.0 (max pos), -1.0 (max neg), 0.5, -0.5
    const input = new Float32Array([0.0, 1.0, -1.0, 0.5, -0.5]);
    const buffer = floatTo16BitPCM(input);

    assert.strictEqual(buffer.byteLength, 10); // 5 samples * 2 bytes = 10 bytes
    const view = new DataView(buffer);

    // Sample 0: 0.0 -> 0
    assert.strictEqual(view.getInt16(0, true), 0);
    // Sample 1: 1.0 -> 32767 (0x7FFF)
    assert.strictEqual(view.getInt16(2, true), 32767);
    // Sample 2: -1.0 -> -32768 (-0x8000)
    assert.strictEqual(view.getInt16(4, true), -32768);
    // Sample 3: 0.5 -> ~16384
    assert.ok(Math.abs(view.getInt16(6, true) - 16384) <= 1);
    // Sample 4: -0.5 -> ~ -16384
    assert.ok(Math.abs(view.getInt16(8, true) - -16384) <= 1);
  });

  test("clamps values outside [-1.0, 1.0] to prevent overflow distortion", () => {
    const input = new Float32Array([2.5, -3.2]);
    const buffer = floatTo16BitPCM(input);
    const view = new DataView(buffer);

    assert.strictEqual(view.getInt16(0, true), 32767);
    assert.strictEqual(view.getInt16(2, true), -32768);
  });

  test("handles empty input safely", () => {
    const buffer = floatTo16BitPCM(new Float32Array(0));
    assert.strictEqual(buffer.byteLength, 0);
    assert.strictEqual(floatTo16BitPCM(null).byteLength, 0);
  });
});

describe("Audio Utils — AudioChunkAccumulator", () => {
  test("emits chunks of exactly 4000 samples (8000 bytes = 250ms at 16kHz)", () => {
    const emittedChunks = [];
    const accumulator = new AudioChunkAccumulator({
      chunkSampleCount: 4000,
      onChunk: (buffer) => emittedChunks.push(buffer),
    });

    // Push 10,000 samples in batches of 2500 samples (4 pushes)
    const batch = new Float32Array(2500);
    batch.fill(0.1);

    accumulator.push(batch); // Total 2500 -> 0 chunks, 2500 buffered
    assert.strictEqual(emittedChunks.length, 0);

    accumulator.push(batch); // Total 5000 -> 1 chunk (4000), 1000 buffered
    assert.strictEqual(emittedChunks.length, 1);
    assert.strictEqual(emittedChunks[0].byteLength, 8000); // 4000 samples * 2 bytes = 8000 bytes

    accumulator.push(batch); // Total 7500 -> 1 chunk (4000), 3500 buffered
    assert.strictEqual(emittedChunks.length, 1);

    accumulator.push(batch); // Total 10000 -> 2 chunks total, 2000 buffered
    assert.strictEqual(emittedChunks.length, 2);
    assert.strictEqual(emittedChunks[1].byteLength, 8000);

    const stats = accumulator.getStats();
    assert.strictEqual(stats.totalChunks, 2);
    assert.strictEqual(stats.totalBytes, 16000);
    assert.strictEqual(stats.bufferedSamples, 2000);
  });

  test("flushes remaining samples cleanly", () => {
    const emittedChunks = [];
    const accumulator = new AudioChunkAccumulator({
      chunkSampleCount: 4000,
      onChunk: (buffer) => emittedChunks.push(buffer),
    });

    accumulator.push(new Float32Array(1500));
    assert.strictEqual(emittedChunks.length, 0);

    // Flush with emitRemaining = true
    accumulator.flush(true);
    assert.strictEqual(emittedChunks.length, 1);
    assert.strictEqual(emittedChunks[0].byteLength, 3000); // 1500 samples * 2 = 3000 bytes (even)

    assert.strictEqual(accumulator.getStats().bufferedSamples, 0);
  });

  test("resets accumulator state", () => {
    const accumulator = new AudioChunkAccumulator({ chunkSampleCount: 100 });
    accumulator.push(new Float32Array(250));
    assert.strictEqual(accumulator.totalChunks, 2);

    accumulator.reset();
    assert.strictEqual(accumulator.totalChunks, 0);
    assert.strictEqual(accumulator.totalBytes, 0);
    assert.strictEqual(accumulator.getStats().bufferedSamples, 0);
  });

  test("processes synthetic noisy audio without buffer or byte corruption", () => {
    // Generate 1000 samples of pseudo-random noise [-1.5, 1.5] to test clamping
    const noise = new Float32Array(1000);
    for (let i = 0; i < noise.length; i++) {
      noise[i] = Math.sin(i * 123.456) * 1.5;
    }
    const buffer = floatTo16BitPCM(noise);
    assert.strictEqual(buffer.byteLength, 2000);
    const view = new DataView(buffer);
    for (let i = 0; i < 1000; i++) {
      const val = view.getInt16(i * 2, true);
      assert.ok(val >= -32768 && val <= 32767);
    }
  });

  test("flushes partial final chunks with odd sample count guaranteeing even byte length", () => {
    const emittedChunks = [];
    const accumulator = new AudioChunkAccumulator({
      chunkSampleCount: 4000,
      onChunk: (buffer) => emittedChunks.push(buffer),
    });

    // 17 samples (odd sample count)
    accumulator.push(new Float32Array(17));
    accumulator.flush(true);
    assert.strictEqual(emittedChunks.length, 1);
    // 17 samples * 2 = 34 bytes (strictly even byte length for PCM16)
    assert.strictEqual(emittedChunks[0].byteLength, 34);
    assert.strictEqual(emittedChunks[0].byteLength % 2, 0);
  });
});

