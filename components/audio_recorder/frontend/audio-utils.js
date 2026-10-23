/**
 * audio-utils.js — Audio Resampling, PCM16 Conversion, and Chunk Accumulator
 *
 * Designed for the Real-Time AI Sales Copilot audio streaming pipeline.
 * Formats: Input from Web Audio API (Float32, arbitrary hardware sample rate)
 *          Output: 16000 Hz, 16-bit signed PCM, little-endian, mono (~250ms chunks)
 */

(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    // Node / CommonJS
    module.exports = factory();
  } else {
    // Browser global
    root.AudioUtils = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  /**
   * Resamples a Float32Array of audio samples from inputSampleRate to targetSampleRate
   * using linear interpolation.
   *
   * @param {Float32Array} inputBuffer - Input audio samples in range [-1.0, 1.0]
   * @param {number} inputSampleRate - Source sample rate (e.g. 48000, 44100)
   * @param {number} targetSampleRate - Desired output sample rate (default 16000)
   * @returns {Float32Array} Resampled audio samples
   */
  function resampleAudio(inputBuffer, inputSampleRate, targetSampleRate = 16000) {
    if (!inputBuffer || inputBuffer.length === 0) {
      return new Float32Array(0);
    }
    if (inputSampleRate <= 0 || targetSampleRate <= 0) {
      throw new Error("Sample rates must be positive numbers");
    }
    if (inputSampleRate === targetSampleRate) {
      return inputBuffer.slice();
    }

    const ratio = inputSampleRate / targetSampleRate;
    const outputLength = Math.round(inputBuffer.length / ratio);
    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const position = i * ratio;
      const index = Math.floor(position);
      const fraction = position - index;
      const sample1 = inputBuffer[index];
      const sample2 = index + 1 < inputBuffer.length ? inputBuffer[index + 1] : sample1;
      output[i] = sample1 + fraction * (sample2 - sample1);
    }

    return output;
  }

  /**
   * Converts Float32 audio samples in [-1.0, 1.0] to signed 16-bit little-endian PCM ArrayBuffer.
   *
   * @param {Float32Array} floatSamples - Input float samples
   * @returns {ArrayBuffer} Raw bytes containing 16-bit signed integers (little-endian)
   */
  function floatTo16BitPCM(floatSamples) {
    if (!floatSamples || floatSamples.length === 0) {
      return new ArrayBuffer(0);
    }

    const buffer = new ArrayBuffer(floatSamples.length * 2);
    const view = new DataView(buffer);

    for (let i = 0; i < floatSamples.length; i++) {
      // Clamp between -1.0 and 1.0 to prevent integer overflow
      const clamped = Math.max(-1.0, Math.min(1.0, floatSamples[i]));
      // Convert to signed 16-bit integer [-32768, 32767]
      const intSample = clamped < 0 ? Math.round(clamped * 0x8000) : Math.round(clamped * 0x7FFF);
      // Little-endian byte order (last argument `true`)
      view.setInt16(i * 2, intSample, true);
    }

    return buffer;
  }

  /**
   * Accumulator for incoming PCM samples that emits chunks of a specified sample count
   * (e.g. 4000 samples = 250ms at 16000Hz = 8000 bytes).
   */
  class AudioChunkAccumulator {
    /**
     * @param {Object} options
     * @param {number} options.chunkSampleCount - Number of 16kHz samples per chunk (default 4000 = 250ms)
     * @param {Function} options.onChunk - Callback invoked with ArrayBuffer for each chunk
     */
    constructor(options = {}) {
      this.chunkSampleCount = options.chunkSampleCount || 4000;
      this.onChunk = options.onChunk || (() => {});
      this._buffer = [];
      this.totalChunks = 0;
      this.totalBytes = 0;
    }

    /**
     * Adds resampled float32 samples, converts to PCM16 chunks when accumulator threshold is reached.
     * @param {Float32Array} resampledSamples - 16kHz float32 samples
     */
    push(resampledSamples) {
      if (!resampledSamples || resampledSamples.length === 0) return;

      for (let i = 0; i < resampledSamples.length; i++) {
        this._buffer.push(resampledSamples[i]);
      }

      while (this._buffer.length >= this.chunkSampleCount) {
        const chunkSamples = this._buffer.splice(0, this.chunkSampleCount);
        const floatArray = new Float32Array(chunkSamples);
        const pcmBuffer = floatTo16BitPCM(floatArray);

        this.totalChunks += 1;
        this.totalBytes += pcmBuffer.byteLength;
        this.onChunk(pcmBuffer);
      }
    }

    /**
     * Flushes remaining samples if any, ensuring even byte length for PCM16.
     * @param {boolean} emitRemaining - Whether to emit the remaining samples as a final chunk
     */
    flush(emitRemaining = false) {
      if (emitRemaining && this._buffer.length > 0) {
        const floatArray = new Float32Array(this._buffer);
        const pcmBuffer = floatTo16BitPCM(floatArray);
        this._buffer = [];
        this.totalChunks += 1;
        this.totalBytes += pcmBuffer.byteLength;
        this.onChunk(pcmBuffer);
      } else {
        this._buffer = [];
      }
    }

    /**
     * Resets the accumulator state.
     */
    reset() {
      this._buffer = [];
      this.totalChunks = 0;
      this.totalBytes = 0;
    }

    /**
     * Returns stats.
     */
    getStats() {
      return {
        bufferedSamples: this._buffer.length,
        totalChunks: this.totalChunks,
        totalBytes: this.totalBytes,
      };
    }
  }

  return {
    resampleAudio,
    floatTo16BitPCM,
    AudioChunkAccumulator,
  };
});
