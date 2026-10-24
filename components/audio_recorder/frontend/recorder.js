/**
 * recorder.js — Browser-Side Microphone Capture & WebSocket Audio Streamer
 *
 * Implements Step 2.4 for the Real-Time AI Sales Copilot.
 * - Managed directly by the main Streamlit Call Controls via custom component commands.
 * - One authoritative WebSocket connection & session owned by this browser client.
 * - Captures audio via navigator.mediaDevices.getUserMedia on explicit user start.
 * - Resamples from hardware sample rate to 16000 Hz mono.
 * - Converts samples to signed 16-bit little-endian PCM.
 * - Emits ~250ms chunks (4000 samples = 8000 bytes) directly to FastAPI WebSocket.
 * - Communicates lifecycle state and error events back to Streamlit via setComponentValue.
 */

(function () {
  "use strict";

  // Configuration defaults
  const DEFAULT_BACKEND_URL = "ws://127.0.0.1:8000/ws";
  const TARGET_SAMPLE_RATE = 16000;
  const CHUNK_DURATION_MS = 250;
  const SAMPLES_PER_CHUNK = Math.round((TARGET_SAMPLE_RATE * CHUNK_DURATION_MS) / 1000); // 4000 samples = 8000 bytes

  // Bounded timeouts (Step 2.6)
  const WS_CONNECT_TIMEOUT_MS = 5000;
  const CONTROL_TIMEOUT_MS = 5000;
  const SHUTDOWN_TIMEOUT_MS = 3000;
  const AUDIO_STALL_TIMEOUT_MS = 5000;

  // State enum
  const State = {
    IDLE: "idle",
    REQUESTING_PERMISSION: "requesting_permission",
    RECORDING: "recording",
    STOPPING: "stopping",
    STOPPED: "stopped",
    ERROR: "error",
  };

  class AudioStreamRecorder {
    constructor() {
      // Internal state
      this.state = State.IDLE;
      this.backendWsUrl = DEFAULT_BACKEND_URL;
      this.sessionId = null;
      this.isAudioStreaming = false;
      this.lastCommandId = null;
      this.errorMessage = null;
      this.statusMessage = null;

      // Audio resources
      this.mediaStream = null;
      this.audioContext = null;
      this.sourceNode = null;
      this.processorNode = null;
      this.muteNode = null;
      this.accumulator = null;

      // WebSocket
      this.ws = null;

      // Metrics & Monotonic Timings (Step 2.6)
      this.chunksSent = 0;
      this.bytesSent = 0;
      this.startCallTime = null;
      this.audioStreamingStartTime = null;
      this.stopCallTime = null;
      this.startupLatencyMs = 0;
      this.shutdownLatencyMs = 0;
      this.recordingDurationMs = 0;
      this.lastChunkSentTime = null;
      this.interruptionCount = 0;

      // Real-time STT Transcripts (Step 3.2B)
      this.transcripts = [];
      this.latestTranscript = null;

      // Bounded Timers
      this.connectTimer = null;
      this.controlTimer = null;
      this.shutdownTimer = null;
      this.stallCheckInterval = null;

      // DOM Elements (bound on init)
      this.elements = {};

      this._bindElements();
      this._attachEventListeners();
      this._setupStreamlitBridge();
      this._updateUI();
    }

    _bindElements() {
      this.elements = {
        card: document.getElementById("recorder-card"),
        statusPill: document.getElementById("status-pill"),
        statusText: document.getElementById("status-text"),
        errorBox: document.getElementById("error-box"),
        errorMessage: document.getElementById("error-message"),
        sessionIdVal: document.getElementById("val-session-id"),
        audioStateVal: document.getElementById("val-audio-state"),
        chunksVal: document.getElementById("val-chunks"),
        bytesVal: document.getElementById("val-bytes"),
        valDuration: document.getElementById("val-duration"),
        valLatency: document.getElementById("val-latency"),
        meterFill: document.getElementById("meter-fill"),
        endpointText: document.getElementById("val-endpoint"),
      };
    }

    _attachEventListeners() {
      // Clean up resources if window unloads or component is dismantled
      window.addEventListener("beforeunload", () => this._cleanupResources());
    }

    _setupStreamlitBridge() {
      // Listen for messages from Streamlit
      window.addEventListener("message", (event) => {
        if (!event.data || event.data.type !== "streamlit:render") return;

        const args = event.data.args || {};
        if (args.backend_ws_url && typeof args.backend_ws_url === "string") {
          this.backendWsUrl = args.backend_ws_url;
          if (this.elements.endpointText) {
            this.elements.endpointText.textContent = this.backendWsUrl;
          }
        }

        // Process incoming control command from Python (Start / Stop / Reset)
        if (args.command) {
          const cmd = args.command;
          const cmdId =
            typeof cmd === "object" && cmd !== null
              ? (cmd.id || JSON.stringify(cmd))
              : String(cmd);
          if (cmdId && cmdId !== this.lastCommandId) {
            this.lastCommandId = cmdId;
            this._handleCommand(cmd);
          }
        }

        this._notifyFrameHeight();
      });

      // Notify parent Streamlit app that component is ready
      if (window.parent && window.parent !== window) {
        window.parent.postMessage(
          { isStreamlitMessage: true, type: "streamlit:componentReady", apiVersion: 1 },
          "*"
        );
      }
      this._notifyFrameHeight();
    }

    _notifyFrameHeight() {
      if (window.parent && window.parent !== window) {
        const height = document.body ? Math.max(document.body.scrollHeight, 185) : 185;
        window.parent.postMessage(
          { isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: height },
          "*"
        );
      }
    }

    /**
     * Sends lifecycle state back to Streamlit Python process.
     * Note: Only called on state transitions, NEVER on individual audio chunks!
     */
    _notifyStreamlit(overrideState = null) {
      if (window.parent && window.parent !== window) {
        let durationMs = this.recordingDurationMs;
        if (!durationMs && this.audioStreamingStartTime) {
          durationMs = Math.round(performance.now() - this.audioStreamingStartTime);
        }
        const payload = {
          state: overrideState || this.state,
          session_id: this.sessionId,
          audio_state: this.isAudioStreaming ? "streaming" : "idle",
          chunks_sent: this.chunksSent,
          bytes_sent: this.bytesSent,
          metrics: {
            startup_latency_ms: Math.round(this.startupLatencyMs || 0),
            shutdown_latency_ms: Math.round(this.shutdownLatencyMs || 0),
            recording_duration_ms: Math.round(durationMs || 0),
            expected_bytes: Math.round(((durationMs || 0) / 1000) * 32000),
            interruptions: this.interruptionCount,
          },
          transcript: this.latestTranscript || null,
          transcripts: this.transcripts ? [...this.transcripts] : [],
          error: this.errorMessage || null,
          status_message: this.statusMessage || null,
        };
        window.parent.postMessage(
          {
            isStreamlitMessage: true,
            type: "streamlit:setComponentValue",
            value: payload,
          },
          "*"
        );
      }
    }

    _handleCommand(cmd) {
      const action =
        typeof cmd === "object" && cmd !== null ? cmd.action : String(cmd);
      console.log("[AudioRecorder] Processing command:", action, cmd);
      switch (action) {
        case "start":
          if (this.state !== State.RECORDING && this.state !== State.REQUESTING_PERMISSION) {
            this.startRecording();
          }
          break;

        case "stop":
          if (this.state === State.RECORDING || this.state === State.REQUESTING_PERMISSION) {
            this.stopRecording();
          }
          break;

        case "reset":
          this.resetRecorder();
          break;

        default:
          console.warn("[AudioRecorder] Unknown command action:", action);
      }
    }

    _setState(newState, statusMessage = null) {
      this.state = newState;
      this.statusMessage = statusMessage;
      this._updateUI(statusMessage);
      this._notifyFrameHeight();
    }

    _showError(message) {
      console.warn("[AudioRecorder] Error:", message);
      this.errorMessage = message;
      if (this.elements.errorBox && this.elements.errorMessage) {
        this.elements.errorMessage.textContent = message;
        this.elements.errorBox.style.display = "flex";
      }
      this._setState(State.ERROR, message);
    }

    _clearError() {
      this.errorMessage = null;
      if (this.elements.errorBox) {
        this.elements.errorBox.style.display = "none";
      }
    }

    _updateUI(customStatusMessage = null) {
      const {
        statusPill,
        statusText,
        sessionIdVal,
        audioStateVal,
        chunksVal,
        bytesVal,
        valDuration,
        valLatency,
        meterFill,
        endpointText,
      } = this.elements;

      if (endpointText) {
        endpointText.textContent = this.backendWsUrl;
      }

      if (sessionIdVal) {
        sessionIdVal.textContent = this.sessionId || "None";
      }

      if (audioStateVal) {
        audioStateVal.textContent = this.isAudioStreaming ? "streaming" : "idle";
        audioStateVal.className = this.isAudioStreaming ? "val-highlight-green" : "";
      }

      if (chunksVal) {
        chunksVal.textContent = this.chunksSent.toLocaleString();
      }

      if (bytesVal) {
        if (this.bytesSent >= 1024 * 1024) {
          bytesVal.textContent = (this.bytesSent / (1024 * 1024)).toFixed(2) + " MB";
        } else if (this.bytesSent >= 1024) {
          bytesVal.textContent = (this.bytesSent / 1024).toFixed(1) + " KB";
        } else {
          bytesVal.textContent = this.bytesSent + " B";
        }
      }

      if (valDuration) {
        let dur = 0;
        if (this.audioStreamingStartTime) {
          dur = ((this.stopCallTime || performance.now()) - this.audioStreamingStartTime) / 1000;
        }
        valDuration.textContent = Math.max(0, dur).toFixed(1) + " s";
      }

      if (valLatency) {
        valLatency.textContent = this.startupLatencyMs > 0 ? `${this.startupLatencyMs} ms` : "--";
      }

      if (meterFill && !this.isAudioStreaming) {
        meterFill.style.width = "0%";
      }

      // Update status pill and text
      switch (this.state) {
        case State.IDLE:
          if (statusPill) {
            statusPill.textContent = "IDLE";
            statusPill.className = "status-pill status-idle";
          }
          if (statusText) {
            statusText.textContent =
              customStatusMessage || "Ready. Click 'Start Call' above to begin call.";
          }
          break;

        case State.REQUESTING_PERMISSION:
          if (statusPill) {
            statusPill.textContent = "REQUESTING MIC";
            statusPill.className = "status-pill status-requesting";
          }
          if (statusText) {
            statusText.textContent =
              customStatusMessage || "Requesting microphone permission...";
          }
          break;

        case State.RECORDING:
          if (statusPill) {
            statusPill.textContent = "RECORDING";
            statusPill.className = "status-pill status-recording";
          }
          if (statusText) {
            statusText.textContent =
              customStatusMessage || "Microphone active. Streaming live 16kHz audio to backend.";
          }
          break;

        case State.STOPPING:
          if (statusPill) {
            statusPill.textContent = "STOPPING";
            statusPill.className = "status-pill status-stopping";
          }
          if (statusText) {
            statusText.textContent =
              customStatusMessage || "Stopping audio stream and terminating session...";
          }
          break;

        case State.STOPPED:
          if (statusPill) {
            statusPill.textContent = "STOPPED";
            statusPill.className = "status-pill status-idle";
          }
          if (statusText) {
            statusText.textContent =
              customStatusMessage || "Call ended cleanly. All audio resources released.";
          }
          break;

        case State.ERROR:
          if (statusPill) {
            statusPill.textContent = "ERROR";
            statusPill.className = "status-pill status-error";
          }
          if (statusText) {
            statusText.textContent = customStatusMessage || "An error occurred.";
          }
          break;
      }
    }

    /**
     * Start the unified recording and session workflow.
     */
    async startRecording() {
      if (this.state === State.RECORDING || this.state === State.REQUESTING_PERMISSION) {
        return; // Idempotent check
      }

      this._clearError();
      this._clearTimers();
      this.chunksSent = 0;
      this.bytesSent = 0;
      this.sessionId = null;
      this.isAudioStreaming = false;
      this.transcripts = [];
      this.latestTranscript = null;

      // Monotonic timing initialization
      this.startCallTime = performance.now();
      this.audioStreamingStartTime = null;
      this.stopCallTime = null;
      this.startupLatencyMs = 0;
      this.shutdownLatencyMs = 0;
      this.recordingDurationMs = 0;
      this.lastChunkSentTime = null;

      // 1. Verify Browser Web Audio & MediaDevices Support
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const err = "Microphone capture is not supported in this browser or iframe environment.";
        this.interruptionCount++;
        this._showError(err);
        this._notifyStreamlit("error");
        return;
      }

      this._setState(State.REQUESTING_PERMISSION);
      this._notifyStreamlit();

      // 2. Request Microphone Access
      try {
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
      } catch (err) {
        let msg = "Microphone access failed.";
        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
          msg = "Microphone access denied. Please allow microphone permissions in your browser.";
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
          msg = "No microphone hardware detected on this device.";
        } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
          msg = "Microphone is already in use by another application or tab.";
        } else {
          msg = `Microphone access error: ${err.message || err.name}`;
        }
        this.interruptionCount++;
        this._cleanupResources();
        this._showError(msg);
        this._notifyStreamlit("error");
        return;
      }

      // 3. Initialize Audio Context
      try {
        const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtxClass) {
          throw new Error("Web Audio API is not supported in this browser.");
        }
        this.audioContext = new AudioCtxClass();
        if (this.audioContext.state === "suspended") {
          await this.audioContext.resume();
        }
      } catch (err) {
        this.interruptionCount++;
        this._cleanupResources();
        const msg = `Failed to initialize Web Audio API: ${err.message}`;
        this._showError(msg);
        this._notifyStreamlit("error");
        return;
      }

      // 4. Initialize Accumulator
      this.accumulator = new AudioUtils.AudioChunkAccumulator({
        chunkSampleCount: SAMPLES_PER_CHUNK,
        onChunk: (pcmBuffer) => this._handleOutgoingChunk(pcmBuffer),
      });

      // 5. Connect to Backend WebSocket
      this._connectWebSocket();
    }

    _connectWebSocket() {
      try {
        this.ws = new WebSocket(this.backendWsUrl);
        this.ws.binaryType = "arraybuffer";
      } catch (err) {
        this.interruptionCount++;
        this._cleanupResources();
        const msg = `Failed to connect to backend at ${this.backendWsUrl}: ${err.message}`;
        this._showError(msg);
        this._notifyStreamlit("error");
        return;
      }

      // Bounded connection timer
      this.connectTimer = setTimeout(() => {
        if (this.state === State.REQUESTING_PERMISSION || this.state === State.RECORDING) {
          console.warn("[AudioRecorder] WebSocket connection timed out.");
          this.interruptionCount++;
          this._cleanupResources();
          this._showError(`WebSocket connection timed out (${WS_CONNECT_TIMEOUT_MS}ms). Backend may be down.`);
          this._notifyStreamlit("error");
        }
      }, WS_CONNECT_TIMEOUT_MS);

      this.ws.onopen = () => {
        if (this.connectTimer) {
          clearTimeout(this.connectTimer);
          this.connectTimer = null;
        }
        console.log("[AudioRecorder] WebSocket connected. Sending start_session...");
        this.ws.send(JSON.stringify({ type: "start_session" }));

        // Bounded control response timer for session_started
        this._startControlTimer("session_started");
      };

      this.ws.onmessage = (event) => {
        if (typeof event.data === "string") {
          try {
            const data = JSON.parse(event.data);
            this._handleControlMessage(data);
          } catch (e) {
            console.warn("[AudioRecorder] Non-JSON message:", event.data);
          }
        }
      };

      this.ws.onerror = (err) => {
        console.error("[AudioRecorder] WebSocket error:", err);
        this._clearTimers();
        if (this.state === State.RECORDING || this.state === State.REQUESTING_PERMISSION) {
          this.interruptionCount++;
          const msg = "WebSocket connection error.";
          this._cleanupResources();
          this._showError(msg);
          this._notifyStreamlit("error");
        }
      };

      this.ws.onclose = (event) => {
        console.log(`[AudioRecorder] WebSocket closed (code: ${event.code})`);
        this._clearTimers();
        if (this.state === State.RECORDING) {
          this.interruptionCount++;
          this._showError("Backend WebSocket connection closed unexpectedly.");
          this._cleanupResources();
          this._notifyStreamlit("error");
        } else {
          this._cleanupResources();
        }
      };
    }

    _startControlTimer(expectedType) {
      if (this.controlTimer) clearTimeout(this.controlTimer);
      this.controlTimer = setTimeout(() => {
        console.warn(`[AudioRecorder] Timed out waiting for '${expectedType}' response.`);
        this.interruptionCount++;
        this._cleanupResources();
        this._showError(`Server response timed out waiting for '${expectedType}'.`);
        this._notifyStreamlit("error");
      }, CONTROL_TIMEOUT_MS);
    }

    _clearControlTimer() {
      if (this.controlTimer) {
        clearTimeout(this.controlTimer);
        this.controlTimer = null;
      }
    }

    _handleControlMessage(data) {
      switch (data.type) {
        case "session_started":
          this._clearControlTimer();
          this.sessionId = data.session_id;
          console.log("[AudioRecorder] Session confirmed by backend:", this.sessionId);
          // Request audio start with protocol specifications
          this.ws.send(
            JSON.stringify({
              type: "start_audio",
              session_id: this.sessionId,
              format: "pcm16",
              sample_rate: TARGET_SAMPLE_RATE,
              channels: 1,
            })
          );
          this._startControlTimer("audio_started");
          break;

        case "audio_started":
          this._clearControlTimer();
          console.log("[AudioRecorder] Audio stream started by backend. Wiring audio nodes.");
          this.audioStreamingStartTime = performance.now();
          if (this.startCallTime) {
            this.startupLatencyMs = Math.round(this.audioStreamingStartTime - this.startCallTime);
          }
          this.isAudioStreaming = true;
          this._startAudioPipeline();
          this._startStallDetector();
          this._setState(State.RECORDING, "Microphone streaming 16kHz PCM16 audio.");
          this._notifyStreamlit("recording");
          break;

        case "audio_stopped":
          this._clearControlTimer();
          console.log(
            `[AudioRecorder] Audio stopped: chunks=${data.audio_chunk_count}, bytes=${data.audio_bytes_received}`
          );
          this.isAudioStreaming = false;
          // Step 10: Terminate session on backend
          if (this.sessionId && this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(
              JSON.stringify({
                type: "stop_session",
                session_id: this.sessionId,
              })
            );
            this._startControlTimer("session_stopped");
          } else {
            this._finishTeardown();
          }
          break;

        case "session_stopped":
          this._clearControlTimer();
          console.log("[AudioRecorder] Session terminated confirmation received.");
          this._finishTeardown();
          break;

        case "error":
          this._clearTimers();
          console.warn("[AudioRecorder] Backend error:", data.message);
          this.interruptionCount++;
          this._cleanupResources();
          this._showError(`Server error: ${data.message}`);
          this._notifyStreamlit("error");
          break;

        case "transcript":
          this._handleTranscriptMessage(data);
          break;

        default:
          console.log("[AudioRecorder] Received message type:", data.type);
      }
    }

    _handleTranscriptMessage(data) {
      if (!data || typeof data !== "object") return;
      const segId = data.segment_id;
      const text = (data.text || "").trim();
      const isFinal = Boolean(data.is_final);
      const timestamp = data.timestamp || new Date().toISOString();

      if (!text && !isFinal) return;

      this.latestTranscript = {
        segment_id: segId,
        speaker: "Sales Rep",
        text: text,
        is_final: isFinal,
        status: data.status || (isFinal ? "final" : "partial"),
        timestamp: timestamp,
        latency_ms: data.latency_ms || 0,
      };

      const existingIdx = this.transcripts.findIndex(
        (t) => String(t.segment_id) === String(segId)
      );
      if (existingIdx >= 0) {
        if (this.transcripts[existingIdx].is_final && !isFinal) {
          // Do not overwrite an already finalized segment with a late partial
          return;
        }
        this.transcripts[existingIdx] = this.latestTranscript;
      } else {
        this.transcripts.push(this.latestTranscript);
      }

      // Live transcript indicator in recorder card
      if (this.elements.statusText && this.state === State.RECORDING) {
        const badge = isFinal ? "✓" : "💬";
        this.elements.statusText.textContent = `${badge} [Live]: "${text}"`;
      }

      this._notifyStreamlit();
    }

    _startAudioPipeline() {
      if (!this.mediaStream || !this.audioContext) return;

      const sourceSampleRate = this.audioContext.sampleRate;
      this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

      const bufferSize = 4096;
      this.processorNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

      this.processorNode.onaudioprocess = (e) => {
        if (!this.isAudioStreaming) return;

        const inputChannelData = e.inputBuffer.getChannelData(0);

        // Calculate live RMS volume for the visualizer meter
        let sum = 0;
        for (let i = 0; i < inputChannelData.length; i++) {
          sum += inputChannelData[i] * inputChannelData[i];
        }
        const rms = Math.sqrt(sum / inputChannelData.length);
        const levelPercent = Math.min(100, Math.round(rms * 400));
        if (this.elements.meterFill) {
          this.elements.meterFill.style.width = `${levelPercent}%`;
        }

        // Resample from hardware rate to 16000 Hz
        const resampled = AudioUtils.resampleAudio(
          inputChannelData,
          sourceSampleRate,
          TARGET_SAMPLE_RATE
        );

        // Accumulate ~250ms chunks
        if (this.accumulator) {
          this.accumulator.push(resampled);
        }
      };

      this.sourceNode.connect(this.processorNode);
      // Mute local speaker playback to prevent acoustic feedback loop
      this.muteNode = this.audioContext.createGain();
      this.muteNode.gain.value = 0;
      this.processorNode.connect(this.muteNode);
      this.muteNode.connect(this.audioContext.destination);
    }

    _handleOutgoingChunk(pcmBuffer) {
      if (!this.isAudioStreaming) return;
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

      try {
        this.ws.send(pcmBuffer);
        this.lastChunkSentTime = performance.now();
        this.chunksSent += 1;
        this.bytesSent += pcmBuffer.byteLength;
        // Update DOM metrics directly (fast, no Streamlit rerun)
        this._updateUI();
      } catch (err) {
        console.warn("[AudioRecorder] Error sending chunk:", err);
      }
    }

    _startStallDetector() {
      this._stopStallDetector();
      this.stallCheckInterval = setInterval(() => {
        if (this.state === State.RECORDING && this.isAudioStreaming) {
          const now = performance.now();
          const lastActivity = this.lastChunkSentTime || this.audioStreamingStartTime;
          if (lastActivity && (now - lastActivity) > AUDIO_STALL_TIMEOUT_MS) {
            console.warn("[AudioRecorder] Warning: No audio chunks emitted in 5s. Microphone may be muted or stalled.");
            if (this.elements.statusText) {
              this.elements.statusText.textContent = "Warning: Microphone input stalled (no audio for 5s).";
            }
          }
        }
      }, 1000);
    }

    _stopStallDetector() {
      if (this.stallCheckInterval) {
        clearInterval(this.stallCheckInterval);
        this.stallCheckInterval = null;
      }
    }

    _clearTimers() {
      if (this.connectTimer) {
        clearTimeout(this.connectTimer);
        this.connectTimer = null;
      }
      if (this.controlTimer) {
        clearTimeout(this.controlTimer);
        this.controlTimer = null;
      }
      if (this.shutdownTimer) {
        clearTimeout(this.shutdownTimer);
        this.shutdownTimer = null;
      }
    }

    /**
     * Stop the recording and session workflow.
     */
    stopRecording() {
      if (this.state !== State.RECORDING && this.state !== State.REQUESTING_PERMISSION) {
        return; // Idempotent check
      }

      this.stopCallTime = performance.now();
      if (this.audioStreamingStartTime) {
        this.recordingDurationMs = Math.round(this.stopCallTime - this.audioStreamingStartTime);
      }

      this._setState(State.STOPPING, "Stopping audio stream...");
      this.isAudioStreaming = false;
      this._stopStallDetector();
      this._clearTimers();

      // Flush remaining accumulator samples
      if (this.accumulator) {
        this.accumulator.flush(true);
      }

      // Stop audio hardware immediately
      this._stopAudioHardware();

      // Bounded shutdown timer: guarantees teardown completes within SHUTDOWN_TIMEOUT_MS even if backend network freezes
      this.shutdownTimer = setTimeout(() => {
        console.warn("[AudioRecorder] Shutdown acknowledgement timed out. Forcing teardown.");
        this._finishTeardown();
      }, SHUTDOWN_TIMEOUT_MS);

      // Send stop_audio request to backend
      if (this.ws && this.ws.readyState === WebSocket.OPEN && this.sessionId) {
        console.log("[AudioRecorder] Sending stop_audio for session:", this.sessionId);
        this.ws.send(
          JSON.stringify({
            type: "stop_audio",
            session_id: this.sessionId,
          })
        );
      } else {
        this._finishTeardown();
      }
    }

    /**
     * Complete reset of the recorder and state.
     */
    resetRecorder() {
      this._cleanupResources();
      this.chunksSent = 0;
      this.bytesSent = 0;
      this.sessionId = null;
      this.startCallTime = null;
      this.audioStreamingStartTime = null;
      this.stopCallTime = null;
      this.startupLatencyMs = 0;
      this.shutdownLatencyMs = 0;
      this.recordingDurationMs = 0;
      this.lastChunkSentTime = null;
      this.interruptionCount = 0;
      this.transcripts = [];
      this.latestTranscript = null;
      this._clearError();
      this._setState(State.IDLE, "Session reset. Ready to start.");
      this._notifyStreamlit("idle");
    }

    _stopAudioHardware() {
      if (this.processorNode) {
        this.processorNode.onaudioprocess = null;
        this.processorNode.disconnect();
        this.processorNode = null;
      }
      if (this.muteNode) {
        this.muteNode.disconnect();
        this.muteNode = null;
      }
      if (this.sourceNode) {
        this.sourceNode.disconnect();
        this.sourceNode = null;
      }

      if (this.mediaStream) {
        this.mediaStream.getTracks().forEach((track) => track.stop());
        this.mediaStream = null;
      }

      if (this.audioContext && this.audioContext.state !== "closed") {
        this.audioContext.close().catch(() => {});
        this.audioContext = null;
      }

      if (this.elements.meterFill) {
        this.elements.meterFill.style.width = "0%";
      }
    }

    _finishTeardown() {
      if (this.shutdownTimer) {
        clearTimeout(this.shutdownTimer);
        this.shutdownTimer = null;
      }
      if (this.stopCallTime && !this.shutdownLatencyMs) {
        this.shutdownLatencyMs = Math.round(performance.now() - this.stopCallTime);
      }
      this._cleanupResources();
      const finalMsg = `Call stopped cleanly. Sent ${this.chunksSent} chunks (${(this.bytesSent / 1024).toFixed(1)} KB) in ${(this.recordingDurationMs / 1000).toFixed(1)}s.`;
      this._setState(State.STOPPED, finalMsg);
      this._notifyStreamlit("stopped");
    }

    _cleanupResources() {
      this._clearTimers();
      this._stopStallDetector();
      this.isAudioStreaming = false;
      this._stopAudioHardware();

      if (this.accumulator) {
        this.accumulator.reset();
        this.accumulator = null;
      }

      if (this.ws) {
        this.ws.onopen = null;
        this.ws.onmessage = null;
        this.ws.onerror = null;
        this.ws.onclose = null;
        if (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING) {
          this.ws.close(1000, "Component cleanup");
        }
        this.ws = null;
      }

      if (this.elements.meterFill) {
        this.elements.meterFill.style.width = "0%";
      }
    }
  }

  // Initialize when DOM is ready
  document.addEventListener("DOMContentLoaded", () => {
    window.salesCoachAudioRecorder = new AudioStreamRecorder();
  });
})();
