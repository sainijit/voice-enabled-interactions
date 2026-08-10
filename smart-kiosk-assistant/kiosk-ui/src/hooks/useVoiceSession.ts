import { useCallback, useEffect, useRef, useState } from 'react';
import { tuning } from '../constants';
import {
  endAudioStream,
  pollSession,
  pushAudioChunk,
  startStreamSession,
  ttsAudioUrl,
} from '../api/kioskApi';
import { concatFloat32, encodeWav, resampleLinear } from '../api/audioUtils';
import { useAudioQueue } from './useAudioQueue';
import type { ChatMessage, HistoryTurn, VoicePhase } from '../types';

interface UseVoiceSessionOptions {
  deviceId: string;
  enabled: boolean; // false while a knowledge-base ingest is in progress
  onTurnComplete?: () => void;
}

const TARGET_RATE = tuning.sampleRate; // 16000

export function useVoiceSession({ deviceId, enabled, onTurnComplete }: UseVoiceSessionOptions) {
  const [phase, setPhase] = useState<VoicePhase>('idle');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [partialUser, setPartialUser] = useState('');
  const [partialAssistant, setPartialAssistant] = useState('');
  const [statusText, setStatusText] = useState('Tap the mic and ask a question');
  const [error, setError] = useState<string | null>(null);
  // Hands-free "continuous conversation" mode: once on, a completed turn
  // automatically re-arms the mic instead of returning to idle. Exposed as
  // state (not just a ref) purely so the UI can show "listening…"/"End
  // conversation" instead of the push-to-talk labels.
  const [conversationMode, setConversationMode] = useState(false);
  const conversationModeRef = useRef(false);
  // How many consecutive turns in a row produced no speech at all — guards
  // against an empty room looping forever on ambient noise/silence.
  const noSpeechStreakRef = useRef(0);

  // `start` is defined further down (it needs `audioQueue`, `pollLoop`, etc.)
  // but `audioQueue`'s onAllDone callback needs to call it to auto-resume
  // listening once TTS finishes. A ref breaks that circular dependency.
  const startRef = useRef<(() => Promise<void>) | null>(null);

  const audioQueue = useAudioQueue({
    onAllDone: () => {
      // After the assistant finishes speaking, allow a KPI/order refresh.
      onTurnComplete?.();
      // Hands-free loop: the mic was torn down before TTS started (see the
      // completion branch in pollLoop), so there is no echo risk in resuming
      // now that playback has finished.
      if (conversationModeRef.current) {
        void startRef.current?.();
      }
    },
  });

  // ── Mutable streaming refs ────────────────────────────────────────────────
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const framesRef = useRef<Float32Array[]>([]);
  const ctxRateRef = useRef<number>(48000);
  const sessionIdRef = useRef<string | null>(null);
  // Persistent conversation ID — generated once per hook mount and reused
  // across all voice turns so the agent retains order context (cart, order_id).
  const conversationIdRef = useRef<string>(crypto.randomUUID());
  const recordingRef = useRef(false);
  const eosRef = useRef(false);
  const pollTimerRef = useRef<number | null>(null);
  const messagesRef = useRef<ChatMessage[]>([]);
  messagesRef.current = messages;

  const buildHistory = useCallback((): HistoryTurn[] => {
    const recent = messagesRef.current.slice(-tuning.maxHistoryTurns);
    return recent
      .filter((m) => m.text.trim())
      .map((m) => ({ role: m.role, content: m.text }));
  }, []);

  const flushChunk = useCallback(async (force = false) => {
    // Nothing can be sent until the session exists. Crucially we must NOT
    // drain framesRef before knowing that: audio recorded between the mic
    // going live and startStreamSession resolving is still real speech, and
    // clearing it here used to destroy the opening words of the sentence.
    const sid = sessionIdRef.current;
    if (!sid) return;

    const frames = framesRef.current;
    if (frames.length === 0) return;
    const total = frames.reduce((acc, f) => acc + f.length, 0);
    const haveSeconds = total / ctxRateRef.current;
    if (!force && haveSeconds < tuning.chunkSeconds) return;

    framesRef.current = [];
    const merged = concatFloat32(frames);
    const resampled = resampleLinear(merged, ctxRateRef.current, TARGET_RATE);
    const wav = encodeWav(resampled, TARGET_RATE);

    try {
      await pushAudioChunk(sid, wav);
    } catch {
      // A transient push failure must not silently swallow several seconds of
      // speech. Put the audio back at the head of the buffer so the next flush
      // retries it; ordering is preserved because frames captured meanwhile
      // were appended after this batch was taken.
      framesRef.current = [merged, ...framesRef.current];
    }
  }, []);

  const teardownCapture = useCallback(() => {
    recordingRef.current = false;
    try {
      workletRef.current?.disconnect();
      sourceRef.current?.disconnect();
    } catch {
      /* ignore */
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    if (ctxRef.current && ctxRef.current.state !== 'closed') {
      ctxRef.current.close().catch(() => undefined);
    }
    workletRef.current = null;
    sourceRef.current = null;
    streamRef.current = null;
    ctxRef.current = null;
  }, []);

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current !== null) {
      window.clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  // ── Single poll loop: drives partial transcript, response, TTS, completion ──
  const pollLoop = useCallback(async () => {
    const sid = sessionIdRef.current;
    if (!sid) return;

    let snapshot;
    try {
      snapshot = await pollSession(sid);
    } catch {
      pollTimerRef.current = window.setTimeout(pollLoop, tuning.pollIntervalMs);
      return;
    }

    const transcript = (snapshot.transcript ?? '').trim();
    const response = (snapshot.response ?? '').trim();
    const running = snapshot.status === 'running' || snapshot.status === 'stopping';

    if (transcript) setPartialUser(transcript);
    if (response) setPartialAssistant(response);

    // Enqueue any new TTS audio segments.
    const segs = snapshot.tts_audio_segments ?? [];
    if (segs.length > 0) {
      const urls = segs.map((s) => ttsAudioUrl(sid, String(s.audio_file)));
      audioQueue.enqueue(urls);
    }

    // Status text mirrors the Gradio state machine.
    if (eosRef.current || (conversationModeRef.current && !running)) {
      if (segs.length) setStatusText(`🔊 Speaking… (${segs.length})`);
      else if (response) setStatusText('💬 Generating response…');
      else if (transcript) setStatusText('📝 Querying knowledge base…');
      else setStatusText('⏳ Processing speech…');
    }

    // Completion: either the frontend explicitly ended the stream (manual
    // push-to-talk), or the backend's own silence-timeout VAD ended the turn
    // on its own (hands-free conversation mode never calls endAudioStream —
    // see startConversation). Either way, once the session is no longer
    // running the turn is done and must be finalised the same way.
    if (!running && (eosRef.current || conversationModeRef.current)) {
      stopPolling();
      // The backend may have ended this turn on its own (silence-timeout VAD)
      // without the frontend ever calling `stop()` — the mic/AudioContext/
      // worklet are still live and `recordingRef` is still true in that case.
      // Tear them down now so the next `start()` (manual or auto-resume) is
      // not silently ignored by its `recordingRef.current` guard, and so the
      // browser doesn't keep sending frames into a session that's already gone.
      recordingRef.current = false;
      teardownCapture();
      const finalTranscript = transcript;
      const finalResponse = response;
      const hadSpeech = Boolean(finalTranscript);
      setMessages((prev) => {
        const next = [...prev];
        if (finalTranscript) next.push({ role: 'user', text: finalTranscript });
        if (finalResponse) next.push({ role: 'assistant', text: finalResponse });
        return next;
      });
      setPartialUser('');
      setPartialAssistant('');
      sessionIdRef.current = null;
      eosRef.current = false;
      setPhase('idle');

      if (conversationModeRef.current) {
        // Guard against looping forever on an empty room: if several turns in
        // a row captured no speech at all, drop out of conversation mode
        // instead of silently polling/re-listening indefinitely.
        noSpeechStreakRef.current = hadSpeech ? 0 : noSpeechStreakRef.current + 1;
        if (noSpeechStreakRef.current >= 3) {
          conversationModeRef.current = false;
          setConversationMode(false);
          setStatusText('No speech detected — conversation ended. Tap to start again.');
        } else if (segs.length > 0) {
          // Wait for TTS playback — resumed from audioQueue's onAllDone so the
          // mic never re-arms while the kiosk's own voice is still playing.
          setStatusText('🔊 Speaking…');
        } else {
          // Nothing to play back (e.g. a silent/no-speech turn) — nothing will
          // fire onAllDone, so re-arm the mic directly after a short pause.
          setStatusText(hadSpeech ? '💬 …' : '🎧 Listening…');
          window.setTimeout(() => {
            if (conversationModeRef.current) void startRef.current?.();
          }, 300);
        }
      } else {
        setStatusText('✓ Done — tap 🎤 for another question');
      }
      // onTurnComplete also fires when TTS finishes (onAllDone); fire here too
      // in case there was no audio to play.
      if (segs.length === 0) onTurnComplete?.();
      return;
    }

    pollTimerRef.current = window.setTimeout(pollLoop, tuning.pollIntervalMs);
  }, [audioQueue, onTurnComplete, stopPolling, teardownCapture]);

  // ── Public: start recording ───────────────────────────────────────────────
  const start = useCallback(async () => {
    if (!enabled) {
      setStatusText('⏳ Ingestion in progress — please wait…');
      return;
    }
    if (recordingRef.current || phase !== 'idle') return;
    setError(null);
    audioQueue.reset();
    framesRef.current = [];
    eosRef.current = false;
    setPartialUser('🎤 Listening…');
    setPartialAssistant('');
    setPhase('listening');
    // Do NOT invite speech yet — the mic permission, AudioContext, worklet
    // module fetch and the session POST all still have to complete. Prompting
    // here made customers start talking before capture existed.
    setStatusText('🎙 Preparing microphone…');

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Microphone access requires HTTPS or localhost.');
      }
      const constraints: MediaStreamConstraints = {
        audio: deviceId
          ? {
              deviceId: { exact: deviceId },
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
            }
          : { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      const ctx = new AudioContext();
      ctxRef.current = ctx;
      ctxRateRef.current = ctx.sampleRate;
      await ctx.audioWorklet.addModule('/pcm-capture-processor.js');

      const source = ctx.createMediaStreamSource(stream);
      sourceRef.current = source;
      const worklet = new AudioWorkletNode(ctx, 'pcm-capture-processor');
      workletRef.current = worklet;

      worklet.port.onmessage = (ev: MessageEvent<Float32Array>) => {
        if (!recordingRef.current) return;
        framesRef.current.push(ev.data);
        // Push-to-talk uploads nothing mid-utterance: the whole recording is
        // sent as ONE chunk when Stop is pressed (see flushChunk(true) in
        // stop()). Streaming it in 2.5s slices meant the backend re-split the
        // audio at fixed boundaries and words straddling a cut were lost
        // ("Aloo Tikki Burger" decoded as "and 2,000"). One uncut chunk gives
        // Whisper the entire utterance and its full context.
        //
        // Hands-free conversation mode keeps streaming: it has no Stop button
        // and depends on the backend's silence endpointing to end a turn, so
        // withholding audio until "stop" would mean the turn never ends.
        if (conversationModeRef.current) void flushChunk(false);
      };

      source.connect(worklet);
      // Connect to destination so the worklet's process() runs; output is silent.
      worklet.connect(ctx.destination);
      // Arm capture before opening the session so nothing spoken during the
      // session round-trip is lost. flushChunk retains frames until the
      // session id exists, so this buffering is safe.
      recordingRef.current = true;

      const { session_id } = await startStreamSession(
        TARGET_RATE,
        buildHistory(),
        conversationIdRef.current,
        !conversationModeRef.current,
      );
      sessionIdRef.current = session_id;

      // Everything is live — only now is it honest to ask for speech.
      setStatusText('🎙 Listening — speak now');

      // Begin the poll loop (partial transcript while listening).
      stopPolling();
      pollTimerRef.current = window.setTimeout(pollLoop, tuning.pollIntervalMs);
    } catch (err) {
      teardownCapture();
      setPhase('idle');
      setPartialUser('');
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setStatusText(`❌ ${msg}`);
    }
  }, [
    enabled,
    phase,
    deviceId,
    audioQueue,
    buildHistory,
    flushChunk,
    pollLoop,
    stopPolling,
    teardownCapture,
  ]);

  // ── Public: stop recording → finalise ─────────────────────────────────────
  const stop = useCallback(async () => {
    if (!recordingRef.current) return;
    recordingRef.current = false;
    setPhase('processing');
    setStatusText('⏳ Processing…');
    setPartialUser((p) => (p === '🎤 Listening…' ? '⏳ Processing…' : p));

    // Flush remaining audio, then signal end-of-stream.
    await flushChunk(true);
    teardownCapture();

    const sid = sessionIdRef.current;
    if (!sid) {
      setPhase('idle');
      setStatusText('No audio — try again');
      setPartialUser('');
      return;
    }
    try {
      await endAudioStream(sid);
      eosRef.current = true;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setStatusText(`❌ ${msg}`);
      setPhase('idle');
      return;
    }
    // The poll loop (already running) will detect completion.
  }, [flushChunk, teardownCapture]);

  // startRef must stay pointed at the latest `start` closure so the
  // conversation-mode auto-resume paths (audioQueue.onAllDone, the no-TTS
  // setTimeout above) always call a `start` that sees the current `phase`.
  startRef.current = start;

  // ── Public: hands-free conversation mode ──────────────────────────────────
  // Turns the mic button into a "start conversation" control: one tap begins
  // listening, and every subsequent turn (silence-timeout end → reply → TTS)
  // automatically re-arms the mic with no further taps, until endConversation
  // is called. The mic is fully torn down before each reply is spoken (see
  // `stop`/finalisation above) and only re-created after playback finishes,
  // so there is no acoustic echo risk from the kiosk hearing its own voice.
  const startConversation = useCallback(() => {
    conversationModeRef.current = true;
    setConversationMode(true);
    noSpeechStreakRef.current = 0;
    void start();
  }, [start]);

  const endConversation = useCallback(() => {
    conversationModeRef.current = false;
    setConversationMode(false);
    // Barge-in may have left TTS playing (see interruptSpeaking); a full exit
    // should always fall silent immediately rather than let the reply finish.
    audioQueue.stop();
    if (recordingRef.current) {
      // Finalise the in-flight utterance normally (customer may be mid-order);
      // it just won't loop again once the reply/TTS for this turn are done.
      void stop();
    } else if (phase === 'idle') {
      setStatusText('Tap the mic and ask a question');
    }
  }, [audioQueue, phase, stop]);

  // ── Public: barge-in — stop the kiosk speaking and listen immediately ─────
  // Lets the customer interrupt a reply mid-playback ("stop, I want to ask
  // something else") without leaving conversation mode. This is a manual tap
  // gesture, not open-mic-during-playback: the mic stays off until this
  // fires, so there is no risk of the kiosk hearing its own voice. Silences
  // the current + any queued TTS segments, then re-arms the mic right away
  // instead of waiting for onAllDone (which audioQueue.stop() deliberately
  // does not fire, to avoid a double-resume race with this call).
  const interruptSpeaking = useCallback(() => {
    audioQueue.stop();
    if (!recordingRef.current && phase === 'idle') {
      void startRef.current?.();
    }
  }, [audioQueue, phase]);

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      conversationModeRef.current = false;
      stopPolling();
      teardownCapture();
      audioQueue.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    phase,
    messages,
    partialUser,
    partialAssistant,
    statusText,
    error,
    playbackState: audioQueue.state,
    conversationMode,
    start,
    stop,
    startConversation,
    endConversation,
    interruptSpeaking,
  };
}
