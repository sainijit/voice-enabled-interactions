import type { ChatMessage, TtsPlaybackState, VoicePhase } from '../../types';

/**
 * Spoken examples shown under the Ask button while the kiosk is idle.
 *
 * These do double duty: they tell a first-time customer that the kiosk
 * understands natural speech (not just keywords), and they advertise the
 * capabilities that are otherwise invisible on a voice-only screen — browsing,
 * ordering, removing an item, and asking a knowledge-base question.
 * They are deliberately NOT buttons: this screen has no text input path, so a
 * tappable chip would promise an interaction the kiosk cannot honour.
 */
const EXAMPLE_PROMPTS = [
  'What burgers do you have?',
  "I'd like a chicken burger and fries",
  'Remove the fries from my cart',
  'Are any items vegetarian?',
  "That's all, confirm my order",
];

interface AskBarProps {
  phase: VoicePhase;
  playbackState: TtsPlaybackState;
  statusText: string;
  partialUser: string;
  partialAssistant: string;
  /** Full turn history; only the most recent exchange is shown above the button. */
  messages: ChatMessage[];
  onStart: () => void;
  onStop: () => void;
}

/**
 * Full-width voice control docked to the bottom of the customer kiosk screen.
 * This is the sole ordering input on this screen (menu/cart are display-only):
 * tap to start speaking, tap again to submit. Shows the most recent exchange
 * (or the live partial transcript/response) directly above the button so the
 * customer gets visual confirmation of what was heard and said back, without
 * needing a full chat transcript.
 */
export function AskBar({
  phase,
  playbackState,
  statusText,
  partialUser,
  partialAssistant,
  messages,
  onStart,
  onStop,
}: AskBarProps) {
  const recording = phase === 'listening';
  const processing = phase === 'processing';
  const speaking = playbackState !== 'idle';
  const idle = !recording && !processing && !speaking;

  const lastUser = [...messages].reverse().find((m) => m.role === 'user')?.text;
  const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant')?.text;

  const showPartialUser = (recording || processing) && !!partialUser;
  const showPartialAssistant = processing && !!partialAssistant;

  const displayUser = showPartialUser ? partialUser : lastUser;
  const displayAssistant = showPartialAssistant ? partialAssistant : lastAssistant;

  const handleClick = () => {
    if (processing) return;
    if (recording) onStop();
    else onStart();
  };

  const buttonClass = recording
    ? 'bg-red-500 text-white kiosk-pulse-recording hover:bg-red-600 focus:ring-red-500/30'
    : processing
      ? 'bg-amber-500 text-white cursor-wait focus:ring-amber-500/30'
      : 'bg-intel-blue text-white hover:bg-intel-blue-dark hover:scale-[1.02] focus:ring-intel-blue/30 active:scale-95';

  // Suggestions are shown only when the kiosk is genuinely idle and the
  // customer has not spoken yet — once a conversation is under way they are
  // noise, and during listening/processing they compete for attention.
  const showExamples = idle && messages.length === 0;

  return (
    <div className="shrink-0 border-t border-gray-200 bg-white px-6 py-4">
      {showExamples && (
        <div className="mx-auto mb-3 max-w-3xl">
          <p className="mb-2 text-center text-xs font-medium uppercase tracking-wider text-gray-400">
            Try saying
          </p>
          <ul className="flex flex-wrap items-center justify-center gap-2">
            {EXAMPLE_PROMPTS.map((example) => (
              <li
                key={example}
                className="rounded-full border border-intel-blue/20 bg-intel-blue/5 px-4 py-1.5 text-sm text-intel-blue"
              >
                “{example}”
              </li>
            ))}
          </ul>
        </div>
      )}

      {(displayUser || displayAssistant) && (
        <div className="mx-auto mb-3 max-w-3xl space-y-1.5 text-center">
          {displayUser && (
            <p className="truncate text-sm text-gray-500">
              🗣️ <span className={showPartialUser ? 'italic' : ''}>{displayUser}</span>
            </p>
          )}
          {displayAssistant && (
            <p className="truncate text-sm font-medium text-intel-dark">
              🤖 <span className={showPartialAssistant ? 'italic' : ''}>{displayAssistant}</span>
            </p>
          )}
        </div>
      )}

      <div className="flex flex-col items-center gap-2">
        <button
          type="button"
          onClick={handleClick}
          disabled={processing}
          aria-pressed={recording}
          aria-label={recording ? 'Stop and submit your question' : 'Tap to ask a question'}
          className={`relative flex w-full max-w-2xl items-center justify-center gap-3 overflow-hidden rounded-2xl px-8 py-5 text-xl font-bold shadow-lg transition-all duration-200 focus:outline-none focus:ring-4 ${buttonClass}`}
        >
          {processing ? (
            /* Three sequential dots: motion without rotation, so the control
               reads as "working" rather than "stuck in a loading spinner". */
            <span className="flex h-8 w-8 items-center justify-center gap-1" aria-hidden="true">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  className="h-1.5 w-1.5 rounded-full bg-current animate-thinking-dot"
                  style={{ animationDelay: `${i * 0.16}s` }}
                />
              ))}
            </span>
          ) : (
            <svg className="h-8 w-8" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              {recording ? (
                <rect x="6" y="6" width="12" height="12" rx="2" />
              ) : (
                <>
                  <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
                  <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
                </>
              )}
            </svg>
          )}
          <span>{recording ? 'Stop' : processing ? 'Processing…' : speaking ? 'Speaking…' : 'Ask'}</span>
          {processing && (
            /* Indeterminate sweep along the bottom edge: conveys forward
               progress for an operation whose duration we cannot predict. */
            <span
              className="pointer-events-none absolute bottom-0 left-0 h-1 w-1/4 rounded-full bg-white/70 animate-progress-sweep"
              aria-hidden="true"
            />
          )}
        </button>
        <p className="min-h-[1.25rem] text-xs text-gray-400" role="status" aria-live="polite">
          {statusText}
        </p>
      </div>
    </div>
  );
}

export default AskBar;
