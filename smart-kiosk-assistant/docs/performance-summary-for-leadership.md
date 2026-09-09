# Smart AI Kiosk — Voice Latency Performance Review

**Audience:** Leadership / Engineering Management
**Date:** 10 September 2026
**Scope:** `voice-enabled-interactions` (kiosk) and `edge-ai-libraries` (audio-analyzer, text-to-speech)

---

## 1. Executive summary

The kiosk lets a customer order by speaking. The metric that matters is
**voice-to-voice (V2V)**: the gap between the customer's last word and the
first sound coming out of the speaker. Anything above ~1.5 s feels like the
machine has ignored you.

| | Starting point | Today | Change |
|---|---:|---:|---:|
| **Voice-to-voice (live kiosk turn)** | ~3.9 s | **~1.1 s** | **−71%** |
| Speech recognition (ASR) | 881 ms | 445 ms | −49% |
| Time to first audio | ~1.5 s | ~0.95 s | −37% |
| Order accuracy / truthfulness | baseline | no regressions | maintained |

**One-line summary for the room:** a customer now hears the kiosk begin to
answer in about one second instead of about four, achieved without buying
hardware and without loosening any correctness guarantee.

Two things are worth stating up front because they are easy to get wrong in
a leadership conversation:

1. **We fixed the measurement as well as the system.** The dashboard used to
   report 1.6 s when the true figure was 1.9 s, because the clock stopped
   when a sentence was *queued* for speech synthesis rather than when audio
   actually existed. The 1.1 s quoted above is measured with the corrected
   metric, so it is comparable to a stopwatch.
2. **Not every improvement is a speed-up.** Roughly a third of the work was
   removing ways the system could tell a customer something untrue. Those
   changes are listed in §7 and are the ones most worth defending.

---

## 2. How a single order flows through the system

```
Customer speaks
   │
   ▼
[kiosk-core]  captures mic audio, detects end of speech (Silero VAD)
   │  streams 1.5 s chunks WHILE the customer is still talking
   ▼
[audio-analyzer]  Whisper ASR (OpenVINO INT8, Intel NPU) + speaker diarization
   │  returns transcript
   ▼
[rag-service]  Qwen3-4B (OpenVINO Model Server, GPU) decides what to do
   │  calls ordering tools on kiosk-core over MCP
   ▼
[text-to-speech]  Kokoro synthesizes the reply, sentence by sentence
   │
   ▼
Customer hears the answer
```

The critical insight behind most of the work: **only the last chunk of audio
is on the critical path.** Everything the customer said before they stopped
talking can be transcribed while they are still speaking. Much of the effort
went into making sure the system genuinely exploits that.

---

## 3. Unit economics — what each stage costs per unit of work

These are measured on our own hardware against the running stack, not
vendor claims. This is the section to use when someone asks "why is it
1 second and not 200 ms?"

### 3.1 LLM — Qwen3-4B int4 on OpenVINO Model Server (GPU)

| Measurement | Value |
|---|---|
| **Decode speed** | **27.5 ms per generated token** (≈ 36 tokens/sec) |
| Prefill (reading the prompt) | 0.139 ms per prompt token |
| Prompt size for an ordering turn | ~2,000 tokens → ~275 ms |
| Fixed request overhead | ~11 ms |

Verified linear across output lengths — 4, 26, 64 and 128 tokens all landed
between 27.5 and 29.9 ms/token, so the model's cost is fully predictable:

```
LLM time ≈ 275 ms (read the prompt) + 27.5 ms × (tokens generated)
```

**This is the single most important number in the deck.** It means every
token we can avoid generating is worth 27.5 ms, and it is why several of our
optimisations are about making the model say *less*, not run faster.

### 3.2 ASR — Whisper small, INT8, on Intel NPU

| Measurement | Value |
|---|---|
| Whisper inference itself | ~165 ms per 1.5 s chunk |
| Real-time factor | ~0.11 (transcribes 9× faster than real time) |
| Full round trip from kiosk-core | ~220 ms per chunk |
| Overhead outside the model | ~55 ms (was ~300 ms) |

### 3.3 TTS — Kokoro (character-based, not token-based)

| Text | Measured |
|---|---|
| `"Done."` (5 chars) | 250 ms |
| `"Got it. One Classic Chicken Burger."` (35 chars) | 508 ms |
| 63 chars | 918 ms |
| 93 chars | 1,489 ms |

```
TTS time ≈ 210 ms fixed + ~9-19 ms per character
```

The fixed 210 ms start-up cost per call is why we split the reply into
sentences and synthesize the *first, shortest* one immediately — covered in
§5.3.

---

## 4. Where the ~1.1 seconds actually goes

Measured on a live kiosk turn — the exact interaction to demo:

> **Customer:** "I would like to order one classic chicken burger"
> **Kiosk:** "Got it. One Classic Chicken Burger. Your total is now ₹169.
> Would you also like Classic French Fries (Regular) (₹89)?"

Trace `f90c1c6e`, `voice_to_voice_ms = 1135.5`:

| Stage | What happens | ms | Share |
|---|---|---:|---:|
| **1. ASR (final chunk)** | Last 0.45 s of speech → text | **183** | 16% |
| **2. LLM** | Decide + generate `<act>add\|Classic Chicken Burger\|1</act>Got it.` | **616** | 54% |
| **3. TTS** | Synthesize the 7-character "Got it." + write audio | **337** | 30% |
| | **Total voice-to-voice** | **1,135** | 100% |

### Reconciling the LLM's 616 ms against the unit costs

```
  275 ms   prefill (~2,000-token prompt)
+ 330 ms   12 tokens × 27.5 ms  ("<act>add|Classic Chicken Burger|1</act>Got it.")
─────────
  605 ms   predicted   vs.   616 ms measured
```

The model is **not** slow — it is doing the minimum work we could arrange for
it. Getting below ~600 ms requires either a smaller prompt or fewer tokens.

### What is deliberately NOT on the critical path

| Work | Cost | Why it's free |
|---|---:|---|
| First two ASR chunks | 249 + 216 ms | Ran while the customer was still speaking |
| `place_order` database write | ~103 ms | Overlaps LLM generation of the spoken words |
| Remaining 3 TTS segments | ~1,900 ms | Synthesized while "Got it." is already playing |
| Price + upsell text | 0 ms | Appended by code from the database, never generated |

This is why the raw "turn total" of ~3.1 s in the logs is **not** what the
customer experiences. Do not quote that number without this context.

---

## 5. What we changed — the four themes

### 5.1 Stop paying for work nobody uses *(this round — biggest ASR win)*

The kiosk sends ~3 audio chunks per utterance. Only the last one needs
speaker identification (working out which voice in a noisy food court is the
actual customer). We had always *intended* to skip that work on the earlier
"preview" chunks — kiosk-core sends a `diarization: false` flag to say so.

**The flag never worked.** The analyzer's HTTP endpoint had no matching
parameter, so the web framework silently discarded it, and every chunk paid
for full speaker diarization and voice enrollment. This had been true since
the feature was written.

We also found the analyzer launching **four separate helper programs per
request** (two `ffprobe`, two `ffmpeg`) just to answer questions like "how
long is this audio?" — which can be read directly from the file header in
microseconds.

| | Before | After |
|---|---:|---:|
| ASR total per turn | 881 ms | **445 ms** |
| Per chunk | ~440 ms | ~220 ms |
| Helper processes per request | 4 | 1 |
| Whisper itself | 165 ms | 165 ms (unchanged) |

**85% of ASR time was overhead, not AI.** Nothing about the model changed.

### 5.2 Make the model generate fewer tokens

At 27.5 ms/token, output length is latency. Three changes:

- **Directive mode.** Instead of the model producing a large structured
  function call and then waiting for a second round-trip to narrate the
  result, it emits one compact inline instruction followed immediately by the
  words to speak. Measured in isolation: first content at 137 ms vs 453 ms;
  full turn 670 ms vs 1,574 ms.
- **Tool description compaction.** Stripped `Returns:`/`Raises:` blocks from
  the tool documentation sent to the model on every single request — pure
  prompt weight that never influenced a decision.
- **Output cap** reduced from 192 to 128 tokens.

### 5.3 Start speaking sooner

TTS costs a fixed ~210 ms per call plus per-character time, and playback
cannot start until the first segment is finished. So we make the first
sentence deliberately tiny.

The model is instructed to open with **"Got it."** — 7 characters, ~250 ms —
instead of leading with a 35-character sentence (~510 ms). Everything after
it is synthesized in parallel while the customer is already hearing audio.
**Worth ~370 ms, and it costs nothing.**

### 5.4 Skip the second LLM call entirely

A tool-using turn normally needs two model round-trips: one to decide the
action, one to describe the result. The second is pure narration of data we
already hold — so for common outcomes we render the sentence directly from
the database result using authored templates.

**Saves a full ~2.7 s inference on those turns.** It is also *more* accurate
than the model: prices and item names are copied verbatim and cannot be
misquoted, dropped or renamed.

---

## 6. Changes in `edge-ai-libraries` (shared microservices)

| Service | Change | Effect |
|---|---|---|
| audio-analyzer | Per-request `diarization` flag plumbed end to end | −220 ms/chunk |
| audio-analyzer | New in-process WAV header reader; 4 subprocess spawns → 1 | −80 ms/request |
| audio-analyzer | Single-chunk short-circuit for short clips | avoids silence scan |
| text-to-speech | `TEXT_TO_SPEECH_WORKERS` — multiple worker processes | 2 parallel synths: 1.77 s → 1.18 s |

The TTS change is worth explaining: the speech engine holds a global lock
because its underlying phonemizer library is not thread-safe, so adding
threads achieves nothing. Separate OS processes each get their own copy and
give genuine concurrency. Defaults to 1 process, so it is opt-in per
deployment.

---

## 7. Correctness work — the part that matters most

Speed is worthless if the kiosk lies about an order. These were found and
fixed during this work; several were latent bugs that predate it.

| Risk | What could have happened | Status |
|---|---|---|
| **False order confirmation** | Model says "your order is confirmed" while the database still says draft — customer walks away, kitchen has nothing | Blocked: an order claim with no matching database write is replaced with an honest failure message |
| **Duplicate charges** | "Add a burger, remove the fries" — the add succeeds, the remove fails, the system retries and adds a **second** burger | Fixed: a turn that has already committed a write can no longer be retried |
| **Phantom confirmation** | A background "draft" prediction, run on a half-finished sentence, really confirming a customer's order mid-utterance | Fixed: writes forced into no-op mode; recovery path disabled for predictions |
| **"Your cart is empty"** | Told to a customer with a full cart, whenever a response failed to decode | Fixed: requires positive proof of an empty cart |
| **Wrong opening hours** | Model merged distinct weekday/weekend hours and invented "closed on public holidays" — at temperature 0 | Fixed: hours and restaurant name answered directly from source data, no model paraphrase |
| **Denial of service** | A malformed 32 KB audio upload could trigger ~9,000 subprocess launches | Fixed: audio length is bounded by real file size |

All 125 unit tests pass. Both repositories are committed on the
`performance-improvement` branch.

---

## 8. Where the remaining time is, and what's next

Today's 1,135 ms splits as **LLM 54% / TTS 30% / ASR 16%**. ASR is no longer
the bottleneck; the model is.

| Opportunity | Est. saving | Confidence | Notes |
|---|---:|---|---|
| Shrink the ~2,000-token prompt | 100–150 ms | High | Directly measurable at 0.139 ms/token |
| Pre-synthesize "Got it." once and cache | ~250 ms | High | It is a fixed string — same audio every time |
| Speculative drafting (built, off) | 200–400 ms | Medium | Starts the LLM on a partial transcript; needs the safety work in §7 verified under load |
| Smaller/faster model for ordering turns | 200–300 ms | Medium | Trade-off against language understanding |
| Disable audio denoising | ~40 ms | Low | Would enable a zero-copy path; needs accuracy A/B first |

A realistic near-term target is **~750–850 ms**. Below that requires
architectural change (single-process, no HTTP between stages), which the
reference prototype achieves at ~700 ms — but it gives up the service
isolation, independent scaling and shared-microservice reuse that this
architecture was chosen for. **That is a product decision, not a bug.**

---

## 9. Talking points if challenged

**"Why was it slow in the first place?"**
It wasn't the AI models. Whisper takes 165 ms and always did. The time was
in scaffolding around them — a flag that silently didn't work, helper
processes launched per request, and the model being asked to generate text we
already had in the database.

**"How do we know the 1.1 s is real?"**
We fixed the measurement first and proved it: two independently-computed
timers that used to disagree by 315 ms now report the identical figure.
The number is reconstructed from timestamped logs and reconciles with the
per-token unit costs to within 2%.

**"Did you trade accuracy for speed?"**
The opposite. The largest single change — rendering replies from database
values instead of asking the model to describe them — is both faster and
strictly more faithful, because prices and item names are copied rather than
regenerated. Six separate ways the kiosk could have made a false statement
were closed during this work.

**"What did it cost?"**
No new hardware. No new licences. The changes are configuration, removed
redundant work, and better use of the Intel NPU/GPU already in the box.
