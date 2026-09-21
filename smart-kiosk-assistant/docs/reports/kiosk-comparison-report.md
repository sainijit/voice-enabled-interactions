# Smart AI Kiosk vs QSR-Lab — Voice Response Speed

**Prepared by:** Engineering — Smart AI Kiosk
**Date:** 2026-09-16
**Audience:** BU leadership

**How to read this:** "voice-to-voice" means the gap between the customer
finishing their sentence and the kiosk starting to speak. Lower is better.
Anything under about one second feels like a normal conversation.

Two systems are compared:

- **Smart AI Kiosk** — our product. Containerised, deployable, runs a real
  ordering flow that writes to a live cart.
- **QSR-Lab** — the research prototype (`kiosk-voice-lab-main`). A single
  Python program used to test ideas quickly. Not a product.

QSR-Lab is where several of our best ideas came from. This document shows what
we took from it, what we changed, and where we now stand.

---

## 1. Headline result

| | **Smart AI Kiosk** | **QSR-Lab** |
|---|---|---|
| Voice-to-voice (typical) | **754 ms** | 1037 ms |
| Test set | 12 runs, 1 order phrase | 75 turns, 80 different questions |
| Task | Taking an order, writing to a live cart | Answering retail questions |

**The Smart AI Kiosk replies in about 0.75 seconds. QSR-Lab takes about
1.04 seconds. We are roughly 280 ms faster — about 27%.**

Both systems were measured today, on the same machine, on the same day, with
the other system completely shut down.

### The one caveat to state up front

The two benchmarks do not ask the question in quite the same way.

QSR-Lab has to *work out* when the customer stopped talking. It waits for half a
second of silence to be sure. Our benchmark is *told* when the customer stopped.

That difference is worth a few hundred milliseconds, and it flatters us. So the
fair way to describe the result is:

> Up to the moment the reply starts being written, the two systems are close
> (542 ms vs 512 ms — QSR-Lab is actually slightly ahead). **Our entire
> advantage comes from what happens after that**: we start speaking in 212 ms
> where QSR-Lab needs 522 ms.

That second half is genuine engineering, and it is the story worth telling.

---

## 2. Where the time goes

Every turn has two halves. Splitting them shows exactly who wins where.

| Half of the turn | Smart AI Kiosk | QSR-Lab | Who is faster |
|---|---|---|---|
| **1. Listening** — customer stops talking → reply ready to be written | 542 ms | **512 ms** | QSR-Lab, by 30 ms |
| **2. Speaking** — reply ready → first sound comes out | **212 ms** | 522 ms | **Smart AI Kiosk, by 310 ms** |
| **Total** | **754 ms** | 1037 ms | **Smart AI Kiosk, by 283 ms** |

*(QSR-Lab's two halves add to 1034 rather than 1037 — these are typical values,
and typical values never sum exactly. The 3 ms difference is not meaningful.)*

Read that as: *we are level on listening, and about 2.5× faster on speaking.*

### Why we win the second half

Turning text into speech normally takes time. QSR-Lab needs **426 ms** to
produce its first spoken phrase.

QSR-Lab tries to hide this by guessing the opening phrase early and preparing
the audio in advance. Good idea — but the guess is only right **21% of the
time** (16 turns out of 75). The other ~79% of the time the customer waits the
full 426 ms.

We solved the same problem a different way: **we reuse the audio we already
made.** The detail matters, so it is worth walking through.

#### Step 1 — we force every reply to start the same way

The instruction we give the model requires it to *"begin with a TWO-WORD
acknowledgement sentence ending in a full stop"* — *"Got it."*, *"Sure thing."*,
*"All set."* — before it says anything else.

This is not a cosmetic choice. It exists so that the first thing the customer
hears is drawn from a **tiny, predictable set of phrases** rather than being
different every time. (The idea came from QSR-Lab, which measured that a
seven-word opener costs ~340 ms to speak while three words costs about 80 ms.)

#### Step 2 — the first time we say it, we keep the audio

When the system speaks *"Got it."* for the first time after starting up, it
synthesises it normally and then **keeps a copy of the finished sound file**.

The important property: **we never generate audio speculatively.** We only ever
retain something a real customer turn had already produced. So the cache cannot
slow anything down — worst case, it simply isn't there and we do what we did
before.

#### Step 3 — every later turn replays that file

On the next turn the system checks whether the sentence it is about to speak is
already in the cache. *"Got it."* is, so instead of calling the speech engine it
**copies the file** — which takes about 1.7 ms.

#### What we allow into the cache, and why

Only phrases that will genuinely recur:

| Rule | Reason |
|---|---|
| **24 characters or shorter** | Openers are short. Long sentences are order-specific and won't repeat. |
| **No digits** | *"Your total is now ₹169."* is short enough to qualify, but the next order has a different total. Without this rule the cache fills with dead price variants and stops accepting real openers. |
| **Maximum 32 phrases** | A small, bounded amount of memory. The handful of real openers are learned within the first few turns. |

#### How we guarantee the right audio is played

The cache is keyed not just on the words, but on **the words plus the voice,
model, language and speaking-style settings**. If a session asks for a different
voice, the key is different and it will not match — so a clip made for one voice
can never be replayed in another. Text is matched after normalising case,
spacing and punctuation, so *"Got it."* and *"got it"* are treated as the same
phrase.

The stored file is also the **finished** article: it has already been trimmed
and volume-adjusted exactly as a fresh one would be. Replaying it is
indistinguishable from synthesising it again.

#### So why is the cost "about zero"?

It is not literally zero — it is **1.7 ms**, the time to copy a ~0.7-second
audio file. That is what "effectively free" means here.

For comparison, on this same system **synthesising that same phrase takes
335 ms** — measured on the first turn after startup, before the cache is
populated.

| | Time to produce *"Got it."* |
|---|---|
| Speech engine (first turn after startup) | **335 ms** |
| Cache replay (every turn after that) | **1.7 ms** |

**So the cache removes about 333 ms from every turn.**

The evidence that speech synthesis has left the customer's wait entirely:

- Time until the model has written the first sentence: **210 ms**
- Time until sound is ready to play: **212 ms**

The gap between those two numbers *is* the whole cost of speaking. It is 1.7 ms
(measured across 12 consecutive turns: 1.5 – 3.5 ms).

**And unlike QSR-Lab's 21%, this works on every turn** — verified over 12
consecutive turns: 21 cache hits, 0 failures.

### Why QSR-Lab wins the first half

QSR-Lab transcribes speech continuously while the customer is still talking, so
by the time it decides the turn is over, the transcript is usually already done.

We also transcribe continuously during speech — and we use QSR-Lab's own fast
model (`distil-small.en` on the GPU) to do it. The difference is what happens at
the end: those running transcripts are used only to judge *when the customer has
finished*. For the words we actually act on, we run one more, more accurate pass
after the turn ends, which costs us about **260 ms**.

Two honest points about this:

- The gap is **only 30 ms overall**, not 260 ms. QSR-Lab's transcription is not
  free — it is simply hidden underneath the half-second it spends waiting for
  silence. It is doing the same work, just earlier.
- We cannot simply copy QSR-Lab's approach today, because it depends on a smaller,
  faster speech model that is **broken on our hardware**. See §6.

---

## 3. What we took from QSR-Lab

This is the collaboration working as intended. Most of our speed foundation came
from QSR-Lab's research.

| # | What we adopted | Why it matters |
|---|---|---|
| 1 | **Which chip runs which model** — language model on GPU, final speech recognition on NPU, speech synthesis on CPU (plus QSR-Lab's own fast `distil-small.en` on GPU for our end-of-speech check) | The single biggest factor in overall speed |
| 2 | **Keep speech synthesis on the CPU** — QSR-Lab found the GPU is the wrong choice here. We confirmed it independently: 8.7 seconds on GPU versus 0.92 seconds on CPU | Avoided a severe regression |
| 3 | **Reuse the repeated part of the prompt** instead of reprocessing it every turn | Cuts thinking time on every single turn |
| 4 | **Keep the opening phrase very short** — QSR-Lab measured that a 7-word opener costs ~340 ms to speak, while 3 words costs about 80 ms | Directly led to our two-word *"Got it."* |
| 5 | **Start speaking the first sentence while the rest is still being written** | The customer hears something much sooner |
| 6 | **One model call per order instead of two** — the model writes the reply and the cart instruction together | Removes a full round trip from every order |
| 7 | **Judge whether the customer actually finished** rather than waiting a fixed time | Fewer interruptions |
| 8 | **Transcribe while the customer is still speaking** | Less work left at the end |

---

## 4. Where we improved on QSR-Lab

### 4.1 A guaranteed opening phrase instead of a guess — worth about 280 ms

QSR-Lab guesses the opening phrase in advance; the guess lands 21% of the time.
We store the finished audio and replay it, which works essentially every time.

**We could not simply copy QSR-Lab's method — in our system it actively makes
things worse.** Our speech and language services handle one request at a time.
Guessing ahead means sending extra work that then sits in the queue *in front of*
the real customer request. When we tested it, a turn that normally takes about
5.8 seconds took **38 seconds**. That feature is deliberately switched off.

Our cache avoids the problem entirely because **it never creates extra work** —
it only keeps a copy of audio the system had already produced for a real
customer. If the phrase is not in the cache, we simply do what we did before.

> **The judgement call:** QSR-Lab's *goal* — don't make the customer wait for
> the opening phrase — was right. Their *method* was wrong for our architecture.
> We kept the goal and replaced the method.

### 4.2 Cart instruction at the end of the reply, not the beginning

QSR-Lab puts the machine-readable cart instruction at the *start* of the reply.
That suits them because it is hidden inside a pre-prepared guess.

We do not guess, so putting it first would mean the model writes machine text
before it writes anything speakable — delaying the first sound. **We require it
at the very end**, so *"Got it."* is ready within the first few words.

### 4.3 We carry production weight that QSR-Lab does not

QSR-Lab is one program on one machine. The Smart AI Kiosk is a set of separate
deployable services with health checks, APIs and independent scaling.

**We are faster while also paying the overhead of being a real distributed
product** — roughly 2 ms per service hop. This is the point most worth making to
leadership: this is not a lab number, it is a product number.

### 4.4 Smaller fixes found during this work

- **Removed a redundant network redirect** on every cart call (each one was
  silently taking two round trips instead of one).
- **Turned off the model's internal "thinking" output**, which was delaying
  speakable words.
- **Reused the connection to the language model** instead of opening a new one
  each turn — this alone had been costing about 390 ms per turn.
- **Fixed a bug that silently dropped the opening phrase** — see §7.

---

## 5. What produced the 754 ms

Split by type, because these are not equally creditable.

### Genuine engineering improvements

| # | Decision | Effect |
|---|---|---|
| 1 | Replay cached opening-phrase audio instead of guessing | **−280 ms** end-to-end (removes ~333 ms of synthesis; the two differ because some of it overlapped other work) |
| 2 | One model call per order instead of two | Removes a round trip |
| 3 | Cart instruction at the end of the reply | Protects time-to-first-sound |
| 4 | Keep speculative pre-generation switched **off** | Avoided a ~32 s regression |
| 5 | Don't cache phrases containing numbers (e.g. prices) | Keeps the cache useful |
| 6 | Removed the redundant network redirect | −13 ms |
| 7 | Keep speech synthesis on CPU | Avoided a ~7.8 s regression |

### Environment fixes — recovered lost ground, not new gains

| # | Action | Effect |
|---|---|---|
| 8 | Stopped two unrelated services that had been left running and were consuming most of the machine's CPU | **−900 ms** |

This one is large, and it will be asked about. It is **not** a speed improvement
we invented — those services should never have been running. We were measuring a
degraded machine. The lesson is operational: always check machine load before
trusting a measurement.

### Measurement corrections — the number, not the product

| # | Change | Effect |
|---|---|---|
| 9 | Reduced artificial padding in the test harness | **−170 ms** |

The product did not get 170 ms faster; our *measurement* became 170 ms more
honest. A real customer never paid this cost.

**Summary: the cache (−280 ms) is the real engineering win. The −900 ms was
repairing the environment and the −170 ms was correcting the measurement.**

---

## 6. What we are *not* claiming

1. **The two tests are not identical.** QSR-Lab answers retail questions; we take
   orders. QSR-Lab has an ordering mode, but its benchmark does not use it, so no
   directly comparable QSR-Lab ordering figure exists.
2. **Different sample sizes.** 75 QSR-Lab turns across 80 questions versus 12 runs
   of one phrase for us. Our figure is precise *for that phrase*, not a broad
   distribution across varied speech.
3. **We do not claim to be more consistent.** Our results vary less, but that is
   because we repeat one sentence while QSR-Lab uses 80 different ones. Any
   system looks steady when given the same input repeatedly.
4. **The end-of-speech detection is not matched** (§1). We do not know what
   QSR-Lab would score under our settings, and we deliberately have not guessed.
5. **Hands-free mode is slower than this headline.** 754 ms reflects
   press-to-talk. In hands-free the end-of-speech shortcut is not firing and
   falls back to a ~1.1 s timeout. **If the demo is hands-free, expect roughly
   1.7 s.** This is our top open item.
6. **First turn after startup is slow** (~0.7–3.5 s). Always run one throwaway
   turn before demonstrating.
7. **QSR-Lab beats us on raw model speed** — it produces its first word in 96 ms
   versus our 210 ms. We win overall because we turn that word into sound almost
   instantly.
8. **We have only partly adopted QSR-Lab's faster speech recognition.** Their
   model (`distil-small.en`) is roughly twice as fast as ours. We run it
   **today on the GPU** for the "has the customer finished speaking?" check,
   where it takes ~40–60 ms per call instead of ~500–1100 ms. What we have
   *not* been able to do is use it for the final, committed transcript. On our
   NPU it **repeats the previous customer's words instead of transcribing the
   new ones** — reproduced three times, including in isolation with no network
   layer involved. The supplier's library offers no way to reset it. It behaves
   correctly on the GPU, but moving the final pass there too would put it in
   direct competition with the language model and speech synthesis for the same
   chip. So the final pass stays on `whisper-small`/NPU, which is proven
   reliable. This is a third-party defect, not a design shortcoming.

   *Carried risk:* the GPU preview pool has only been validated in isolation,
   not under simultaneous language-model and speech-synthesis load. That is
   exactly the condition a live demo creates.

---

## 7. A measurement bug we found and fixed

While preparing this document we found that our own opening-phrase cache was
failing on **every** turn: the audio file for the first sentence was never
written, because the code did not create the folder it wrote into.

Two consequences, both now fixed:

- **The customer never actually heard *"Got it."*** Playback began at the second
  sentence.
- **Our stopwatch stopped at the wrong moment** — it recorded when we *decided*
  to speak rather than when sound was *ready*.

We rebuilt, redeployed and re-measured. The corrected figure is **754 ms**
(previously reported as 732 ms). The headline is essentially unchanged, but the
number is now real rather than accidentally close. Verified after the fix:
21 cache hits, 0 failures, all four audio segments produced on every turn.

We are reporting this because the earlier figure had been circulated internally.

---

## 8. What comes next

| Priority | Item | Expected gain |
|---|---|---|
| **1** | Fix hands-free end-of-speech detection | up to −950 ms in hands-free |
| **2** | Recover the ~260 ms final transcription pass — needs a faster model for the *final* pass. `distil-small.en` already serves our end-of-speech check on GPU, but is unusable on NPU (supplier defect) and would contend for the GPU if moved there | up to −260 ms |
| 3 | Adopt QSR-Lab's approach of reusing a transcript produced *during* speech | Partial; limited by item 2 |
| 4 | Cache more stock phrases beyond *"Got it."* | Small |
| 5 | Re-run QSR-Lab's benchmark under matched end-of-speech settings | Better comparison quality |

---

## 9. How this was measured

- **Never run at the same time.** Both systems want the same GPU and NPU. The
  entire Smart AI Kiosk stack was shut down for the QSR-Lab run, and QSR-Lab had
  exited before ours. Timestamps confirm no overlap.
- **Machine load checked** before each run.
- **QSR-Lab's benchmark was not modified** — its own script, its own questions.
  Nothing was tuned in our favour.
- **Deployed build verified** — confirmed the running container was
  `false/kiosk-core:2026.2.0-rc2` and contained the fix, after discovering that a
  rebuild can silently leave the previous version running.
- **Repeatable.** The QSR-Lab result (1037 ms) closely matches its earlier run
  (1060 ms).

---

## 10. One paragraph for the presentation

> Measured today on the same hardware, with each system given exclusive access,
> the Smart AI Kiosk replies to a customer in **0.75 seconds** against the
> QSR-Lab research prototype's **1.04 seconds**. Up to the point where the reply
> begins to be written the two are close — in fact QSR-Lab is marginally ahead.
> Our advantage comes entirely from the second half of the turn: QSR-Lab spends
> about 426 ms turning text into speech and its workaround only succeeds on one
> turn in five, whereas we replay a guaranteed opening phrase and start speaking
> in under 2 ms, every time. We built on QSR-Lab's research — chip allocation,
> prompt reuse, short openers and streaming speech — and improved on its weakest
> stage, while carrying the extra weight of being a real containerised product
> rather than a single research program. Two things we state openly: the two
> benchmarks run different tasks and different end-of-speech rules, and our
> remaining gap — a faster final transcription pass — is blocked by a
> third-party model defect rather than by our own design.
