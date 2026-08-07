import { useEffect, useRef, useState } from 'react';

// ---------------------------------------------------------------------------
// Shared queue-count hooks.
//
// Extracted from the operator QsrPanel so both the operator "QSR" tab and the
// customer-facing kiosk screen read the queue-service the exact same way and
// can never drift out of sync.
//
// `useQueueStream` is the default: queue-service pushes a Server-Sent Event
// the moment the count changes, so the UI updates in well under 100 ms instead
// of the 0-2000 ms lag the old poll had. `useQueueCount` is kept as the
// fallback for environments where the SSE connection cannot be established
// (e.g. a proxy that buffers text/event-stream).
// ---------------------------------------------------------------------------

export type QueueStatus = 'LOW' | 'MEDIUM' | 'HIGH' | 'unknown';

export interface QueueInfo {
  count: number;
  status: QueueStatus;
}

export const QUEUE_STREAM_URL = '/queue-svc/stream';
export const QUEUE_COUNT_URL = '/queue-svc/api/v1/queue/count';
export const QUEUE_EVENTS_URL = '/queue-svc/api/v1/queue/events';
export const QUEUE_POLL_MS = 2_000;

// Consecutive EventSource failures tolerated before giving up on SSE and
// falling back to polling. EventSource retries on its own, so this only trips
// when the transport is genuinely unusable rather than briefly interrupted.
const QUEUE_SSE_MAX_ERRORS = 3;

export const QUEUE_STATUS_STYLE: Record<QueueStatus, string> = {
  LOW: 'bg-green-50  border-green-200  text-green-800',
  MEDIUM: 'bg-amber-50  border-amber-200  text-amber-800',
  HIGH: 'bg-red-50    border-red-200    text-red-800',
  unknown: 'bg-gray-50   border-gray-200   text-gray-500',
};

export const QUEUE_STATUS_ICON: Record<QueueStatus, string> = {
  LOW: '🟢',
  MEDIUM: '🟡',
  HIGH: '🔴',
  unknown: '⚪',
};

/** Polls the queue-service count endpoint every `intervalMs`, fails soft.
 *
 * Kept as the fallback transport for `useQueueStream`. Pass `enabled=false`
 * to keep the hook mounted (hooks cannot be called conditionally) while
 * suppressing the network traffic.
 */
export function useQueueCount(
  onData: (info: QueueInfo) => void,
  url: string = QUEUE_COUNT_URL,
  intervalMs: number = QUEUE_POLL_MS,
  enabled: boolean = true,
) {
  const onDataRef = useRef(onData);
  useEffect(() => {
    onDataRef.current = onData;
  }, [onData]);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const res = await fetch(url, { signal: AbortSignal.timeout(4000) });
        if (!res.ok || cancelled) return;
        const data = (await res.json()) as { count: number; status: string };
        onDataRef.current({
          count: data.count ?? 0,
          status: (data.status as QueueStatus) ?? 'unknown',
        });
      } catch {
        // queue-service unavailable — caller keeps its last-known state hidden
      }
    };

    void poll();
    const id = window.setInterval(() => {
      void poll();
    }, intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [url, intervalMs, enabled]);
}

/** Subscribes to queue-service Server-Sent Events, falling back to polling.
 *
 * queue-service emits an event only when the count actually changes, so an
 * idle queue costs nothing beyond a periodic heartbeat comment. `onData` is
 * held in a ref so an un-memoised callback cannot tear down and reopen the
 * connection on every render.
 */
export function useQueueStream(
  onData: (info: QueueInfo) => void,
  eventsUrl: string = QUEUE_EVENTS_URL,
  fallbackUrl: string = QUEUE_COUNT_URL,
  fallbackIntervalMs: number = QUEUE_POLL_MS,
) {
  const onDataRef = useRef(onData);
  useEffect(() => {
    onDataRef.current = onData;
  }, [onData]);

  const [degraded, setDegraded] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.EventSource === 'undefined') {
      setDegraded(true);
      return;
    }

    let errors = 0;
    const source = new EventSource(eventsUrl);

    const handle = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data) as { count: number; status: string };
        errors = 0;
        setDegraded(false);
        onDataRef.current({
          count: data.count ?? 0,
          status: (data.status as QueueStatus) ?? 'unknown',
        });
      } catch {
        // Malformed frame — ignore and wait for the next one.
      }
    };

    source.addEventListener('queue', handle as EventListener);
    source.onerror = () => {
      // EventSource reconnects by itself; only treat a sustained run of
      // failures as "SSE is unusable here" and switch to polling.
      errors += 1;
      if (errors >= QUEUE_SSE_MAX_ERRORS) setDegraded(true);
    };

    return () => {
      source.removeEventListener('queue', handle as EventListener);
      source.close();
    };
  }, [eventsUrl]);

  // Only actually polls once the stream has proven unusable.
  useQueueCount(onData, fallbackUrl, fallbackIntervalMs, degraded);
}
