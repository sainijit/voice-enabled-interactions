import { useEffect } from 'react';

// ---------------------------------------------------------------------------
// Shared queue-count polling hook.
//
// Extracted from the operator QsrPanel so both the operator "QSR" tab and the
// customer-facing kiosk screen poll the queue-service the exact same way and
// can never drift out of sync. Pure move — behaviour is unchanged.
// ---------------------------------------------------------------------------

export type QueueStatus = 'LOW' | 'MEDIUM' | 'HIGH' | 'unknown';

export interface QueueInfo {
  count: number;
  status: QueueStatus;
}

export const QUEUE_STREAM_URL = '/queue-svc/stream';
export const QUEUE_COUNT_URL = '/queue-svc/api/v1/queue/count';
export const QUEUE_POLL_MS = 2_000;

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

/** Polls the queue-service count endpoint every `intervalMs`, fails soft. */
export function useQueueCount(
  onData: (info: QueueInfo) => void,
  url: string = QUEUE_COUNT_URL,
  intervalMs: number = QUEUE_POLL_MS,
) {
  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const res = await fetch(url, { signal: AbortSignal.timeout(4000) });
        if (!res.ok || cancelled) return;
        const data = (await res.json()) as { count: number; status: string };
        onData({ count: data.count ?? 0, status: (data.status as QueueStatus) ?? 'unknown' });
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
  }, [url, intervalMs, onData]);
}
