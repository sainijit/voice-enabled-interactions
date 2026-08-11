import { useState } from 'react';

import { QUEUE_STREAM_URL } from '../../hooks/useQueue';

interface LiveQueueFeedProps {
  /** Fixed height for the video frame, e.g. "100%", "280px". Defaults to full height of parent. */
  height?: string;
  /** Show the small "LIVE" badge overlay in the top-left corner. Defaults to true. */
  showBadge?: boolean;
  className?: string;
}

/**
 * Live MJPEG queue feed from queue-service (bounding boxes + count overlay
 * baked into the stream server-side).
 *
 * Extracted out of QsrPanel so the same feed can also be shown above the
 * chat pane on the main operator screen (App.tsx) without duplicating the
 * stream URL / error-fallback logic. Each mount opens its own MJPEG
 * connection to queue-service — acceptable for a single-kiosk demo.
 */
export function LiveQueueFeed({ height = '100%', showBadge = true, className = '' }: LiveQueueFeedProps) {
  const [streamErr, setStreamErr] = useState(false);

  return (
    <div className={`relative overflow-hidden bg-black ${className}`} style={{ height }}>
      {streamErr ? (
        <div className="flex h-full items-center justify-center text-xs text-gray-400">
          📷 Queue feed unavailable
        </div>
      ) : (
        <img
          src={QUEUE_STREAM_URL}
          alt="Live queue feed with person detection"
          className="h-full w-full object-contain"
          onError={() => setStreamErr(true)}
        />
      )}
      {showBadge && !streamErr && (
        <span className="absolute left-2 top-2 flex items-center gap-1 rounded-full bg-black/60 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-white">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-red-500" />
          Live Queue
        </span>
      )}
    </div>
  );
}

export default LiveQueueFeed;
