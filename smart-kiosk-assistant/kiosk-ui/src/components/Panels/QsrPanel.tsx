import { useCallback, useState } from 'react';

import MenuPanel from '../Order/MenuPanel';
import OrderPanel from '../Order/OrderPanel';
import { LiveQueueFeed } from '../Chat/LiveQueueFeed';
import {
  QUEUE_STATUS_ICON as STATUS_ICON,
  QUEUE_STATUS_STYLE as STATUS_STYLE,
  useQueueStream,
  type QueueInfo,
} from '../../hooks/useQueue';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type SubTab = 'menu' | 'cart';

interface QsrPanelProps {
  orderActive: boolean;
}

// ---------------------------------------------------------------------------
// QsrPanel — top-level QSR tab
//
// Layout (top → bottom):
//   1. Live MJPEG feed from queue-service (bounding boxes + count overlay)
//   2. Queue status banner  (count + LOW/MEDIUM/HIGH), updates every 2 s
//   3. Peak-hour notice     (MEDIUM/HIGH only, while showing express menu)
//   4. Sub-tab pills        (Menu / Cart)
//   5. MenuPanel + OrderPanel — BOTH always mounted so OrderPanel keeps
//      its polling interval alive; CSS `hidden` class toggles visibility.
// ---------------------------------------------------------------------------

export function QsrPanel({ orderActive }: QsrPanelProps) {
  const [subTab, setSubTab]             = useState<SubTab>('menu');
  const [queueInfo, setQueueInfo]       = useState<QueueInfo | null>(null);
  const [showFullMenu, setShowFullMenu] = useState(false);

  const onQueueData = useCallback((info: QueueInfo) => {
    setQueueInfo(info);
  }, []);

  useQueueStream(onQueueData);

  const status  = queueInfo?.status ?? 'unknown';
  const isPeak  = status === 'MEDIUM' || status === 'HIGH';
  const peakOnly = isPeak && !showFullMenu;

  const subTabs: { id: SubTab; label: string; icon: string }[] = [
    { id: 'menu', label: 'Menu', icon: '🍔' },
    { id: 'cart', label: 'Cart', icon: '🛒' },
  ];

  return (
    <div className="space-y-3 p-4">

      {/* 1 ── Live queue feed (MJPEG) ───────────────────────────────────── */}
      <div className="overflow-hidden rounded-lg border border-gray-200 shadow-sm">
        <LiveQueueFeed height="280px" />
      </div>

      {/* 2 ── Queue status banner ───────────────────────────────────────── */}
      {queueInfo !== null && (
        <div
          className={`flex items-center justify-between rounded-lg border px-3 py-2 text-xs font-medium ${STATUS_STYLE[status]}`}
        >
          <span>
            {STATUS_ICON[status]}&nbsp;Queue:&nbsp;
            <strong>{queueInfo.count}</strong>&nbsp;
            {queueInfo.count === 1 ? 'person' : 'people'}&nbsp;·&nbsp;{status}
          </span>
          {isPeak && (
            <button
              type="button"
              onClick={() => setShowFullMenu((v) => !v)}
              className="ml-2 rounded border border-current px-2 py-0.5 text-[10px] opacity-75 hover:opacity-100"
            >
              {showFullMenu ? '⚡ Peak menu' : '📋 Full menu'}
            </button>
          )}
        </div>
      )}

      {/* 3 ── Peak-hour notice ──────────────────────────────────────────── */}
      {peakOnly && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          ⚡ <strong>Peak hours</strong> — express menu shown (Burgers · Sides · Beverages).
          Tap "Full menu" to see all items.
        </div>
      )}

      {/* 4 ── Sub-tab pills ─────────────────────────────────────────────── */}
      <div className="flex gap-2">
        {subTabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setSubTab(tab.id)}
            className={`
              flex flex-1 items-center justify-center gap-1.5 rounded-lg border py-2 text-xs font-semibold
              transition-colors duration-150
              ${subTab === tab.id
                ? 'border-intel-blue bg-blue-50/60 text-intel-blue'
                : 'border-gray-200 bg-white text-gray-400 hover:text-gray-600'}
            `}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* 5 ── Panel bodies — BOTH always mounted; CSS toggles visibility ── */}
      <div className={subTab === 'menu' ? '' : 'hidden'}>
        <MenuPanel peakOnly={peakOnly} />
      </div>
      <div className={subTab === 'cart' ? '' : 'hidden'}>
        <OrderPanel active={orderActive} />
      </div>
    </div>
  );
}

export default QsrPanel;
