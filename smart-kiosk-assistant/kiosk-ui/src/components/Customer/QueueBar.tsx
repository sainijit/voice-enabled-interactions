import BrandSlot from '../../assets/BrandSlot.svg';
import type { QueueInfo, QueueStatus } from '../../hooks/useQueue';
import { QUEUE_STATUS_ICON, QUEUE_STATUS_STYLE } from '../../hooks/useQueue';

interface QueueBarProps {
  brand?: string;
  queueInfo: QueueInfo | null;
  isPeak: boolean;
  showFullMenu: boolean;
  onToggleFullMenu: () => void;
}

/**
 * Top header for the customer kiosk screen: brand + live queue pill (updates
 * every 2s via useQueueCount) + a peak-hours banner with an escape hatch back
 * to the full menu. Colour and copy mirror the operator QSR tab exactly.
 */
export function QueueBar({
  brand = 'Smart Kiosk',
  queueInfo,
  isPeak,
  showFullMenu,
  onToggleFullMenu,
}: QueueBarProps) {
  const status: QueueStatus = queueInfo?.status ?? 'unknown';

  return (
    <header className="shrink-0 border-b border-gray-200 bg-white">
      <div className="flex items-center justify-between gap-4 px-6 py-4">
        <div className="flex items-center gap-3">
          <img src={BrandSlot} alt="" className="h-9 w-auto object-contain" />
          <h1 className="text-xl font-bold text-intel-dark tracking-tight">{brand}</h1>
        </div>

        {queueInfo !== null && (
          <div
            className={`flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-semibold ${QUEUE_STATUS_STYLE[status]}`}
          >
            <span>{QUEUE_STATUS_ICON[status]}</span>
            <span>
              Queue: <strong>{queueInfo.count}</strong> {queueInfo.count === 1 ? 'person' : 'people'}
            </span>
            <span className="opacity-60">·</span>
            <span>{status}</span>
          </div>
        )}
      </div>

      {isPeak && (
        <div className="flex items-center justify-between gap-3 bg-amber-50 px-6 py-2.5 text-sm text-amber-800 border-t border-amber-200">
          <span>
            ⚡ <strong>Peak hours</strong> — showing our express menu (Burgers · Sides · Beverages) to
            keep the line moving.
          </span>
          <button
            type="button"
            onClick={onToggleFullMenu}
            className="shrink-0 min-h-[44px] rounded-md border border-amber-300 bg-white px-4 py-2.5 text-sm font-semibold text-amber-800 hover:bg-amber-100 transition-colors focus:outline-none focus:ring-2 focus:ring-amber-400"
          >
            {showFullMenu ? 'Show express menu' : 'Show full menu'}
          </button>
        </div>
      )}
    </header>
  );
}

export default QueueBar;
