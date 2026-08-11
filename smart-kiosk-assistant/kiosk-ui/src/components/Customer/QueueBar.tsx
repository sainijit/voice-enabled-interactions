interface QueueBarProps {
  isPeak: boolean;
  showFullMenu: boolean;
  onToggleFullMenu: () => void;
}

/**
 * Peak-hours banner for the customer kiosk screen.
 *
 * The brand + live queue count used to live in a second header bar here,
 * duplicating the Intel brand row already shown in the blue top header and
 * adding a redundant "Smart Kiosk" title customers never needed. The queue
 * count now lives as a pill in the top header (see Header.tsx); this
 * component keeps only the one thing that genuinely needs its own banner —
 * the peak-hours notice with its "show full menu" escape hatch — and renders
 * nothing at all outside of peak hours, so no empty bar sits under the header.
 */
export function QueueBar({ isPeak, showFullMenu, onToggleFullMenu }: QueueBarProps) {
  if (!isPeak) return null;

  return (
    <div className="flex shrink-0 items-center justify-between gap-3 border-b border-amber-200 bg-amber-50 px-6 py-2.5 text-sm text-amber-800">
      <span>
        ⚡ <strong>Peak hours</strong> — showing our express menu (Burgers · Sides · Beverages) to keep
        the line moving.
      </span>
      <button
        type="button"
        onClick={onToggleFullMenu}
        className="shrink-0 min-h-[44px] rounded-md border border-amber-300 bg-white px-4 py-2.5 text-sm font-semibold text-amber-800 hover:bg-amber-100 transition-colors focus:outline-none focus:ring-2 focus:ring-amber-400"
      >
        {showFullMenu ? 'Show express menu' : 'Show full menu'}
      </button>
    </div>
  );
}

export default QueueBar;

