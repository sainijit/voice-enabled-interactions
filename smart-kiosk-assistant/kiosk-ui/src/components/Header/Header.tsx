import BrandSlot from '../../assets/BrandSlot.svg';
import { constants } from '../../constants';
import type { QueueInfo, QueueStatus } from '../../hooks/useQueue';
import { QUEUE_STATUS_ICON } from '../../hooks/useQueue';

// Queue-pill colours tuned for the dark blue header bar — the light/white
// backgrounds in QUEUE_STATUS_STYLE (designed for a white bar) would lose
// contrast here, so this on-dark variant keeps the same status colours
// (green/amber/red) but as a translucent chip with a light border.
const QUEUE_STATUS_STYLE_ON_DARK: Record<QueueStatus, string> = {
  LOW: 'bg-white/10 border-white/25 text-white',
  MEDIUM: 'bg-amber-400/20 border-amber-300/40 text-amber-50',
  HIGH: 'bg-red-400/20 border-red-300/40 text-red-50',
  unknown: 'bg-white/10 border-white/20 text-white/70',
};

interface HeaderProps {
  /** Live queue count, shown as a pill on the right of the header when
   * present. Omitted entirely on screens with no queue concept (operator
   * dashboard tabs other than the customer kiosk view). */
  queueInfo?: QueueInfo | null;
}

const Header = ({ queueInfo }: HeaderProps) => {
  const status: QueueStatus = queueInfo?.status ?? 'unknown';

  return (
    <header
      className="sticky top-0 left-0 right-0 z-50 bg-intel-blue w-full flex items-center justify-between px-4 sm:px-6 border-b border-intel-blue-dark"
      style={{ height: '60px' }}
    >
      {/* Brand */}
      <div className="flex items-center gap-3 sm:gap-4">
        <img src={BrandSlot} alt="Intel" className="h-[44px] w-auto object-contain" />
        <div className="flex flex-col">
          <span className="text-sm sm:text-base font-semibold text-white font-display leading-tight">
            {constants.TITLE}
          </span>
          <span className="text-[10px] text-white/50 font-mono tracking-widest uppercase hidden sm:block">
            AI Kiosk Assistant · v{constants.VERSION}
          </span>
        </div>
      </div>

      {/* Live queue count — lives here (not a second header bar) so the
          customer kiosk screen has a single, unambiguous header. */}
      {queueInfo !== null && queueInfo !== undefined && (
        <div
          className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs sm:text-sm font-semibold ${QUEUE_STATUS_STYLE_ON_DARK[status]}`}
        >
          <span aria-hidden="true">{QUEUE_STATUS_ICON[status]}</span>
          <span>
            <strong>{queueInfo.count}</strong> {queueInfo.count === 1 ? 'person' : 'people'} in queue
          </span>
        </div>
      )}
    </header>
  );
};

export default Header;

