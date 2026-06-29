import { useState } from 'react';

import MenuPanel from '../Order/MenuPanel';
import OrderPanel from '../Order/OrderPanel';

type SubTab = 'menu' | 'cart';

interface QsrPanelProps {
  orderActive: boolean;
}

/**
 * QsrPanel — the "QSR" top-level tab content.
 *
 * Holds two sub-tabs:
 *   • Menu — the restaurant catalogue grouped by category (read-only)
 *   • Cart — the live order and the confirmed-order receipt
 */
export function QsrPanel({ orderActive }: QsrPanelProps) {
  const [subTab, setSubTab] = useState<SubTab>('menu');

  const subTabs: { id: SubTab; label: string; icon: string }[] = [
    { id: 'menu', label: 'Menu', icon: '🍔' },
    { id: 'cart', label: 'Cart', icon: '🛒' },
  ];

  return (
    <div className="space-y-3 p-4">
      {/* ── Sub-tab pills ──────────────────────────────────────────────────── */}
      <div className="flex gap-2">
        {subTabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setSubTab(tab.id)}
            className={`
              flex flex-1 items-center justify-center gap-1.5 rounded-lg border py-2 text-xs font-semibold
              transition-colors duration-150
              ${
                subTab === tab.id
                  ? 'border-intel-blue bg-blue-50/60 text-intel-blue'
                  : 'border-gray-200 bg-white text-gray-400 hover:text-gray-600'
              }
            `}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* ── Sub-tab body ───────────────────────────────────────────────────── */}
      {subTab === 'menu' ? <MenuPanel /> : <OrderPanel active={orderActive} />}
    </div>
  );
}

export default QsrPanel;
