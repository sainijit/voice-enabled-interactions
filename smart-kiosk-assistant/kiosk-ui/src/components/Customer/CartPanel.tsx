import { useOrderTracking } from '../../hooks/useOrderTracking';
import { formatCurrency, formatOrderId } from '../../api/orderingApi';

interface CartPanelProps {
  active: boolean;
}

/**
 * Display-only cart for the customer kiosk screen. Reuses the exact same
 * live polling/state machine as the operator OrderPanel (useOrderTracking):
 * adaptive 2s/10s cadence, confirmed-receipt retention, upsell suggestions.
 * No add/remove controls — per the voice-only requirement, items only change
 * via the Ask button / agent.
 */
export function CartPanel({ active }: CartPanelProps) {
  const { order, suggestions } = useOrderTracking(active);
  const items = order?.items ?? [];
  const isConfirmed = order?.status === 'confirmed';

  return (
    <aside className="flex h-full flex-col border-l border-gray-200 bg-white">
      <div className="flex items-center justify-between gap-2 border-b border-gray-100 px-5 py-4">
        <h2 className="text-base font-bold text-intel-dark">
          {isConfirmed ? '✅ Order Confirmed' : '🛒 Your Order'}
        </h2>
        {order?.order_id !== undefined && (
          <span className="text-xs font-medium text-gray-400">#{formatOrderId(order.order_id)}</span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-4">
        {!order ? (
          <p className="text-sm text-gray-400">
            Your cart is empty. Tap <strong>Ask</strong> below and tell us what you'd like.
          </p>
        ) : (
          <>
            <div className="space-y-3">
              {items.map((item) => (
                <div key={item.id} className="flex items-start justify-between gap-3">
                  <span className="text-sm text-intel-dark">
                    <span className="mr-1.5 inline-block min-w-[1.5rem] rounded bg-gray-100 px-1.5 text-center text-xs font-semibold text-gray-500">
                      {item.quantity}×
                    </span>
                    {item.product_name}
                  </span>
                  <span className="shrink-0 text-sm font-semibold text-intel-dark">
                    {formatCurrency(item.subtotal)}
                  </span>
                </div>
              ))}
            </div>

            <div className="my-4 border-t border-gray-200" />

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-sm font-bold text-intel-dark">Total</span>
                <span
                  className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                    isConfirmed ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'
                  }`}
                >
                  {order.status}
                </span>
              </div>
              <span className="text-lg font-bold text-intel-blue">{formatCurrency(order.total)}</span>
            </div>

            {isConfirmed ? (
              <p className="mt-4 rounded-md bg-green-50 px-3 py-2 text-center text-xs text-green-700">
                🎉 Thank you! Your order {formatOrderId(order.order_id)} is confirmed.
              </p>
            ) : suggestions.length > 0 ? (
              <div className="mt-4">
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
                  ✨ You might also like
                </h3>
                <div className="space-y-1.5">
                  {suggestions.map((s) => (
                    <div
                      key={s.product.product_id}
                      className="rounded-md border border-gray-200 bg-gray-50 px-2.5 py-1.5 text-xs text-intel-dark"
                    >
                      {s.product.name} — {s.reason}
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </>
        )}
      </div>
    </aside>
  );
}

export default CartPanel;
