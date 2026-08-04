import { useOrderTracking } from '../../hooks/useOrderTracking';
import { formatCurrency, formatOrderId } from '../../api/orderingApi';

interface OrderPanelProps {
  active: boolean;
}

export function OrderPanel({ active }: OrderPanelProps) {
  const { order, suggestions: visibleSuggestions } = useOrderTracking(active);

  const items = order?.items ?? [];
  const isConfirmed = order?.status === 'confirmed';

  return (
    <section className="rounded-lg border border-kiosk-border bg-white p-4">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-intel-dark">
          {isConfirmed ? '✅ Order Confirmed' : '🛒 Current Order'}
        </h2>
        {order?.order_id !== undefined ? (
          <span className="text-xs text-kiosk-textlo">#{formatOrderId(order.order_id)}</span>
        ) : null}
      </div>

      {!order ? (
        <p className="py-3 text-sm text-kiosk-textlo">No active order yet. Start ordering by voice.</p>
      ) : (
        <div className="mt-3">
          <div className="space-y-2">
            {items.map((item) => (
              <div key={item.id} className="flex justify-between gap-3">
                <span className="text-sm text-intel-dark">
                  <span className="text-xs">{item.quantity}×</span> {item.product_name}
                </span>
                <span className="text-sm font-medium">{formatCurrency(item.subtotal)}</span>
              </div>
            ))}
          </div>

          <div className="my-3 border-t border-kiosk-border" />

          <div className="flex items-center justify-between gap-3 text-sm font-bold text-intel-dark">
            <div className="flex items-center gap-2">
              <span>Total</span>
              <span
                className={`rounded-full px-2 py-0.5 text-[10px] ${
                  isConfirmed ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'
                }`}
              >
                {order.status}
              </span>
            </div>
            <span>{formatCurrency(order.total)}</span>
          </div>

          {isConfirmed ? (
            <p className="mt-3 rounded-md bg-green-50 px-2 py-1.5 text-center text-xs text-green-700">
              🎉 Thank you! Your order {formatOrderId(order.order_id)} is confirmed.
            </p>
          ) : visibleSuggestions.length > 0 ? (
            <div>
              <h3 className="mb-1 mt-3 text-xs font-semibold text-kiosk-textmd">✨ You might also like</h3>
              {visibleSuggestions.map((suggestion) => (
                <div
                  key={suggestion.product.product_id}
                  className="mb-1 rounded-md border border-kiosk-border bg-kiosk-asst px-2 py-1 text-xs text-intel-dark"
                >
                  {suggestion.product.name} — {suggestion.reason}
                </div>
              ))}
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
}

export default OrderPanel;
