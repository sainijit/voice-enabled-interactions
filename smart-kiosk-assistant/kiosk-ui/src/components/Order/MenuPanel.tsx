import { useEffect, useMemo, useRef, useState } from 'react';

import { fetchMenu } from '../../api/orderingApi';
import { groupByCategory, formatPrice } from '../../menu/categories';
import type { Product } from '../../types';

interface MenuPanelProps {
  /** When true, only fast-prep peak categories are shown. */
  peakOnly?: boolean;
}

export function MenuPanel({ peakOnly = false }: MenuPanelProps) {
  const [products, setProducts] = useState<Product[] | null>(null);
  const mountedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    void (async () => {
      const data = await fetchMenu();
      if (mountedRef.current) setProducts(data);
    })();
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const grouped = useMemo(
    () => (products ? groupByCategory(products, peakOnly) : []),
    [products, peakOnly],
  );

  if (products === null) {
    return <p className="px-1 py-3 text-sm text-kiosk-textlo">Loading menu…</p>;
  }

  if (products.length === 0) {
    return <p className="px-1 py-3 text-sm text-kiosk-textlo">Menu is currently unavailable.</p>;
  }

  return (
    <div className="space-y-3">
      {grouped.map((cat) => (
        <div
          key={cat.key}
          className="overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm"
        >
          <div className="flex items-center justify-between border-b border-gray-100 bg-gray-50 px-3 py-2">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
              {cat.icon} {cat.label}
            </span>
            <span className="text-[10px] text-gray-400">{cat.items.length}</span>
          </div>
          <div className="divide-y divide-gray-100">
            {cat.items.map((item) => (
              <div
                key={item.product_id}
                className="flex items-center justify-between gap-3 px-3 py-2"
              >
                <span className="min-w-0 truncate text-sm text-intel-dark">{item.name}</span>
                <span className="shrink-0 text-sm font-medium text-intel-dark">
                  {formatPrice(item.price)}
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export default MenuPanel;
