import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { fetchCurrentOrder, fetchOrder, fetchUpsell } from '../api/orderingApi';
import { tuning } from '../constants';
import type { Order, UpsellSuggestion } from '../types';

// ---------------------------------------------------------------------------
// Shared live-order polling/state machine.
//
// Extracted from the operator OrderPanel so both the operator "QSR → Cart"
// tab and the customer-facing kiosk CartPanel track the same draft/confirmed
// order the exact same way (adaptive poll cadence, confirmed-receipt
// retention, upsell suggestions) and can never drift apart. Pure move —
// behaviour is unchanged for the operator UI.
// ---------------------------------------------------------------------------

// Poll quickly during an active ordering session; fall back to the same
// cadence as the performance dashboard so we don't spam the backend when idle.
const ACTIVE_POLL_MS = 2000;
const IDLE_POLL_MS = tuning.perfRefreshMs; // 10 s

export interface OrderTracking {
  order: Order | null;
  suggestions: UpsellSuggestion[];
}

/**
 * Tracks the current user's draft order (or the just-confirmed receipt) plus
 * up to 3 upsell suggestions, polling at `ACTIVE_POLL_MS` while `active` is
 * true and `IDLE_POLL_MS` otherwise.
 */
export function useOrderTracking(active: boolean): OrderTracking {
  const [order, setOrder] = useState<Order | null>(null);
  const [suggestions, setSuggestions] = useState<UpsellSuggestion[]>([]);
  const mountedRef = useRef(false);
  // Remember the order currently on screen so we can keep showing the confirmed
  // receipt after the draft query stops returning it.
  const shownOrderRef = useRef<Order | null>(null);

  const applyOrder = useCallback(async (next: Order | null) => {
    if (!mountedRef.current) return;
    shownOrderRef.current = next;
    setOrder(next);

    const productIds =
      next && next.status === 'draft' ? next.items?.map((item) => item.product_id) ?? [] : [];
    if (productIds.length > 0) {
      const nextSuggestions = await fetchUpsell(productIds);
      if (!mountedRef.current) return;
      setSuggestions(nextSuggestions);
    } else {
      setSuggestions([]);
    }
  }, []);

  const loadOrder = useCallback(async () => {
    const draft = await fetchCurrentOrder(tuning.userId);
    if (!mountedRef.current) return;

    if (draft) {
      // A live draft exists — always show it (this also replaces a stale receipt
      // once the customer starts a brand-new order).
      await applyOrder(draft);
      return;
    }

    // No draft. If we were showing one, it was just confirmed — fetch the frozen
    // confirmed order by id and keep it on screen as a receipt instead of blanking.
    const shown = shownOrderRef.current;
    if (shown && shown.status !== 'confirmed') {
      const confirmed = await fetchOrder(shown.order_id);
      if (!mountedRef.current) return;
      if (confirmed) {
        await applyOrder(confirmed);
        return;
      }
    }
    // Already showing a confirmed receipt (or nothing) — leave it untouched.
    if (!shown) {
      await applyOrder(null);
    }
  }, [applyOrder]);

  // Single interval whose cadence adapts to the active state.
  useEffect(() => {
    mountedRef.current = true;
    void loadOrder();

    const intervalMs = active ? ACTIVE_POLL_MS : IDLE_POLL_MS;
    const intervalId = window.setInterval(() => {
      void loadOrder();
    }, intervalMs);

    return () => {
      mountedRef.current = false;
      window.clearInterval(intervalId);
    };
  }, [active, loadOrder]);

  const visibleSuggestions = useMemo(() => suggestions.slice(0, 3), [suggestions]);

  return { order, suggestions: visibleSuggestions };
}
