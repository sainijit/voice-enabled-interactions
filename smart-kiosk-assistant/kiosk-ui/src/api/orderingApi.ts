import { endpoints } from '../constants';
import type { Order, PaymentIntent, Product, UpsellSuggestion } from '../types';

/** Format a value in Indian Rupees, dropping a trailing .0 for whole values (matches the agent's replies). */
export const formatCurrency = (value: number | undefined): string => {
  const rounded = Math.round((value ?? 0) * 100) / 100;
  return `₹${Number.isInteger(rounded) ? rounded : rounded.toFixed(2)}`;
};

/** Matches the order id the agent speaks (e.g. "ORD-11"), no zero-padding. */
export const formatOrderId = (orderId: number): string => `ORD-${orderId}`;

/** Fetch the full product catalogue (restaurant menu). */
export async function fetchMenu(): Promise<Product[]> {
  try {
    const res = await fetch(endpoints.products, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return [];
    return await res.json();
  } catch {
    return [];
  }
}

/** Fetch a single order by id (any status), or null if not found. */
export async function fetchOrder(orderId: number): Promise<Order | null> {
  try {
    const res = await fetch(endpoints.order(orderId), { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return null;
    const data: Order | null = await res.json();
    return data ?? null;
  } catch {
    return null;
  }
}

/** Fetch the current draft order for a user, or null if none exists. */
export async function fetchCurrentOrder(userId: string): Promise<Order | null> {
  try {
    const res = await fetch(endpoints.currentOrder(userId), { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return null;
    // Backend returns JSON null (200) when no draft order exists.
    const data: Order | null = await res.json();
    return data ?? null;
  } catch {
    return null;
  }
}

/**
 * Discard the user's open draft cart.
 *
 * Called once when the customer screen mounts so a new conversation always
 * starts from an empty cart: the cart lives server-side in SQLite, so without
 * this a page refresh would resurface the previous customer's abandoned items.
 * Returns true when the backend acknowledged the reset.
 */
export async function clearCurrentOrder(userId: string): Promise<boolean> {
  try {
    const res = await fetch(endpoints.currentOrder(userId), {
      method: 'DELETE',
      signal: AbortSignal.timeout(4000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/** Fetch rule-based upsell suggestions for a cart's product IDs. */
export async function fetchUpsell(productIds: string[]): Promise<UpsellSuggestion[]> {
  if (productIds.length === 0) return [];
  try {
    const res = await fetch(endpoints.upsell, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ product_ids: productIds }),
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) return [];
    return await res.json();
  } catch {
    return [];
  }
}

/**
 * Fetch the demo payment intent (incl. QR code) for a CONFIRMED order.
 *
 * Returns null when the order is still a draft (422), when the payment
 * feature is disabled (404), or on any network error — callers simply omit
 * the payment panel in that case, so a payment outage can never block the
 * customer from seeing their confirmed receipt.
 */
export async function fetchPaymentIntent(orderId: number): Promise<PaymentIntent | null> {
  try {
    const res = await fetch(endpoints.orderPayment(orderId), {
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) return null;
    const data: PaymentIntent | null = await res.json();
    return data ?? null;
  } catch {
    return null;
  }
}
