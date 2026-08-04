import { endpoints } from '../constants';
import type { Order, Product, UpsellSuggestion } from '../types';

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
