import type { Product } from '../types';

// ---------------------------------------------------------------------------
// Shared category metadata.
//
// Extracted from the operator MenuPanel so both the operator "QSR" tab and
// the customer-facing kiosk screen group/label/order categories identically
// and share the same peak-hours ("express menu") rule. Pure move — behaviour
// is unchanged for the operator UI.
// ---------------------------------------------------------------------------

export interface CategoryMeta {
  key: string;
  label: string;
  /** Emoji fallback, used wherever an icon library entry isn't available. */
  icon: string;
}

// Display order + icon for each catalogue category. Categories not listed
// here still render (alphabetically, with a default icon) so the menu never
// hides items just because it's an unrecognised category.
export const CATEGORY_META: CategoryMeta[] = [
  { key: 'burgers', label: 'Burgers', icon: '🍔' },
  { key: 'pizza', label: 'Pizza', icon: '🍕' },
  { key: 'wraps', label: 'Wraps', icon: '🌯' },
  { key: 'sides', label: 'Sides', icon: '🍟' },
  { key: 'beverages', label: 'Beverages', icon: '🥤' },
  { key: 'desserts', label: 'Desserts', icon: '🍰' },
];

/** Categories shown during peak hours (fast-prep, low-queue impact). */
export const PEAK_CATEGORIES = new Set(['burgers', 'beverages', 'sides']);

export const categoryMeta = (key: string): CategoryMeta =>
  CATEGORY_META.find((c) => c.key === key) ?? {
    key,
    label: key.charAt(0).toUpperCase() + key.slice(1),
    icon: '🍽',
  };

export interface GroupedCategory extends CategoryMeta {
  items: Product[];
}

/**
 * Group products by category, ordered per CATEGORY_META then any extra
 * categories alphabetically, optionally restricted to the peak-hours subset.
 */
export function groupByCategory(products: Product[], peakOnly: boolean): GroupedCategory[] {
  const filtered = peakOnly ? products.filter((p) => PEAK_CATEGORIES.has(p.category)) : products;
  const byCat = new Map<string, Product[]>();
  for (const p of filtered) {
    const list = byCat.get(p.category) ?? [];
    list.push(p);
    byCat.set(p.category, list);
  }
  const ordered: string[] = [
    ...CATEGORY_META.map((c) => c.key).filter((k) => byCat.has(k)),
    ...[...byCat.keys()].filter((k) => !CATEGORY_META.some((c) => c.key === k)).sort(),
  ];
  return ordered.map((key) => ({
    ...categoryMeta(key),
    items: (byCat.get(key) ?? []).sort((a, b) => a.name.localeCompare(b.name)),
  }));
}

/** Format a price in Indian Rupees, dropping trailing .0 for whole values. */
export const formatPrice = (value: number): string => {
  const rounded = Math.round(value * 100) / 100;
  return `₹${Number.isInteger(rounded) ? rounded : rounded.toFixed(2)}`;
};
