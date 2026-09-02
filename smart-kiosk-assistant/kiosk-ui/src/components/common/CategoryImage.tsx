import { useState } from 'react';
import { addCollection, Icon } from '@iconify/react';
import categoryIconData from '../../assets/category-icons.json';

// ---------------------------------------------------------------------------
// Offline category imagery.
//
// Uses a small hand-picked subset of the Noto Emoji set (@iconify-json/noto),
// pre-extracted at dev time into src/assets/category-icons.json (~30 KB for
// 7 icons vs. the ~25 MB full collection) and bundled with the app — no CDN,
// no network calls, works fully air-gapped. @iconify/react renders the raw
// SVG bodies registered via addCollection, so there is zero runtime fetch.
// ---------------------------------------------------------------------------

const CATEGORY_ICON_NAME: Record<string, string> = {
  burgers: 'hamburger',
  pizza: 'pizza',
  wraps: 'burrito',
  sides: 'french-fries',
  beverages: 'cup-with-straw',
  'hot-beverages': 'hot-beverage',
  desserts: 'shortcake',
};
const DEFAULT_ICON_NAME = 'fork-and-knife-with-plate';

// ---------------------------------------------------------------------------
// Per-item imagery.
//
// A single shared icon per category (e.g. one hamburger for every burger)
// made every card in a section look identical. Each product now resolves to
// its own icon: an explicit product_id override where we want a precise
// match (e.g. "Aloo Tikki Burger" → potato), then a name-keyword fallback,
// then a deterministic per-category rotation (hash of product_id) so any
// future/unmapped item still gets visual variety instead of collapsing back
// to one repeated icon. All icons are Apache-2.0 Noto Emoji — free to use,
// not copyrighted/branded artwork.
// ---------------------------------------------------------------------------

/** Exact icon for specific catalogue items where keyword matching alone would be ambiguous. */
const PRODUCT_ICON_OVERRIDE: Record<string, string> = {
  'BURGER-VEG-001': 'falafel', // Crispy Veg Patty Burger
  'BURGER-VEG-002': 'cheese-wedge', // Paneer Tikka Burger
  'BURGER-VEG-003': 'potato', // Aloo Tikki Burger
  'BURGER-VEG-004': 'hamburger', // Double Stack Veg
  'BURGER-NV-001': 'poultry-leg', // Classic Chicken Burger
  'BURGER-NV-002': 'hot-pepper', // Spicy Chicken Crunch Burger
  'BURGER-NV-003': 'meat-on-bone', // Double Chicken Tower
  'WRAP-VEG-001': 'stuffed-flatbread', // Paneer Bhurji Kathi Roll
  'WRAP-VEG-002': 'burrito', // Mixed Veg Fajita Wrap
  'WRAP-NV-001': 'taco', // Chicken Tikka Kathi Roll
  'WRAP-NV-002': 'sandwich', // Peri Peri Chicken Wrap
  'PIZZA-VEG-001': 'pizza', // Margherita Pizza
  'PIZZA-VEG-002': 'cheese-wedge', // Paneer Makhani Pizza
  'PIZZA-NV-001': 'poultry-leg', // Chicken BBQ Pizza
  'PIZZA-NV-002': 'meat-on-bone', // Pepperoni-Style Chicken Pizza
  'SIDE-001': 'french-fries', // Classic French Fries
  'SIDE-002': 'hot-pepper', // Peri Peri Fries
  'SIDE-003': 'onion', // Onion Rings
  'SIDE-004': 'garlic', // Garlic Bread
  'DRINK-001': 'tumbler-glass', // Pepsi
  'DRINK-002': 'cup-with-straw', // 7UP
  'DRINK-003': 'mango', // Mango Lassi
  'DRINK-004': 'hot-beverage', // Cold Coffee
  'DRINK-005': 'lime', // Fresh Lime Soda
  'TEA-001': 'teapot', // Masala Chai
  'TEA-002': 'hot-beverage', // Ginger Tea
  'TEA-003': 'teapot', // Cardamom Elaichi Tea
  'TEA-004': 'teacup-without-handle', // Green Tea
  'TEA-005': 'lime', // Lemon Honey Tea
  'TEA-006': 'teacup-without-handle', // Kashmiri Kahwa
  'COFFEE-001': 'hot-beverage', // Filter Coffee
  'COFFEE-002': 'hot-beverage', // Espresso
  'COFFEE-003': 'glass-of-milk', // Cappuccino
  'COFFEE-004': 'glass-of-milk', // Cafe Latte
  'COFFEE-005': 'hot-beverage', // Americano
  'COFFEE-006': 'chocolate-bar', // Hazelnut Mocha
  'DESSERT-001': 'chocolate-bar', // Chocolate Brownie
  'DESSERT-002': 'ice-cream', // Vanilla Soft Serve
};

/** Keyword → icon, checked in order, for items not in the explicit override map above. */
const NAME_KEYWORD_ICON: Array<[RegExp, string]> = [
  [/paneer/i, 'cheese-wedge'],
  [/(chicken|meat)/i, 'poultry-leg'],
  [/(spicy|peri peri|pepperoni)/i, 'hot-pepper'],
  [/aloo|potato/i, 'potato'],
  [/(taco)/i, 'taco'],
  [/burrito|fajita/i, 'burrito'],
  [/roll|kathi|flatbread/i, 'stuffed-flatbread'],
  [/wrap|sandwich/i, 'sandwich'],
  [/pizza/i, 'pizza'],
  [/fries/i, 'french-fries'],
  [/onion/i, 'onion'],
  [/garlic/i, 'garlic'],
  [/mango/i, 'mango'],
  [/lime/i, 'lime'],
  [/(latte|cappuccino|mocha|macchiato)/i, 'glass-of-milk'],
  [/(chai|masala tea)/i, 'teapot'],
  [/coffee|espresso|americano/i, 'hot-beverage'],
  // Generic tea last, so the milk/spiced variants above win first.
  [/tea|kahwa|kadha/i, 'teacup-without-handle'],
  [/(cola|soda|pepsi|7up|fizz)/i, 'tumbler-glass'],
  [/chocolate/i, 'chocolate-bar'],
  [/(vanilla|ice cream|soft serve)/i, 'ice-cream'],
];

/** Per-category pool used as a last-resort deterministic fallback so unmapped items still vary. */
const CATEGORY_ICON_POOL: Record<string, string[]> = {
  burgers: ['hamburger', 'cheese-wedge', 'poultry-leg', 'meat-on-bone', 'falafel', 'potato', 'sandwich'],
  pizza: ['pizza', 'cheese-wedge', 'poultry-leg', 'meat-on-bone'],
  wraps: ['stuffed-flatbread', 'burrito', 'taco', 'sandwich'],
  sides: ['french-fries', 'hot-pepper', 'onion', 'garlic'],
  beverages: ['tumbler-glass', 'cup-with-straw', 'mango', 'hot-beverage', 'lime'],
  'hot-beverages': ['hot-beverage', 'teacup-without-handle', 'teapot', 'glass-of-milk', 'bubble-tea'],
  desserts: ['chocolate-bar', 'ice-cream', 'shortcake'],
};

/** Small stable string hash (djb2) — deterministic across renders/reloads for a given product_id. */
function stableHash(value: string): number {
  let hash = 5381;
  for (let i = 0; i < value.length; i++) {
    hash = (hash * 33) ^ value.charCodeAt(i);
  }
  return Math.abs(hash);
}

/** Resolves the icon name for one specific menu item, never the same shared per-category icon. */
function resolveItemIconName(productId: string, name: string, category: string): string {
  const override = PRODUCT_ICON_OVERRIDE[productId];
  if (override) return override;

  const keywordMatch = NAME_KEYWORD_ICON.find(([pattern]) => pattern.test(name));
  if (keywordMatch) return keywordMatch[1];

  const pool = CATEGORY_ICON_POOL[category];
  if (pool && pool.length > 0) {
    return pool[stableHash(productId || name) % pool.length];
  }
  return CATEGORY_ICON_NAME[category] ?? DEFAULT_ICON_NAME;
}

// Registered once at module load — statically bundled, no runtime fetch.
addCollection(categoryIconData);

interface CategoryImageProps {
  /** Product category key, e.g. 'burgers'. Falls back to a generic plate icon. */
  category: string;
  /** Stable product identifier — enables a precise per-item icon override. */
  productId?: string;
  /** Product/item name — used for keyword-based icon matching when no productId override exists. */
  itemName?: string;
  /** Emoji to show if the icon set fails to render for any reason. */
  fallbackEmoji?: string;
  className?: string;
  size?: number;
}

/**
 * Renders a small illustrative image for a menu item or category. When
 * `productId`/`itemName` are given (item cards), each item resolves to its
 * own distinct icon instead of one icon shared by the whole category.
 * Section headers can omit them to keep showing the category-level icon.
 * Falls back to an emoji glyph so an unknown/future category or item never
 * breaks the layout.
 */
export function CategoryImage({
  category,
  productId,
  itemName,
  fallbackEmoji = '🍽',
  className,
  size = 28,
}: CategoryImageProps) {
  const [failed, setFailed] = useState(false);

  const iconName =
    productId || itemName
      ? resolveItemIconName(productId ?? '', itemName ?? '', category)
      : (CATEGORY_ICON_NAME[category] ?? DEFAULT_ICON_NAME);

  if (failed) {
    return (
      <span className={className} style={{ fontSize: size }} role="img" aria-label={category}>
        {fallbackEmoji}
      </span>
    );
  }

  return (
    <Icon
      icon={`noto:${iconName}`}
      width={size}
      height={size}
      className={className}
      aria-label={category}
      onError={() => setFailed(true)}
    />
  );
}

export default CategoryImage;
