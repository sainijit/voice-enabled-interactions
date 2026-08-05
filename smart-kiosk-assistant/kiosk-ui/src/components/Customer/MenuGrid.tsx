import { CategoryImage } from '../common/CategoryImage';
import { formatPrice, type GroupedCategory } from '../../menu/categories';
import type { Product } from '../../types';

/** A single display-only menu item card — image + name + price, no controls (voice-only ordering). */
function MenuCard({ item, categoryIcon }: { item: Product; categoryIcon: string }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-xl border border-gray-200 bg-white p-4 text-center shadow-sm transition-shadow hover:shadow-md">
      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-gray-50">
        <CategoryImage
          category={item.category}
          productId={item.product_id}
          itemName={item.name}
          fallbackEmoji={categoryIcon}
          size={40}
        />
      </div>
      <span className="text-sm font-semibold text-intel-dark leading-tight">{item.name}</span>
      <span className="text-sm font-bold text-intel-blue">{formatPrice(item.price)}</span>
    </div>
  );
}

interface MenuGridProps {
  categories: GroupedCategory[];
  /** Called with each category's DOM node as it mounts, so CategoryRail can scroll to it. */
  registerSection: (key: string, node: HTMLElement | null) => void;
}

/**
 * Display-only scrollable menu, grouped into one section per category
 * (queue-driven filtering already applied by the caller). No add/remove
 * controls — this screen is voice-only; customers order by tapping the Ask
 * button and speaking.
 */
export function MenuGrid({ categories, registerSection }: MenuGridProps) {
  if (categories.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-gray-400">Menu is currently unavailable.</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 p-6">
      {categories.map((cat) => (
        <section key={cat.key} ref={(node) => registerSection(cat.key, node)} aria-label={cat.label}>
          <div className="mb-3 flex items-center gap-2">
            <CategoryImage category={cat.key} fallbackEmoji={cat.icon} size={24} />
            <h2 className="text-base font-bold text-intel-dark">{cat.label}</h2>
          </div>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4">
            {cat.items.map((item) => (
              <MenuCard key={item.product_id} item={item} categoryIcon={cat.icon} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

export default MenuGrid;
