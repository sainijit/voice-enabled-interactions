import { CategoryImage } from '../common/CategoryImage';
import type { GroupedCategory } from '../../menu/categories';

interface CategoryRailProps {
  categories: GroupedCategory[];
  activeKey: string | null;
  onSelect: (key: string) => void;
}

/**
 * Category navigation rail. Purely a scroll-to-section jump list — the
 * customer kiosk screen is voice-only for ordering (see AskBar), so tapping a
 * category never adds anything to the cart, it just scrolls the menu.
 * Renders as a vertical rail in landscape (the primary 1920x1080 kiosk
 * layout) and collapses to a horizontal snap-scroll strip on narrow/portrait
 * screens (see CustomerApp's responsive classes).
 */
export function CategoryRail({ categories, activeKey, onSelect }: CategoryRailProps) {
  return (
    <nav
      aria-label="Menu categories"
      className="flex gap-2 overflow-x-auto lg:flex-col lg:overflow-visible lg:gap-1.5 lg:overflow-y-auto"
    >
      {categories.map((cat) => {
        const isActive = cat.key === activeKey;
        return (
          <button
            key={cat.key}
            type="button"
            onClick={() => onSelect(cat.key)}
            aria-current={isActive}
            className={`flex shrink-0 items-center gap-2.5 rounded-xl border px-3.5 py-3 text-left transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-intel-blue/40 lg:w-full
              ${
                isActive
                  ? 'border-intel-blue bg-blue-50 text-intel-blue shadow-sm'
                  : 'border-gray-200 bg-white text-gray-600 hover:border-intel-blue/40 hover:bg-blue-50/40'
              }`}
          >
            <CategoryImage category={cat.key} fallbackEmoji={cat.icon} size={26} />
            <span className="text-sm font-semibold whitespace-nowrap">{cat.label}</span>
          </button>
        );
      })}
    </nav>
  );
}

export default CategoryRail;
