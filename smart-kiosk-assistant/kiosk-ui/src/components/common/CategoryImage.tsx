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
  desserts: 'shortcake',
};
const DEFAULT_ICON_NAME = 'fork-and-knife-with-plate';

// Registered once at module load — statically bundled, no runtime fetch.
addCollection(categoryIconData);

interface CategoryImageProps {
  /** Product category key, e.g. 'burgers'. Falls back to a generic plate icon. */
  category: string;
  /** Emoji to show if the icon set fails to render for any reason. */
  fallbackEmoji?: string;
  className?: string;
  size?: number;
}

/**
 * Renders a small illustrative image for a menu category (e.g. all burger
 * items show one hamburger icon). Falls back to an emoji glyph so an unknown
 * or future category never breaks the layout.
 */
export function CategoryImage({
  category,
  fallbackEmoji = '🍽',
  className,
  size = 28,
}: CategoryImageProps) {
  const [failed, setFailed] = useState(false);

  const iconName = CATEGORY_ICON_NAME[category] ?? DEFAULT_ICON_NAME;

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
