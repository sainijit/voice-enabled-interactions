import { useCallback, useEffect, useRef, useState } from 'react';

// ---------------------------------------------------------------------------
// Draggable split between the chat pane and the right-hand dashboard.
//
// The operator screen is used on very different displays (a laptop during
// development, a wide demo monitor on a stand), and the "right" split differs
// per display and per task -- reading a long transcript wants a wide chat,
// watching the hardware charts wants a wide dashboard. This hook owns that
// ratio so the operator can set it once and have it stick.
//
// Only the committed value lives in React state. While the pointer is down,
// ResizeHandle writes the column width straight to the DOM and calls
// `preview()`; committing on every pointermove would re-render the recharts
// graphs dozens of times a second and make the drag stutter.
// ---------------------------------------------------------------------------

/** Chat pane width as a percentage of the split container. */
export const SPLIT_DEFAULT_PERCENT = 60;

// Clamps. Below ~25% a pane is too narrow to be useful (the dashboard's KPI
// cards and the chat's message bubbles both start wrapping badly), and letting
// a pane reach 0 would hide it with no obvious way to get it back.
export const SPLIT_MIN_PERCENT = 25;
export const SPLIT_MAX_PERCENT = 75;

/** Percentage points moved per arrow-key press (keyboard resizing). */
export const SPLIT_KEYBOARD_STEP = 2;

const STORAGE_KEY = 'kiosk-ui.splitPercent';

export function clampSplit(value: number): number {
  if (!Number.isFinite(value)) return SPLIT_DEFAULT_PERCENT;
  return Math.min(SPLIT_MAX_PERCENT, Math.max(SPLIT_MIN_PERCENT, value));
}

function readStoredSplit(): number {
  // localStorage throws in private-mode/sandboxed contexts; a missing
  // preference must never stop the UI from rendering.
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw === null) return SPLIT_DEFAULT_PERCENT;
    return clampSplit(Number.parseFloat(raw));
  } catch {
    return SPLIT_DEFAULT_PERCENT;
  }
}

export interface ResizablePanel {
  /** Committed chat-pane width, as a percentage. Drives the grid template. */
  percent: number;
  /** Commit a new width (persisted). Used on drag end and keyboard resize. */
  setPercent: (value: number) => void;
  /** Restore the default split. Bound to double-click on the handle. */
  reset: () => void;
}

/**
 * Owns the chat/dashboard split ratio, persisted across reloads.
 *
 * Returns a clamped percentage plus setters; the DOM work lives in
 * ResizeHandle so this stays testable and free of layout concerns.
 */
export function useResizablePanel(): ResizablePanel {
  // Lazy initialiser: read localStorage once on mount rather than on every
  // render.
  const [percent, setPercentState] = useState<number>(readStoredSplit);

  // Avoid writing back the value we just read on first render.
  const hydrated = useRef(false);
  useEffect(() => {
    if (!hydrated.current) {
      hydrated.current = true;
      return;
    }
    try {
      window.localStorage.setItem(STORAGE_KEY, String(percent));
    } catch {
      // Preference is a nicety, not a requirement.
    }
  }, [percent]);

  const setPercent = useCallback((value: number) => {
    setPercentState(clampSplit(value));
  }, []);

  const reset = useCallback(() => {
    setPercentState(SPLIT_DEFAULT_PERCENT);
  }, []);

  return { percent, setPercent, reset };
}

export default useResizablePanel;
