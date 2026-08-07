import { useCallback, useRef, useState } from 'react';

import {
  SPLIT_KEYBOARD_STEP,
  SPLIT_MAX_PERCENT,
  SPLIT_MIN_PERCENT,
  clampSplit,
} from '../../hooks/useResizablePanel';

interface ResizeHandleProps {
  /** Current chat-pane width, as a percentage of the container. */
  percent: number;
  /** Commit a new width. Called on drag end and on each keyboard step. */
  onCommit: (percent: number) => void;
  /** Restore the default split (double-click). */
  onReset: () => void;
  /** The grid element being resized; its width converts px drag -> percent. */
  containerRef: React.RefObject<HTMLElement | null>;
  /**
   * Applies a width without committing it to React state, for live feedback
   * during a drag. Kept separate from `onCommit` so pointermove never triggers
   * a re-render of the charts.
   */
  onPreview: (percent: number) => void;
}

/**
 * Draggable divider between the chat pane and the dashboard.
 *
 * Uses Pointer Events rather than mouse/touch handlers so mouse, touch and pen
 * share one code path, and calls `setPointerCapture` so the drag keeps
 * tracking even when the pointer moves faster than layout can follow and
 * leaves the handle -- without capture the divider "sticks" mid-drag.
 *
 * Hidden below `lg`, where the layout stacks vertically and a horizontal
 * split has no meaning.
 */
export function ResizeHandle({
  percent,
  onCommit,
  onReset,
  containerRef,
  onPreview,
}: ResizeHandleProps) {
  const [dragging, setDragging] = useState(false);
  // Holds the latest previewed value so pointerup can commit it without
  // needing a state update on every move.
  const latest = useRef(percent);

  const percentFromEvent = useCallback(
    (clientX: number, handleWidth: number): number | null => {
      const el = containerRef.current;
      if (!el) return null;
      const rect = el.getBoundingClientRect();
      // The grid container is padded (p-3), and the columns are laid out in
      // its *content* box. Measuring against the border box would offset the
      // divider from the cursor by the padding width.
      const style = window.getComputedStyle(el);
      const padLeft = Number.parseFloat(style.paddingLeft) || 0;
      const padRight = Number.parseFloat(style.paddingRight) || 0;
      const contentLeft = rect.left + padLeft;
      const contentWidth = rect.width - padLeft - padRight;
      if (contentWidth <= 0) return null;
      // The pointer grabs the middle of the handle, but the chat column ends
      // at the handle's leading edge.
      const x = clientX - contentLeft - handleWidth / 2;
      return clampSplit((x / contentWidth) * 100);
    },
    [containerRef],
  );

  const handlePointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      // Ignore secondary buttons so a right-click never starts a drag.
      if (event.button !== 0) return;
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      latest.current = percent;
      setDragging(true);
      // Without this the drag selects chat text and the I-beam cursor flickers
      // over the panes.
      document.body.style.userSelect = 'none';
      document.body.style.cursor = 'col-resize';
    },
    [percent],
  );

  const handlePointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!dragging) return;
      const next = percentFromEvent(event.clientX, event.currentTarget.offsetWidth);
      if (next === null) return;
      latest.current = next;
      onPreview(next);
    },
    [dragging, onPreview, percentFromEvent],
  );

  const endDrag = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!dragging) return;
      setDragging(false);
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
      onCommit(latest.current);
    },
    [dragging, onCommit],
  );

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      // Keyboard resizing: a pointer-only divider is unusable for anyone not
      // using a mouse, and this is the operator-facing screen.
      let next: number | null = null;
      if (event.key === 'ArrowLeft') next = percent - SPLIT_KEYBOARD_STEP;
      else if (event.key === 'ArrowRight') next = percent + SPLIT_KEYBOARD_STEP;
      else if (event.key === 'Home') next = SPLIT_MIN_PERCENT;
      else if (event.key === 'End') next = SPLIT_MAX_PERCENT;
      else if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        onReset();
        return;
      }
      if (next === null) return;
      event.preventDefault();
      onCommit(next);
    },
    [onCommit, onReset, percent],
  );

  return (
    <div
      // The visible line is 2px, but a 2px pointer target is genuinely hard to
      // grab, so the element itself is the full grid column and the line is
      // drawn by the inner span.
      className={`hidden lg:flex items-center justify-center group relative
                  cursor-col-resize touch-none select-none
                  focus:outline-none`}
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize chat and dashboard panels"
      aria-valuenow={Math.round(percent)}
      aria-valuemin={SPLIT_MIN_PERCENT}
      aria-valuemax={SPLIT_MAX_PERCENT}
      tabIndex={0}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onDoubleClick={onReset}
      onKeyDown={handleKeyDown}
      title="Drag to resize · double-click to reset"
    >
      {/* Visible divider line. */}
      <span
        className={`h-full w-[2px] rounded-full transition-colors
                    ${dragging ? 'bg-intel-blue' : 'bg-gray-300 group-hover:bg-intel-blue/60'}
                    group-focus-visible:bg-intel-blue`}
      />
      {/* Grip dots -- without an affordance the divider reads as decoration
          and operators never discover it is draggable. */}
      <span
        className={`absolute flex flex-col gap-[3px] rounded-full px-[3px] py-1.5
                    transition-colors
                    ${dragging ? 'bg-intel-blue' : 'bg-gray-300 group-hover:bg-intel-blue/60'}`}
        aria-hidden="true"
      >
        <span className="block h-[3px] w-[3px] rounded-full bg-white/90" />
        <span className="block h-[3px] w-[3px] rounded-full bg-white/90" />
        <span className="block h-[3px] w-[3px] rounded-full bg-white/90" />
      </span>
    </div>
  );
}

export default ResizeHandle;
