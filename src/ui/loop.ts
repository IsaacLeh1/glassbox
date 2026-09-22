import { useEffect, useRef, useState } from 'react';

/**
 * A frame loop that keeps running when the tab is in the background.
 *
 * requestAnimationFrame is suspended entirely in a hidden tab, which would
 * silently freeze a training run the moment someone switches away. That is a
 * poor experience for a long job, so this falls back to a timer when hidden
 * and returns to the animation frame when the tab comes back.
 *
 * The work done per tick is still bounded by the caller's own time budget, so
 * a background run uses no more CPU than a foreground one.
 */
export function useFrameLoop(active: boolean, onFrame: () => void) {
  const cb = useRef(onFrame);
  cb.current = onFrame;

  useEffect(() => {
    if (!active) return;
    let stopped = false;
    let rafId = 0;
    let timerId: number | undefined;

    const schedule = () => {
      if (stopped) return;
      if (document.hidden) {
        // Browsers clamp background timers to about once a second, which is
        // slow but keeps the run alive rather than stopping it dead.
        timerId = window.setTimeout(run, 16);
      } else {
        rafId = requestAnimationFrame(run);
      }
    };

    const run = () => {
      if (stopped) return;
      cb.current();
      schedule();
    };

    const onVisibility = () => {
      // Cancel whichever scheduler is pending and pick the right one again.
      cancelAnimationFrame(rafId);
      if (timerId !== undefined) clearTimeout(timerId);
      schedule();
    };

    schedule();
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      stopped = true;
      cancelAnimationFrame(rafId);
      if (timerId !== undefined) clearTimeout(timerId);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [active]);
}

/** True while this tab is the one on screen. */
export function usePageVisible(): boolean {
  const [visible, setVisible] = useState(() => !document.hidden);
  useEffect(() => {
    const on = () => setVisible(!document.hidden);
    document.addEventListener('visibilitychange', on);
    return () => document.removeEventListener('visibilitychange', on);
  }, []);
  return visible;
}
