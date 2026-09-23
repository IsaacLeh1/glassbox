import { useEffect, useRef, useState } from 'react';

/**
 * A frame loop that keeps running when the browser stops painting.
 *
 * The obvious implementation is requestAnimationFrame, and it fails in a way
 * that is hard to diagnose: a long training run simply stops, with no error
 * and no indication why. There are several reasons a frame may never arrive,
 * and `document.hidden` only covers one of them:
 *
 *  - the tab is in the background, which document.hidden does report,
 *  - the window is behind another window, so nothing is being composited,
 *  - the display has gone to sleep,
 *  - the window is minimised.
 *
 * In every case but the first, `document.hidden` stays false while frames stop
 * completely. So rather than trying to detect the cause, each cycle races the
 * animation frame against a timer and takes whichever arrives first. When
 * frames are flowing the timer never wins and this behaves exactly like a
 * plain rAF loop; when they stop, the timer takes over and the run continues.
 *
 * The work done per tick is still bounded by the caller's own time budget, so
 * a background run uses no more CPU than a foreground one.
 */

/**
 * How long to wait for an animation frame before falling back to a timer.
 * Comfortably longer than a 60Hz frame, so the timer does not win races on a
 * healthy page, and short enough that a stalled run barely slows down. A
 * genuinely hidden tab has its timers clamped to roughly one a second by the
 * browser, which is slow but keeps the run alive rather than stopping it dead.
 */
const FALLBACK_MS = 32;

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
      let fired = false;
      const fire = () => {
        if (stopped || fired) return;
        fired = true;
        cancelAnimationFrame(rafId);
        if (timerId !== undefined) clearTimeout(timerId);
        run();
      };
      rafId = requestAnimationFrame(fire);
      timerId = window.setTimeout(fire, FALLBACK_MS);
    };

    const run = () => {
      if (stopped) return;
      try {
        cb.current();
      } catch (err) {
        // A throw used to freeze the loop silently, because the frame that
        // failed never scheduled a successor. Stop deliberately and let the
        // error surface, rather than leaving a run that merely looks slow.
        stopped = true;
        throw err;
      }
      schedule();
    };

    schedule();
    return () => {
      stopped = true;
      cancelAnimationFrame(rafId);
      if (timerId !== undefined) clearTimeout(timerId);
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
