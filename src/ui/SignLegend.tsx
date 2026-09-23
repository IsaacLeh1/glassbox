import { Callout, Depth, Eq, M, Panel } from './kit';

function Swatch({ color, label, sub }: { color: string; label: string; sub: string }) {
  return (
    <div className="flex items-start gap-2.5">
      <span
        className="mt-[3px] block shrink-0 rounded"
        style={{ width: 26, height: 14, background: color, border: '1px solid var(--border-2)' }}
      />
      <div className="min-w-0">
        <div className="text-[12px] font-semibold" style={{ color: 'var(--text)' }}>
          {label}
        </div>
        <div className="text-[11.5px] leading-snug" style={{ color: 'var(--text-3)' }}>
          {sub}
        </div>
      </div>
    </div>
  );
}

/**
 * What the two colours mean, everywhere in the application.
 *
 * Blue and amber are used consistently for negative and positive throughout
 * Glassbox, so this is explained once, properly, and referenced from the
 * places that use it. The pair is chosen over red and green because it stays
 * distinguishable for viewers with red-green colour blindness.
 */
export default function SignLegend({ compact = false }: { compact?: boolean }) {
  const swatches = (
    <div className="grid gap-3 sm:grid-cols-2">
      <Swatch
        color="rgba(76, 126, 243, 0.95)"
        label="Blue means negative"
        sub="This number is below zero. Whatever it is attached to argues against the answer."
      />
      <Swatch
        color="rgba(245, 158, 11, 0.95)"
        label="Amber means positive"
        sub="This number is above zero. Whatever it is attached to argues for the answer."
      />
      <Swatch
        color="rgba(245, 158, 11, 0.22)"
        label="Pale means weak"
        sub="Close to zero. It barely matters either way, and the model would hardly notice if it vanished."
      />
      <Swatch
        color="rgba(76, 126, 243, 1)"
        label="Strong means strong"
        sub="Far from zero. This one is doing real work and changing it will visibly change the output."
      />
    </div>
  );

  if (compact) return swatches;

  return (
    <Panel title="What blue and amber mean" subtitle="The same two colours are used for this everywhere in the app">
      {swatches}

      <div className="mt-4">
        <Depth
          plain={
            <div className="space-y-3">
              <p>
                Every number in a neural network can be positive or negative, and the colour tells you which
                without having to read the digits. The strength of the colour tells you how big it is.
              </p>

              <div className="rounded-lg p-3" style={{ background: 'var(--bg-2)' }}>
                <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.07em]" style={{ color: 'var(--text-3)' }}>
                  What actually changes in the model
                </div>
                <p className="mb-2">
                  <strong style={{ color: 'var(--pos)' }}>Drag a weight toward amber</strong> and you are
                  telling the model: when this input goes up, push the answer up too. Evidence in favour.
                  Somewhere in a real language model, this is the connection that makes the word
                  &ldquo;not&rdquo; raise the chance of a negative word coming next.
                </p>
                <p className="mb-2">
                  <strong style={{ color: 'var(--neg)' }}>Drag it toward blue</strong> and you are telling it
                  the opposite: when this input goes up, push the answer <em>down</em>. Evidence against. This
                  is just as important — a great deal of what a model knows is what to rule out.
                </p>
                <p>
                  <strong>Drag it to the middle, where the colour disappears</strong>, and the connection is
                  effectively cut. The input still arrives, but it is multiplied by nothing and has no effect
                  at all.
                </p>
              </div>

              <p>
                In the square picture of the input space the same rule applies to the answer rather than to a
                weight: amber regions are where the model says yes, blue regions are where it says no, and the
                pale band between them is where it genuinely is not sure. That pale band is the boundary, and
                almost everything in training is about moving it to the right place.
              </p>

              <p>
                One thing worth internalising: a big negative number is not a mistake or a broken value. In a
                trained model, roughly half of all weights are negative. &ldquo;Push down&rdquo; and
                &ldquo;push up&rdquo; are equally useful things to have learned.
              </p>
            </div>
          }
          math={
            <div className="space-y-2">
              <p>
                Colour encodes <M>sign(x)</M> and opacity encodes <M>|x| / max|x|</M> within the panel being
                drawn, so intensity is always relative to the largest magnitude on screen rather than to a
                fixed absolute scale.
              </p>
              <Eq note="From ui/kit.tsx. Magnitude drives alpha, so a value near zero fades into the background.">
                hue = x ≥ 0 ? amber : blue,  alpha = min(1, |x| / max)
              </Eq>
              <p>
                For a probability field the mapping is recentred on <M>0.5</M>: the displayed quantity is{' '}
                <M>2(p − 0.5)</M>, so <M>p = 0.5</M> renders as the neutral background and the decision
                boundary is exactly the pale line.
              </p>
              <p>
                Blue and amber are used rather than red and green because they remain distinguishable under
                deuteranopia and protanopia, and because they differ in luminance as well as hue.
              </p>
            </div>
          }
          code={
            <Eq>
              <div className="space-y-1 text-[11px]">
                <div>const NEG = [76, 126, 243]; // blue</div>
                <div>const POS = [245, 158, 11]; // amber</div>
                <div>const t = Math.min(1, Math.abs(v) / max);</div>
                <div>const [r, g, b] = v &gt;= 0 ? POS : NEG;</div>
                <div>return `rgba(${'{r}'}, ${'{g}'}, ${'{b}'}, ${'{t}'})`;</div>
              </div>
            </Eq>
          }
        />
      </div>

      <div className="mt-3">
        <Callout tone="insight" title="Why sign matters more than size">
          If you flip one weight from amber to blue you have not made the model slightly worse at its job, you
          have reversed what that piece of evidence means to it. Sign changes behaviour; magnitude changes
          confidence. When you are dragging sliders in this step, watch which of those two you are doing.
        </Callout>
      </div>
    </Panel>
  );
}
