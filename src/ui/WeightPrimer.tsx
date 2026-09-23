import { Callout, Depth, Eq, M, Panel } from './kit';

/**
 * The starting point for someone with no background at all.
 *
 * Deliberately uses one everyday decision the whole way through instead of
 * switching metaphors, and never uses a word it has not already explained.
 */
export default function WeightPrimer() {
  return (
    <Panel title="Start here: what is a weight?" subtitle="No maths needed for this part">
      <Depth
        plain={
          <div className="space-y-3">
            <p>
              Imagine you are deciding whether to take an umbrella. Two things tell you something: how dark
              the sky looks, and what the forecast said.
            </p>
            <p>
              Neither one decides on its own. You give each one an amount of <em>say</em>. Perhaps you trust
              the forecast a lot and the look of the sky only a little. Those two amounts of say are the
              weights.
            </p>
            <p>
              That is all a weight is: a number that decides how much one piece of information counts. You
              multiply each input by its weight, add the results together, and you get a single score. High
              score, take the umbrella.
            </p>

            <div className="rounded-lg p-3" style={{ background: 'var(--bg-2)' }}>
              <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.07em]" style={{ color: 'var(--text-3)' }}>
                There are exactly three things you can do to a weight
              </div>
              <ul className="space-y-1.5">
                <li>
                  <strong style={{ color: 'var(--pos)' }}>Make it bigger.</strong> That input now has more say.
                  A small change in it swings the answer further.
                </li>
                <li>
                  <strong style={{ color: 'var(--text-2)' }}>Make it smaller.</strong> Less say. At exactly
                  zero the input is ignored completely, as though it were never measured.
                </li>
                <li>
                  <strong style={{ color: 'var(--neg)' }}>Make it negative.</strong> The input now argues the
                  other way. A darkening sky would start pushing you toward leaving the umbrella behind.
                </li>
              </ul>
            </div>

            <p>
              There is one more number, called the <strong>bias</strong>, and it does not belong to any input.
              It is your starting opinion before you look at anything — whether you are the sort of person who
              takes an umbrella by default. It nudges the final score up or down no matter what the inputs
              say, and you can watch it do exactly that in a moment.
            </p>

            <p>
              Below, the two sliders are the weights and the third is the bias. The square picture is the
              answer for every possible combination of the two inputs at once. Left to right is one input,
              bottom to top is the other, so every point in the square is one particular day.
            </p>

            <div className="rounded-lg p-3" style={{ background: 'var(--bg-2)' }}>
              <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.07em]" style={{ color: 'var(--text-3)' }}>
                Which colour means take it
              </div>
              <p className="mb-2">
                <strong style={{ color: 'var(--pos)' }}>Orange is take the umbrella.</strong> The score came
                out high on those days, and the deeper the orange the more certain the answer.
              </p>
              <p className="mb-2">
                <strong style={{ color: 'var(--neg)' }}>Blue is leave it at home.</strong> The score came out
                low. Again, deeper means more certain.
              </p>
              <p>
                <strong>The pale band where the two meet is the days it cannot call.</strong> The score landed
                near zero, the evidence is balanced, and a small change in the weather would flip the answer.
                Almost everything that happens in training is about moving that band to the right place.
              </p>
            </div>

            <p>
              Now the <strong>bias</strong>, which is the easiest of the three to see at work. It belongs to
              neither input, so changing it cannot tilt that band &mdash; it slides the whole picture bodily.
              Push the bias up and orange floods across the square: you are becoming the sort of person who
              takes an umbrella on days that do not really warrant it. Push it down and blue takes over,
              until you would walk into a downpour rather than carry one. The weights decide{' '}
              <em>which way the band leans</em>. The bias decides <em>how much convincing you need</em>.
            </p>

            <p>
              Move any slider and watch the whole picture respond. Nothing is being replayed &mdash; the
              square is recomputed point by point, every frame.
            </p>

            <Callout tone="insight" title="What any of this has to do with AI">
              <p className="mb-2">
                Swap the umbrella for the next word in a sentence and you have a language model. The inputs
                stop being sky and forecast and become numbers standing for the words so far. The answer
                stops being take it or leave it and becomes a score for every word in the vocabulary. The
                calculation in between does not change at all: multiply each input by its weight, add up,
                decide.
              </p>
              <p className="mb-2">
                A real model is this, with billions of weights instead of two, stacked in layers so that the
                answers of one become the inputs of the next. Nothing cleverer is added later. When a model
                &ldquo;knows&rdquo; that the word after <em>not</em> tends to be a negative one, that
                knowledge <em>is</em> a weight, sitting somewhere in the pile, orange because it pushes that
                answer up. When a model declines to say something, some set of weights is pushing those words
                down, blue.
              </p>
              <p>
                Which is why nobody can point to the line of code where a model learned a fact. There is no
                line. There is this picture, several billion times over, and what the model knows lives in
                the shape of it.
              </p>
            </Callout>
          </div>
        }
        math={
          <div className="space-y-2">
            <p>
              A neuron is an affine map followed by a nonlinearity. For inputs <M>x ∈ ℝⁿ</M>, weights{' '}
              <M>w ∈ ℝⁿ</M> and bias <M>b ∈ ℝ</M>:
            </p>
            <Eq note="The weighted sum z is the only place the inputs and the weights meet.">
              z = wᵀx + b = Σᵢ wᵢxᵢ + b,   a = f(z)
            </Eq>
            <p>
              The set of inputs where <M>z = 0</M> is a hyperplane with normal vector <M>w</M>, offset{' '}
              <M>|b| / ‖w‖</M> from the origin. Scaling <M>w</M> rotates nothing and only sharpens{' '}
              <M>f</M> across that surface; changing the direction of <M>w</M> rotates it; changing{' '}
              <M>b</M> translates it along its own normal.
            </p>
            <p>
              A layer applies this to many neurons at once as <M>a = f(Wx + b)</M>, which is the matrix
              expression step 02 takes apart.
            </p>
          </div>
        }
        code={
          <div className="space-y-2">
            <p>The entire neuron, as it is actually evaluated for the picture below:</p>
            <Eq>
              <div className="space-y-1 text-[11.5px]">
                <div>const z = w1 * x1 + w2 * x2 + b;</div>
                <div>const a = ACTIVATIONS[act].f(z);</div>
              </div>
            </Eq>
            <p>
              A whole layer is the same thing with the loop hoisted into a matrix multiply, from{' '}
              <span className="mono">engine/mlp.ts</span>:
            </p>
            <Eq>
              <div className="space-y-1 text-[11.5px]">
                <div>const z = addRowVec(matmul(cur, this.W[l]), this.b[l]);</div>
                <div>cur = mapMat(z, ACTIVATIONS[this.actAt(l)].f);</div>
              </div>
            </Eq>
          </div>
        }
      />
    </Panel>
  );
}
