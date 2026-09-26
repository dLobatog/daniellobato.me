/* Python exercise for the optimizers lesson. */
window.OPT_EXERCISES = {
  opt: {
    id: 'optimizers-v1',
    starter: `
import numpy as np


def momentum_step(w, g, v, lr, beta):
    """Heavy-ball momentum: v <- beta * v + g, then w <- w - lr * v. returns (w, v)."""
    raise NotImplementedError


def adam_step(w, g, m, s, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """One Adam step with bias correction. t is the 1-based step number.

    m <- b1 m + (1 - b1) g ;  s <- b2 s + (1 - b2) g^2
    w <- w - lr * mhat / (sqrt(shat) + eps), with mhat = m / (1 - b1^t), shat = s / (1 - b2^t)
    returns (w, m, s)
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np

def test_momentum_first_steps():
    w = np.array([1.0, -2.0]); g = np.array([0.5, 0.5]); v = np.zeros(2)
    w1, v1 = momentum_step(w, g, v, lr=0.1, beta=0.9)
    assert np.allclose(v1, g) and np.allclose(w1, w - 0.1 * g), f"first step: v = g, w -= lr*g; got w={w1}, v={v1}"
    w2, v2 = momentum_step(w1, g, v1, lr=0.1, beta=0.9)
    assert np.allclose(v2, 1.9 * g), f"with a constant gradient, v should become (1 + beta) g; got {v2}"

def test_momentum_converges_on_the_valley():
    H = np.array([1.0, 20.0]); w = np.array([-3.0, 1.5]); v = np.zeros(2)
    for _ in range(100):
        w, v = momentum_step(w, H * w, v, lr=0.1, beta=0.5)
    assert np.linalg.norm(w) < 1e-3, f"should converge on the lesson's valley; ended at {w}"

def test_adam_first_step_ignores_gradient_scale():
    for scale in [1e-3, 1.0, 1e3]:
        w1, m, s = adam_step(np.zeros(2), np.array([scale, -scale]), np.zeros(2), np.zeros(2), t=1, lr=0.01)
        assert np.allclose(w1, [-0.01, 0.01], atol=1e-6), f"with bias correction the first step is ~lr*sign(g) for any scale; got {w1} at scale {scale}"

def test_adam_matches_reference_over_several_steps():
    rng = np.random.default_rng(0)
    w = rng.normal(size=3); m = np.zeros(3); s = np.zeros(3)
    rw, rm, rs = w.copy(), m.copy(), s.copy()
    for t in range(1, 6):
        g = rng.normal(size=3)
        w, m, s = adam_step(w, g, m, s, t, lr=0.05)
        rm = 0.9 * rm + 0.1 * g; rs = 0.999 * rs + 0.001 * g * g
        rw = rw - 0.05 * (rm / (1 - 0.9 ** t)) / (np.sqrt(rs / (1 - 0.999 ** t)) + 1e-8)
    assert np.allclose(w, rw) and np.allclose(m, rm) and np.allclose(s, rs), "Adam state or weights differ from the reference after 5 steps"

def test_does_not_modify_inputs_in_place():
    w = np.array([1.0, 1.0]); m = np.zeros(2); s = np.zeros(2); g = np.array([1.0, 2.0])
    w_before = w.copy()
    adam_step(w, g, m, s, t=1)
    assert np.array_equal(w, w_before) and not m.any() and not s.any(), "return new arrays; don't mutate the caller's w, m, s"
`,
    solution: `
import numpy as np


def momentum_step(w, g, v, lr, beta):
    v = beta * v + g
    return w - lr * v, v


def adam_step(w, g, m, s, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * g
    s = b2 * s + (1 - b2) * g * g
    mhat = m / (1 - b1 ** t)              # undo the bias toward the zero initialization
    shat = s / (1 - b2 ** t)
    return w - lr * mhat / (np.sqrt(shat) + eps), m, s
`,
  },
};
