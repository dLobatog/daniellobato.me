/* Python exercises for the cross-entropy lesson. Tests run in the browser via Pyodide. */
window.CE_EXERCISES = {
  ce: {
    id: 'cross-entropy-v1',
    starter: `
import numpy as np


def cross_entropy(logits, targets):
    """Mean cross-entropy in nats.

    logits:  float array, shape (N, V), one row of unnormalized scores per example
    targets: int array, shape (N,), index of the observed class in each row
    returns: float, mean over n of -log softmax(logits[n])[targets[n]]
    Must stay finite for large logits (e.g. 1000).
    """
    raise NotImplementedError


def cross_entropy_grad(logits, targets):
    """Gradient of cross_entropy(logits, targets) with respect to logits.

    returns: float array, shape (N, V)
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np
_rng = np.random.default_rng(0)

def _ref(z, y):
    m = z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(y)), y]))

def test_uniform_logits_give_ln_V():
    z = np.zeros((3, 4)); y = np.array([0, 1, 3])
    got = cross_entropy(z, y)
    assert abs(got - np.log(4)) < 1e-9, f"expected ln 4 = {np.log(4):.6f}, got {got}"

def test_matches_reference_on_random_batch():
    z = _rng.normal(size=(5, 7)) * 3; y = _rng.integers(0, 7, size=5)
    got = cross_entropy(z, y); exp = _ref(z, y)
    assert abs(got - exp) < 1e-9, f"expected {exp:.6f}, got {got}"

def test_is_a_mean_not_a_sum():
    z = _rng.normal(size=(1, 4)); y = np.array([2])
    one = cross_entropy(z, y)
    many = cross_entropy(np.repeat(z, 6, axis=0), np.repeat(y, 6))
    assert abs(one - many) < 1e-9, "repeating the batch 6 times should not change a mean"

def test_stable_for_large_logits():
    z = np.array([[1000.0, 0.0, -1000.0], [0.0, 1000.0, 999.0]]); y = np.array([0, 2])
    got = cross_entropy(z, y)
    assert np.isfinite(got), f"got {got}: exp(1000) overflows, so subtract each row's max first"
    exp = _ref(z, y)
    assert abs(got - exp) < 1e-6, f"expected {exp:.6f}, got {got}"

def test_shift_invariance():
    z = _rng.normal(size=(4, 5)); y = _rng.integers(0, 5, size=4)
    a = cross_entropy(z, y); b = cross_entropy(z + 50.0, y)
    assert abs(a - b) < 1e-9, "adding a constant to every logit in a row must not change the loss"

def test_grad_shape_and_rows_sum_to_zero():
    z = _rng.normal(size=(3, 6)); y = np.array([0, 5, 2])
    g = np.asarray(cross_entropy_grad(z, y))
    assert g.shape == z.shape, f"gradient shape {g.shape}, expected {z.shape}"
    assert np.allclose(g.sum(axis=1), 0, atol=1e-12), "each row of dL/dz should sum to 0"

def test_grad_matches_finite_differences():
    z = _rng.normal(size=(3, 4)); y = np.array([1, 0, 3])
    g = np.asarray(cross_entropy_grad(z, y)); eps = 1e-6; num = np.zeros_like(z)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            zp = z.copy(); zp[i, j] += eps
            zm = z.copy(); zm[i, j] -= eps
            num[i, j] = (_ref(zp, y) - _ref(zm, y)) / (2 * eps)
    err = np.max(np.abs(g - num))
    assert err < 1e-6, f"max |analytic - numeric| = {err:.2e} (did you divide by N?)"

def test_grad_stable_for_large_logits():
    z = np.array([[1000.0, 0.0, -1000.0]]); y = np.array([1])
    g = np.asarray(cross_entropy_grad(z, y))
    assert np.all(np.isfinite(g)), f"gradient has non-finite values: {g}"
    assert np.allclose(g, [[1.0, -1.0, 0.0]], atol=1e-9), f"expected [[1, -1, 0]], got {g}"
`,
    solution: `
import numpy as np


def _log_softmax(logits):
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=1, keepdims=True)          # largest entry becomes 0: exp can't overflow
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def cross_entropy(logits, targets):
    logp = _log_softmax(logits)
    n = logp.shape[0]
    return float(-logp[np.arange(n), targets].mean())


def cross_entropy_grad(logits, targets):
    q = np.exp(_log_softmax(logits))              # softmax, computed stably
    n = q.shape[0]
    q[np.arange(n), targets] -= 1.0               # q - one_hot(y)
    return q / n                                  # the loss is a mean over N rows
`,
  },
  kl: {
    id: 'kl-divergence-v1',
    starter: `
import numpy as np


def kl_divergence(p, q):
    """KL(p || q) in nats for two probability vectors of the same length.

    Conventions:
      - a term with p[i] == 0 contributes 0, even if q[i] == 0;
      - if p[i] > 0 and q[i] == 0 for any i, return np.inf.
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np
_rng = np.random.default_rng(1)

def test_zero_for_identical_distributions():
    p = np.array([0.5, 0.25, 0.2, 0.05])
    got = kl_divergence(p, p)
    assert abs(got) < 1e-12, f"KL(p||p) should be 0, got {got}"

def test_lesson_value_uniform_model():
    p = np.array([0.5, 0.25, 0.2, 0.05]); q = np.full(4, 0.25)
    exp = float(np.sum(p * np.log(p / q)))
    got = kl_divergence(p, q)
    assert abs(got - exp) < 1e-9, f"expected {exp:.6f} nats (0.3195 bits), got {got}"

def test_not_symmetric():
    p = np.array([0.9, 0.1]); q = np.array([0.5, 0.5])
    a = kl_divergence(p, q); b = kl_divergence(q, p)
    assert abs(a - b) > 1e-3, f"KL(p||q)={a:.4f} and KL(q||p)={b:.4f} should differ"

def test_zero_in_p_contributes_nothing():
    got = kl_divergence(np.array([0.5, 0.5, 0.0]), np.array([0.25, 0.25, 0.5]))
    assert abs(got - np.log(2)) < 1e-9, f"expected ln 2 = {np.log(2):.6f}, got {got}"

def test_zero_in_both_is_fine():
    got = kl_divergence(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    assert got == 0 or abs(got) < 1e-12, f"expected 0, got {got}"

def test_q_zero_where_p_positive_is_infinite():
    got = kl_divergence(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
    assert got == np.inf, f"expected inf, got {got}"

def test_equals_cross_entropy_minus_entropy():
    for _ in range(5):
        p = _rng.dirichlet(np.ones(6)); q = _rng.dirichlet(np.ones(6))
        ce = -np.sum(p * np.log(q)); h = -np.sum(p * np.log(p))
        got = kl_divergence(p, q)
        assert abs(got - (ce - h)) < 1e-9, f"expected H(p,q) - H(p) = {ce - h:.6f}, got {got}"
        assert got >= 0, f"KL must be non-negative, got {got}"
`,
    solution: `
import numpy as np


def kl_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    support = p > 0                      # 0 * log(0 / q) = 0 by convention
    if np.any(q[support] == 0):
        return np.inf                    # the model rules out something that happens
    return float(np.sum(p[support] * np.log(p[support] / q[support])))
`,
  },
};
