/* Python exercises for the backprop lesson. */
window.BP_EXERCISES = {
  mlp: {
    id: 'mlp-backward-v1',
    starter: `
import numpy as np


def mlp_loss_and_grads(X, y, W1, b1, w2, b2):
    """One hidden ReLU layer, sigmoid output, mean binary cross-entropy.

    X: (N, D)   y: (N,) of 0/1   W1: (D, H)   b1: (H,)   w2: (H,)   b2: float
      h = relu(X @ W1 + b1)      z = h @ w2 + b2      p = sigmoid(z)
    loss = mean over N of -[y log p + (1 - y) log(1 - p)], computed stably from z.
    returns (loss, grads) with grads = {"W1": ..., "b1": ..., "w2": ..., "b2": ...}
    (same shapes as the parameters; b2's gradient is a float).
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np

def _ref(X, y, W1, b1, w2, b2):
    z = np.maximum(X @ W1 + b1, 0) @ w2 + b2
    return float(np.mean(np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z)))))

def _lesson():
    X = np.array([[1.0, 0.6]]); y = np.array([0.0])
    W1 = np.array([[1.0, -0.5], [0.5, -1.0]]); b1 = np.array([0.1, 0.2]); w2 = np.array([1.5, 0.8]); b2 = 0.1
    return X, y, W1, b1, w2, b2

def test_lesson_example_loss():
    loss, _ = mlp_loss_and_grads(*_lesson())
    assert abs(loss - 2.305083) < 1e-5, f"expected 2.305083, got {loss}"

def test_lesson_example_gradients():
    _, g = mlp_loss_and_grads(*_lesson())
    assert np.allclose(g["w2"], [1.260349, 0.0], atol=1e-5), f"w2 grad {g['w2']}"
    assert np.allclose(g["W1"], [[1.350374, 0.0], [0.810225, 0.0]], atol=1e-5), f"W1 grad {g['W1']}"
    assert np.allclose(g["b1"], [1.350374, 0.0], atol=1e-5), f"b1 grad {g['b1']}"
    assert abs(float(g["b2"]) - 0.900250) < 1e-5, f"b2 grad {g['b2']}"

def test_gradients_match_finite_differences_on_a_batch():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 3)); y = (rng.random(7) > 0.5).astype(float)
    W1 = rng.normal(size=(3, 4)); b1 = rng.normal(size=4) * 0.1; w2 = rng.normal(size=4); b2 = 0.2
    _, g = mlp_loss_and_grads(X, y, W1, b1, w2, b2)
    h = 1e-6
    for name, arr in [("W1", W1), ("b1", b1), ("w2", w2)]:
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            arr[idx] = old + h; lp = _ref(X, y, W1, b1, w2, b2)
            arr[idx] = old - h; lm = _ref(X, y, W1, b1, w2, b2)
            arr[idx] = old
            num[idx] = (lp - lm) / (2 * h)
        err = np.max(np.abs(np.asarray(g[name]) - num))
        assert err < 1e-5, f"{name}: max |analytic - numeric| = {err:.2e} (did you average over N?)"
    nb = (_ref(X, y, W1, b1, w2, b2 + h) - _ref(X, y, W1, b1, w2, b2 - h)) / (2 * h)
    assert abs(float(g["b2"]) - nb) < 1e-5, f"b2: analytic {g['b2']} vs numeric {nb}"

def test_stable_for_huge_logits():
    X = np.array([[100.0, 100.0]]); y = np.array([0.0])
    loss, g = mlp_loss_and_grads(X, y, np.ones((2, 2)), np.zeros(2), np.ones(2), 0.0)
    assert np.isfinite(loss), f"loss is {loss}: compute it from the logit, not from log(1 - sigmoid(z))"
    assert abs(loss - 400.0) < 1e-6, f"expected 400.0, got {loss}"
    assert all(np.all(np.isfinite(np.asarray(v))) for v in g.values()), "gradients must be finite"

def test_shapes():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 3)); y = np.ones(5)
    _, g = mlp_loss_and_grads(X, y, rng.normal(size=(3, 6)), np.zeros(6), rng.normal(size=6), 0.0)
    assert np.shape(g["W1"]) == (3, 6) and np.shape(g["b1"]) == (6,) and np.shape(g["w2"]) == (6,), "gradient shapes must match parameter shapes"
    assert np.ndim(g["b2"]) == 0, "b2's gradient should be a scalar"
`,
    solution: `
import numpy as np


def mlp_loss_and_grads(X, y, W1, b1, w2, b2):
    N = X.shape[0]
    a = X @ W1 + b1                       # (N, H) pre-activations
    h = np.maximum(a, 0)                  # (N, H)
    z = h @ w2 + b2                       # (N,)
    # Stable BCE from logits: softplus(z) - y*z
    loss = np.mean(np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z))))
    p = 0.5 * (1 + np.tanh(z / 2))        # stable sigmoid
    dz = (p - y) / N                      # (N,)  blame at each logit, mean over N
    gw2 = h.T @ dz                        # (H,)
    gb2 = float(dz.sum())
    dh = np.outer(dz, w2) * (a > 0)       # (N, H) through w2, gated by ReLU
    gW1 = X.T @ dh                        # (D, H)
    gb1 = dh.sum(axis=0)                  # (H,)
    return float(loss), {"W1": gW1, "b1": gb1, "w2": gw2, "b2": gb2}
`,
  },
  num: {
    id: 'numerical-grad-v1',
    starter: `
import numpy as np


def numerical_grad(f, x, h=1e-5):
    """Central-difference gradient of a scalar function f at array x (any shape).

    returns an array shaped like x whose entry i is (f(x + h e_i) - f(x - h e_i)) / (2h).
    Must not modify the caller's x.
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np

def test_sum_of_squares():
    x = np.array([1.0, -2.0, 3.0])
    g = np.asarray(numerical_grad(lambda v: float(np.sum(v ** 2)), x))
    assert np.allclose(g, 2 * x, atol=1e-6), f"expected {2 * x}, got {g}"

def test_product_of_entries():
    x = np.array([3.0, 4.0])
    g = np.asarray(numerical_grad(lambda v: float(v[0] * v[1]), x))
    assert np.allclose(g, [4.0, 3.0], atol=1e-6), f"expected [4, 3], got {g}"

def test_matrix_input_keeps_shape():
    x = np.arange(6, dtype=float).reshape(2, 3)
    g = np.asarray(numerical_grad(lambda v: float(np.sum(np.sin(v))), x))
    assert g.shape == (2, 3), f"expected shape (2, 3), got {g.shape}"
    assert np.allclose(g, np.cos(x), atol=1e-6), "gradient of sum(sin(x)) should be cos(x)"

def test_does_not_modify_input():
    x = np.array([0.5, 1.5]); before = x.copy()
    numerical_grad(lambda v: float(np.sum(v ** 3)), x)
    assert np.array_equal(x, before), f"x was modified: {before} -> {x}"
`,
    solution: `
import numpy as np


def numerical_grad(f, x, h=1e-5):
    x = np.array(x, dtype=float)          # a private copy
    g = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + h; fp = f(x)
        x[idx] = old - h; fm = f(x)
        x[idx] = old
        g[idx] = (fp - fm) / (2 * h)
    return g
`,
  },
};
