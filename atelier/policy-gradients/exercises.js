/* Python exercises for the policy-gradient lesson. */
window.PG_EXERCISES = {
  adv: {
    id: 'grpo-advantages-v1',
    starter: `
import numpy as np


def grpo_advantages(rewards, eps=1e-6):
    """Group-relative advantages.

    rewards: float array, shape (P, G): G sampled completions for each of P prompts
    returns: float array, shape (P, G): (r - mean) / (std + eps), with mean and the
             population std (ddof=0) taken over each prompt's own group
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np

def test_shape_is_preserved():
    r = np.zeros((3, 5)); a = np.asarray(grpo_advantages(r))
    assert a.shape == (3, 5), f"expected shape (3, 5), got {a.shape}"

def test_one_right_out_of_four():
    a = np.asarray(grpo_advantages(np.array([[1.0, 0.0, 0.0, 0.0]])))
    exp = np.array([[np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)]])
    assert np.allclose(a, exp, atol=1e-4), f"expected {np.round(exp, 3)}, got {np.round(a, 3)}"

def test_equal_rewards_give_zero_not_nan():
    a = np.asarray(grpo_advantages(np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])))
    assert np.all(np.isfinite(a)), f"got non-finite values {a}: guard the zero std"
    assert np.allclose(a, 0), f"expected zeros for equal rewards, got {a}"

def test_each_prompt_is_normalized_separately():
    r = np.array([[1.0, 0.0, 0.0, 0.0], [10.0, 20.0, 30.0, 40.0]])
    a = np.asarray(grpo_advantages(r))
    assert np.allclose(a.mean(axis=1), 0, atol=1e-6), f"each row should have mean 0, got {a.mean(axis=1)}"
    assert abs(a[0, 0] - np.sqrt(3)) < 1e-3, "row 0 must not be affected by row 1's scale (normalize per row)"

def test_invariant_to_shifting_and_scaling_rewards():
    rng = np.random.default_rng(0); r = rng.random((4, 6))
    a = np.asarray(grpo_advantages(r)); b = np.asarray(grpo_advantages(10 * r + 3))
    assert np.allclose(a, b, atol=1e-4), "advantages should not change when every reward is scaled and shifted"

def test_population_std():
    a = np.asarray(grpo_advantages(np.array([[1.0, 0.0]])))
    assert np.allclose(a, [[1.0, -1.0]], atol=1e-4), f"with ddof=0, [1, 0] gives [+1, -1]; got {a}"
`,
    solution: `
import numpy as np


def grpo_advantages(rewards, eps=1e-6):
    r = np.asarray(rewards, dtype=float)
    mean = r.mean(axis=1, keepdims=True)
    std = r.std(axis=1, keepdims=True)        # population std (ddof=0)
    return (r - mean) / (std + eps)           # equal rewards -> 0 / eps = 0
`,
  },
  clip: {
    id: 'ppo-clip-v1',
    starter: `
import numpy as np


def clipped_surrogate(logp_new, logp_old, adv, eps=0.2):
    """PPO/GRPO clipped surrogate, per token (to be maximized).

    logp_new, logp_old, adv: float arrays of the same shape (N,)
    returns (obj, grad):
      obj  = min(rho * adv, clip(rho, 1 - eps, 1 + eps) * adv), rho = exp(logp_new - logp_old)
      grad = d obj / d logp_new, elementwise
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np

def _call(ln, lo, a, eps=0.2):
    obj, grad = clipped_surrogate(np.array(ln, float), np.array(lo, float), np.array(a, float), eps)
    return np.asarray(obj, float), np.asarray(grad, float)

def test_ratio_one_gives_advantage_and_gradient_A():
    obj, g = _call([np.log(0.3)], [np.log(0.3)], [1.5])
    assert np.allclose(obj, [1.5]) and np.allclose(g, [1.5]), f"rho=1: expected obj=A=1.5, grad=A=1.5; got {obj}, {g}"

def test_good_completion_past_upper_clip_gets_no_gradient():
    obj, g = _call([np.log(1.3)], [0.0], [1.0])
    assert np.allclose(obj, [1.2]), f"A>0, rho=1.3: obj should be 1.2*A, got {obj}"
    assert np.allclose(g, [0.0]), f"A>0, rho=1.3: clipped, gradient should be 0, got {g}"

def test_bad_completion_that_became_likelier_is_not_clipped():
    obj, g = _call([np.log(1.3)], [0.0], [-1.0])
    assert np.allclose(obj, [-1.3]), f"A<0, rho=1.3: min picks the unclipped -1.3, got {obj}"
    assert np.allclose(g, [-1.3]), f"A<0, rho=1.3: gradient is rho*A = -1.3, got {g}"

def test_bad_completion_past_lower_clip_gets_no_gradient():
    obj, g = _call([np.log(0.7)], [0.0], [-1.0])
    assert np.allclose(obj, [-0.8]), f"A<0, rho=0.7: obj should be 0.8*A = -0.8, got {obj}"
    assert np.allclose(g, [0.0]), f"A<0, rho=0.7: clipped, gradient 0, got {g}"

def test_good_completion_below_range_is_not_clipped():
    obj, g = _call([np.log(0.7)], [0.0], [1.0])
    assert np.allclose(obj, [0.7]) and np.allclose(g, [0.7]), f"A>0, rho=0.7: expected obj=0.7, grad=0.7; got {obj}, {g}"

def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    lo = np.log(rng.uniform(0.1, 0.9, 40)); ln = lo + rng.uniform(-0.5, 0.5, 40); a = rng.normal(size=40)
    rho = np.exp(ln - lo); keep = (np.abs(rho - 0.8) > 0.02) & (np.abs(rho - 1.2) > 0.02)
    ln, lo, a = ln[keep], lo[keep], a[keep]
    _, g = _call(ln, lo, a); h = 1e-6
    op, _ = _call(ln + h, lo, a); om, _ = _call(ln - h, lo, a)
    num = (op - om) / (2 * h)
    err = np.max(np.abs(g - num))
    assert err < 1e-5, f"max |analytic - numeric| = {err:.2e}"
`,
    solution: `
import numpy as np


def clipped_surrogate(logp_new, logp_old, adv, eps=0.2):
    rho = np.exp(np.asarray(logp_new) - np.asarray(logp_old))
    unclipped = rho * adv
    clipped = np.clip(rho, 1 - eps, 1 + eps) * adv
    obj = np.minimum(unclipped, clipped)
    # d(rho * adv)/d logp_new = rho * adv; the clipped branch is constant where it is active.
    active = unclipped <= clipped
    grad = np.where(active, rho * adv, 0.0)
    return obj, grad
`,
  },
  kl: {
    id: 'k3-kl-v1',
    starter: `
import numpy as np


def k3_kl(logp, logp_ref):
    """Per-token estimate of KL(pi || pi_ref) from tokens sampled from pi.

    logp, logp_ref: log-probabilities of the sampled tokens under pi and pi_ref, shape (N,)
    returns: shape (N,), each estimate = r - log(r) - 1 with r = pi_ref / pi
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np

def test_zero_when_policies_agree():
    lp = np.log(np.array([0.2, 0.5, 0.9]))
    got = np.asarray(k3_kl(lp, lp))
    assert np.allclose(got, 0), f"expected zeros, got {got}"

def test_every_estimate_is_non_negative():
    rng = np.random.default_rng(0)
    lp = np.log(rng.uniform(0.01, 1, 1000)); lr = np.log(rng.uniform(0.01, 1, 1000))
    got = np.asarray(k3_kl(lp, lr))
    assert got.shape == (1000,), f"expected shape (1000,), got {got.shape}"
    assert np.all(got >= -1e-12), f"k3 is never negative; min was {got.min()}"

def test_mean_estimates_the_true_kl():
    rng = np.random.default_rng(1)
    pi = np.array([0.5, 0.25, 0.2, 0.05]); ref = np.array([0.25, 0.25, 0.25, 0.25])
    true = float(np.sum(pi * np.log(pi / ref)))
    x = rng.choice(4, size=200000, p=pi)
    est = float(np.mean(k3_kl(np.log(pi[x]), np.log(ref[x]))))
    assert abs(est - true) < 0.01, f"sample mean {est:.4f} should be close to KL(pi||ref) = {true:.4f}"
`,
    solution: `
import numpy as np


def k3_kl(logp, logp_ref):
    log_r = np.asarray(logp_ref) - np.asarray(logp)   # log(pi_ref / pi)
    return np.exp(log_r) - log_r - 1.0
`,
  },
};
