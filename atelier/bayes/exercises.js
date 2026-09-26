/* Python exercises for the Bayes lesson. */
window.BAYES_EXERCISES = {
  post: {
    id: 'bayes-posterior-v1',
    starter: `
import math


def posterior(prior, sensitivity, false_positive_rate, flagged=True):
    """P(positive | test result) for one binary test.

    sensitivity = P(flag | positive), false_positive_rate = P(flag | negative).
    flagged=False means the test did NOT flag.
    """
    raise NotImplementedError


def update_odds(prior, likelihood_ratios):
    """Posterior probability after multiplying the prior odds by each likelihood ratio.

    Must stay finite and correct for extreme ratios (work in log-odds).
    """
    raise NotImplementedError
`,
    tests: `
import math

def test_lesson_example():
    got = posterior(0.02, 0.9, 0.05, True)
    assert abs(got - 18 / 67) < 1e-9, f"expected 18/67 = {18/67:.4f}, got {got}"

def test_not_flagged():
    got = posterior(0.02, 0.9, 0.05, False)
    exp = 0.02 * 0.1 / (0.02 * 0.1 + 0.98 * 0.95)
    assert abs(got - exp) < 1e-12, f"expected {exp:.5f}, got {got}"

def test_uninformative_test_returns_prior():
    got = posterior(0.3, 0.4, 0.4, True)
    assert abs(got - 0.3) < 1e-12, f"a test that flags everything equally tells you nothing: expected 0.3, got {got}"

def test_certain_priors_stay_certain():
    assert posterior(0.0, 0.9, 0.05) == 0.0 and posterior(1.0, 0.9, 0.05) == 1.0, "prior 0 or 1 cannot be moved by evidence"

def test_odds_update_matches_two_flags():
    got = update_odds(0.02, [0.9 / 0.05, 0.8 / 0.1])
    exp = (0.02 * 0.9 * 0.8) / (0.02 * 0.9 * 0.8 + 0.98 * 0.05 * 0.1)
    assert abs(got - exp) < 1e-9, f"expected {exp:.4f}, got {got}"

def test_odds_update_is_stable_for_extreme_ratios():
    got = update_odds(0.5, [1e200, 1e200])
    assert math.isfinite(got) and abs(got - 1.0) < 1e-12, f"expected 1.0, got {got}: multiplying 1e200 * 1e200 overflows; add log ratios instead"
    got2 = update_odds(0.5, [1e-200, 1e-200])
    assert math.isfinite(got2) and got2 < 1e-300, f"expected about 0, got {got2}"
`,
    solution: `
import math


def posterior(prior, sensitivity, false_positive_rate, flagged=True):
    if flagged:
        a, b = sensitivity * prior, false_positive_rate * (1 - prior)
    else:
        a, b = (1 - sensitivity) * prior, (1 - false_positive_rate) * (1 - prior)
    return a / (a + b)            # (spam and evidence) / (all with evidence)


def update_odds(prior, likelihood_ratios):
    log_odds = math.log(prior) - math.log1p(-prior)
    log_odds += sum(math.log(r) for r in likelihood_ratios)
    # stable sigmoid of the log-odds
    if log_odds >= 0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    e = math.exp(log_odds)
    return e / (1.0 + e)
`,
  },
  recal: {
    id: 'downsample-recal-v1',
    starter: `
import numpy as np


def recalibrate(p, neg_keep_rate):
    """Undo negative downsampling.

    The model was trained keeping every positive but only a fraction neg_keep_rate (w)
    of the negatives. p: predicted probabilities (float or numpy array).
    returns calibrated probabilities with the same shape.
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np

def test_no_downsampling_is_identity():
    p = np.array([0.01, 0.3, 0.9])
    assert np.allclose(recalibrate(p, 1.0), p), "w = 1 should change nothing"

def test_lesson_example():
    got = float(recalibrate(0.5, 0.1))
    assert abs(got - 1 / 11) < 1e-12, f"odds 1 x 0.1 -> p = 1/11 = 0.0909; got {got}"

def test_keeps_order_and_shape():
    p = np.linspace(0.01, 0.99, 50)
    q = np.asarray(recalibrate(p, 0.05))
    assert q.shape == p.shape, f"shape {q.shape} != {p.shape}"
    assert np.all(np.diff(q) > 0), "recalibration must be monotone (ranking unchanged)"
    assert np.all(q < p), "fewer negatives in training means the model over-predicts: q must be below p"

def test_matches_logit_shift():
    p = np.array([0.2, 0.6]); w = 0.25
    logit = np.log(p / (1 - p)) + np.log(w)
    assert np.allclose(recalibrate(p, w), 1 / (1 + np.exp(-logit))), "should equal adding log(w) to the logit"
`,
    solution: `
import numpy as np


def recalibrate(p, neg_keep_rate):
    p = np.asarray(p, dtype=float)
    w = neg_keep_rate
    # Training odds were inflated by 1/w; true odds = model odds * w.
    return p / (p + (1 - p) / w)
`,
  },
};
