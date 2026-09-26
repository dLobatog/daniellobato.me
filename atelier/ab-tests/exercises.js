/* Python exercises for the A/B testing lesson (standard library only). */
window.AB_EXERCISES = {
  z: {
    id: 'ab-ztest-v1',
    starter: `
import math


def two_proportion_z(clicks_a, n_a, clicks_b, n_b):
    """Two-sided two-proportion z-test with a pooled standard error.

    returns (z, p_value), z positive when B's rate is higher.
    """
    raise NotImplementedError
`,
    tests: `
import math

def test_equal_rates_give_z_zero_and_p_one():
    z, p = two_proportion_z(500, 10000, 500, 10000)
    assert abs(z) < 1e-12 and abs(p - 1) < 1e-12, f"expected (0, 1), got ({z}, {p})"

def test_known_value():
    z, p = two_proportion_z(500, 10000, 560, 10000)
    pool = 1060 / 20000; se = math.sqrt(pool * (1 - pool) * (2 / 10000)); ez = 0.006 / se
    assert abs(z - ez) < 1e-9, f"expected z = {ez:.4f}, got {z}"
    assert abs(p - math.erfc(ez / math.sqrt(2))) < 1e-9, f"expected p = {math.erfc(ez / math.sqrt(2)):.4f}, got {p}"

def test_sign_follows_b_minus_a():
    z1, p1 = two_proportion_z(500, 10000, 560, 10000)
    z2, p2 = two_proportion_z(560, 10000, 500, 10000)
    assert z1 > 0 > z2 and abs(p1 - p2) < 1e-12, "z flips sign when A and B swap; the two-sided p does not"

def test_unequal_arm_sizes():
    z, p = two_proportion_z(250, 5000, 1100, 20000)
    pool = 1350 / 25000; se = math.sqrt(pool * (1 - pool) * (1 / 5000 + 1 / 20000))
    assert abs(z - (1100 / 20000 - 250 / 5000) / se) < 1e-9, "use 1/n_a + 1/n_b in the standard error"
`,
    solution: `
import math


def two_proportion_z(clicks_a, n_a, clicks_b, n_b):
    pa, pb = clicks_a / n_a, clicks_b / n_b
    pool = (clicks_a + clicks_b) / (n_a + n_b)             # rate under H0: no difference
    se = math.sqrt(pool * (1 - pool) * (1 / n_a + 1 / n_b))
    z = (pb - pa) / se
    return z, math.erfc(abs(z) / math.sqrt(2))              # two-sided normal tail
`,
  },
  ss: {
    id: 'ab-samplesize-v1',
    starter: `
import math
from statistics import NormalDist


def sample_size_per_arm(base_rate, rel_lift, alpha=0.05, power=0.8):
    """Users per arm to detect base_rate -> base_rate * (1 + rel_lift) with a two-sided test.

    n = (z_{1-alpha/2} + z_{power})^2 * (p1(1-p1) + p2(1-p2)) / (p2 - p1)^2, rounded up.
    """
    raise NotImplementedError


def srm_pvalue(n_a, n_b, expected_share_a=0.5):
    """Chi-square (1 degree of freedom) p-value that the observed split matches the expected one."""
    raise NotImplementedError
`,
    tests: `
import math

def test_lesson_sample_size():
    n = sample_size_per_arm(0.05, 0.02)
    assert isinstance(n, int), f"return an int, got {type(n).__name__}"
    assert abs(n - 752700) <= 5, f"expected about 752,700 per arm, got {n}"

def test_halving_the_effect_needs_about_4x():
    a = sample_size_per_arm(0.10, 0.10); b = sample_size_per_arm(0.10, 0.05)
    assert 3.7 < b / a < 4.3, f"halving the lift should need ~4x the users; got {b / a:.2f}x"

def test_more_power_needs_more_users():
    assert sample_size_per_arm(0.05, 0.05, power=0.9) > sample_size_per_arm(0.05, 0.05, power=0.8)

def test_srm_flags_a_broken_split():
    p = srm_pvalue(50900, 49100)
    assert p < 1e-7, f"50,900 vs 49,100 should be a clear mismatch; got p = {p}"

def test_srm_accepts_a_fine_split():
    p = srm_pvalue(50030, 49970)
    assert p > 0.5, f"50,030 vs 49,970 is well within noise; got p = {p}"

def test_srm_uneven_design():
    p = srm_pvalue(90000, 10000, expected_share_a=0.9)
    assert abs(p - 1.0) < 1e-9, f"an exact 90/10 split of a 90/10 design should give p = 1; got {p}"
`,
    solution: `
import math
from statistics import NormalDist


def sample_size_per_arm(base_rate, rel_lift, alpha=0.05, power=0.8):
    p1 = base_rate
    p2 = base_rate * (1 + rel_lift)
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    n = z ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2
    return math.ceil(n)


def srm_pvalue(n_a, n_b, expected_share_a=0.5):
    total = n_a + n_b
    ea, eb = total * expected_share_a, total * (1 - expected_share_a)
    chi2 = (n_a - ea) ** 2 / ea + (n_b - eb) ** 2 / eb
    return math.erfc(math.sqrt(chi2 / 2))                   # chi-square(1) tail
`,
  },
};
