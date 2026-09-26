/* Python exercises for the leakage lesson. */
window.LEAK_EXERCISES = {
  auc: {
    id: 'auc-ranks-v1',
    starter: `
import numpy as np


def roc_auc(y, scores):
    """ROC AUC = probability that a random positive scores above a random negative (ties count 1/2).

    y: 0/1 array, scores: float array. Use ranks (Mann-Whitney U), O(n log n), not a double loop.
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np

def _brute(y, s):
    pos = s[y == 1]; neg = s[y == 0]
    return float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))

def test_perfect_and_inverted():
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(roc_auc(y, s) - 1.0) < 1e-12 and abs(roc_auc(y, -s)) < 1e-12

def test_ties_count_half():
    y = np.array([0, 1, 0, 1]); s = np.array([0.5, 0.5, 0.5, 0.5])
    assert abs(roc_auc(y, s) - 0.5) < 1e-12, "all scores tied: AUC must be 0.5"

def test_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(int); s = np.round(rng.normal(size=400) + y, 1)
    assert abs(roc_auc(y, s) - _brute(y, s)) < 1e-12, "use average ranks for tied scores"

def test_invariant_to_monotone_transform():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.5).astype(int); s = rng.normal(size=300) + y
    assert abs(roc_auc(y, s) - roc_auc(y, np.exp(3 * s))) < 1e-12, "AUC depends only on the ordering of scores"
`,
    solution: `
import numpy as np


def roc_auc(y, scores):
    y = np.asarray(y); s = np.asarray(scores, dtype=float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):                                   # average ranks over ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2   # Mann-Whitney U for positives
    return float(u / (n_pos * n_neg))
`,
  },
  pit: {
    id: 'point-in-time-v1',
    starter: `
import numpy as np


def prior_mean(entity, time, value, default=0.0):
    """Point-in-time feature: for each row, the mean of 'value' over the SAME entity's rows
    with a strictly EARLIER time. Rows with no earlier history get 'default'.

    entity, time, value: arrays of length n (not necessarily sorted). returns an array of length n.
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np

def test_simple_history():
    ent = np.array([1, 1, 1, 2, 2]); t = np.array([1, 2, 3, 1, 2]); v = np.array([1.0, 0.0, 1.0, 5.0, 7.0])
    got = np.asarray(prior_mean(ent, t, v, default=-1.0))
    assert np.allclose(got, [-1.0, 1.0, 0.5, -1.0, 5.0]), f"expected [-1, 1, 0.5, -1, 5], got {got}"

def test_unsorted_input_keeps_row_order():
    ent = np.array([2, 1, 1, 2, 1]); t = np.array([2, 3, 1, 1, 2]); v = np.array([7.0, 1.0, 1.0, 5.0, 0.0])
    got = np.asarray(prior_mean(ent, t, v, default=-1.0))
    assert np.allclose(got, [5.0, 0.5, -1.0, -1.0, 1.0]), f"results must line up with the input rows; got {got}"

def test_never_uses_the_rows_own_value_or_the_future():
    ent = np.zeros(4, int); t = np.array([1, 2, 3, 4]); v = np.array([0.0, 0.0, 0.0, 100.0])
    got = np.asarray(prior_mean(ent, t, v))
    assert got[3] == 0.0 and np.all(got[:3] == 0.0), f"the last row's own value (100) must not appear anywhere; got {got}"

def test_same_time_rows_do_not_see_each_other():
    ent = np.array([1, 1, 1]); t = np.array([1, 1, 2]); v = np.array([2.0, 4.0, 9.0])
    got = np.asarray(prior_mean(ent, t, v, default=0.0))
    assert np.allclose(got, [0.0, 0.0, 3.0]), f"rows at the same time are not 'earlier'; expected [0, 0, 3], got {got}"
`,
    solution: `
import numpy as np


def prior_mean(entity, time, value, default=0.0):
    entity = np.asarray(entity); time = np.asarray(time); value = np.asarray(value, dtype=float)
    out = np.full(len(value), float(default))
    order = np.lexsort((time, entity))              # group by entity, then time
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and entity[order[j]] == entity[order[i]]:
            j += 1
        idx = order[i:j]                            # one entity, sorted by time
        total, count, k = 0.0, 0, 0
        while k < len(idx):
            m = k
            while m < len(idx) and time[idx[m]] == time[idx[k]]:
                m += 1
            if count:
                out[idx[k:m]] = total / count       # only strictly earlier rows
            total += value[idx[k:m]].sum(); count += m - k
            k = m
        i = j
    return out
`,
  },
};
