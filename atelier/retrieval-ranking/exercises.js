/* Python exercises for the retrieval vs ranking lesson. */
window.RR_EXERCISES = {
  ndcg: {
    id: 'ndcg-v1',
    starter: `
import numpy as np


def ndcg_at_k(ranked_relevance, k, all_relevance=None):
    """NDCG@k with exponential gain (2^rel - 1) and discount log2(rank + 1), ranks starting at 1.

    ranked_relevance: graded relevance of the items in the order you ranked them.
    all_relevance: relevance of every relevant item for this query (for the ideal ranking).
                   If None, use ranked_relevance itself.
    Return 0.0 when the ideal DCG is 0.
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np

def test_perfect_ranking_is_one():
    assert abs(ndcg_at_k([3, 2, 1, 0], 4) - 1.0) < 1e-12

def test_lesson_example_binary():
    got = ndcg_at_k([1, 0, 0, 1, 0, 0, 0, 0, 0, 0], 10, all_relevance=[1, 1, 1])
    dcg = 1 / np.log2(2) + 1 / np.log2(5); idcg = 1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4)
    assert abs(got - dcg / idcg) < 1e-12, f"expected {dcg / idcg:.4f}, got {got}"

def test_missing_relevant_items_are_penalized():
    a = ndcg_at_k([1, 1], 10)
    b = ndcg_at_k([1, 1], 10, all_relevance=[1, 1, 1, 1])
    assert abs(a - 1) < 1e-12 and b < 1, "items never retrieved must lower NDCG when all_relevance is given"

def test_graded_gain_and_cutoff():
    got = ndcg_at_k([0, 3, 2, 3], 2)
    ideal = (2**3 - 1) / np.log2(2) + (2**3 - 1) / np.log2(3)
    dcg = 0 + (2**3 - 1) / np.log2(3)
    assert abs(got - dcg / ideal) < 1e-12, f"expected {dcg / ideal:.4f} (gain 2^rel - 1, top-k only), got {got}"

def test_no_relevant_items_gives_zero():
    assert ndcg_at_k([0, 0, 0], 3) == 0.0
`,
    solution: `
import numpy as np


def _dcg(rels, k):
    rels = np.asarray(rels, dtype=float)[:k]
    ranks = np.arange(1, len(rels) + 1)
    return float(np.sum((2 ** rels - 1) / np.log2(ranks + 1)))


def ndcg_at_k(ranked_relevance, k, all_relevance=None):
    ideal_pool = ranked_relevance if all_relevance is None else all_relevance
    idcg = _dcg(sorted(ideal_pool, reverse=True), k)
    return 0.0 if idcg == 0 else _dcg(ranked_relevance, k) / idcg
`,
  },
  inbatch: {
    id: 'inbatch-softmax-v1',
    starter: `
import numpy as np


def in_batch_softmax_loss(U, V, log_q=None):
    """Sampled softmax with in-batch negatives.

    U: (B, d) user embeddings, V: (B, d) embeddings of the item each user clicked (row b is user b's positive).
    logits[b, c] = U[b] . V[c], minus log_q[c] if log_q is given (the logQ correction).
    returns the mean over b of -log softmax(logits[b])[b]. Must be numerically stable.
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np

def _ref(U, V, log_q=None):
    L = U @ V.T
    if log_q is not None:
        L = L - log_q[None, :]
    L = L - L.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(L).sum(axis=1)) - np.diag(L)))

def test_uninformative_embeddings_give_log_B():
    U = np.zeros((8, 4)); V = np.zeros((8, 4))
    assert abs(in_batch_softmax_loss(U, V) - np.log(8)) < 1e-12, "all logits equal: loss should be log B"

def test_matches_reference():
    rng = np.random.default_rng(0)
    U = rng.normal(size=(6, 5)); V = rng.normal(size=(6, 5))
    assert abs(in_batch_softmax_loss(U, V) - _ref(U, V)) < 1e-9

def test_logq_correction():
    rng = np.random.default_rng(1)
    U = rng.normal(size=(6, 5)); V = rng.normal(size=(6, 5)); lq = np.log(rng.dirichlet(np.ones(6)))
    assert abs(in_batch_softmax_loss(U, V, lq) - _ref(U, V, lq)) < 1e-9, "subtract log_q from each column (item) before the softmax"

def test_aligned_embeddings_have_low_loss():
    V = np.eye(4) * 10.0
    assert in_batch_softmax_loss(V.copy(), V) < 1e-3, "users identical to their own item should have near-zero loss"

def test_stable_for_large_scores():
    U = np.eye(3) * 100.0; V = np.eye(3) * 100.0
    got = in_batch_softmax_loss(U, V)
    assert np.isfinite(got), f"got {got}: subtract each row's max before exponentiating"
`,
    solution: `
import numpy as np


def in_batch_softmax_loss(U, V, log_q=None):
    logits = U @ V.T                                  # (B, B): row = user, column = item
    if log_q is not None:
        logits = logits - np.asarray(log_q)[None, :]  # undo popularity-proportional sampling
    logits = logits - logits.max(axis=1, keepdims=True)
    log_softmax = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return float(-np.mean(np.diag(log_softmax)))      # positives sit on the diagonal
`,
  },
};
