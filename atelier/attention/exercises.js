/* Python exercises for the attention lesson. */
window.ATT_EXERCISES = {
  attn: {
    id: 'attention-v1',
    starter: `
import numpy as np


def attention(Q, K, V, causal=False):
    """Scaled dot-product attention.

    Q: (n, d) queries   K: (m, d) keys   V: (m, dv) values
    returns: (n, dv) = softmax(Q K^T / sqrt(d)) V, softmax over the m keys of each row.
    causal=True (requires n == m): query i may only attend to keys j <= i.
    Must stay finite for very large scores.
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np
_rng = np.random.default_rng(0)

def _ref_weights(Q, K, causal):
    s = Q @ K.T / np.sqrt(Q.shape[1])
    if causal:
        s = np.where(np.tril(np.ones_like(s, dtype=bool)), s, -np.inf)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)

def test_output_shape():
    out = np.asarray(attention(_rng.normal(size=(3, 4)), _rng.normal(size=(5, 4)), _rng.normal(size=(5, 2))))
    assert out.shape == (3, 2), f"expected (3, 2), got {out.shape}"

def test_identical_keys_give_the_mean_value():
    K = np.ones((4, 3)); V = _rng.normal(size=(4, 2)); Q = _rng.normal(size=(2, 3))
    out = np.asarray(attention(Q, K, V))
    assert np.allclose(out, V.mean(axis=0)), f"equal scores should average the values; got {out}"

def test_weights_match_reference():
    # With V = identity, the output IS the weight matrix.
    Q = _rng.normal(size=(4, 8)); K = _rng.normal(size=(6, 8))
    W = np.asarray(attention(Q, K, np.eye(6)))
    assert np.allclose(W, _ref_weights(Q, K, False), atol=1e-9), "weights differ from softmax(QK^T / sqrt(d)); did you scale by sqrt(d)?"
    assert np.allclose(W.sum(axis=1), 1), "each row of weights must sum to 1"

def test_causal_mask():
    X = _rng.normal(size=(5, 4))
    W = np.asarray(attention(X, X, np.eye(5), causal=True))
    assert np.allclose(np.triu(W, k=1), 0), "weights above the diagonal (the future) must be exactly 0"
    assert np.allclose(W.sum(axis=1), 1), "masked rows must still sum to 1: mask the scores before the softmax"
    assert np.allclose(W, _ref_weights(X, X, True), atol=1e-9), "causal weights differ from the reference"

def test_first_position_reads_only_itself():
    X = _rng.normal(size=(4, 3)); V = _rng.normal(size=(4, 2))
    out = np.asarray(attention(X, X, V, causal=True))
    assert np.allclose(out[0], V[0]), f"position 0 can only see itself; expected {V[0]}, got {out[0]}"

def test_stable_for_huge_scores():
    Q = np.array([[1000.0, 0.0]]); K = np.array([[1000.0, 0.0], [0.0, 1000.0]]); V = np.array([[1.0], [2.0]])
    out = np.asarray(attention(Q, K, V))
    assert np.all(np.isfinite(out)), f"got {out}: subtract the row max before exponentiating"
    assert np.allclose(out, [[1.0]]), f"expected [[1.0]], got {out}"
`,
    solution: `
import numpy as np


def attention(Q, K, V, causal=False):
    d = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d)                             # (n, m)
    if causal:
        future = np.triu(np.ones(scores.shape, dtype=bool), k=1)
        scores = np.where(future, -np.inf, scores)            # mask BEFORE softmax
    scores = scores - scores.max(axis=1, keepdims=True)       # stability
    w = np.exp(scores)
    w = w / w.sum(axis=1, keepdims=True)                       # rows sum to 1
    return w @ V                                               # (n, dv)
`,
  },
  decode: {
    id: 'kv-decode-v1',
    starter: `
import numpy as np


def decode_step(q, K_cache, V_cache, k_new, v_new):
    """One generation step with a KV cache.

    q, k_new: (d,)  v_new: (dv,)  K_cache: (t, d)  V_cache: (t, dv), the t earlier tokens
    Append the new token's key and value, then attend from q over all t + 1 positions.
    returns (out, K_cache, V_cache): out has shape (dv,), and the caches now have t + 1 rows.
    """
    raise NotImplementedError


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per_value):
    """Total bytes needed to cache K and V for every layer, head and token."""
    raise NotImplementedError
`,
    tests: `
import numpy as np

def _full_causal(Q, K, V):
    s = Q @ K.T / np.sqrt(Q.shape[1])
    s = np.where(np.tril(np.ones_like(s, dtype=bool)), s, -np.inf)
    s -= s.max(axis=1, keepdims=True); w = np.exp(s); w /= w.sum(axis=1, keepdims=True)
    return w @ V

def test_matches_full_causal_attention_token_by_token():
    rng = np.random.default_rng(0)
    T, d, dv = 6, 4, 3
    Q = rng.normal(size=(T, d)); K = rng.normal(size=(T, d)); V = rng.normal(size=(T, dv))
    full = _full_causal(Q, K, V)
    Kc = np.zeros((0, d)); Vc = np.zeros((0, dv))
    for t in range(T):
        out, Kc, Vc = decode_step(Q[t], Kc, Vc, K[t], V[t])
        assert np.allclose(np.asarray(out), full[t], atol=1e-9), f"step {t}: cached decoding differs from full causal attention"
    assert np.shape(Kc) == (T, d) and np.shape(Vc) == (T, dv), f"final cache shapes {np.shape(Kc)}, {np.shape(Vc)}"

def test_kv_cache_bytes_lesson_example():
    got = kv_cache_bytes(32, 32, 128, 8192, 1, 2)
    assert got == 4294967296, f"expected 4,294,967,296 bytes (about 4.3 GB), got {got}"

def test_grouped_query_attention_shrinks_the_cache():
    full = kv_cache_bytes(32, 32, 128, 8192, 4, 2)
    gqa = kv_cache_bytes(32, 8, 128, 8192, 4, 2)
    assert full == 4 * gqa, f"8 KV heads instead of 32 should cut the cache 4x; got {full} vs {gqa}"
`,
    solution: `
import numpy as np


def decode_step(q, K_cache, V_cache, k_new, v_new):
    K = np.vstack([K_cache, k_new[None, :]])                  # (t+1, d)
    V = np.vstack([V_cache, v_new[None, :]])                  # (t+1, dv)
    s = K @ q / np.sqrt(q.shape[0])                            # (t+1,)  no mask needed:
    s = s - s.max()                                            # the cache holds only the past
    w = np.exp(s); w = w / w.sum()
    return w @ V, K, V


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per_value):
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_value
`,
  },
};
