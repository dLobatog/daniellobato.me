/* Python exercises for the PCA/SVD lesson. */
window.PCA_EXERCISES = {
  pca: {
    id: 'pca-v1',
    starter: `
import numpy as np


def pca(X, k):
    """PCA via the SVD of the centered data.

    X: (n, d). returns (components, explained_variance_ratio, Z):
      components: (k, d), orthonormal rows, largest variance first
      explained_variance_ratio: (k,), each component's share of the total variance
      Z: (n, k), the centered data's coordinates along the components
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np

def _data(seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(300, 3)) * np.array([5.0, 2.0, 0.5])
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return Z @ Q.T + shift, Q

def test_shapes_and_orthonormal_components():
    X, _ = _data()
    C, r, Z = pca(X, 2)
    C = np.asarray(C)
    assert C.shape == (2, 3) and np.shape(r) == (2,) and np.shape(Z) == (300, 2), "check the returned shapes"
    assert np.allclose(C @ C.T, np.eye(2), atol=1e-8), "components must be orthonormal rows"

def test_recovers_the_main_direction():
    X, Q = _data(1)
    C, r, Z = pca(X, 1)
    cos = abs(float(np.asarray(C)[0] @ Q[:, 0]))
    assert cos > 0.99, f"first component should align with the largest-variance direction (|cos| = {cos:.3f})"

def test_ratios_descend_and_sum_to_one_with_all_components():
    X, _ = _data(2)
    _, r, _ = pca(X, 3)
    r = np.asarray(r)
    assert np.all(np.diff(r) <= 1e-12), "explained variance ratios should be in decreasing order"
    assert abs(r.sum() - 1) < 1e-9, "all components together explain 100% of the variance"

def test_centering():
    X, _ = _data(3)
    C1, _, _ = pca(X, 2); C2, _, _ = pca(X + 100.0, 2)
    assert np.allclose(np.abs(np.asarray(C1)), np.abs(np.asarray(C2)), atol=1e-8), "shifting the data must not change the components: center first"

def test_reconstruction_error_equals_discarded_variance():
    X, _ = _data(4)
    Xc = X - X.mean(axis=0)
    C, r, Z = pca(X, 2)
    err = np.mean(np.sum((Xc - np.asarray(Z) @ np.asarray(C)) ** 2, axis=1))
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2 / len(X)
    assert abs(err - lam[2]) < 1e-8, f"mean squared reconstruction error {err:.5f} should equal the dropped eigenvalue {lam[2]:.5f}"
`,
    solution: `
import numpy as np


def pca(X, k):
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)     # Xc = U S Vt
    components = Vt[:k]                                    # principal directions
    var = S ** 2 / len(X)                                  # eigenvalues of the covariance
    ratio = var[:k] / var.sum()
    Z = Xc @ components.T                                  # = U[:, :k] * S[:k]
    return components, ratio, Z
`,
  },
  lr: {
    id: 'low-rank-v1',
    starter: `
import numpy as np


def low_rank_approx(M, k):
    """Best rank-k approximation of M in Frobenius norm (truncated SVD)."""
    raise NotImplementedError
`,
    tests: `
import numpy as np

def test_rank_and_shape():
    M = np.random.default_rng(0).normal(size=(8, 6))
    A = np.asarray(low_rank_approx(M, 2))
    assert A.shape == M.shape, f"shape {A.shape} != {M.shape}"
    assert np.linalg.matrix_rank(A, tol=1e-8) <= 2, "result must have rank at most k"

def test_error_equals_discarded_singular_values():
    M = np.random.default_rng(1).normal(size=(10, 7))
    s = np.linalg.svd(M, compute_uv=False)
    A = np.asarray(low_rank_approx(M, 3))
    err2 = np.sum((M - A) ** 2)
    assert abs(err2 - np.sum(s[3:] ** 2)) < 1e-8, "squared Frobenius error must equal the sum of the dropped singular values squared (Eckart-Young)"

def test_full_rank_is_exact():
    M = np.random.default_rng(2).normal(size=(5, 5))
    assert np.allclose(low_rank_approx(M, 5), M), "k = full rank should reproduce M"

def test_beats_a_random_rank_k_guess():
    rng = np.random.default_rng(3); M = rng.normal(size=(12, 9))
    best = np.sum((M - np.asarray(low_rank_approx(M, 2))) ** 2)
    for _ in range(20):
        B = rng.normal(size=(12, 2)) @ rng.normal(size=(2, 9))
        coef = np.sum(M * B) / np.sum(B * B)
        assert best <= np.sum((M - coef * B) ** 2) + 1e-9, "truncated SVD must beat any other rank-k matrix"
`,
    solution: `
import numpy as np


def low_rank_approx(M, k):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k]
`,
  },
};
