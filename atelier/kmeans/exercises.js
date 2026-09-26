/* Python exercises for the k-means lesson. */
window.KM_EXERCISES = {
  init: {
    id: 'kmeans-pp-v1',
    starter: `
import numpy as np


def kmeans_pp_init(X, k, rng):
    """k-means++ seeding.

    X: (n, d) data, k: number of centers, rng: np.random.Generator
    First center: a uniformly random row of X. Each next center: a row of X chosen with
    probability proportional to its squared distance to the nearest center chosen so far.
    returns: (k, d) array of initial centers (rows of X)
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np

def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]])
    return np.vstack([c + 0.3 * rng.normal(size=(30, 2)) for c in centers]), centers

def test_returns_k_rows_of_X():
    X, _ = _blobs()
    C = np.asarray(kmeans_pp_init(X, 3, np.random.default_rng(1)))
    assert C.shape == (3, 2), f"expected shape (3, 2), got {C.shape}"
    for c in C:
        assert np.any(np.all(np.isclose(X, c), axis=1)), f"center {c} is not a row of X"

def test_one_center_per_far_blob():
    X, centers = _blobs()
    hits = 0
    for s in range(50):
        C = np.asarray(kmeans_pp_init(X, 3, np.random.default_rng(s)))
        owner = np.argmin(((C[:, None, :] - centers[None, :, :]) ** 2).sum(-1), axis=1)
        hits += len(set(owner.tolist())) == 3
    assert hits >= 48, f"only {hits}/50 seeds put one center in each far-apart blob; sample by squared distance"

def test_never_repeats_a_chosen_point_while_others_remain():
    X = np.array([[0.0, 0.0]] * 5 + [[1.0, 1.0], [5.0, 5.0]])
    for s in range(30):
        C = np.asarray(kmeans_pp_init(X, 3, np.random.default_rng(s)))
        assert len({tuple(c) for c in C}) == 3, f"picked duplicate centers {C}: points already chosen have distance 0"
`,
    solution: `
import numpy as np


def kmeans_pp_init(X, k, rng):
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    d2 = ((X - centers[0]) ** 2).sum(axis=1)             # squared distance to nearest center
    for _ in range(1, k):
        probs = d2 / d2.sum()
        centers.append(X[rng.choice(n, p=probs)])
        d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(axis=1))
    return np.array(centers)
`,
  },
  km: {
    id: 'kmeans-lloyd-v1',
    starter: `
import numpy as np


def kmeans(X, k, n_iter=100, seed=0):
    """Lloyd's algorithm with k-means++ seeding.

    X: (n, d). returns (centers (k, d), labels (n,), inertia float).
    - initialize with k-means++ using np.random.default_rng(seed)
    - repeat: assign each point to its nearest center; move each center to its points' mean
    - stop when the labels stop changing (or after n_iter iterations)
    - if a cluster becomes empty, re-seed its center at the point farthest from its assigned center
    inertia = sum of squared distances from each point to its assigned center.
    """
    raise NotImplementedError
`,
    tests: `
import numpy as np

def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    X = np.vstack([c + 0.5 * rng.normal(size=(40, 2)) for c in centers])
    y = np.repeat(np.arange(3), 40)
    return X, y

def test_recovers_well_separated_blobs():
    X, y = _blobs()
    C, lab, J = kmeans(X, 3, seed=0)
    lab = np.asarray(lab)
    for b in range(3):
        assert len(set(lab[y == b].tolist())) == 1, f"blob {b} was split across clusters"
    assert len(set(lab.tolist())) == 3, "three blobs should give three different clusters"

def test_centers_are_means_and_inertia_is_consistent():
    X, _ = _blobs(1)
    C, lab, J = kmeans(X, 3, seed=2)
    C = np.asarray(C); lab = np.asarray(lab)
    for j in range(3):
        assert np.allclose(C[j], X[lab == j].mean(axis=0), atol=1e-8), f"center {j} is not the mean of its points"
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    assert np.array_equal(lab, d.argmin(axis=1)), "labels must be nearest-center assignments at convergence"
    assert abs(J - d.min(axis=1).sum()) < 1e-6, f"inertia {J} does not match the sum of squared distances"

def test_k_equals_one_gives_the_mean():
    X, _ = _blobs(2)
    C, lab, J = kmeans(X, 1)
    assert np.allclose(np.asarray(C)[0], X.mean(axis=0)), "with k = 1 the center is the overall mean"

def test_no_empty_clusters_and_zero_inertia_on_distinct_points():
    X = np.repeat(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0], [9.0, 1.0]]), 3, axis=0)
    C, lab, J = kmeans(X, 5, seed=3)
    assert len(set(np.asarray(lab).tolist())) == 5, "all 5 clusters should be used"
    assert J < 1e-9, f"5 distinct locations with k = 5 should give inertia 0, got {J}"

def test_deterministic_given_seed():
    X, _ = _blobs(4)
    a = kmeans(X, 3, seed=7); b = kmeans(X, 3, seed=7)
    assert np.allclose(a[0], b[0]) and abs(a[2] - b[2]) < 1e-12, "same seed must give the same result"
`,
    solution: `
import numpy as np


def _pp(X, k, rng):
    n = X.shape[0]
    C = [X[rng.integers(n)]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(1, k):
        C.append(X[rng.choice(n, p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((X - C[-1]) ** 2).sum(1))
    return np.array(C, dtype=float)


def kmeans(X, k, n_iter=100, seed=0):
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    C = _pp(X, k, rng)
    labels = None
    for _ in range(n_iter):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)      # (n, k)   O(n k d)
        new = d.argmin(axis=1)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = X[labels == j]
            if len(members):
                C[j] = members.mean(axis=0)
            else:                                                 # empty: re-seed at the worst-served point
                far = d[np.arange(len(X)), labels].argmax()
                C[j] = X[far]
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    labels = d.argmin(axis=1)
    return C, labels, float(d.min(axis=1).sum())
`,
  },
};
