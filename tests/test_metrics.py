"""
Mathematical-property tests for all alignment metrics.

Each test verifies a formal guarantee that must hold regardless of
implementation details, so that consolidating metrics.py and measures.py
during refactoring cannot silently change metric behaviour.

The existing measures.sanity_check_metrics() is a manual version of
these tests; here they become reproducible and CI-enforced.
"""
import pytest
import torch

from metrics import AlignmentMetrics
from measures import (
    orthogonal_procrustes_distance,
    mutual_knn_alignment_simple,
    rsa_similarity,
    jaccard_similarity,
    cosine_similarity_batch,
)

N    = 128   # number of samples
D    = 64    # feature dimension
TOPK = 10    # k for kNN metrics


@pytest.fixture(scope="module")
def features():
    """Two independent random feature matrices, shared across this module."""
    torch.manual_seed(42)
    X = torch.randn(N, D)
    Y = torch.randn(N, D)
    return X, Y


# ---------------------------------------------------------------------------
# CKA (AlignmentMetrics)
# ---------------------------------------------------------------------------

class TestCKA:

    def test_identical_is_one(self, features):
        X, _ = features
        assert AlignmentMetrics.cka(X, X) == pytest.approx(1.0, abs=1e-4)

    def test_scale_invariant(self, features):
        """CKA is invariant to isotropic scaling of representations."""
        X, _ = features
        assert AlignmentMetrics.cka(X, X * 5.0) == pytest.approx(
            AlignmentMetrics.cka(X, X), abs=1e-4
        )

    def test_in_unit_range(self, features):
        X, Y = features
        s = AlignmentMetrics.cka(X, Y)
        assert 0.0 <= s <= 1.0 + 1e-6

    def test_symmetric(self, features):
        X, Y = features
        assert AlignmentMetrics.cka(X, Y) == pytest.approx(
            AlignmentMetrics.cka(Y, X), abs=1e-5
        )

    def test_identical_greater_than_random(self, features):
        X, Y = features
        assert AlignmentMetrics.cka(X, X) > AlignmentMetrics.cka(X, Y)

    def test_rbf_identical_is_one(self, features):
        X, _ = features
        assert AlignmentMetrics.cka(X, X, kernel_metric='rbf') == pytest.approx(1.0, abs=1e-4)

    def test_unbiased_identical_near_one(self, features):
        """Unbiased estimator converges to 1 for large N but is not exactly 1."""
        X, _ = features
        assert AlignmentMetrics.unbiased_cka(X, X) > 0.9


# ---------------------------------------------------------------------------
# SVCCA (AlignmentMetrics)
# ---------------------------------------------------------------------------

class TestSVCCA:

    def test_identical_greater_than_random(self, features):
        # svd_lowrank is randomized: two calls on the same tensor can return
        # different subspace orientations, so svcca(X, X) is not guaranteed
        # to be near 1. The meaningful property is the relative ordering.
        X, Y = features
        assert AlignmentMetrics.svcca(X, X, cca_dim=5) > AlignmentMetrics.svcca(X, Y, cca_dim=5)


# ---------------------------------------------------------------------------
# kNN-based metrics (AlignmentMetrics)
# ---------------------------------------------------------------------------

class TestKNNMetrics:

    def test_mutual_knn_identical_is_one(self, features):
        X, _ = features
        assert AlignmentMetrics.mutual_knn(X, X, topk=TOPK) == pytest.approx(1.0, abs=1e-5)

    def test_cycle_knn_identical_is_one(self, features):
        X, _ = features
        assert AlignmentMetrics.cycle_knn(X, X, topk=TOPK) == pytest.approx(1.0, abs=1e-5)

    def test_mutual_knn_identical_greater_than_random(self, features):
        X, Y = features
        assert AlignmentMetrics.mutual_knn(X, X, topk=TOPK) > AlignmentMetrics.mutual_knn(X, Y, topk=TOPK)

    def test_cycle_knn_identical_greater_than_random(self, features):
        X, Y = features
        assert AlignmentMetrics.cycle_knn(X, X, topk=TOPK) > AlignmentMetrics.cycle_knn(X, Y, topk=TOPK)


# ---------------------------------------------------------------------------
# Orthogonal Procrustes (measures.py)
# ---------------------------------------------------------------------------

class TestOrthogonalProcrustes:

    def test_identical_is_zero(self, features):
        # The implementation computes distance_sq = norm_X + norm_Y - 2*nuclear_norm
        # via two different numerical paths (Frobenius vs SVD), so floating-point
        # cancellation means the result is near but not exactly 0.
        # Test relative to input scale instead of absolute tolerance.
        X, _ = features
        d = orthogonal_procrustes_distance(X, X)
        assert d < 1e-3 * float(X.norm())

    def test_rotation_invariant(self, features):
        """Distance should be 0 for any orthogonal transformation of identical inputs."""
        X, _ = features
        torch.manual_seed(0)
        Q, _ = torch.linalg.qr(torch.randn(D, D))
        assert orthogonal_procrustes_distance(X, X @ Q) == pytest.approx(0.0, abs=1e-4)

    def test_nonnegative(self, features):
        X, Y = features
        assert orthogonal_procrustes_distance(X, Y) >= 0.0

    def test_identical_less_than_random(self, features):
        X, Y = features
        assert orthogonal_procrustes_distance(X, X) < orthogonal_procrustes_distance(X, Y)


# ---------------------------------------------------------------------------
# mutual_knn_alignment_simple (measures.py)
# ---------------------------------------------------------------------------

class TestMutualKNNSimple:

    def test_identical_is_one(self, features):
        X, _ = features
        assert mutual_knn_alignment_simple(X, X, k=TOPK) == pytest.approx(1.0, abs=1e-5)

    def test_in_unit_range(self, features):
        X, Y = features
        assert 0.0 <= mutual_knn_alignment_simple(X, Y, k=TOPK) <= 1.0 + 1e-6

    def test_identical_greater_than_random(self, features):
        X, Y = features
        assert mutual_knn_alignment_simple(X, X, k=TOPK) > mutual_knn_alignment_simple(X, Y, k=TOPK)


# ---------------------------------------------------------------------------
# RSA and Jaccard (measures.py)
# ---------------------------------------------------------------------------

class TestRSA:

    def test_identical_near_one(self, features):
        X, _ = features
        assert rsa_similarity(X, X) > 0.9

    def test_in_range(self, features):
        """Spearman correlation is in [-1, 1]."""
        X, Y = features
        s = rsa_similarity(X, Y)
        assert -1.0 - 1e-6 <= s <= 1.0 + 1e-6


class TestJaccard:

    def test_identical_is_one(self, features):
        X, _ = features
        assert jaccard_similarity(X, X, k=TOPK) == pytest.approx(1.0, abs=1e-5)

    def test_in_unit_range(self, features):
        X, Y = features
        assert 0.0 <= jaccard_similarity(X, Y, k=TOPK) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# cosine_similarity_batch (measures.py)
# ---------------------------------------------------------------------------

class TestCosineSimilarityBatch:

    def test_identical_is_one(self, features):
        X, _ = features
        sims = cosine_similarity_batch(X, X)
        assert sims.shape == (N,)
        assert sims.min() == pytest.approx(1.0, abs=1e-5)

    def test_output_shape(self, features):
        X, Y = features
        assert cosine_similarity_batch(X, Y).shape == (N,)

    def test_in_unit_range(self, features):
        X, Y = features
        sims = cosine_similarity_batch(X, Y)
        assert sims.min() >= -1.0 - 1e-6
        assert sims.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# measures.cka — 3-D wrapper around ckatorch.cka_batch
# ---------------------------------------------------------------------------

class TestMeasuresCKA3D:
    """measures.cka expects [bs, tokens, dim] (3-D), unlike AlignmentMetrics.cka."""

    def test_identical_near_one(self):
        from measures import cka as measures_cka
        torch.manual_seed(42)
        X = torch.randn(4, 16, 64)
        assert measures_cka(X, X) > 0.9

    def test_returns_float(self):
        from measures import cka as measures_cka
        X = torch.randn(4, 16, 64)
        assert isinstance(measures_cka(X, X), float)
