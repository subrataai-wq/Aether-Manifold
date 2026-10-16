//! Project Aether-Manifold: Master Pre-Publish Test Suite Manifest — Category V
//!
//! Formal Property-Based Fuzzing of Vectorized Sinkhorn-Knopp Optimal Transport
//! in strict compliance with test_suite_manifest.md Category V (Tests 5.6 & 5.7).
//!
//! Verifies:
//! - 5.6 `prop_sinkhorn_knopp_marginal_convergence`
//! - 5.7 `prop_wasserstein_sinkhorn_metric_properties`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use aether_manifold::{debiased_sinkhorn_divergence, sinkhorn_knopp, SinkhornConfig};
use proptest::prelude::*;

fn normalize_bag(bag: &mut [f32], n: usize, dim: usize) {
    for i in 0..n {
        if let Some(vec) = bag.get_mut(i * dim..(i + 1) * dim) {
            let mut norm_sq = 0.0f32;
            for &x in vec.iter() {
                norm_sq += x * x;
            }
            if norm_sq > 1e-6 {
                let inv_norm = 1.0 / norm_sq.sqrt();
                for x in vec.iter_mut() {
                    *x *= inv_norm;
                }
            } else if let Some(first) = vec.first_mut() {
                *first = 1.0;
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1000,
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    /// 5.6: Verifies that the SIMD Sinkhorn-Knopp solver for W_eps(A, B) converges
    /// to doubly-stochastic transport plan marginals ||P 1_m - mu||_1 <= 10^-4 and
    /// ||P^T 1_n - nu||_1 <= 10^-4 within <= 50 iterations over multi-vector bags on M.
    #[test]
    fn prop_sinkhorn_knopp_marginal_convergence(
        n in 2usize..6usize,
        m in 2usize..6usize,
        dim in 2usize..8usize,
        a_coords in proptest::collection::vec(-10.0f32..10.0f32, 48),
        b_coords in proptest::collection::vec(-10.0f32..10.0f32, 48),
    ) {
        let mut a_bag = match a_coords.get(..n * dim) {
            Some(s) => s.to_vec(),
            None => return Err(TestCaseError::fail("Buffer out of bounds")),
        };
        let mut b_bag = match b_coords.get(..m * dim) {
            Some(s) => s.to_vec(),
            None => return Err(TestCaseError::fail("Buffer out of bounds")),
        };
        normalize_bag(&mut a_bag, n, dim);
        normalize_bag(&mut b_bag, m, dim);

        let config = SinkhornConfig {
            epsilon: 1.0,
            max_iter: 50,
            tolerance: 1e-4,
        };

        let res = sinkhorn_knopp(&a_bag, n, &b_bag, m, dim, &config).unwrap();

        prop_assert!(
            res.converged,
            "Sinkhorn failed to converge in {} iters: err_u={}, err_v={}",
            res.iterations,
            res.marginal_error_u,
            res.marginal_error_v
        );
        prop_assert!(
            res.iterations <= 50,
            "Iterations exceeded 50: {}",
            res.iterations
        );
        prop_assert!(
            res.marginal_error_u <= 1e-4 + 1e-6,
            "Marginal u error exceeded tolerance: {}",
            res.marginal_error_u
        );
        prop_assert!(
            res.marginal_error_v <= 1e-4 + 1e-6,
            "Marginal v error exceeded tolerance: {}",
            res.marginal_error_v
        );
    }

    /// 5.7: Verifies that Debiased Sinkhorn Divergence S_eps(A, B) satisfies
    /// non-negativity S_eps(A, B) >= 0, identity S_eps(A, A) = 0, and
    /// symmetry S_eps(A, B) = S_eps(B, A) over multi-vector bags on M.
    #[test]
    fn prop_wasserstein_sinkhorn_metric_properties(
        n in 2usize..5usize,
        m in 2usize..5usize,
        dim in 2usize..6usize,
        a_coords in proptest::collection::vec(-10.0f32..10.0f32, 30),
        b_coords in proptest::collection::vec(-10.0f32..10.0f32, 30),
    ) {
        let mut a_bag = match a_coords.get(..n * dim) {
            Some(s) => s.to_vec(),
            None => return Err(TestCaseError::fail("Buffer out of bounds")),
        };
        let mut b_bag = match b_coords.get(..m * dim) {
            Some(s) => s.to_vec(),
            None => return Err(TestCaseError::fail("Buffer out of bounds")),
        };
        normalize_bag(&mut a_bag, n, dim);
        normalize_bag(&mut b_bag, m, dim);

        let config = SinkhornConfig {
            epsilon: 1.0,
            max_iter: 50,
            tolerance: 1e-4,
        };

        // 1. Identity of indiscernibles: S_eps(A, A) = 0
        let s_aa = debiased_sinkhorn_divergence(&a_bag, n, &a_bag, n, dim, &config).unwrap();
        prop_assert_eq!(s_aa, 0.0, "Self divergence must be 0");

        // 2. Non-negativity: S_eps(A, B) >= 0
        let s_ab = debiased_sinkhorn_divergence(&a_bag, n, &b_bag, m, dim, &config).unwrap();
        prop_assert!(s_ab >= 0.0, "Divergence must be non-negative: {}", s_ab);

        // 3. Symmetry: S_eps(A, B) = S_eps(B, A)
        let s_ba = debiased_sinkhorn_divergence(&b_bag, m, &a_bag, n, dim, &config).unwrap();
        let diff = (s_ab - s_ba).abs();
        let tol = 2e-3 * (s_ab.max(s_ba)).max(1.0);
        prop_assert!(
            diff <= tol,
            "Symmetry violated: s_ab={}, s_ba={}, diff={}, tol={}",
            s_ab,
            s_ba,
            diff,
            tol
        );
    }
}
