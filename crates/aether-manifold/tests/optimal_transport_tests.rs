//! Project Aether-Manifold: Optimal Transport & Sinkhorn-Knopp Integration Tests
//!
//! Validates Entropy-Regularized Wasserstein Distance $\mathcal{W}_{\varepsilon}(A, B)$,
//! Debiased Sinkhorn Divergence $S_{\varepsilon}(A, B)$, Log-Domain stabilization
//! for small $\varepsilon$, and zero hot-path heap allocations (AGENTS.md Rule 3.1).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use aether_core::alloc_guard::{reset_hot_path_counters, HotPathAllocGuard, TrackingAllocator};
use aether_core::error::AetherError;
use aether_manifold::{
    debiased_sinkhorn_divergence, sinkhorn_knopp, sinkhorn_knopp_with_workspace,
    sinkhorn_workspace_size, SinkhornConfig,
};

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator::new(std::alloc::System);

#[test]
fn gate_zero_hot_path_allocations_on_sinkhorn() {
    let n = 4;
    let m = 4;
    let dim = 8;

    let a_bag = [1.0f32; 32]; // 4 * 8
    let b_bag = [2.0f32; 32]; // 4 * 8

    let req_size = sinkhorn_workspace_size(n, m);
    let mut workspace = vec![0.0f32; req_size];

    let config = SinkhornConfig {
        epsilon: 0.1,
        max_iter: 50,
        tolerance: 1e-4,
    };

    reset_hot_path_counters();
    {
        let guard = HotPathAllocGuard::enter();

        let res = sinkhorn_knopp_with_workspace(
            &a_bag,
            n,
            &b_bag,
            m,
            dim,
            &config,
            &mut workspace,
        )
        .unwrap();

        assert_eq!(
            guard.alloc_count(),
            0,
            "Sinkhorn-Knopp solver triggered hot-path heap allocations!"
        );
        assert!(guard.assert_zero_allocations().is_ok());
        assert!(res.cost > 0.0);
    }
}

#[test]
fn test_sinkhorn_convergence_and_identity() {
    let n = 3;
    let dim = 4;
    let a_bag = [
        1.0f32, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ];

    let config = SinkhornConfig {
        epsilon: 0.1,
        max_iter: 100,
        tolerance: 1e-5,
    };

    // Self-divergence should be exactly 0
    let s_aa = debiased_sinkhorn_divergence(&a_bag, n, &a_bag, n, dim, &config).unwrap();
    assert_eq!(s_aa, 0.0, "Debiased Sinkhorn divergence of identical sets must be 0");

    let b_bag = [
        2.0f32, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 2.0, 0.0,
    ];

    let s_ab = debiased_sinkhorn_divergence(&a_bag, n, &b_bag, n, dim, &config).unwrap();
    let s_ba = debiased_sinkhorn_divergence(&b_bag, n, &a_bag, n, dim, &config).unwrap();

    assert!(s_ab > 0.0, "Distance between distinct bags must be positive");
    assert!((s_ab - s_ba).abs() < 1e-5, "Divergence must be symmetric");
}

#[test]
fn test_sinkhorn_marginal_convergence() {
    let n = 4;
    let m = 3;
    let dim = 4;

    let a_bag = [
        0.5f32, 1.0, 0.2, 0.8,
        1.2, 0.3, 0.9, 0.1,
        0.0, 0.5, 1.5, 0.4,
        0.7, 0.8, 0.3, 0.2,
    ];

    let b_bag = [
        0.1f32, 0.9, 0.4, 0.6,
        1.0, 0.2, 0.8, 0.3,
        0.5, 0.5, 0.5, 0.5,
    ];

    let config = SinkhornConfig {
        epsilon: 0.05,
        max_iter: 50,
        tolerance: 1e-4,
    };

    let res = sinkhorn_knopp(&a_bag, n, &b_bag, m, dim, &config).unwrap();
    assert!(res.converged, "Sinkhorn must converge within max iterations");
    assert!(res.iterations <= 50, "Iterations must not exceed 50");
    assert!(
        res.marginal_error_u <= 1e-4,
        "Source marginal error must satisfy tolerance: {}",
        res.marginal_error_u
    );
    assert!(
        res.marginal_error_v <= 1e-4,
        "Target marginal error must satisfy tolerance: {}",
        res.marginal_error_v
    );
}

#[test]
fn test_sinkhorn_log_domain_stabilization_small_epsilon() {
    // Category V Test 5.8: test_sinkhorn_log_domain_stabilization_small_epsilon
    let n = 3;
    let m = 3;
    let dim = 4;

    let a_bag = [
        10.0f32, 2.0, 5.0, 1.0,
        3.0, 8.0, 0.5, 4.0,
        1.0, 1.0, 9.0, 2.0,
    ];

    let b_bag = [
        0.0f32, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
        8.0, 9.0, 1.0, 2.0,
    ];

    // Extreme small epsilon: 1e-4 where exp(-C/eps) would otherwise underflow to 0
    let config = SinkhornConfig {
        epsilon: 1e-4,
        max_iter: 200,
        tolerance: 1e-3,
    };

    let res = sinkhorn_knopp(&a_bag, n, &b_bag, m, dim, &config).unwrap();

    // Verify Invariant from test_suite_manifest.md Section 6 Test 5.8:
    // "verify zero SIMD floating-point underflow (0.0) or overflow (Inf) during scaling."
    assert!(!res.cost.is_nan(), "Cost must not be NaN");
    assert!(!res.cost.is_infinite(), "Cost must not be Infinite");
    assert!(res.cost > 0.0, "Cost must be strictly positive (no underflow)");
    assert!(!res.regularized_cost.is_nan(), "Regularized cost must not be NaN");
    assert!(!res.regularized_cost.is_infinite(), "Regularized cost must not be Infinite");
    assert!(!res.marginal_error_u.is_nan(), "Marginal error must not be NaN");
    assert!(!res.marginal_error_v.is_nan(), "Marginal error must not be NaN");
    assert!(res.marginal_error_v < 0.05, "Target marginal error must not diverge");
}

#[test]
fn test_sinkhorn_error_handling() {
    let a_bag = [1.0f32; 8];
    let b_bag = [2.0f32; 8];

    // Non-positive epsilon
    let bad_config = SinkhornConfig {
        epsilon: 0.0,
        max_iter: 10,
        tolerance: 1e-4,
    };
    assert!(matches!(
        sinkhorn_knopp(&a_bag, 2, &b_bag, 2, 4, &bad_config),
        Err(AetherError::LorentzHorizonViolation)
    ));

    // Zero dimension
    let ok_config = SinkhornConfig::default();
    assert!(matches!(
        sinkhorn_knopp(&a_bag, 2, &b_bag, 2, 0, &ok_config),
        Err(AetherError::InvalidBufferSize { .. })
    ));

    // Buffer length mismatch
    assert!(matches!(
        sinkhorn_knopp(&a_bag[..7], 2, &b_bag, 2, 4, &ok_config),
        Err(AetherError::InvalidBufferSize { .. })
    ));

    // Insufficient workspace
    let mut tiny_workspace = [0.0f32; 4];
    assert!(matches!(
        sinkhorn_knopp_with_workspace(&a_bag, 2, &b_bag, 2, 4, &ok_config, &mut tiny_workspace),
        Err(AetherError::ScratchpadExhausted { .. })
    ));
}

