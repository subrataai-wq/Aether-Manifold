//! Project Aether-Manifold: Master Pre-Publish Test Suite Manifest — Category III
//!
//! Formal Property-Based Fuzzing of Riemannian Product Manifold & Geodesic Math
//! in strict compliance with test_suite_manifest.md Category III and AGENTS.md Rule 4.1.
//!
//! Verifies:
//! - 3.1 `prop_lorentz_manifold_surface_invariant`
//! - 3.2 `prop_lorentz_reprojection_drift_correction`
//! - 3.3 `prop_product_manifold_metric_non_negativity_and_identity`
//! - 3.4 `prop_product_manifold_metric_symmetry`
//! - 3.5 `prop_product_manifold_riemannian_triangle_inequality`
//! - 3.9 `prop_hypersphere_antipodal_numerical_stability`
//! - 3.10 `test_simd_vs_scalar_manifold_bit_equivalence`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use aether_manifold::{
    lorentz_compute_x0, lorentz_inner_product, lorentz_reproject,
    product_manifold_distance, product_manifold_distance_squared,
    scalar_product_manifold_distance_squared, spherical_distance, spherical_normalize,
    EuclideanSpec, HyperbolicSpec, ProductManifoldConfig, SphericalSpec,
};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5000))]

    /// 3.1: Verifies that for any generated point x in H_kappa^d,
    /// the Lorentzian inner product satisfies |<x, x>_L - 1/kappa| within float32 precision.
    #[test]
    fn prop_lorentz_manifold_surface_invariant(
        spatial in proptest::collection::vec(-50.0f32..50.0f32, 2..16),
        kappa in -5.0f32..-0.05f32,
    ) {
        let x0 = lorentz_compute_x0(&spatial, kappa).unwrap();
        let mut point = Vec::with_capacity(spatial.len() + 1);
        point.push(x0);
        point.extend(spatial);

        let inner = lorentz_inner_product(&point, &point).unwrap();
        let expected = 1.0 / kappa;
        let diff = (inner - expected).abs();
        // IEEE-754 cancellation bound: error is proportional to magnitude x0^2 * epsilon
        let tol = 1e-4 * (x0 * x0).max(1.0);
        prop_assert!(diff <= tol, "Invariant failed: inner={}, expected={}, diff={}, tol={}", inner, expected, diff, tol);
    }

    /// 3.2: Verifies that applying simulated numerical drift and re-projecting via
    /// x_0 = sqrt(1 / -kappa + sum x_l^2) restores the point to H_kappa^d surface.
    #[test]
    fn prop_lorentz_reprojection_drift_correction(
        spatial in proptest::collection::vec(-50.0f32..50.0f32, 2..16),
        kappa in -5.0f32..-0.1f32,
        drift in -1.0f32..1.0f32,
    ) {
        let x0 = lorentz_compute_x0(&spatial, kappa).unwrap();
        let mut point = Vec::with_capacity(spatial.len() + 1);
        // Inject drift into x0
        point.push(x0 + drift);
        point.extend(spatial);

        // Re-project onto manifold surface
        lorentz_reproject(&mut point, kappa).unwrap();

        let inner = lorentz_inner_product(&point, &point).unwrap();
        let expected = 1.0 / kappa;
        let diff = (inner - expected).abs();
        let tol = 1e-4 * (x0 * x0).max(1.0);
        prop_assert!(diff <= tol, "Reprojection failed: inner={}, expected={}, diff={}, tol={}", inner, expected, diff, tol);
    }

    /// 3.3: Verifies non-negativity d_M^2(u, v) >= 0 and identity d_M^2(u, u) <= 1e-7 across product manifold.
    #[test]
    fn prop_product_manifold_metric_non_negativity_and_identity(
        h_spatial_u in proptest::collection::vec(-20.0f32..20.0f32, 3),
        h_spatial_v in proptest::collection::vec(-20.0f32..20.0f32, 3),
        s_raw_u in proptest::collection::vec(-10.0f32..10.0f32, 3),
        s_raw_v in proptest::collection::vec(-10.0f32..10.0f32, 3),
        e_u in proptest::collection::vec(-50.0f32..50.0f32, 2),
        e_v in proptest::collection::vec(-50.0f32..50.0f32, 2),
    ) {
        let kappa = -1.0f32;
        let tau = 1.0f32;

        let config = ProductManifoldConfig {
            hyperbolic: vec![HyperbolicSpec { dim: 4, kappa, weight: 1.0 }],
            spherical: vec![SphericalSpec { dim: 3, tau, weight: 1.0 }],
            euclidean: Some(EuclideanSpec { dim: 2, weight: 1.0 }),
        };

        // Construct point u
        let mut u = Vec::with_capacity(9);
        u.push(lorentz_compute_x0(&h_spatial_u, kappa).unwrap());
        u.extend(&h_spatial_u);
        let mut s_u = s_raw_u;
        if s_u.iter().all(|&x| x == 0.0) { s_u[0] = 1.0; }
        spherical_normalize(&mut s_u).unwrap();
        u.extend(&s_u);
        u.extend(&e_u);

        // Construct point v
        let mut v = Vec::with_capacity(9);
        v.push(lorentz_compute_x0(&h_spatial_v, kappa).unwrap());
        v.extend(&h_spatial_v);
        let mut s_v = s_raw_v;
        if s_v.iter().all(|&x| x == 0.0) { s_v[0] = 1.0; }
        spherical_normalize(&mut s_v).unwrap();
        v.extend(&s_v);
        v.extend(&e_v);

        // 1. Identity: d_M^2(u, u) must be <= 1e-7
        let d2_self = product_manifold_distance_squared(&config, &u, &u).unwrap();
        prop_assert!(d2_self <= 1e-7, "Identity violation: d2_self = {}", d2_self);

        // 2. Non-negativity: d_M^2(u, v) >= 0
        let d2_uv = product_manifold_distance_squared(&config, &u, &v).unwrap();
        prop_assert!(d2_uv >= 0.0, "Non-negativity violation: d2_uv = {}", d2_uv);
    }

    /// 3.4: Verifies metric symmetry |d_M(u, v) - d_M(v, u)| <= 1e-6.
    #[test]
    fn prop_product_manifold_metric_symmetry(
        h_spatial_u in proptest::collection::vec(-15.0f32..15.0f32, 2),
        h_spatial_v in proptest::collection::vec(-15.0f32..15.0f32, 2),
        s_raw_u in proptest::collection::vec(-5.0f32..5.0f32, 3),
        s_raw_v in proptest::collection::vec(-5.0f32..5.0f32, 3),
        e_u in proptest::collection::vec(-20.0f32..20.0f32, 2),
        e_v in proptest::collection::vec(-20.0f32..20.0f32, 2),
    ) {
        let kappa = -0.8f32;
        let tau = 1.5f32;

        let config = ProductManifoldConfig {
            hyperbolic: vec![HyperbolicSpec { dim: 3, kappa, weight: 1.0 }],
            spherical: vec![SphericalSpec { dim: 3, tau, weight: 1.0 }],
            euclidean: Some(EuclideanSpec { dim: 2, weight: 1.0 }),
        };

        let mut u = Vec::new();
        u.push(lorentz_compute_x0(&h_spatial_u, kappa).unwrap());
        u.extend(&h_spatial_u);
        let mut s_u = s_raw_u;
        if s_u.iter().all(|&x| x == 0.0) { s_u[0] = 1.0; }
        spherical_normalize(&mut s_u).unwrap();
        u.extend(&s_u);
        u.extend(&e_u);

        let mut v = Vec::new();
        v.push(lorentz_compute_x0(&h_spatial_v, kappa).unwrap());
        v.extend(&h_spatial_v);
        let mut s_v = s_raw_v;
        if s_v.iter().all(|&x| x == 0.0) { s_v[0] = 1.0; }
        spherical_normalize(&mut s_v).unwrap();
        v.extend(&s_v);
        v.extend(&e_v);

        let d_uv = product_manifold_distance(&config, &u, &v).unwrap();
        let d_vu = product_manifold_distance(&config, &v, &u).unwrap();
        let diff = (d_uv - d_vu).abs();
        prop_assert!(diff <= 1e-6, "Symmetry violation: d_uv={}, d_vu={}, diff={}", d_uv, d_vu, diff);
    }

    /// 3.5: Verifies Riemannian triangle inequality d_M(u, w) <= d_M(u, v) + d_M(v, w) + 1e-4.
    #[test]
    fn prop_product_manifold_riemannian_triangle_inequality(
        h_spatial_u in proptest::collection::vec(-10.0f32..10.0f32, 2),
        h_spatial_v in proptest::collection::vec(-10.0f32..10.0f32, 2),
        h_spatial_w in proptest::collection::vec(-10.0f32..10.0f32, 2),
        s_raw_u in proptest::collection::vec(-5.0f32..5.0f32, 3),
        s_raw_v in proptest::collection::vec(-5.0f32..5.0f32, 3),
        s_raw_w in proptest::collection::vec(-5.0f32..5.0f32, 3),
        e_u in proptest::collection::vec(-10.0f32..10.0f32, 2),
        e_v in proptest::collection::vec(-10.0f32..10.0f32, 2),
        e_w in proptest::collection::vec(-10.0f32..10.0f32, 2),
    ) {
        let kappa = -1.0f32;
        let tau = 1.0f32;

        let config = ProductManifoldConfig {
            hyperbolic: vec![HyperbolicSpec { dim: 3, kappa, weight: 1.0 }],
            spherical: vec![SphericalSpec { dim: 3, tau, weight: 1.0 }],
            euclidean: Some(EuclideanSpec { dim: 2, weight: 1.0 }),
        };

        let mut u = Vec::new();
        u.push(lorentz_compute_x0(&h_spatial_u, kappa).unwrap());
        u.extend(&h_spatial_u);
        let mut s_u = s_raw_u;
        if s_u.iter().all(|&x| x == 0.0) { s_u[0] = 1.0; }
        spherical_normalize(&mut s_u).unwrap();
        u.extend(&s_u);
        u.extend(&e_u);

        let mut v = Vec::new();
        v.push(lorentz_compute_x0(&h_spatial_v, kappa).unwrap());
        v.extend(&h_spatial_v);
        let mut s_v = s_raw_v;
        if s_v.iter().all(|&x| x == 0.0) { s_v[0] = 1.0; }
        spherical_normalize(&mut s_v).unwrap();
        v.extend(&s_v);
        v.extend(&e_v);

        let mut w = Vec::new();
        w.push(lorentz_compute_x0(&h_spatial_w, kappa).unwrap());
        w.extend(&h_spatial_w);
        let mut s_w = s_raw_w;
        if s_w.iter().all(|&x| x == 0.0) { s_w[0] = 1.0; }
        spherical_normalize(&mut s_w).unwrap();
        w.extend(&s_w);
        w.extend(&e_w);

        let d_uw = product_manifold_distance(&config, &u, &w).unwrap();
        let d_uv = product_manifold_distance(&config, &u, &v).unwrap();
        let d_vw = product_manifold_distance(&config, &v, &w).unwrap();

        let tolerance = 1e-4;
        prop_assert!(
            d_uw <= d_uv + d_vw + tolerance,
            "Triangle inequality broken: d(u,w)={}, d(u,v)={}, d(v,w)={}, sum={}",
            d_uw, d_uv, d_vw, d_uv + d_vw
        );
    }

    /// 3.9: Tests hypersphere antipodal numerical stability when y approaches -x.
    #[test]
    fn prop_hypersphere_antipodal_numerical_stability(
        dim in 2usize..16usize,
        tau in 0.1f32..10.0f32,
    ) {
        let mut x = vec![0.0f32; dim];
        x[0] = 1.0;
        spherical_normalize(&mut x).unwrap();

        // Exact antipodal
        let mut neg_x = x.clone();
        for val in neg_x.iter_mut() {
            *val = -*val;
        }

        let d_anti = spherical_distance(&x, &neg_x, tau).unwrap();
        prop_assert!(!d_anti.is_nan(), "Antipodal distance resulted in NaN");
        let expected = core::f32::consts::PI / tau.sqrt();
        let diff = (d_anti - expected).abs();
        prop_assert!(diff <= 1e-5, "Antipodal distance mismatch: actual={}, expected={}", d_anti, expected);
    }

    /// 3.10: Compares AVX/NEON SIMD and scalar implementations, asserting deviation <= 1e-4.
    #[test]
    fn test_simd_vs_scalar_manifold_bit_equivalence(
        h_spatial_u in proptest::collection::vec(-10.0f32..10.0f32, 4),
        h_spatial_v in proptest::collection::vec(-10.0f32..10.0f32, 4),
        s_raw_u in proptest::collection::vec(-5.0f32..5.0f32, 4),
        s_raw_v in proptest::collection::vec(-5.0f32..5.0f32, 4),
        e_u in proptest::collection::vec(-20.0f32..20.0f32, 4),
        e_v in proptest::collection::vec(-20.0f32..20.0f32, 4),
    ) {
        let kappa = -1.2f32;
        let tau = 0.5f32;

        let config = ProductManifoldConfig {
            hyperbolic: vec![HyperbolicSpec { dim: 5, kappa, weight: 1.0 }],
            spherical: vec![SphericalSpec { dim: 4, tau, weight: 1.0 }],
            euclidean: Some(EuclideanSpec { dim: 4, weight: 1.0 }),
        };

        let mut u = Vec::new();
        u.push(lorentz_compute_x0(&h_spatial_u, kappa).unwrap());
        u.extend(&h_spatial_u);
        let mut s_u = s_raw_u;
        if s_u.iter().all(|&x| x == 0.0) { s_u[0] = 1.0; }
        spherical_normalize(&mut s_u).unwrap();
        u.extend(&s_u);
        u.extend(&e_u);

        let mut v = Vec::new();
        v.push(lorentz_compute_x0(&h_spatial_v, kappa).unwrap());
        v.extend(&h_spatial_v);
        let mut s_v = s_raw_v;
        if s_v.iter().all(|&x| x == 0.0) { s_v[0] = 1.0; }
        spherical_normalize(&mut s_v).unwrap();
        v.extend(&s_v);
        v.extend(&e_v);

        let simd_d2 = product_manifold_distance_squared(&config, &u, &v).unwrap();
        let scalar_d2 = scalar_product_manifold_distance_squared(&config, &u, &v).unwrap();

        let diff = (simd_d2 - scalar_d2).abs();
        let tol = 1e-4 * scalar_d2.abs().max(1.0);
        prop_assert!(diff <= tol, "SIMD vs scalar deviation: simd={}, scalar={}, diff={}", simd_d2, scalar_d2, diff);
    }
}
