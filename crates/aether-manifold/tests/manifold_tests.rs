//! Project Aether-Manifold: Manifold Kernels Integration & Unit Tests
//!
//! Validates Euclidean, Lorentz Hyperbolic, and Spherical metric computations,
//! horizon clamping, and zero hot-path heap allocations.

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
    euclidean_distance, euclidean_distance_squared, lorentz_compute_x0, lorentz_distance,
    lorentz_distance_squared, lorentz_inner_product, lorentz_reproject,
    product_manifold_distance, product_manifold_distance_squared, scalar_lorentz_distance,
    scalar_spherical_distance, spherical_distance, spherical_distance_squared,
    spherical_normalize, EuclideanSpec, HyperbolicSpec, ProductManifoldConfig, SphericalSpec,
};

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator::new(std::alloc::System);

#[test]
fn test_euclidean_metrics() {
    let u = [1.0f32, 2.0, 3.0, 4.0];
    let v = [4.0f32, 6.0, 8.0, 4.0];
    // diffs: 3, 4, 5, 0 -> squares: 9 + 16 + 25 + 0 = 50
    let d2 = euclidean_distance_squared(&u, &v).unwrap();
    assert_eq!(d2, 50.0);
    let d = euclidean_distance(&u, &v).unwrap();
    assert!((d - libm::sqrtf(50.0)).abs() < 1e-6);

    // Identity
    assert_eq!(euclidean_distance_squared(&u, &u).unwrap(), 0.0);

    // Dimension mismatch
    let w = [1.0f32, 2.0];
    assert!(matches!(
        euclidean_distance_squared(&u, &w),
        Err(AetherError::InvalidBufferSize { .. })
    ));
}

#[test]
fn test_lorentz_hyperbolic_metrics() {
    let kappa = -1.0f32;
    // Construct valid point in H_{-1}^2: x_0 = sqrt(1 / 1 + 1^2 + 2^2) = sqrt(1 + 1 + 4) = sqrt(6)
    let mut x = [0.0f32, 1.0, 2.0];
    lorentz_reproject(&mut x, kappa).unwrap();
    assert!((x[0] - libm::sqrtf(6.0)).abs() < 1e-6);

    // Check Lorentz surface invariant: <x, x>_L = -x0^2 + x1^2 + x2^2 = -6 + 1 + 4 = -1 = 1 / kappa
    let inner_self = lorentz_inner_product(&x, &x).unwrap();
    assert!((inner_self - (-1.0)).abs() < 1e-6);

    // Distance to self must be 0
    let d_self = lorentz_distance(&x, &x, kappa).unwrap();
    assert_eq!(d_self, 0.0);

    // Another point
    let mut y = [0.0f32, 2.0, 3.0];
    lorentz_reproject(&mut y, kappa).unwrap();
    // y_0 = sqrt(1 + 4 + 9) = sqrt(14)
    assert!((y[0] - libm::sqrtf(14.0)).abs() < 1e-6);

    let d_xy = lorentz_distance(&x, &y, kappa).unwrap();
    let d_yx = lorentz_distance(&y, &x, kappa).unwrap();
    assert!((d_xy - d_yx).abs() < 1e-6);
    assert!(d_xy > 0.0);

    // Compare with scalar implementation
    let scalar_d = scalar_lorentz_distance(&x, &y, kappa).unwrap();
    assert!((d_xy - scalar_d).abs() < 1e-5);

    // Invalid kappa >= 0 must return error
    assert!(matches!(
        lorentz_distance(&x, &y, 1.0),
        Err(AetherError::LorentzHorizonViolation)
    ));
}

#[test]
fn test_spherical_metrics() {
    let tau = 1.0f32;
    let mut u = [1.0f32, 0.0, 0.0];
    let mut v = [0.0f32, 1.0, 0.0];
    spherical_normalize(&mut u).unwrap();
    spherical_normalize(&mut v).unwrap();

    // Orthogonal unit vectors on S^2 with tau = 1: distance = arccos(0) = pi / 2
    let d_uv = spherical_distance(&u, &v, tau).unwrap();
    let expected = core::f32::consts::FRAC_PI_2;
    assert!((d_uv - expected).abs() < 1e-6);

    // Identity: distance to self must be 0
    let d_self = spherical_distance(&u, &u, tau).unwrap();
    assert_eq!(d_self, 0.0);

    // Antipodal stability: v = -u
    let antipodal = [-1.0f32, 0.0, 0.0];
    let d_anti = spherical_distance(&u, &antipodal, tau).unwrap();
    assert!((d_anti - core::f32::consts::PI).abs() < 1e-6);

    // Compare with scalar implementation
    let scalar_d = scalar_spherical_distance(&u, &v, tau).unwrap();
    assert!((d_uv - scalar_d).abs() < 1e-6);

    // Invalid tau <= 0
    assert!(matches!(
        spherical_distance(&u, &v, -1.0),
        Err(AetherError::LorentzHorizonViolation)
    ));
}

#[test]
fn test_product_manifold_metrics() {
    let config = ProductManifoldConfig {
        hyperbolic: vec![HyperbolicSpec {
            dim: 3, // x0 + 2 spatial
            kappa: -1.0,
            weight: 1.0,
        }],
        spherical: vec![SphericalSpec {
            dim: 3,
            tau: 1.0,
            weight: 1.0,
        }],
        euclidean: Some(EuclideanSpec {
            dim: 2,
            weight: 1.0,
        }),
    };

    assert_eq!(config.total_dim(), 3 + 3 + 2); // 8

    let mut u = vec![0.0f32; 8];
    // Hyperbolic part (indices 0..3)
    u[1] = 1.0;
    u[2] = 1.0;
    u[0] = lorentz_compute_x0(&u[1..3], -1.0).unwrap();
    // Spherical part (indices 3..6)
    u[3] = 1.0;
    spherical_normalize(&mut u[3..6]).unwrap();
    // Euclidean part (indices 6..8)
    u[6] = 2.0;
    u[7] = 3.0;

    let mut v = vec![0.0f32; 8];
    v[1] = 2.0;
    v[2] = 0.5;
    v[0] = lorentz_compute_x0(&v[1..3], -1.0).unwrap();
    v[4] = 1.0;
    spherical_normalize(&mut v[3..6]).unwrap();
    v[6] = 5.0;
    v[7] = 7.0;

    let d2 = product_manifold_distance_squared(&config, &u, &v).unwrap();
    assert!(d2 > 0.0);

    let d = product_manifold_distance(&config, &u, &v).unwrap();
    assert!((d - libm::sqrtf(d2)).abs() < 1e-6);

    // Identity
    let d_self = product_manifold_distance_squared(&config, &u, &u).unwrap();
    assert!((d_self - 0.0).abs() < 1e-6);
}

#[test]
fn gate_zero_hot_path_allocations_on_distances() {
    let u = [1.0f32; 128];
    let v = [2.0f32; 128];

    reset_hot_path_counters();
    {
        let guard = HotPathAllocGuard::enter();

        let _ = euclidean_distance_squared(&u, &v).unwrap();
        let _ = euclidean_distance(&u, &v).unwrap();

        assert_eq!(guard.alloc_count(), 0, "Euclidean distance allocated heap memory");
        assert!(guard.assert_zero_allocations().is_ok());
    }

    let mut h_u = [0.0f32; 64];
    h_u[0] = 10.0;
    let mut h_v = [0.0f32; 64];
    h_v[0] = 10.0;

    reset_hot_path_counters();
    {
        let guard = HotPathAllocGuard::enter();

        let _ = lorentz_inner_product(&h_u, &h_v).unwrap();
        let _ = lorentz_distance(&h_u, &h_v, -1.0).unwrap();
        let _ = lorentz_distance_squared(&h_u, &h_v, -1.0).unwrap();

        assert_eq!(guard.alloc_count(), 0, "Lorentz distance allocated heap memory");
        assert!(guard.assert_zero_allocations().is_ok());
    }

    let s_u = [0.5f32; 32];
    let s_v = [0.5f32; 32];

    reset_hot_path_counters();
    {
        let guard = HotPathAllocGuard::enter();

        let _ = spherical_distance(&s_u, &s_v, 2.0).unwrap();
        let _ = spherical_distance_squared(&s_u, &s_v, 2.0).unwrap();

        assert_eq!(guard.alloc_count(), 0, "Spherical distance allocated heap memory");
        assert!(guard.assert_zero_allocations().is_ok());
    }
}

