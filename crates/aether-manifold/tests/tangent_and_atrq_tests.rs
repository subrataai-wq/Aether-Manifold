//! Project Aether-Manifold: Tangent-Space & AT-RQ Integration Tests
//!
//! Validates:
//! 1. Exponential & Logarithmic map bijection across Lorentz, Spherical, and Product manifolds.
//! 2. Lorentz Parallel Transport isometry.
//! 3. MVCC time-travel updates and velocity vector recovery on composite Product Manifolds.
//! 4. Anisotropic residual loss radial penalty dominance (\eta_{\parallel} >> \eta_{\perp}).
//! 5. Zero heap allocations on the hot path for tangent operators and Page ADC (AGENTS.md Rule 3.1).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use aether_core::alloc_guard::{reset_hot_path_counters, HotPathAllocGuard, TrackingAllocator};
use aether_core::page::TensorGraphPage;
use aether_manifold::at_rq::{
    anisotropic_residual_loss, at_rq_build_query_lut, at_rq_decode_neighbor, at_rq_encode_neighbor,
    at_rq_score_page_neighbors, CompactNeighborCode,
};
use aether_manifold::hyperbolic::{lorentz_compute_x0, lorentz_inner_product};
use aether_manifold::product::{
    EuclideanSpec, HyperbolicSpec, ProductManifoldConfig, SphericalSpec,
};
use aether_manifold::simd::{simd_dot_product, simd_euclidean_distance_squared};
use aether_manifold::tangent::{
    lorentz_exp_map, lorentz_log_map, lorentz_parallel_transport, lorentz_project_tangent,
    product_exp_map, product_log_map, product_parallel_transport, spherical_exp_map,
    spherical_log_map, spherical_parallel_transport, spherical_project_tangent,
};

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator::new(std::alloc::System);

#[test]
fn test_lorentz_exp_log_bijection() {
    let kappa = -1.0f32;
    // Base point x on H^3_{-1}: x_0 = sqrt(1.0 + 0.3^2 + 0.4^2 + 0.5^2) = sqrt(1.5)
    let spatial_x = [0.3f32, -0.4, 0.5];
    let x0 = lorentz_compute_x0(&spatial_x, kappa).unwrap();
    let x = [x0, spatial_x[0], spatial_x[1], spatial_x[2]];

    // Verify surface invariant <x, x>_L = 1/kappa = -1.0
    let inner_xx = lorentz_inner_product(&x, &x).unwrap();
    assert!((inner_xx - (-1.0)).abs() < 1e-6);

    // Tangent vector v in T_x H^3
    let raw_w = [0.1f32, 0.8, -0.2, 0.3];
    let mut v = [0.0f32; 4];
    lorentz_project_tangent(&x, &raw_w, kappa, &mut v).unwrap();

    let inner_xv = lorentz_inner_product(&x, &v).unwrap();
    assert!(
        inner_xv.abs() < 1e-6,
        "v must be strictly tangent: {}",
        inner_xv
    );

    // 1. Standard distance bijection
    let mut y = [0.0f32; 4];
    lorentz_exp_map(&x, &v, kappa, &mut y).unwrap();

    let inner_yy = lorentz_inner_product(&y, &y).unwrap();
    assert!(
        (inner_yy - (-1.0)).abs() < 1e-5,
        "y must satisfy surface invariant"
    );

    let mut v_rec = [0.0f32; 4];
    lorentz_log_map(&x, &y, kappa, &mut v_rec).unwrap();

    for i in 0..4 {
        assert!(
            (v_rec[i] - v[i]).abs() < 1e-4,
            "Log(Exp(v)) failed at index {}: v={}, v_rec={}",
            i,
            v[i],
            v_rec[i]
        );
    }

    let mut y_rec = [0.0f32; 4];
    lorentz_exp_map(&x, &v_rec, kappa, &mut y_rec).unwrap();
    for i in 0..4 {
        assert!(
            (y_rec[i] - y[i]).abs() < 1e-4,
            "Exp(Log(y)) failed at index {}: y={}, y_rec={}",
            i,
            y[i],
            y_rec[i]
        );
    }

    // 2. Micro-angle bijection (||v||_L < 10^-6) triggering Taylor expansion
    let mut v_micro = [0.0f32; 4];
    for i in 0..4 {
        v_micro[i] = v[i] * 5e-7;
    }

    let mut y_micro = [0.0f32; 4];
    lorentz_exp_map(&x, &v_micro, kappa, &mut y_micro).unwrap();

    let mut v_micro_rec = [0.0f32; 4];
    lorentz_log_map(&x, &y_micro, kappa, &mut v_micro_rec).unwrap();

    for i in 0..4 {
        assert!(
            (v_micro_rec[i] - v_micro[i]).abs() < 1e-7,
            "Micro-angle bijection failed at {}: orig={}, rec={}",
            i,
            v_micro[i],
            v_micro_rec[i]
        );
    }
}

#[test]
fn test_spherical_exp_log_bijection() {
    let tau = 2.0f32;
    // Unit base point on S^2: x = [1, 0, 0]
    let x = [1.0f32, 0.0, 0.0];

    // Tangent vector v in T_x S^2: v must satisfy <x, v> = 0
    let raw_w = [0.5f32, 0.3, -0.4];
    let mut v = [0.0f32; 3];
    spherical_project_tangent(&x, &raw_w, &mut v).unwrap();

    let dot_xv = simd_dot_product(&x, &v);
    assert!(
        dot_xv.abs() < 1e-6,
        "v must be tangent to sphere: {}",
        dot_xv
    );

    // 1. Standard distance bijection
    let mut y = [0.0f32; 3];
    spherical_exp_map(&x, &v, tau, &mut y).unwrap();

    let norm_y = simd_dot_product(&y, &y).sqrt();
    assert!((norm_y - 1.0).abs() < 1e-6, "y must be a unit vector");

    let mut v_rec = [0.0f32; 3];
    spherical_log_map(&x, &y, tau, &mut v_rec).unwrap();

    for i in 0..3 {
        assert!(
            (v_rec[i] - v[i]).abs() < 1e-4,
            "Spherical Log(Exp(v)) failed at {}: orig={}, rec={}",
            i,
            v[i],
            v_rec[i]
        );
    }

    // 2. Micro-angle bijection on sphere
    let mut v_micro = [0.0f32; 3];
    for i in 0..3 {
        v_micro[i] = v[i] * 5e-7;
    }

    let mut y_micro = [0.0f32; 3];
    spherical_exp_map(&x, &v_micro, tau, &mut y_micro).unwrap();

    let mut v_micro_rec = [0.0f32; 3];
    spherical_log_map(&x, &y_micro, tau, &mut v_micro_rec).unwrap();

    for i in 0..3 {
        assert!(
            (v_micro_rec[i] - v_micro[i]).abs() < 1e-7,
            "Spherical micro-angle bijection failed at {}: orig={}, rec={}",
            i,
            v_micro[i],
            v_micro_rec[i]
        );
    }
}

#[test]
fn test_lorentz_parallel_transport_isometry() {
    let kappa = -1.0f32;
    let x_sp = [0.2f32, -0.3, 0.4];
    let x0 = lorentz_compute_x0(&x_sp, kappa).unwrap();
    let x = [x0, x_sp[0], x_sp[1], x_sp[2]];

    let y_sp = [-0.1f32, 0.5, 0.2];
    let y0 = lorentz_compute_x0(&y_sp, kappa).unwrap();
    let y = [y0, y_sp[0], y_sp[1], y_sp[2]];

    let w1 = [0.0f32, 1.0, 0.0, 0.0];
    let w2 = [0.0f32, 0.0, 1.0, 0.5];
    let mut u = [0.0f32; 4];
    let mut v = [0.0f32; 4];
    lorentz_project_tangent(&x, &w1, kappa, &mut u).unwrap();
    lorentz_project_tangent(&x, &w2, kappa, &mut v).unwrap();

    let inner_uv = lorentz_inner_product(&u, &v).unwrap();

    let mut u_trans = [0.0f32; 4];
    let mut v_trans = [0.0f32; 4];
    lorentz_parallel_transport(&x, &y, &u, kappa, &mut u_trans).unwrap();
    lorentz_parallel_transport(&x, &y, &v, kappa, &mut v_trans).unwrap();

    // Verify transported vectors are tangent at y: <y, u_trans> = 0
    let inner_yu = lorentz_inner_product(&y, &u_trans).unwrap();
    let inner_yv = lorentz_inner_product(&y, &v_trans).unwrap();
    assert!(
        inner_yu.abs() < 1e-5,
        "u_trans must be in T_y H^3: {}",
        inner_yu
    );
    assert!(
        inner_yv.abs() < 1e-5,
        "v_trans must be in T_y H^3: {}",
        inner_yv
    );

    // Verify Riemannian isometry: <u_trans, v_trans>_L == <u, v>_L
    let inner_trans_uv = lorentz_inner_product(&u_trans, &v_trans).unwrap();
    assert!(
        (inner_trans_uv - inner_uv).abs() < 1e-5,
        "Parallel transport must preserve Riemannian inner product: orig={}, trans={}",
        inner_uv,
        inner_trans_uv
    );
}

#[test]
fn test_product_manifold_mvcc_time_travel() {
    let config = ProductManifoldConfig {
        hyperbolic: vec![HyperbolicSpec {
            dim: 4,
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

    assert_eq!(config.total_dim(), 9);

    // Base point x(tau_1) on M
    let x_sp = [0.3f32, -0.4, 0.5];
    let x0 = lorentz_compute_x0(&x_sp, -1.0).unwrap();
    let x_tau1 = [
        x0, x_sp[0], x_sp[1], x_sp[2], // Hyperbolic (dim 4)
        1.0, 0.0, 0.0, // Spherical (dim 3)
        2.0, -1.0, // Euclidean (dim 2)
    ];

    // Velocity vector Delta v in T_{x(tau_1)} M
    let mut delta_v = [0.0f32; 9];
    let raw_vh = [0.1f32, 0.5, -0.2, 0.3];
    lorentz_project_tangent(&x_tau1[..4], &raw_vh, -1.0, &mut delta_v[..4]).unwrap();

    let raw_vs = [0.0f32, 0.2, -0.3];
    spherical_project_tangent(&x_tau1[4..7], &raw_vs, &mut delta_v[4..7]).unwrap();

    delta_v[7] = 0.4;
    delta_v[8] = -0.6;

    // Time-travel step: x(tau_2) = Exp_{x(tau_1)}(Delta v)
    let mut x_tau2 = [0.0f32; 9];
    product_exp_map(&config, &x_tau1, &delta_v, &mut x_tau2).unwrap();

    // Reverse recovery: Delta v_rec = Log_{x(tau_1)}(x(tau_2))
    let mut delta_v_rec = [0.0f32; 9];
    product_log_map(&config, &x_tau1, &x_tau2, &mut delta_v_rec).unwrap();

    for i in 0..9 {
        assert!(
            (delta_v_rec[i] - delta_v[i]).abs() < 1e-4,
            "Product manifold MVCC recovery failed at {}: orig={}, rec={}",
            i,
            delta_v[i],
            delta_v_rec[i]
        );
    }
}

#[test]
fn test_anisotropic_parallel_penalty_dominance() {
    let dim = 24;
    let mut v = [0.0f32; 24];
    for (i, val) in v.iter_mut().enumerate() {
        *val = (i + 1) as f32 * 0.2;
    }

    let v_norm = simd_dot_product(&v, &v).sqrt();

    // Purely parallel error: r_parallel along v direction with norm epsilon
    let eps = 0.2f32;
    let mut v_hat_parallel = [0.0f32; 24];
    for i in 0..dim {
        v_hat_parallel[i] = v[i] - eps * (v[i] / v_norm);
    }

    // Purely orthogonal error: construct an orthogonal vector
    // Swap first two components with signs: (-v1, v0, 0, ...)
    let mut ortho_dir = [0.0f32; 24];
    ortho_dir[0] = -v[1];
    ortho_dir[1] = v[0];
    let ortho_norm = simd_dot_product(&ortho_dir, &ortho_dir).sqrt();

    let mut v_hat_ortho = [0.0f32; 24];
    for i in 0..dim {
        v_hat_ortho[i] = v[i] - eps * (ortho_dir[i] / ortho_norm);
    }

    let eta_parallel = 100.0f32;
    let eta_ortho = 1.0f32;

    let loss_parallel =
        anisotropic_residual_loss(&v, &v_hat_parallel, eta_parallel, eta_ortho).unwrap();
    let loss_ortho = anisotropic_residual_loss(&v, &v_hat_ortho, eta_parallel, eta_ortho).unwrap();

    assert!(
        loss_parallel > 50.0 * loss_ortho,
        "Parallel penalty must heavily dominate orthogonal penalty: parallel={}, ortho={}",
        loss_parallel,
        loss_ortho
    );
}

#[test]
fn gate_zero_hot_path_allocations_tangent_and_page_adc() {
    let config = ProductManifoldConfig {
        hyperbolic: vec![HyperbolicSpec {
            dim: 4,
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

    let base = [1.2247449f32, 0.3, -0.4, 0.5, 1.0, 0.0, 0.0, 2.0, -1.0];
    let tangent = [0.0f32, 0.1, -0.1, 0.2, 0.0, 0.1, 0.1, 0.2, 0.3];
    let mut out_point = [0.0f32; 9];
    let mut out_tangent = [0.0f32; 9];

    // Setup a 4096-byte TensorGraphPage
    let mut page = TensorGraphPage {
        header: aether_core::page::PageHeader {
            crc32c_checksum: 0,
            active_vertex_count: 8,
            lsn: aether_core::hlc::HlcTimestamp::pack(100, 1),
            page_id: 42,
            manifold_curvature_bits: 0,
            csr_offset_table: [0u16; 16],
        },
        at_rq_codebooks: [0u8; 256],
        neighbor_ids: [0u64; 64],
        quantized_neighbors: [0u8; 2048],
        sheaf_edge_weights: [0u8; 1216],
    };

    // Populate 8 active neighbor codes
    for i in 0..8 {
        let code = CompactNeighborCode {
            radial_norm_sq: (i + 1) as f32 * 0.5,
            scale: 1.0,
            subspace_codes: [i as u8; 24],
        };
        let bytes = code.as_bytes();
        let target_slice = &mut page.quantized_neighbors[i * 32..(i + 1) * 32];
        target_slice.copy_from_slice(bytes);
    }

    // Query LUT: 24 subspaces x 16 centroids = 384 floats
    let centroids_per_subspace = 16;
    let query_lut = [0.05f32; 24 * 16];
    let qc_norm_sq = 1.25f32;
    let kappa = -1.0f32;
    let mut out_distances = [0.0f32; 64];

    reset_hot_path_counters();
    {
        let guard = HotPathAllocGuard::enter();

        // 1. Tangent exponential map (0 heap allocations)
        product_exp_map(&config, &base, &tangent, &mut out_point).unwrap();

        // 2. Tangent logarithmic map (0 heap allocations)
        product_log_map(&config, &base, &out_point, &mut out_tangent).unwrap();

        // 3. Page ADC scoring kernel (0 heap allocations)
        at_rq_score_page_neighbors(
            qc_norm_sq,
            &query_lut,
            centroids_per_subspace,
            kappa,
            &page,
            &mut out_distances,
        )
        .unwrap();

        assert_eq!(
            guard.alloc_count(),
            0,
            "Tangent operators and Page ADC triggered hot-path heap allocations!"
        );
        assert!(guard.assert_zero_allocations().is_ok());
    }

    // Verify distance results:
    // Active vertices (0..8) must be finite non-negative distances
    for (i, &dist) in out_distances.iter().enumerate().take(8) {
        assert!(
            dist.is_finite() && dist >= 0.0,
            "Active neighbor {} distance must be finite and non-negative: {}",
            i,
            dist
        );
    }

    // Inactive vertices (8..64) must be INFINITY
    for (i, &dist) in out_distances.iter().enumerate().skip(8) {
        assert_eq!(
            dist,
            f32::INFINITY,
            "Inactive neighbor {} must have INFINITY distance",
            i
        );
    }
}

#[test]
fn test_at_rq_encode_decode_roundtrip() {
    let dim = 48; // 24 subspaces x 2 dimensions each
    let d_sub = 2;
    let centroids_per_subspace = 4;

    // Build codebooks: 24 x 4 x 2 = 192 floats
    let mut codebooks = vec![0.0f32; 24 * centroids_per_subspace * d_sub];
    for (idx, val) in codebooks.iter_mut().enumerate() {
        *val = (idx as f32 * 0.1).sin();
    }

    let mut v_tangent = vec![0.0f32; dim];
    for (idx, val) in v_tangent.iter_mut().enumerate() {
        *val = (idx as f32 * 0.15).cos();
    }

    let mut code = CompactNeighborCode::ZERO;
    at_rq_encode_neighbor(&v_tangent, &codebooks, centroids_per_subspace, &mut code).unwrap();

    assert!(code.radial_norm_sq > 0.0);

    let mut reconstructed = vec![0.0f32; dim];
    at_rq_decode_neighbor(
        &code,
        &codebooks,
        centroids_per_subspace,
        &mut reconstructed,
    )
    .unwrap();

    let diff_sq = simd_euclidean_distance_squared(&v_tangent, &reconstructed);
    assert!(diff_sq.is_finite());

    // Build query LUT and verify dimensions and finite values
    let mut query_lut = vec![0.0f32; 24 * centroids_per_subspace];
    at_rq_build_query_lut(
        &v_tangent,
        &codebooks,
        centroids_per_subspace,
        &mut query_lut,
    )
    .unwrap();
    assert!(query_lut.iter().all(|x| x.is_finite()));
}

#[test]
fn test_spherical_and_product_parallel_transport() {
    // 1. Spherical parallel transport isometry
    let x = [1.0f32, 0.0, 0.0];
    let y = [0.0f32, 1.0, 0.0];
    let u = [0.0f32, 0.0, 1.0];
    let mut u_trans = [0.0f32; 3];
    spherical_parallel_transport(&x, &y, &u, &mut u_trans).unwrap();
    assert!((simd_dot_product(&y, &u_trans)).abs() < 1e-6);
    assert!((simd_dot_product(&u_trans, &u_trans) - 1.0).abs() < 1e-6);

    // 2. Product manifold parallel transport
    let config = ProductManifoldConfig {
        hyperbolic: vec![HyperbolicSpec {
            dim: 4,
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

    let p1 = [1.2247449f32, 0.3, -0.4, 0.5, 1.0, 0.0, 0.0, 2.0, -1.0];
    let p2 = [1.2247449f32, -0.3, 0.4, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0];
    let mut v = [0.0f32; 9];
    let raw_vh = [0.0f32, 1.0, 0.0, 0.0];
    lorentz_project_tangent(&p1[..4], &raw_vh, -1.0, &mut v[..4]).unwrap();
    v[6] = 1.0; // Tangent to p1[4..7] = [1, 0, 0]
    v[7] = 0.5;
    v[8] = -0.5;

    let mut v_trans = [0.0f32; 9];
    product_parallel_transport(&config, &p1, &p2, &v, &mut v_trans).unwrap();
    assert!(v_trans.iter().all(|x| x.is_finite()));
}
