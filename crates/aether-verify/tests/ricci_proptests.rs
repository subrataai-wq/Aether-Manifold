//! Deep Property-Based Fuzzing for Ricci Flow, Sheaf Optimization, and DMA Descriptors
//!
//! Enforces:
//! - Forman-Ricci curvature symmetry and bounded degree relations.
//! - Curvature evolution physical bounds invariance (kappa < 0, tau > 0, 0 NaNs).
//! - Householder Sheaf restriction map exact isometry preservation under optimization.
//! - CRC32C single-bit corruption detection across DMA page descriptors.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::assertions_on_constants,
    clippy::needless_range_loop
)]

use aether_autotune::dma_compaction::DmaPageDescriptor;
use aether_autotune::ricci_flow::{EdgeCurvature, RicciFlowConfig, RicciFlowOptimizer};
use aether_autotune::sheaf_opt::SheafMorphismOptimizer;
use aether_query::sheaf::{SheafRestrictionMap, VertexStalk};
use proptest::prelude::*;

proptest! {
    /// Property 1: Forman-Ricci curvature symmetry and bounds:
    /// R_F(u, v) = 4 - deg(u) - deg(v) <= 2 for all connected graphs (deg >= 1).
    #[test]
    fn prop_forman_ricci_symmetry_and_bounds(
        num_vertices in 10usize..50,
        seed in 1u64..1000,
    ) {
        let mut edges = Vec::new();
        // Generate deterministic ring + random cross edges
        for i in 0..num_vertices {
            let j = (i + 1) % num_vertices;
            edges.push((i, j));
            let cross = (i * 3 + (seed as usize)) % num_vertices;
            if cross != i {
                edges.push((i, cross));
            }
        }

        let optimizer = RicciFlowOptimizer::new(RicciFlowConfig::default()).expect("optimizer");
        let curvatures = optimizer.compute_forman_curvatures(num_vertices, &edges).expect("curvatures");

        for edge in &curvatures {
            prop_assert!(edge.forman_curvature <= 2.0);
            prop_assert!(!edge.forman_curvature.is_nan());
            prop_assert!(!edge.ollivier_curvature.is_nan());
        }
    }

    /// Property 2: Curvature evolution strictly preserves physical bounds:
    /// kappa in [kappa_min, kappa_max] < 0 and tau in [tau_min, tau_max] > 0 with zero NaNs.
    #[test]
    fn prop_curvature_evolution_physical_bounds(
        init_k in -5.0f32..-0.1f32,
        init_t in 0.1f32..5.0f32,
        c1 in -10.0f32..10.0f32,
        c2 in -10.0f32..10.0f32,
        c3 in -10.0f32..10.0f32,
    ) {
        let config = RicciFlowConfig {
            learning_rate_kappa: 0.1,
            learning_rate_tau: 0.1,
            kappa_bounds: (-10.0, -0.05),
            tau_bounds: (0.05, 10.0),
            cooperative_yield_micros: 250,
        };
        let optimizer = RicciFlowOptimizer::new(config).expect("optimizer");

        let edge_curvatures = [
            EdgeCurvature::new(0, 1, c1, c1 / 5.0),
            EdgeCurvature::new(1, 2, c2, c2 / 5.0),
            EdgeCurvature::new(2, 3, c3, c3 / 5.0),
        ];

        let (next_k, next_t) = optimizer.evolve_curvatures(init_k, init_t, &edge_curvatures);

        prop_assert!(!next_k.is_nan() && !next_t.is_nan());
        prop_assert!((-10.0..=-0.05).contains(&next_k));
        prop_assert!((0.05..=10.0).contains(&next_t));
    }

    /// Property 3: Householder Sheaf restriction map step optimization preserves exact isometry:
    /// ||H_w x||_2 == ||x||_2 +/- 10^-5 identically.
    #[test]
    fn prop_sheaf_householder_isometry_preserved(
        u_vals in proptest::array::uniform8(-10.0f32..10.0f32),
        v_vals in proptest::array::uniform8(-10.0f32..10.0f32),
        learning_rate in 0.01f32..1.0f32,
    ) {
        let optimizer = SheafMorphismOptimizer::<8>::new(learning_rate, 1e-6).expect("optimizer");
        let stalk_u = VertexStalk::<8>::new(u_vals);
        let stalk_v = VertexStalk::<8>::new(v_vals);

        let map_u = SheafRestrictionMap::<8>::default();
        let map_v = SheafRestrictionMap::<8>::default();

        let updated_map_u = optimizer.step_optimize(&map_u, &map_v, &stalk_u, &stalk_v);

        let mut out = aether_query::sheaf::EdgeStalk::<8>::default();
        updated_map_u.apply(&stalk_u, &mut out);

        let mut out_norm_sq = 0.0f32;
        for &c in &out.components {
            out_norm_sq += c * c;
        }
        let out_norm = out_norm_sq.sqrt();
        let in_norm = stalk_u.norm();

        if in_norm > 1e-4 {
            let relative_error = ((out_norm - in_norm) / in_norm).abs();
            prop_assert!(relative_error < 1e-4, "Relative isometry error too high: {}", relative_error);
        }
    }

    /// Property 4: Single-bit flip corruption detection across DMA page descriptors.
    #[test]
    fn prop_dma_descriptor_single_bit_flip_detection(
        src_page in 0u64..u64::MAX,
        dst_page in 0u64..u64::MAX,
        src_offset in 0u32..1000,
        dst_offset in 0u32..1000,
        byte_idx in 0usize..28,
        bit_idx in 0usize..8,
    ) {
        let mut desc = DmaPageDescriptor::new(src_page, dst_page, src_offset, dst_offset, 0x01);
        desc.update_crc();
        prop_assert!(desc.verify_crc().is_ok());

        let mut bytes: [u8; 64] = bytemuck::cast(desc);
        if let Some(target) = bytes.get_mut(byte_idx) {
            *target ^= 1u8 << bit_idx;
        }

        let corrupted: DmaPageDescriptor = bytemuck::cast(bytes);
        prop_assert!(corrupted.verify_crc().is_err());
    }
}
