//! Unit and integration test suite for `aether-autotune`.
//!
//! Verifies:
//! - Discrete Forman and Ollivier Ricci curvature estimation.
//! - Continuous Product Manifold curvature evolution within physical bounds.
//! - Sheaf restriction map Dirichlet energy minimization with exact Householder isometry.
//! - DMA compaction page descriptor CRC32C validation and zero-alloc execution.
//! - Autonomous churn coordinator atomic snapshot wait-free loads.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::assertions_on_constants,
    clippy::needless_range_loop
)]

use aether_autotune::churn_controller::AutonomousChurnCoordinator;
use aether_autotune::dma_compaction::{DmaCompactionEngine, DmaPageDescriptor};
use aether_autotune::ricci_flow::{RicciFlowConfig, RicciFlowOptimizer};
use aether_autotune::sheaf_opt::SheafMorphismOptimizer;
use aether_core::alloc_guard::{reset_hot_path_counters, HotPathAllocGuard};
use aether_core::page::TensorGraphPage;
use aether_manifold::product::{HyperbolicSpec, ProductManifoldConfig};
use aether_query::sheaf::{SheafRestrictionMap, VertexStalk};
use aether_storage::voronoi_compaction::VoronoiCellCertificate;
use bytemuck::Zeroable;

#[test]
fn test_ricci_flow_forman_curvatures() {
    let config = RicciFlowConfig::default();
    let optimizer = RicciFlowOptimizer::new(config).expect("Valid config");

    // Star graph (1 central hub node 0 connected to 5 leaves 1..5)
    // Hub degree = 5, leaf degrees = 1.
    // Forman curvature R_F(0, leaf) = 4 - 5 - 1 = -2.0 (strictly negative, hyperbolic branching)
    let num_vertices = 6;
    let edges = vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)];

    let curvatures = optimizer
        .compute_forman_curvatures(num_vertices, &edges)
        .expect("Curvatures computed");

    assert_eq!(curvatures.len(), 5);
    for edge in &curvatures {
        assert_eq!(edge.forman_curvature, -2.0);
        assert!(edge.ollivier_curvature < 0.0);
    }

    // Evolve curvatures: negative curvature must shift kappa more negative
    let initial_kappa = -1.0;
    let initial_tau = 1.0;
    let (next_kappa, _next_tau) =
        optimizer.evolve_curvatures(initial_kappa, initial_tau, &curvatures);

    assert!(
        next_kappa < initial_kappa,
        "Hyperbolic curvature must deepen under branching: next={}, initial={}",
        next_kappa,
        initial_kappa
    );
}

#[test]
fn test_curvature_evolution_bounds() {
    let config = RicciFlowConfig {
        learning_rate_kappa: 0.5,
        learning_rate_tau: 0.5,
        kappa_bounds: (-5.0, -0.1),
        tau_bounds: (0.1, 5.0),
        cooperative_yield_micros: 250,
    };
    let optimizer = RicciFlowOptimizer::new(config).expect("Valid config");

    // Test extreme negative edge curvatures
    let neg_edges = vec![
        aether_autotune::ricci_flow::EdgeCurvature::new(0, 1, -100.0, -50.0),
        aether_autotune::ricci_flow::EdgeCurvature::new(1, 2, -100.0, -50.0),
    ];
    let (k1, _) = optimizer.evolve_curvatures(-1.0, 1.0, &neg_edges);
    assert!((-5.0..=-0.1).contains(&k1));

    // Test extreme positive edge curvatures
    let pos_edges = vec![
        aether_autotune::ricci_flow::EdgeCurvature::new(0, 1, 100.0, 50.0),
        aether_autotune::ricci_flow::EdgeCurvature::new(1, 2, 100.0, 50.0),
    ];
    let (_, t1) = optimizer.evolve_curvatures(-1.0, 1.0, &pos_edges);
    assert!((0.1..=5.0).contains(&t1));
}

#[test]
fn test_sheaf_morphism_optimization_energy_reduction() {
    let optimizer = SheafMorphismOptimizer::<8>::new(0.5, 1e-6).expect("Valid config");

    let stalk_u = VertexStalk::<8>::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let stalk_v = VertexStalk::<8>::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    // Initial maps: identity
    let map_u = SheafRestrictionMap::<8>::default();
    let map_v = SheafRestrictionMap::<8>::default();

    let initial_energy = optimizer.compute_dirichlet_energy(&map_u, &map_v, &stalk_u, &stalk_v);
    assert!(initial_energy > 0.0);

    // Optimize map_u towards map_v * stalk_v
    let updated_map_u = optimizer.step_optimize(&map_u, &map_v, &stalk_u, &stalk_v);
    let new_energy = optimizer.compute_dirichlet_energy(&updated_map_u, &map_v, &stalk_u, &stalk_v);

    assert!(
        new_energy < initial_energy,
        "Energy must decrease: initial={}, new={}",
        initial_energy,
        new_energy
    );

    // Verify isometry: ||H_w x||_2 == ||x||_2
    let mut out_stalk = aether_query::sheaf::EdgeStalk::<8>::default();
    updated_map_u.apply(&stalk_u, &mut out_stalk);
    let out_norm = {
        let mut sum = 0.0f32;
        for &c in &out_stalk.components {
            sum += c * c;
        }
        sum.sqrt()
    };
    assert!(
        (out_norm - stalk_u.norm()).abs() < 1e-5,
        "Householder map must preserve exact isometry"
    );
}

#[test]
fn test_dma_compaction_zero_alloc() {
    reset_hot_path_counters();

    let engine = DmaCompactionEngine::new();

    let mut desc = DmaPageDescriptor::new(100, 200, 0, 0, 0x01);
    desc.update_crc();
    assert!(desc.verify_crc().is_ok());

    let source_page = TensorGraphPage::zeroed();
    let mut target_pages = [TensorGraphPage::zeroed()];

    let cert = VoronoiCellCertificate::<8> {
        cell_id: 1,
        centroid: [0.0f32; 8],
        max_geodesic_radius: 1.0,
        modified: false,
    };

    let config = ProductManifoldConfig {
        hyperbolic: vec![HyperbolicSpec {
            dim: 8,
            kappa: -1.0,
            weight: 1.0,
        }],
        spherical: vec![],
        euclidean: None,
    };

    let vertex_coords = [[0.1f32; 8]];

    let guard = HotPathAllocGuard::enter();
    let telemetry = engine
        .execute_dma_compaction(
            &config,
            &[desc],
            &[source_page],
            &mut target_pages,
            &cert,
            &vertex_coords,
        )
        .expect("DMA compaction succeeded");
    core::mem::drop(guard);

    assert_eq!(telemetry.total_pages_processed, 1);
    assert_eq!(telemetry.interior_bypassed_pages, 1);
}

#[test]
fn test_autonomous_churn_coordinator() {
    let ricci_opt = RicciFlowOptimizer::new(RicciFlowConfig::default()).expect("Valid config");
    let mut coordinator = AutonomousChurnCoordinator::new(-1.0, 1.0, ricci_opt);

    // Initial snapshot load without locking
    let (k0, t0) = coordinator.snapshot().load();
    assert_eq!(k0, -1.0);
    assert_eq!(t0, 1.0);

    // Record mutations and trigger Ricci step
    coordinator.record_mutations(500);
    let edge_curvatures = vec![
        aether_autotune::ricci_flow::EdgeCurvature::new(0, 1, -2.0, -0.5),
        aether_autotune::ricci_flow::EdgeCurvature::new(1, 2, -3.0, -0.8),
    ];

    let (k1, _) = coordinator
        .trigger_ricci_step(&edge_curvatures)
        .expect("Ricci step succeeded");
    assert!(k1 < -1.0);

    let telemetry = coordinator.telemetry();
    assert_eq!(telemetry.total_mutations, 500);
    assert_eq!(telemetry.ricci_steps_executed, 1);
    assert_eq!(telemetry.active_kappa, k1);
}
