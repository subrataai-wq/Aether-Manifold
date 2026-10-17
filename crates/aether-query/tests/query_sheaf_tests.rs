//! Project Aether-Manifold: Phase 2 Query & Sheaf Tests
//!
//! Validates:
//! 1. Householder isometry ($\|H_w x\| = \|x\|$) and involution ($H_w H_w x = x$).
//! 2. Zero-copy packing and unpacking of 64 sheaf morphisms in 4KB TensorGraphPage.
//! 3. Sheaf Coboundary and Laplacian Positive Semi-Definiteness ($x^T L_{\mathcal{F}} x \ge 0$).
//! 4. Sheaf Dirichlet Energy Solver convergence.
//! 5. Semiring SpMV multi-hop query execution across Tropical, Diffusion, and Bottleneck.
//! 6. Zero hot-path heap allocations inside `HotPathAllocGuard`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::needless_range_loop
)]

use aether_core::alloc_guard::HotPathAllocGuard;
use aether_core::page::TensorGraphPage;
use aether_query::compiler::{FrontierNode, QueryPlan, SemiringSpMvEngine, MAX_QUERY_FRONTIER};
use aether_query::dirichlet::{DirichletSolverConfig, SheafDirichletSolver};
use aether_query::laplacian::{BlockSheafLaplacian, SheafCoboundary};
use aether_query::semiring::{
    semiring_eval, BottleneckSemiring, DiffusionSemiring, SemiringAlgebra, SemiringMode,
    TropicalSemiring,
};
use aether_query::sheaf::{
    pack_page_sheaf_morphisms, unpack_page_sheaf_morphism, SheafRestrictionMap, VertexStalk,
    DEFAULT_STALK_DIM,
};

#[test]
fn test_householder_sheaf_isometry_and_involution() {
    let reflector = [0.1, -0.3, 0.5, 0.2, -0.4, 0.6, -0.1, 0.2];
    let morphism = SheafRestrictionMap::<DEFAULT_STALK_DIM>::from_reflector(&reflector, 1.0);

    let x = VertexStalk::new([1.0, 2.0, -1.5, 0.5, -2.0, 3.0, 0.1, -0.8]);
    let initial_norm = x.norm();

    let mut y = aether_query::sheaf::EdgeStalk::default();
    morphism.apply(&x, &mut y);

    let mut sum_sq_y = 0.0f32;
    for val in &y.components {
        sum_sq_y += val * val;
    }
    let transformed_norm = libm::sqrtf(sum_sq_y);

    // Isometry check: ||H x|| == ||x||
    assert!(
        (transformed_norm - initial_norm).abs() < 1e-5,
        "Householder reflection must preserve stalk L2 norm"
    );

    // Involution check: H (H x) == x
    let mut x_reconstructed = VertexStalk::default();
    morphism.apply_adjoint(&y, &mut x_reconstructed);

    for (orig, rec) in x.components.iter().zip(x_reconstructed.components.iter()) {
        assert!(
            (orig - rec).abs() < 1e-5,
            "Householder reflection must be an involution"
        );
    }
}

#[test]
fn test_pack_unpack_page_sheaf_morphisms() {
    let mut page = TensorGraphPage {
        header: aether_core::page::PageHeader {
            crc32c_checksum: 0,
            active_vertex_count: 64,
            lsn: aether_core::hlc::HlcTimestamp::default(),
            page_id: 42,
            manifold_curvature_bits: 0,
            csr_offset_table: [0; 15],
            flags: 0,
        },
        at_rq_codebooks: [0; 256],
        neighbor_ids: [0; 64],
        quantized_neighbors: [0; 2048],
        sheaf_edge_weights: [0; 1216],
    };

    let mut morphisms = [SheafRestrictionMap::<DEFAULT_STALK_DIM>::default(); 64];
    let mut weights = [0.0f32; 64];

    for (i, m) in morphisms.iter_mut().enumerate() {
        let val = (i as f32 + 1.0) * 0.01;
        let ref_arr = [val, -val, val * 0.5, val, -val, val, 0.1, -0.2];
        let attenuation = 0.85 + (i as f32 * 0.002);
        *m = SheafRestrictionMap::from_reflector(&ref_arr, attenuation);
        if let Some(w) = weights.get_mut(i) {
            *w = attenuation;
        }
    }

    pack_page_sheaf_morphisms(&mut page, &morphisms, &weights).expect("Packing must succeed");

    // Unpack and assert fidelity within BF16 tolerance
    for (i, original) in morphisms.iter().enumerate() {
        let unpacked =
            unpack_page_sheaf_morphism(&page, i).expect("Unpacking neighbor morphism must succeed");

        for (o_w, u_w) in original.reflector_w.iter().zip(unpacked.reflector_w.iter()) {
            assert!(
                (o_w - u_w).abs() < 0.01,
                "BF16 reflector components must roundtrip within 1% error"
            );
        }

        assert!(
            (original.attenuation - unpacked.attenuation).abs() < 0.01,
            "Attenuation weight must roundtrip within BF16 tolerance"
        );
    }
}

#[test]
fn test_sheaf_coboundary_and_laplacian_psd() {
    let map_u = SheafRestrictionMap::<DEFAULT_STALK_DIM>::from_reflector(
        &[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        1.0,
    );
    let map_v = SheafRestrictionMap::<DEFAULT_STALK_DIM>::identity();

    let x_u = VertexStalk::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let x_v = VertexStalk::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    let energy = SheafCoboundary::edge_dirichlet_energy(&map_u, &x_u, &map_v, &x_v);
    assert!(energy >= 0.0, "Sheaf Dirichlet energy must be non-negative");

    // Triangle graph: 3 vertices, 3 edges
    let x_w = VertexStalk::new([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let edges = [
        (map_u, x_u, map_v, x_v),
        (map_v, x_v, map_u, x_w),
        (map_u, x_w, map_u, x_u),
    ];

    let total_energy = BlockSheafLaplacian::total_dirichlet_energy(&edges);
    assert!(
        total_energy >= 0.0,
        "Total graph Sheaf Dirichlet energy must be >= 0 (PSD)"
    );
}

#[test]
fn test_sheaf_dirichlet_solver_convergence() {
    let mut stalks = [
        VertexStalk::<DEFAULT_STALK_DIM>::new([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        VertexStalk::<DEFAULT_STALK_DIM>::new([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ];

    let target_attractions = [
        VertexStalk::<DEFAULT_STALK_DIM>::new([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        VertexStalk::<DEFAULT_STALK_DIM>::new([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ];

    let id_map = SheafRestrictionMap::<DEFAULT_STALK_DIM>::identity();

    let edges_0 = [aether_query::dirichlet::ClusterEdgeDescriptor {
        neighbor_index: 1,
        map_u: id_map,
        map_v: id_map,
    }];
    let edges_1 = [aether_query::dirichlet::ClusterEdgeDescriptor {
        neighbor_index: 0,
        map_u: id_map,
        map_v: id_map,
    }];
    let adj: [&[aether_query::dirichlet::ClusterEdgeDescriptor<DEFAULT_STALK_DIM>]; 2] =
        [&edges_0, &edges_1];

    let config = DirichletSolverConfig {
        lambda: 1.0,
        step_size: 0.1,
        max_iterations: 30,
        tolerance: 1e-4,
    };

    let iters =
        SheafDirichletSolver::solve_local_cluster(&mut stalks, &target_attractions, &adj, &config)
            .expect("Solver must succeed");

    assert!(iters > 0, "Solver must execute at least 1 iteration");
    assert!(
        (stalks[0].components[0] - 0.5).abs() < 0.25,
        "Stalk 0 must move toward target attraction"
    );
    assert!(
        (stalks[1].components[1] - 0.5).abs() < 0.25,
        "Stalk 1 must move toward target attraction"
    );
}

#[test]
fn test_semiring_algebra_axioms() {
    // Tropical (min, +)
    assert_eq!(TropicalSemiring::zero(), f32::INFINITY);
    assert_eq!(TropicalSemiring::one(), 0.0);
    assert_eq!(TropicalSemiring::oplus(3.0, 5.0), 3.0);
    assert_eq!(TropicalSemiring::otimes(3.0, 5.0), 8.0);

    // Diffusion (+, *)
    assert_eq!(DiffusionSemiring::zero(), 0.0);
    assert_eq!(DiffusionSemiring::one(), 1.0);
    assert!((DiffusionSemiring::oplus(0.3, 0.4) - 0.7).abs() < 1e-6);
    assert!((DiffusionSemiring::otimes(0.3, 0.4) - 0.12).abs() < 1e-6);

    // Bottleneck (max, min)
    assert_eq!(BottleneckSemiring::zero(), 0.0);
    assert_eq!(BottleneckSemiring::one(), f32::INFINITY);
    assert_eq!(BottleneckSemiring::oplus(0.3, 0.8), 0.8);
    assert_eq!(BottleneckSemiring::otimes(0.3, 0.8), 0.3);

    // Dynamic dispatch
    assert_eq!(semiring_eval(SemiringMode::Tropical, 2.0, 4.0, false), 2.0);
    assert_eq!(semiring_eval(SemiringMode::Tropical, 2.0, 4.0, true), 6.0);
}

static STATIC_PAGES: [TensorGraphPage; 4] = [
    // Page 0: Node 1 -> Neighbors [2, 3]
    TensorGraphPage {
        header: aether_core::page::PageHeader {
            crc32c_checksum: 0,
            active_vertex_count: 2,
            lsn: aether_core::hlc::HlcTimestamp::pack(1, 0),
            page_id: 1,
            manifold_curvature_bits: 0,
            csr_offset_table: [0; 15],
            flags: 0,
        },
        at_rq_codebooks: [0; 256],
        neighbor_ids: {
            let mut ids = [0u64; 64];
            ids[0] = 2;
            ids[1] = 3;
            ids
        },
        quantized_neighbors: [0; 2048],
        sheaf_edge_weights: [0; 1216],
    },
    // Page 1: Node 2 -> Neighbor [4]
    TensorGraphPage {
        header: aether_core::page::PageHeader {
            crc32c_checksum: 0,
            active_vertex_count: 1,
            lsn: aether_core::hlc::HlcTimestamp::pack(1, 0),
            page_id: 2,
            manifold_curvature_bits: 0,
            csr_offset_table: [0; 15],
            flags: 0,
        },
        at_rq_codebooks: [0; 256],
        neighbor_ids: {
            let mut ids = [0u64; 64];
            ids[0] = 4;
            ids
        },
        quantized_neighbors: [0; 2048],
        sheaf_edge_weights: [0; 1216],
    },
    // Page 2: Node 3 -> Neighbor [4]
    TensorGraphPage {
        header: aether_core::page::PageHeader {
            crc32c_checksum: 0,
            active_vertex_count: 1,
            lsn: aether_core::hlc::HlcTimestamp::pack(1, 0),
            page_id: 3,
            manifold_curvature_bits: 0,
            csr_offset_table: [0; 15],
            flags: 0,
        },
        at_rq_codebooks: [0; 256],
        neighbor_ids: {
            let mut ids = [0u64; 64];
            ids[0] = 4;
            ids
        },
        quantized_neighbors: [0; 2048],
        sheaf_edge_weights: [0; 1216],
    },
    // Page 3: Node 4 (Target)
    TensorGraphPage {
        header: aether_core::page::PageHeader {
            crc32c_checksum: 0,
            active_vertex_count: 0,
            lsn: aether_core::hlc::HlcTimestamp::pack(1, 0),
            page_id: 4,
            manifold_curvature_bits: 0,
            csr_offset_table: [0; 15],
            flags: 0,
        },
        at_rq_codebooks: [0; 256],
        neighbor_ids: [0; 64],
        quantized_neighbors: [0; 2048],
        sheaf_edge_weights: [0; 1216],
    },
];

fn mock_page_lookup(node_id: u64) -> Option<&'static TensorGraphPage> {
    match node_id {
        1 => Some(&STATIC_PAGES[0]),
        2 => Some(&STATIC_PAGES[1]),
        3 => Some(&STATIC_PAGES[2]),
        4 => Some(&STATIC_PAGES[3]),
        _ => None,
    }
}

fn mock_distance_oracle(node_id: u64) -> f32 {
    match node_id {
        4 => 0.05, // Nearest to query target
        2 => 0.5,
        3 => 0.3,
        1 => 1.0,
        _ => 2.0,
    }
}

#[test]
fn test_semiring_spmv_query_execution() {
    let plan = QueryPlan {
        semiring_mode: SemiringMode::Tropical,
        max_hops: 2,
        top_k: 5,
        beam_width: 64,
        beta: 1.0,
        gamma: 1.0,
        prune_threshold: 1e-4,
    };

    let mut results = [FrontierNode::default(); MAX_QUERY_FRONTIER];
    let mut result_count = 0;

    SemiringSpMvEngine::execute_query(
        &plan,
        &[1], // Start from seed node 1
        mock_page_lookup,
        mock_distance_oracle,
        &mut results,
        &mut result_count,
    )
    .expect("Query execution must succeed");

    assert!(result_count > 0, "Query must return candidate results");
    // Target node 4 should be found at hop 2
    let found_target = results[..result_count].iter().any(|n| n.node_id == 4);
    assert!(found_target, "2-hop traversal must reach target node 4");
}

#[test]
fn gate_zero_hot_path_allocations_query_engine() {
    let guard = HotPathAllocGuard::enter();

    // 1. Householder isometry & application
    let morphism = SheafRestrictionMap::<DEFAULT_STALK_DIM>::from_reflector(&[0.1; 8], 1.0);
    let stalk_u = VertexStalk::<DEFAULT_STALK_DIM>::new([1.0; 8]);
    let stalk_v = VertexStalk::<DEFAULT_STALK_DIM>::new([0.5; 8]);
    let mut edge = aether_query::sheaf::EdgeStalk::<DEFAULT_STALK_DIM>::default();
    morphism.apply(&stalk_u, &mut edge);

    // 2. Sheaf Coboundary
    let energy = SheafCoboundary::edge_dirichlet_energy(&morphism, &stalk_u, &morphism, &stalk_v);
    assert!(energy >= 0.0);

    // 3. SpMV Query Step
    let plan = QueryPlan {
        semiring_mode: SemiringMode::Tropical,
        max_hops: 2,
        top_k: 5,
        beam_width: 64,
        beta: 1.0,
        gamma: 1.0,
        prune_threshold: 1e-4,
    };

    let mut results = [FrontierNode::default(); MAX_QUERY_FRONTIER];
    let mut result_count = 0;

    let res = SemiringSpMvEngine::execute_query(
        &plan,
        &[1],
        mock_page_lookup,
        mock_distance_oracle,
        &mut results,
        &mut result_count,
    );
    assert!(res.is_ok());

    // 4. Assert zero heap allocations on the hot path
    guard
        .assert_zero_allocations()
        .expect("Allocations must be 0");
}
