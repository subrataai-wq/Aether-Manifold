//! Project Aether-Manifold: Quickstart Notebook Rust Snippet Verification Suite
//!
//! Validates that every embedded Rust API code snippet shown in `notebooks/aether_manifold_quickstart.md`
//! (Cells 1D, 2D, 3D, 4C, 5C, 6A, 7D, 8D, 9B) compiles and runs correctly.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::needless_range_loop
)]

use aether_consensus::consensus::{LeaderlessQuorumEngine, QuorumConfig};
use aether_consensus::morphism_replicator::MorphismReplicator;
use aether_consensus::partitioner::{GraphEdge, PartitionerConfig, SpectralGeodesicPartitioner};
use aether_consensus::rdma_fabric::{RdmaFabric, RdmaMemoryWindow};
use aether_core::alloc_guard::{reset_hot_path_counters, HotPathAllocGuard, TrackingAllocator};
use aether_core::hlc::HlcTimestamp;
use aether_core::page::{PageHeader, TensorGraphPage};
use aether_manifold::at_rq::{AnisotropicTangentQuantizer, CompactNeighborCode};
use aether_manifold::optimal_transport::{
    debiased_sinkhorn_divergence_with_workspace, SinkhornConfig,
};
use aether_manifold::product::{
    EuclideanSpec, HyperbolicSpec, ProductManifoldConfig, SphericalSpec,
};
use aether_query::compiler::{FrontierNode, QueryPlan, SemiringSpMvEngine, MAX_QUERY_FRONTIER};
use aether_query::semiring::SemiringMode;
use aether_query::sheaf::{pack_page_sheaf_morphisms, SheafRestrictionMap};
use aether_storage::l0_mvcc::L0MorphismLog;
use aether_storage::page_arena::AlignedPageArena;
use aether_storage::voronoi_compaction::{
    evaluate_voronoi_boundary_certificate, CompactionDecision, VoronoiCellCertificate,
};

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator::new(std::alloc::System);

// -----------------------------------------------------------------------------
// Cell 1D
// -----------------------------------------------------------------------------
pub fn build_research_manifold() -> (ProductManifoldConfig, AnisotropicTangentQuantizer) {
    const _: () = assert!(core::mem::size_of::<PageHeader>() == 64);
    const _: () = assert!(core::mem::size_of::<TensorGraphPage>() == 4096);
    const _: () = assert!(core::mem::size_of::<CompactNeighborCode>() == 32);

    let config = ProductManifoldConfig {
        hyperbolic: vec![HyperbolicSpec {
            dim: 65,
            kappa: -1.0_f32,
            weight: 1.5_f32,
        }],
        spherical: vec![SphericalSpec {
            dim: 768,
            tau: 1.0_f32,
            weight: 1.0_f32,
        }],
        euclidean: Some(EuclideanSpec {
            dim: 64,
            weight: 0.5_f32,
        }),
    };

    let codebooks = vec![0.0f32; 24 * 16 * 2];
    let quantizer = AnisotropicTangentQuantizer::new(48, 16, codebooks)
        .expect("Valid AT-RQ quantizer configuration");

    (config, quantizer)
}

// -----------------------------------------------------------------------------
// Cell 2D
// -----------------------------------------------------------------------------
pub fn ingest_zero_alloc_hot_path(
    arena: &mut AlignedPageArena,
    clock_state: &core::sync::atomic::AtomicU64,
    physical_now_micros: u64,
) {
    let _guard = HotPathAllocGuard::enter();

    let ts = HlcTimestamp::now_or_tick(clock_state, physical_now_micros);

    let slot = arena
        .allocate_page(1001, ts, 0)
        .expect("Pre-allocated 4KB slot available");
    let page = arena.get_page_mut(slot).expect("Valid slot in arena");

    let maps = [SheafRestrictionMap::identity(); 4];
    let weights = [1.0f32; 4];
    pack_page_sheaf_morphisms(page, &maps, &weights)
        .expect("Morphisms packed into 4KB page without allocation");
}

// -----------------------------------------------------------------------------
// Cell 3D
// -----------------------------------------------------------------------------
pub fn execute_sheaf_query_zero_alloc(
    page_lookup: impl Fn(u64) -> Option<&'static TensorGraphPage>,
    distance_oracle: impl Fn(u64) -> f32,
    start_vertex: u64,
) -> ([FrontierNode; MAX_QUERY_FRONTIER], usize) {
    let plan = QueryPlan {
        semiring_mode: SemiringMode::Tropical,
        max_hops: 3,
        top_k: 10,
        beam_width: 64,
        beta: 1.0_f32,
        gamma: 1.5_f32,
        prune_threshold: 1e-4_f32,
    };

    let mut results = [FrontierNode::default(); MAX_QUERY_FRONTIER];
    let mut result_count = 0;

    let _guard = HotPathAllocGuard::enter();

    SemiringSpMvEngine::execute_query(
        &plan,
        &[start_vertex],
        page_lookup,
        distance_oracle,
        &mut results,
        &mut result_count,
    )
    .expect("Query executed in p99 < 5ms");

    (results, result_count)
}

// -----------------------------------------------------------------------------
// Cell 4C
// -----------------------------------------------------------------------------
pub fn compute_subgraph_ot_zero_alloc(
    a_bag: &[f32],
    n: usize,
    b_bag: &[f32],
    m: usize,
    dim: usize,
    workspace: &mut [f32],
) -> f32 {
    let config = SinkhornConfig {
        epsilon: 0.05_f32,
        max_iter: 100,
        tolerance: 1e-4_f32,
    };

    let _guard = HotPathAllocGuard::enter();

    debiased_sinkhorn_divergence_with_workspace(a_bag, n, b_bag, m, dim, &config, workspace)
        .expect("Converged within tolerance with zero heap allocations")
}

// -----------------------------------------------------------------------------
// Cell 5C
// -----------------------------------------------------------------------------
pub fn reconstruct_time_travel_state_zero_alloc<const D: usize, const CAP: usize>(
    config: &ProductManifoldConfig,
    l0_log: &L0MorphismLog<D, CAP>,
    vertex_id: u64,
    base_state_tau1: &[f32; D],
    query_hlc: HlcTimestamp,
) -> [f32; D] {
    let mut reconstructed = [0.0_f32; D];
    let _guard = HotPathAllocGuard::enter();

    let entry = l0_log
        .get_latest_entry(vertex_id)
        .expect("Causal HLC timestamp resolved in L0MorphismLog");

    l0_log
        .reconstruct_state_at(
            config,
            base_state_tau1,
            entry,
            query_hlc,
            &mut reconstructed,
        )
        .expect("Causal HLC timestamp resolved in L0MorphismLog");

    reconstructed
}

// -----------------------------------------------------------------------------
// Cell 6A
// -----------------------------------------------------------------------------
pub fn check_o1_pointer_splice_eligibility<const D: usize>(
    config: &ProductManifoldConfig,
    u_coords: &[f32; D],
    vertex_kth_neighbor_geodesic_dist: f32,
    modified_cell_max_radius: f32,
    centroid: [f32; D],
) -> bool {
    let cert = VoronoiCellCertificate {
        cell_id: 42,
        centroid,
        max_geodesic_radius: modified_cell_max_radius,
        modified: true,
    };

    let decision = evaluate_voronoi_boundary_certificate(
        config,
        u_coords,
        vertex_kth_neighbor_geodesic_dist,
        &cert,
    );

    decision == Ok(CompactionDecision::BypassInteriorSplice)
}

// -----------------------------------------------------------------------------
// Cell 7D
// -----------------------------------------------------------------------------
pub fn partition_cluster_topology(
    num_vertices: usize,
    edges: &[GraphEdge],
) -> aether_consensus::partitioner::PartitionPlan {
    let config = PartitionerConfig {
        num_shards: 2,
        geodesic_weight_mu: 1.0_f32,
        balance_tolerance: 0.05_f32,
        max_power_iterations: 150,
    };

    let partitioner =
        SpectralGeodesicPartitioner::new(config).expect("Valid partitioner configuration");

    let plan = partitioner
        .partition(num_vertices, edges)
        .expect("Topology successfully partitioned");

    assert!(plan.is_balanced(0.05));
    plan
}

// -----------------------------------------------------------------------------
// Cell 8D
// -----------------------------------------------------------------------------
pub fn execute_consensus_lifecycle_zero_alloc() {
    let config0 = QuorumConfig::new(0, 3, 2, 2).expect("Valid N=3, W=2, R=2");
    let config1 = QuorumConfig::new(1, 3, 2, 2).expect("Valid N=3, W=2, R=2");

    let mut node0 = LeaderlessQuorumEngine::new(config0, 1_000_000);
    let mut node1 = LeaderlessQuorumEngine::new(config1, 1_000_000);
    let mut replicator0 = MorphismReplicator::new(0);

    let _guard = HotPathAllocGuard::enter();

    let proposal = node0.propose(5001, [0xEE; 24], 1_000_010).unwrap();
    let ack = node1.handle_remote_propose(&proposal, 1_000_012).unwrap();
    let commit = node0
        .handle_ack(&ack)
        .unwrap()
        .expect("Write quorum reached");
    replicator0.apply_commit(&commit).unwrap();
}

// -----------------------------------------------------------------------------
// Cell 9B
// -----------------------------------------------------------------------------
pub fn fetch_remote_page_zero_copy(
    fabric: &RdmaFabric,
    target_node_id: u64,
    page_index: usize,
) -> [u8; 4096] {
    let mut page_buf = [0u8; 4096];
    let _guard = HotPathAllocGuard::enter();

    fabric
        .read_remote_page(target_node_id, page_index, &mut page_buf)
        .expect("Remote page fetched via RDMA fabric without allocation");

    page_buf
}

#[test]
fn test_all_quickstart_notebook_snippets_compile_and_run() {
    reset_hot_path_counters();

    // 1. Cell 1D
    let (config, quantizer) = build_research_manifold();
    assert_eq!(config.hyperbolic.len(), 1);
    assert_eq!(quantizer.dimension(), 48);

    // 2. Cell 2D
    let mut arena = AlignedPageArena::with_capacity(8);
    let clock = core::sync::atomic::AtomicU64::new(100);
    ingest_zero_alloc_hot_path(&mut arena, &clock, 200);
    assert_eq!(arena.allocated_count(), 1);

    // 3. Cell 3D
    let dummy_page: TensorGraphPage = bytemuck::Zeroable::zeroed();
    let page_ref: &'static TensorGraphPage = Box::leak(Box::new(dummy_page));
    let page_lookup = move |_nid| Some(page_ref);
    let dist_oracle = |_nid| 0.5f32;
    let (_results, count) = execute_sheaf_query_zero_alloc(page_lookup, dist_oracle, 1);
    assert!(count > 0);

    // 4. Cell 4C
    let a_bag = [1.0f32, 2.0];
    let b_bag = [1.1f32, 2.1];
    let mut ws = vec![0.0f32; 1024];
    let ot_cost = compute_subgraph_ot_zero_alloc(&a_bag, 1, &b_bag, 1, 2, &mut ws);
    assert!(ot_cost >= 0.0);

    // 5. Cell 5C
    let mut l0 = L0MorphismLog::<2, 16>::new();
    let m_cfg = ProductManifoldConfig {
        hyperbolic: vec![],
        spherical: vec![],
        euclidean: Some(EuclideanSpec {
            dim: 2,
            weight: 1.0,
        }),
    };
    l0.record_geodesic_delta(
        &m_cfg,
        1001,
        HlcTimestamp(100),
        &[0.0, 0.0],
        HlcTimestamp(200),
        &[0.5, 0.5],
    )
    .unwrap();
    let rec =
        reconstruct_time_travel_state_zero_alloc(&m_cfg, &l0, 1001, &[0.0, 0.0], HlcTimestamp(150));
    assert!(rec[0] > 0.0);

    // 6. Cell 6A
    let is_splice =
        check_o1_pointer_splice_eligibility(&m_cfg, &[0.0, 0.0], 0.2, 0.5, [10.0, 10.0]);
    assert!(is_splice);

    // 7. Cell 7D
    let edges = vec![
        GraphEdge::new(0, 1, 0.1, 0.2),
        GraphEdge::new(1, 2, 0.2, 0.3),
        GraphEdge::new(2, 3, 0.1, 0.1),
    ];
    let plan = partition_cluster_topology(4, &edges);
    assert_eq!(plan.num_shards(), 2);

    // 8. Cell 8D
    execute_consensus_lifecycle_zero_alloc();

    // 9. Cell 9B
    let mut fabric = RdmaFabric::new();
    let window = RdmaMemoryWindow::with_capacity(4);
    fabric.register_node(1, window).unwrap();
    let pg = fetch_remote_page_zero_copy(&fabric, 1, 0);
    assert_eq!(pg.len(), 4096);
}
