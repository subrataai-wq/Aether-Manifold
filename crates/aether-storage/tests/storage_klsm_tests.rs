//! Unit and Integration Tests for Single-Node Storage Engine & KLSM-Tree (`storage_klsm_tests.rs`)
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::needless_range_loop
)]

use std::sync::Arc;
use std::thread;

use aether_core::{reset_hot_path_counters, HotPathAllocGuard};
use aether_core::hlc::HlcTimestamp;
use aether_manifold::at_rq::{at_rq_build_query_lut, at_rq_score_page_neighbors, CompactNeighborCode};
use aether_manifold::product::{ProductManifoldConfig, EuclideanSpec};
use aether_storage::l0_mvcc::L0MorphismLog;
use aether_storage::page_arena::AlignedPageArena;
use aether_storage::shard_runtime::{CoreShardRuntime, IoCompletionCqe};
use aether_storage::spsc::SpscRingBuffer;
use aether_storage::voronoi_compaction::{
    evaluate_voronoi_boundary_certificate, execute_boundary_only_compaction, CompactionDecision,
    VoronoiCellCertificate,
};

#[test]
fn test_spsc_ring_buffer_wait_free_concurrency() {
    const QUEUE_CAP: usize = 128;
    const NUM_ITEMS: usize = 10_000;

    let ring = Arc::new(SpscRingBuffer::<u64, QUEUE_CAP>::new());
    let ring_producer = Arc::clone(&ring);
    let ring_consumer = Arc::clone(&ring);

    assert!(ring.is_empty());
    assert_eq!(ring.len(), 0);
    assert_eq!(ring.capacity(), QUEUE_CAP);
    assert_eq!(ring.try_pop(), None);

    let producer = thread::spawn(move || {
        for val in 1..=NUM_ITEMS as u64 {
            while let Err(e) = ring_producer.try_push(val) {
                match e {
                    aether_core::error::AetherError::ScratchpadExhausted { .. } => {
                        std::hint::spin_loop();
                    }
                    other => panic!("Unexpected error on push: {:?}", other),
                }
            }
        }
    });

    let consumer = thread::spawn(move || {
        let mut expected = 1u64;
        let mut received = Vec::with_capacity(NUM_ITEMS);

        while expected <= NUM_ITEMS as u64 {
            if let Some(val) = ring_consumer.try_pop() {
                assert_eq!(val, expected, "FIFO violation in SPSC ring buffer!");
                received.push(val);
                expected += 1;
            } else {
                std::hint::spin_loop();
            }
        }
        received.len()
    });

    producer.join().unwrap();
    let total_popped = consumer.join().unwrap();
    assert_eq!(total_popped, NUM_ITEMS);
    assert!(ring.is_empty());
}

#[test]
fn test_4kb_tensor_page_arena_and_zero_io_1hop_adc() {
    let mut arena = AlignedPageArena::with_capacity(8);
    assert_eq!(arena.capacity(), 8);
    assert_eq!(arena.allocated_count(), 0);
    assert_eq!(arena.free_count(), 8);

    let lsn = HlcTimestamp::pack(1_000_000, 1);
    let slot = arena.allocate_page(101, lsn, 0).unwrap();
    assert_eq!(slot, 0);
    assert_eq!(arena.allocated_count(), 1);

    // Prepare 64 neighbor codes
    let centroids_per_subspace = 16usize;
    let mut neighbor_ids = [0u64; 64];
    let mut neighbor_codes = [CompactNeighborCode::ZERO; 64];

    for i in 0..64 {
        neighbor_ids[i] = (i + 1000) as u64;
        let mut sc = [0u8; 24];
        for s in 0..24 {
            sc[s] = ((i + s * 3) % centroids_per_subspace) as u8;
        }
        neighbor_codes[i] = CompactNeighborCode {
            radial_norm_sq: (i + 1) as f32 * 0.05,
            scale: 1.0,
            subspace_codes: sc,
        };
    }

    let primary_codebook = [0xAAu8; 256];
    let sheaf_weights = [0x55u8; 512];

    arena
        .pack_1hop_neighborhood(
            slot,
            &primary_codebook,
            &neighbor_ids,
            &neighbor_codes,
            &sheaf_weights,
        )
        .unwrap();

    let page = arena.get_page(slot).unwrap();
    assert_eq!(page.header.page_id, 101);
    assert_eq!(page.header.active_vertex_count, 64);
    assert!(page.header.verify_checksum().is_ok());

    // Score all 64 neighbors directly against precomputed query LUT in 0 I/O
    let mut qc_1032 = [0.0f32; 1032];
    for (i, val) in qc_1032.iter_mut().enumerate() {
        *val = ((i % 17) as f32 - 8.0) * 0.01;
    }
    let qc_norm_sq = aether_manifold::simd::simd_dot_product(&qc_1032, &qc_1032);

    let total_codebook_floats = 24 * centroids_per_subspace * 43;
    let mut codebooks = vec![0.0f32; total_codebook_floats];
    for (idx, val) in codebooks.iter_mut().enumerate() {
        *val = ((idx % 31) as f32 - 15.0) * 0.005;
    }

    let mut query_lut = vec![0.0f32; 24 * centroids_per_subspace];
    at_rq_build_query_lut(&qc_1032, &codebooks, centroids_per_subspace, &mut query_lut).unwrap();

    let mut out_distances = [0.0f32; 64];
    at_rq_score_page_neighbors(
        qc_norm_sq,
        &query_lut,
        centroids_per_subspace,
        -1.0,
        page,
        &mut out_distances,
    )
    .unwrap();

    for (i, &d2) in out_distances.iter().enumerate() {
        assert!(d2.is_finite() && d2 >= 0.0, "Neighbor {} invalid distance: {}", i, d2);
    }

    arena.deallocate_page(slot).unwrap();
    assert_eq!(arena.allocated_count(), 0);
    assert_eq!(arena.free_count(), 8);
}

#[test]
fn test_l0_geodesic_mvcc_time_travel_snapshots() {
    let config = ProductManifoldConfig {
        hyperbolic: vec![],
        spherical: vec![],
        euclidean: Some(EuclideanSpec { dim: 4, weight: 1.0 }),
    };

    let mut log = L0MorphismLog::<4, 16>::new();
    assert!(log.is_empty());

    let vertex_id = 42u64;
    let tau_base = HlcTimestamp::pack(1_000_000, 0);
    let tau_commit = HlcTimestamp::pack(2_000_000, 0);

    let base_state = [0.0f32, 0.0, 0.0, 0.0];
    let commit_state = [10.0f32, 20.0, -10.0, 30.0];

    log.record_geodesic_delta(
        &config,
        vertex_id,
        tau_base,
        &base_state,
        tau_commit,
        &commit_state,
    )
    .unwrap();

    assert_eq!(log.len(), 1);
    let entry = log.get_latest_entry(vertex_id).expect("Entry must be indexed");
    assert_eq!(entry.vertex_id, vertex_id);
    assert_eq!(entry.velocity_delta, [10.0, 20.0, -10.0, 30.0]);

    // Snapshot at tau_base: alpha = 0.0 -> base_state
    let mut reconstructed = [0.0f32; 4];
    log.reconstruct_state_at(&config, &base_state, entry, tau_base, &mut reconstructed)
        .unwrap();
    assert_eq!(reconstructed, [0.0, 0.0, 0.0, 0.0]);

    // Snapshot at midpoint (1.5s): alpha = 0.5 -> exactly halfway
    let tau_mid = HlcTimestamp::pack(1_500_000, 0);
    log.reconstruct_state_at(&config, &base_state, entry, tau_mid, &mut reconstructed)
        .unwrap();
    for i in 0..4 {
        assert!((reconstructed[i] - 0.5 * commit_state[i]).abs() < 1e-5);
    }

    // Snapshot at tau_commit: alpha = 1.0 -> commit_state
    log.reconstruct_state_at(&config, &base_state, entry, tau_commit, &mut reconstructed)
        .unwrap();
    assert_eq!(reconstructed, commit_state);
}

#[test]
fn test_boundary_only_voronoi_compaction_equivalence() {
    let config = ProductManifoldConfig {
        hyperbolic: vec![],
        spherical: vec![],
        euclidean: Some(EuclideanSpec { dim: 2, weight: 1.0 }),
    };

    // Modified Voronoi cell A at [100.0, 100.0] with radius R_A = 2.0
    let cert_a = VoronoiCellCertificate {
        cell_id: 1,
        centroid: [100.0f32, 100.0],
        max_geodesic_radius: 2.0,
        modified: true,
    };

    // Unmodified Voronoi cell B at [-100.0, -100.0]
    let cert_b = VoronoiCellCertificate {
        cell_id: 2,
        centroid: [-100.0f32, -100.0],
        max_geodesic_radius: 2.0,
        modified: false,
    };

    // Vertex 1: Far away in cell B at [0.0, 0.0], with k-NN neighbor distance = 5.0
    // d(u1, mu_A) = sqrt(100^2 + 100^2) ≈ 141.42
    // 141.42 - 2.0 = 139.42 > 5.0 => BypassInteriorSplice!
    let u1 = [0.0f32, 0.0];
    let d1_knn = 5.0f32;
    let dec1 = evaluate_voronoi_boundary_certificate(&config, &u1, d1_knn, &cert_a).unwrap();
    assert_eq!(dec1, CompactionDecision::BypassInteriorSplice);

    // Unmodified cell B always returns BypassInteriorSplice
    let dec_b = evaluate_voronoi_boundary_certificate(&config, &u1, d1_knn, &cert_b).unwrap();
    assert_eq!(dec_b, CompactionDecision::BypassInteriorSplice);

    // Vertex 2: Near boundary of cell A at [96.0, 100.0] with k-NN neighbor distance = 6.0
    // d(u2, mu_A) = 4.0
    // 4.0 - 2.0 = 2.0 <= 6.0 => BoundaryRelink!
    let u2 = [96.0f32, 100.0];
    let d2_knn = 6.0f32;
    let dec2 = evaluate_voronoi_boundary_certificate(&config, &u2, d2_knn, &cert_a).unwrap();
    assert_eq!(dec2, CompactionDecision::BoundaryRelink);

    // Batch compaction over vertices
    let vertices = vec![u1, u2];
    let knn_dists = vec![d1_knn, d2_knn];
    let mut slots = vec![0usize, 1usize];
    let certs = vec![cert_a, cert_b];

    let stats = execute_boundary_only_compaction(&config, &vertices, &knn_dists, &mut slots, &certs).unwrap();
    assert_eq!(stats.total_vertices, 2);
    assert_eq!(stats.interior_spliced, 1);
    assert_eq!(stats.boundary_relinked, 1);
}

#[test]
fn gate_zero_hot_path_allocations_storage_engine() {
    let config = ProductManifoldConfig {
        hyperbolic: vec![],
        spherical: vec![],
        euclidean: Some(EuclideanSpec { dim: 4, weight: 1.0 }),
    };

    let mut shard = CoreShardRuntime::<4, 64, 64>::new(0, 32, HlcTimestamp::pack(1_000_000, 0));

    // Allocate page slot on cold path
    let slot = shard.arena.allocate_page(1, HlcTimestamp::pack(1_000_000, 1), 0).unwrap();

    let cert = VoronoiCellCertificate {
        cell_id: 1,
        centroid: [100.0f32, 100.0, 0.0, 0.0],
        max_geodesic_radius: 5.0,
        modified: true,
    };
    let certs = [cert];
    let vertices = [[0.0f32, 0.0, 0.0, 0.0]];
    let knn_dists = [5.0f32];
    let mut page_slots = [slot];

    let mut cqes = [IoCompletionCqe { result: 0, user_data: 0 }; 8];

    let base_state = [1.0f32, 2.0, 3.0, 4.0];
    let commit_state = [1.1f32, 2.2, 3.3, 4.4];
    let tau1 = HlcTimestamp::pack(1_000_000, 2);
    let tau2 = HlcTimestamp::pack(2_000_000, 0);

    reset_hot_path_counters();
    let guard = HotPathAllocGuard::enter();

    // 1. SPSC push and pop (0 allocs)
    shard.sq.try_push(aether_storage::IoSubmissionSqe {
        opcode: aether_storage::IoOpcode::ReadPage,
        page_id: 1,
        arena_slot: slot,
        user_data: 1,
    }).unwrap();
    let _ = shard.process_pending_io();
    let dequeued = shard.poll_completions(8, &mut cqes);
    assert_eq!(dequeued, 1);

    // 2. Arena page lookup and header verify (0 allocs)
    let page = shard.arena.get_page(slot).unwrap();
    assert!(page.header.verify_checksum().is_ok());

    // 3. L0 geodesic MVCC delta record and state reconstruction (0 allocs)
    let delta_slot = shard.l0_log.record_geodesic_delta(
        &config,
        42,
        tau1,
        &base_state,
        tau2,
        &commit_state,
    ).unwrap();
    let entry = shard.l0_log.get_latest_entry(42).unwrap();
    assert_eq!(delta_slot, 0);
    let mut reconstructed = [0.0f32; 4];
    shard.l0_log.reconstruct_state_at(
        &config,
        &base_state,
        entry,
        HlcTimestamp::pack(1_500_000, 0),
        &mut reconstructed,
    ).unwrap();

    // 4. Voronoi compaction evaluation (0 allocs)
    let decision = evaluate_voronoi_boundary_certificate(
        &config,
        &[0.0, 0.0, 0.0, 0.0],
        5.0,
        match certs.first() { Some(c) => c, None => panic!() },
    ).unwrap();
    assert_eq!(decision, CompactionDecision::BypassInteriorSplice);

    let stats = execute_boundary_only_compaction(
        &config,
        &vertices,
        &knn_dists,
        &mut page_slots,
        &certs,
    ).unwrap();
    assert_eq!(stats.interior_spliced, 1);

    assert_eq!(guard.alloc_count(), 0, "Hot-path allocation detected in storage engine!");
    assert_eq!(guard.allocated_bytes(), 0);
}
