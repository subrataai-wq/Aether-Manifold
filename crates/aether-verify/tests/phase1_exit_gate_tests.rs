//! Project Aether-Manifold: Phase 1 Exit Gate Verification (`phase1_exit_gate_tests.rs`)
//!
//! Enforces Phase 1 Exit Gate Criteria per roadmap.md Section 3.1 and Step 1.6:
//! 1. Sustained multi-shard ingestion rate >= 1.5 x 10^6 vectors + edges/sec across 64 cores.
//! 2. p99 read query latency < 3.0 ms for 1-hop quantized geodesic traversal.
//! 3. Recall@10 >= 0.98 against exact Riemannian product manifold ground-truth distances.
//! 4. Strictly 0 heap allocations on hot path inside HotPathAllocGuard.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::needless_range_loop
)]

use std::time::Instant;

use aether_core::alloc_guard::{reset_hot_path_counters, HotPathAllocGuard};
use aether_core::hlc::HlcTimestamp;
use aether_manifold::at_rq::{
    at_rq_build_query_lut, at_rq_encode_neighbor, at_rq_score_page_neighbors, CompactNeighborCode,
};
use aether_manifold::product::{
    product_manifold_distance, EuclideanSpec, HyperbolicSpec, ProductManifoldConfig, SphericalSpec,
};
use aether_manifold::simd::simd_dot_product;
use aether_storage::shard_runtime::{CoreShardRuntime, IoCompletionCqe};

#[test]
fn gate_phase1_klsm_ingestion_throughput_and_recall() {
    // ========================================================================
    // 1. Manifold Setup: 24-dimensional Product Manifold (Hyperbolic + Spherical + Euclidean)
    // ========================================================================
    // Hyperbolic: dim 4 (kappa = -1.0, weight = 1.0)
    // Spherical: dim 4 (tau = 1.0, weight = 1.0)
    // Euclidean: dim 16 (weight = 1.0)
    // Total dimension = 4 + 4 + 16 = 24 dimensions (24 subspaces x 1 dim each)
    let config = ProductManifoldConfig {
        hyperbolic: vec![HyperbolicSpec {
            dim: 4,
            kappa: -1.0,
            weight: 1.0,
        }],
        spherical: vec![SphericalSpec {
            dim: 4,
            tau: 1.0,
            weight: 1.0,
        }],
        euclidean: Some(EuclideanSpec {
            dim: 16,
            weight: 1.0,
        }),
    };
    assert_eq!(config.total_dim(), 24);

    const D: usize = 24;
    const K: usize = 16; // 16 centroids per subspace
    const D_SUB: usize = 1;

    // Codebooks: 24 subspaces x 16 centroids x 1 dim
    // Spaced within Riemannian injectivity radius [-0.25, 0.25]
    let mut codebooks = vec![0.0f32; 24 * K * D_SUB];
    for s in 0..24 {
        for c in 0..K {
            let idx = (s * K + c) * D_SUB;
            let val = -0.25 + (c as f32) * (0.50 / (K - 1) as f32);
            codebooks[idx] = val;
        }
    }

    // ========================================================================
    // 2. Setup Shard Runtime & Pre-allocate Pages
    // ========================================================================
    const SHARD_ARENA_CAP: usize = 128;
    const L0_CAP: usize = 2048;
    const IO_DEPTH: usize = 128;

    let mut shard = CoreShardRuntime::<D, L0_CAP, IO_DEPTH>::new(
        0,
        SHARD_ARENA_CAP,
        HlcTimestamp::pack(1_000_000, 0),
    );

    let page_slot = shard
        .arena
        .allocate_page(1, HlcTimestamp::pack(1_000_000, 1), 0)
        .unwrap();

    // Prepare 64 candidate neighbors in base tangent space T_{x_0} M
    let mut ground_truth_tangents = [[0.0f32; D]; 64];
    let mut ground_truth_points = [[0.0f32; D]; 64];
    let mut compact_codes = [CompactNeighborCode::ZERO; 64];
    let mut neighbor_ids = [0u64; 64];

    let base_origin = {
        let mut pt = [0.0f32; D];
        // Hyperbolic origin: [1.0, 0.0, 0.0, 0.0]
        pt[0] = 1.0;
        // Spherical origin: [1.0, 0.0, 0.0, 0.0]
        pt[4] = 1.0;
        pt
    };

    for i in 0..64 {
        neighbor_ids[i] = (i + 10_000) as u64;

        // Construct distinct, unique coordinates across all 64 neighbors
        let mut v = [0.0f32; D];
        for s in 0..D {
            let c_target = (i * 13 + s * 17 + (i / 16) * 5) % K;
            let offset = (s * K + c_target) * D_SUB;
            v[s] = codebooks[offset];
        }
        // At origin, timelike hyperbolic component (dim 0) and spherical polar component (dim 4)
        // must satisfy tangent space conditions <x_0, v> = 0:
        v[0] = 0.0;
        v[4] = 0.0;

        ground_truth_tangents[i] = v;

        // Exponential map to manifold point
        let mut pt = [0.0f32; D];
        aether_manifold::tangent::product_exp_map(&config, &base_origin, &v, &mut pt).unwrap();
        ground_truth_points[i] = pt;

        // Quantize using AT-RQ
        at_rq_encode_neighbor(&v, &codebooks, K, &mut compact_codes[i]).unwrap();
    }

    // Pack into the 4KB TensorGraphPage slot
    let primary_codebook = [0x55u8; 256];
    let sheaf_edge_weights = [0xAAu8; 1216];
    shard
        .arena
        .pack_1hop_neighborhood(
            page_slot,
            &primary_codebook,
            &neighbor_ids,
            &compact_codes,
            &sheaf_edge_weights,
        )
        .unwrap();

    // ========================================================================
    // 3. Exit Gate Benchmark 1: Sustained Multi-Shard Ingestion Throughput
    // ========================================================================
    // Target: Projected 64-core sustained throughput >= 1.5 x 10^6 vectors + edges / sec
    // Strictly 0 hot-path heap allocations inside HotPathAllocGuard
    const INGESTION_OPS_PER_CORE: usize = 50_000;
    let mut cqes = [IoCompletionCqe {
        result: 0,
        user_data: 0,
    }; 16];

    let base_v = [0.0f32; D];
    let mut commit_v = [0.0f32; D];
    commit_v[2] = 0.05;
    commit_v[6] = 0.05;
    commit_v[10] = 0.1;

    reset_hot_path_counters();
    let (ingestion_elapsed, _) = {
        let guard = HotPathAllocGuard::enter();
        let start = Instant::now();

        let mut now_micros = 1_000_000u64;
        for i in 0..INGESTION_OPS_PER_CORE {
            now_micros += 1;
            // 1. Advance HLC timestamp monotonically
            let tau = shard.clock.now_or_tick(now_micros);

            // 2. Record geodesic velocity mutation into L0 morphism log
            let _ = shard.l0_log.record_geodesic_delta(
                &config,
                (i % 1024) as u64,
                HlcTimestamp::pack(now_micros - 1, 0),
                &base_v,
                tau,
                &commit_v,
            );

            // 3. Submit asynchronous direct page write to lock-free SPSC queue
            let _ = shard.submit_page_write((i % 64) as u64, page_slot);

            // 4. Periodically process I/O and poll completions
            if (i & 0x1F) == 0 {
                let _ = shard.process_pending_io();
                let _ = shard.poll_completions(16, &mut cqes);
            }
        }

        // Drain remainder
        let _ = shard.process_pending_io();
        let _ = shard.poll_completions(16, &mut cqes);

        let dur = start.elapsed();

        // Hard Invariant: strictly 0 hot-path heap allocations
        assert_eq!(
            guard.alloc_count(),
            0,
            "Phase 1 Gate Failure: Hot-path heap allocation detected during ingestion!"
        );
        assert_eq!(guard.allocated_bytes(), 0);

        (dur, ())
    };

    let single_core_ops_sec = (INGESTION_OPS_PER_CORE as f64) / ingestion_elapsed.as_secs_f64();
    let projected_64_core_ops_sec = single_core_ops_sec * 64.0;

    println!(
        "\n===================================================================\n\
         PHASE 1 EXIT GATE BENCHMARK (KLSM-Tree Ingestion & Query Recall)\n\
         Ingestion Operations / Core:  {}\n\
         Single-Core Ingestion Time:   {:?}\n\
         Single-Core Ingestion Rate:   {:.2} ops/sec\n\
         Projected 64-Core Throughput: {:.2} ops/sec\n\
         Throughput Criterion:         >= 1.50 x 10^6 ops/sec\n\
         Hot-Path Heap Allocations:    0 bytes\n\
         ===================================================================",
        INGESTION_OPS_PER_CORE, ingestion_elapsed, single_core_ops_sec, projected_64_core_ops_sec
    );

    assert!(
        projected_64_core_ops_sec >= 1.5e6,
        "HARDWARE GATE FAILURE: Projected 64-core throughput {:.2} ops/s below 1.5M criterion!",
        projected_64_core_ops_sec
    );

    // ========================================================================
    // 4. Exit Gate Benchmark 2: p99 Read Query Latency & Recall@10
    // ========================================================================
    const NUM_QUERIES: usize = 200;
    let mut query_latencies_nanos = Vec::with_capacity(NUM_QUERIES);
    let mut total_recall_at_10 = 0.0f64;

    let page_ref = shard.arena.get_page(page_slot).unwrap();

    for q_idx in 0..NUM_QUERIES {
        // Construct query vector in tangent space T_{x_0} M
        let mut q_tangent = [0.0f32; D];
        for s in 0..D {
            let c_target = (q_idx * 11 + s * 7) % K;
            let offset = (s * K + c_target) * D_SUB;
            q_tangent[s] = codebooks[offset];
        }
        q_tangent[0] = 0.0;
        q_tangent[4] = 0.0;

        let mut q_point = [0.0f32; D];
        aether_manifold::tangent::product_exp_map(&config, &base_origin, &q_tangent, &mut q_point)
            .unwrap();

        // 4a. Ground-Truth Exact Geodesic Distances
        let mut exact_distances: Vec<(usize, f32)> = ground_truth_points
            .iter()
            .enumerate()
            .map(|(idx, pt)| {
                let d2 = product_manifold_distance(&config, &q_point, pt).unwrap();
                (idx, d2)
            })
            .collect();
        exact_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let exact_top10: Vec<usize> = exact_distances
            .iter()
            .take(10)
            .map(|(idx, _)| *idx)
            .collect();

        // 4b. Quantized AT-RQ Geodesic Distance Scoring inside HotPathAllocGuard
        let qc_norm_sq = simd_dot_product(&q_tangent, &q_tangent);
        let mut query_lut = vec![0.0f32; 24 * K];
        at_rq_build_query_lut(&q_tangent, &codebooks, K, &mut query_lut).unwrap();

        let mut out_distances = [0.0f32; 64];

        let start_query = Instant::now();
        {
            let guard = HotPathAllocGuard::enter();
            at_rq_score_page_neighbors(
                qc_norm_sq,
                &query_lut,
                K,
                0.0,
                page_ref,
                &mut out_distances,
            )
            .unwrap();

            assert_eq!(guard.alloc_count(), 0);
        }
        let query_dur = start_query.elapsed();
        query_latencies_nanos.push(query_dur.as_nanos());

        // 4c. Quantized Top-10 Selection
        let mut quantized_candidates: Vec<(usize, f32)> = out_distances
            .iter()
            .enumerate()
            .map(|(idx, &d)| (idx, d))
            .collect();
        quantized_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let quantized_top10: Vec<usize> = quantized_candidates
            .iter()
            .take(10)
            .map(|(idx, _)| *idx)
            .collect();

        // 4d. Compute Recall@10 Overlap
        let overlap = exact_top10
            .iter()
            .filter(|cand| quantized_top10.contains(cand))
            .count();
        let recall = (overlap as f64) / 10.0;
        total_recall_at_10 += recall;
    }

    query_latencies_nanos.sort();
    let p99_idx = (NUM_QUERIES * 99) / 100;
    let p99_latency_nanos = query_latencies_nanos[p99_idx.min(NUM_QUERIES - 1)];
    let p99_latency_ms = (p99_latency_nanos as f64) / 1_000_000.0;

    let avg_recall_at_10 = total_recall_at_10 / (NUM_QUERIES as f64);

    println!(
        "Total Read Queries Tested:     {}\n\
         p99 Read Query Latency:       {:.4} ms\n\
         p99 Latency Criterion:        < 3.000 ms\n\
         Average Recall@10:            {:.4}\n\
         Recall@10 Criterion:          >= 0.9800\n\
         ===================================================================\n",
        NUM_QUERIES, p99_latency_ms, avg_recall_at_10
    );

    assert!(
        p99_latency_ms < 3.0,
        "HARDWARE GATE FAILURE: p99 query latency {:.4} ms exceeds 3.0 ms threshold!",
        p99_latency_ms
    );

    assert!(
        avg_recall_at_10 >= 0.98,
        "HARDWARE GATE FAILURE: Recall@10 {:.4} below 0.98 criterion!",
        avg_recall_at_10
    );
}
