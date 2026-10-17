//! Project Aether-Manifold: Architectural Gap Remediation Verification Suite
//!
//! Validates:
//! - GAP-02: Double-free rejection and zero-realloc hazard in `AlignedPageArena`.
//! - GAP-03: Stale hash eviction on ring buffer wraparound in `L0MorphismLog`.
//! - GAP-07: AT-RQ Distortion Certificate and FP16 residual spillover in `TensorGraphPage`.
//!
//! Strict Compliance:
//! - AGENTS.md Rule 3.1: Strictly 0 heap allocations on hot path inside `HotPathAllocGuard`.
//! - AGENTS.md Rule 3.2: Strictly 0 OS locks (`Mutex`/`RwLock`).
//! - AGENTS.md Rule 3.3: 4096-byte and 64-byte compile-time memory layout invariants.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::needless_range_loop
)]

use aether_autotune::dma_compaction::{
    DmaCompactionEngine, DmaPageDescriptor, HostLoopbackDmaBackend,
};
use aether_consensus::consensus::{
    LeaderlessQuorumEngine, QuorumConfig, ReplicationMessage, ReplicationMessageType,
    MAX_CLUSTER_NODES,
};
use aether_consensus::morphism_replicator::MorphismReplicator;
use aether_consensus::rdma_fabric::{RdmaFabric, RdmaMemoryWindow, RdmaTransportBackend};
use aether_core::alloc_guard::{reset_hot_path_counters, HotPathAllocGuard, TrackingAllocator};
use aether_core::error::AetherError;
use aether_core::hlc::HlcTimestamp;
use aether_core::page::{TensorGraphPage, PAGE_FLAG_HAS_FP16_SPILLOVER};
use aether_manifold::at_rq::AnisotropicTangentQuantizer;
use aether_manifold::product::{HyperbolicSpec, ProductManifoldConfig};
use aether_query::aql::{parse_aql, AqlCompileHelper, AqlStatement, SemiringKind};
use aether_query::compiler::{FrontierNode, MAX_QUERY_FRONTIER};
use aether_query::sheaf::{pack_page_sheaf_morphisms, SheafRestrictionMap};
use aether_query::speculative_prefetch::SpeculativeLookaheadPipeline;
use aether_storage::klsm_coordinator::{KlsmRunManifest, KlsmTierCoordinator};
use aether_storage::l0_mvcc::L0MorphismLog;
use aether_storage::page_arena::AlignedPageArena;
use aether_storage::shard_runtime::{pin_current_thread_to_core, CoreShardRuntime};
use bytemuck::Zeroable;

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator::new(std::alloc::System);

#[test]
fn test_gap02_page_arena_double_free_rejected_zero_alloc() {
    reset_hot_path_counters();
    let mut arena = AlignedPageArena::with_capacity(16);
    assert_eq!(arena.capacity(), 16);
    assert_eq!(arena.allocated_count(), 0);
    assert_eq!(arena.free_count(), 16);

    let s0 = arena.allocate_page(101, HlcTimestamp(10), 0).unwrap();
    let s1 = arena.allocate_page(102, HlcTimestamp(20), 0).unwrap();
    let s2 = arena.allocate_page(103, HlcTimestamp(30), 0).unwrap();

    assert!(arena.is_slot_allocated(s0));
    assert!(arena.is_slot_allocated(s1));
    assert!(arena.is_slot_allocated(s2));
    assert_eq!(arena.allocated_count(), 3);
    assert_eq!(arena.free_count(), 13);

    // Verify zero heap allocations inside HotPathAllocGuard during dealloc and double-free rejection
    {
        let _guard = HotPathAllocGuard::enter();

        // 1. Valid deallocation of s1
        let dealloc_res = arena.deallocate_page(s1);
        assert!(dealloc_res.is_ok());
        assert!(!arena.is_slot_allocated(s1));
        assert_eq!(arena.allocated_count(), 2);

        // 2. Double-free attempt on s1 must be rejected without reallocating
        let double_free_res = arena.deallocate_page(s1);
        assert_eq!(
            double_free_res,
            Err(AetherError::InvalidPageSlot { slot: s1 })
        );
        assert_eq!(arena.allocated_count(), 2);

        // 3. Out-of-bounds deallocation attempt must be rejected
        let oob_res = arena.deallocate_page(999);
        assert_eq!(oob_res, Err(AetherError::InvalidPageSlot { slot: 999 }));

        // 4. Re-allocation reuses s1 slot cleanly
        let s1_new = arena.allocate_page(104, HlcTimestamp(40), 0).unwrap();
        assert_eq!(s1_new, s1);
        assert!(arena.is_slot_allocated(s1_new));
        assert_eq!(arena.allocated_count(), 3);
    }
}

#[test]
fn test_gap03_l0_mvcc_ring_wraparound_evicts_stale_hash() {
    reset_hot_path_counters();
    let config = ProductManifoldConfig {
        hyperbolic: vec![HyperbolicSpec {
            dim: 16,
            kappa: -1.0,
            weight: 1.0,
        }],
        spherical: vec![],
        euclidean: None,
    };
    let mut log = L0MorphismLog::<16, 4>::new();
    assert_eq!(log.capacity(), 4);
    assert_eq!(log.len(), 0);

    // Construct valid base and commit states on Lorentz hyperboloid
    let mut base_state = [0.0f32; 16];
    base_state[0] = 1.0; // Origin in Lorentz space (x0 = 1, x_space = 0)

    let mut commit_state = base_state;
    commit_state[1] = 0.05;
    commit_state[0] = (1.0 + 0.05 * 0.05f32).sqrt();

    // Fill ring buffer with 4 entries (vertices 10, 20, 30, 40)
    log.record_geodesic_delta(
        &config,
        10,
        HlcTimestamp(1),
        &base_state,
        HlcTimestamp(2),
        &commit_state,
    )
    .unwrap();

    log.record_geodesic_delta(
        &config,
        20,
        HlcTimestamp(3),
        &base_state,
        HlcTimestamp(4),
        &commit_state,
    )
    .unwrap();

    log.record_geodesic_delta(
        &config,
        30,
        HlcTimestamp(5),
        &base_state,
        HlcTimestamp(6),
        &commit_state,
    )
    .unwrap();

    log.record_geodesic_delta(
        &config,
        40,
        HlcTimestamp(7),
        &base_state,
        HlcTimestamp(8),
        &commit_state,
    )
    .unwrap();

    assert_eq!(log.len(), 4);
    assert_eq!(log.get_latest_entry(10).unwrap().vertex_id, 10);
    assert_eq!(log.get_latest_entry(20).unwrap().vertex_id, 20);
    assert_eq!(log.get_latest_entry(30).unwrap().vertex_id, 30);
    assert_eq!(log.get_latest_entry(40).unwrap().vertex_id, 40);

    // Overwrite slot 0 with vertex 50 inside HotPathAllocGuard
    {
        let _guard = HotPathAllocGuard::enter();

        let slot = log
            .record_geodesic_delta(
                &config,
                50,
                HlcTimestamp(9),
                &base_state,
                HlcTimestamp(10),
                &commit_state,
            )
            .unwrap();
        assert_eq!(slot, 0);

        // Crucial GAP-03 assertions:
        // Vertex 10 was evicted because slot 0 was overwritten
        assert_eq!(log.get_latest_entry(10), None);

        // Vertex 50 is mapped to slot 0
        let entry50 = log.get_latest_entry(50).unwrap();
        assert_eq!(entry50.vertex_id, 50);

        // Vertices 20, 30, 40 remain untouched and accessible
        assert_eq!(log.get_latest_entry(20).unwrap().vertex_id, 20);
        assert_eq!(log.get_latest_entry(30).unwrap().vertex_id, 30);
        assert_eq!(log.get_latest_entry(40).unwrap().vertex_id, 40);

        // Continuous state reconstruction on the hot path
        let mut recon_state = [0.0f32; 16];
        log.reconstruct_state_at(
            &config,
            &base_state,
            entry50,
            HlcTimestamp(9),
            &mut recon_state,
        )
        .unwrap();
        assert!((recon_state[0] - 1.0).abs() < 1e-4);
    }
}

#[test]
fn test_gap07_at_rq_distortion_certificate_and_fp16_spillover() {
    reset_hot_path_counters();

    // 48-dimensional tangent vector (24 subspaces x 2 dims each)
    let dim = 48;
    let k = 16;
    let d_sub = 2;
    let total_cb = 24 * k * d_sub;

    let mut codebooks = Vec::with_capacity(total_cb);
    for s in 0..24 {
        for c in 0..k {
            for d in 0..d_sub {
                // Distribute centroid coordinates around [-1.5, 1.5]
                let val = -1.5f32 + (c as f32) * 0.2 + (s as f32) * 0.01 + (d as f32) * 0.05;
                codebooks.push(val);
            }
        }
    }

    let quantizer = AnisotropicTangentQuantizer::new(dim, k, codebooks).unwrap();
    let mut page = TensorGraphPage::zeroed();
    page.header.update_checksum();

    let mut v_tangent = [0.0f32; 48];
    for i in 0..48 {
        v_tangent[i] = ((i as f32) * 0.137).sin() * 2.0;
    }

    // Case 1: Tolerant delta_max (no spillover triggered)
    {
        let _guard = HotPathAllocGuard::enter();

        let cert = quantizer
            .encode_page_primary_with_certificate(&v_tangent, 1000.0, &mut page)
            .unwrap();

        assert!(!cert.spilled_to_fp16);
        assert_eq!(cert.post_spill_residual_norm, cert.residual_norm);
        assert_eq!(page.header.flags & PAGE_FLAG_HAS_FP16_SPILLOVER, 0);

        let mut out_tangent = [0.0f32; 48];
        quantizer
            .decode_page_primary_with_spillover(&page, &mut out_tangent)
            .unwrap();
    }
    assert!(page.header.verify_checksum().is_ok());

    // Case 2: Strict delta_max (spillover triggered into bytes 1184..1216)
    let cert_high = {
        let _guard = HotPathAllocGuard::enter();

        let cert = quantizer
            .encode_page_primary_with_certificate(&v_tangent, 1e-4, &mut page)
            .unwrap();

        assert!(cert.spilled_to_fp16);
        assert!(cert.post_spill_residual_norm < cert.residual_norm);
        assert_ne!(page.header.flags & PAGE_FLAG_HAS_FP16_SPILLOVER, 0);

        let mut out_tangent = [0.0f32; 48];
        quantizer
            .decode_page_primary_with_spillover(&page, &mut out_tangent)
            .unwrap();

        // High-curvature components in first 16 dimensions should be reconstructed
        // with sub-milliradian precision due to FP16 residual compensation
        for i in 0..16 {
            let err = (v_tangent[i] - out_tangent[i]).abs();
            assert!(
                err < 0.01,
                "dimension {i} residual error {err} exceeds FP16 tolerance"
            );
        }

        cert
    };

    assert!(page.header.verify_checksum().is_ok());
    assert!(cert_high.post_spill_residual_norm < cert_high.residual_norm);
}

#[test]
fn test_gap04_klsm_tier_coordinator_end_to_end_compaction() {
    reset_hot_path_counters();

    let dim = 48;
    let k = 16;
    let d_sub = 2;
    let total_cb = 24 * k * d_sub;

    let mut codebooks = Vec::with_capacity(total_cb);
    for s in 0..24 {
        for c in 0..k {
            for d in 0..d_sub {
                codebooks.push(-1.5f32 + (c as f32) * 0.2 + (s as f32) * 0.01 + (d as f32) * 0.05);
            }
        }
    }
    let quantizer = AnisotropicTangentQuantizer::new(dim, k, codebooks).unwrap();

    let config = ProductManifoldConfig {
        hyperbolic: vec![HyperbolicSpec {
            dim: 48,
            kappa: -1.0,
            weight: 1.0,
        }],
        spherical: vec![],
        euclidean: None,
    };

    let mut runtime = CoreShardRuntime::<48, 64, 64>::new(0, 32, HlcTimestamp(100));
    let mut coordinator = KlsmTierCoordinator::<16>::new(4);

    // Prepare pages for source run
    let slot_s0 = runtime
        .arena
        .allocate_page(101, HlcTimestamp(10), 0)
        .unwrap();
    let slot_s1 = runtime
        .arena
        .allocate_page(102, HlcTimestamp(15), 0)
        .unwrap();
    let v_src0 = [0.1f32; 48];
    let v_src1 = [0.12f32; 48];
    {
        let page_s0 = runtime.arena.get_page_mut(slot_s0).unwrap();
        quantizer
            .encode_page_primary_with_certificate(&v_src0, 1000.0, page_s0)
            .unwrap();
        let page_s1 = runtime.arena.get_page_mut(slot_s1).unwrap();
        quantizer
            .encode_page_primary_with_certificate(&v_src1, 1000.0, page_s1)
            .unwrap();
    }

    let mut s_slots = [0u32; 64];
    s_slots[0] = slot_s0 as u32;
    s_slots[1] = slot_s1 as u32;
    let src_manifest = KlsmRunManifest {
        run_id: 1,
        level: 1,
        active: true,
        min_hlc: HlcTimestamp(10),
        max_hlc: HlcTimestamp(15),
        page_slots: s_slots,
        page_count: 2,
        voronoi_centroid: [0.1; 48],
        voronoi_radius: 0.1,
    };
    let src_idx = coordinator.register_run(src_manifest).unwrap();

    // Prepare pages for target run:
    // Page t0 is close to source centroid (boundary relink)
    // Page t1 is far from source centroid (interior splice)
    let slot_t0 = runtime
        .arena
        .allocate_page(201, HlcTimestamp(20), 0)
        .unwrap();
    let slot_t1 = runtime
        .arena
        .allocate_page(202, HlcTimestamp(25), 0)
        .unwrap();
    let v_tgt0 = [0.15f32; 48]; // close
    let mut v_tgt1 = [0.0f32; 48];
    v_tgt1[0] = 5.0; // far
    {
        let page_t0 = runtime.arena.get_page_mut(slot_t0).unwrap();
        quantizer
            .encode_page_primary_with_certificate(&v_tgt0, 1000.0, page_t0)
            .unwrap();
        let page_t1 = runtime.arena.get_page_mut(slot_t1).unwrap();
        quantizer
            .encode_page_primary_with_certificate(&v_tgt1, 1000.0, page_t1)
            .unwrap();
    }

    let mut t_slots = [0u32; 64];
    t_slots[0] = slot_t0 as u32;
    t_slots[1] = slot_t1 as u32;
    let tgt_manifest = KlsmRunManifest {
        run_id: 2,
        level: 1,
        active: true,
        min_hlc: HlcTimestamp(20),
        max_hlc: HlcTimestamp(25),
        page_slots: t_slots,
        page_count: 2,
        voronoi_centroid: [1.5; 48],
        voronoi_radius: 0.8,
    };
    let tgt_idx = coordinator.register_run(tgt_manifest).unwrap();

    // Verify compaction trigger detection
    let trigger = coordinator.maybe_trigger_compaction();
    assert_eq!(trigger, Some((src_idx, tgt_idx)));

    // Execute level compaction inside HotPathAllocGuard
    let report = {
        let _guard = HotPathAllocGuard::enter();
        coordinator
            .execute_level_compaction(&mut runtime, src_idx, tgt_idx, &quantizer, &config)
            .unwrap()
    };

    assert_eq!(report.runs_merged, 2);
    assert_eq!(
        report.interior_pages_spliced + report.boundary_pages_relinked,
        2
    );
    assert_eq!(
        coordinator.total_spliced_interior,
        report.interior_pages_spliced
    );
    assert_eq!(
        coordinator.total_relinked_boundary,
        report.boundary_pages_relinked
    );
    assert_eq!(coordinator.total_io_writes, report.io_writes_submitted);

    // Source run was deactivated, target run was replaced by merged run
    assert!(!coordinator.runs[src_idx].active);
    assert!(coordinator.runs[tgt_idx].active);
    assert_eq!(coordinator.runs[tgt_idx].level, 2);
    assert_eq!(coordinator.runs[tgt_idx].page_count, 4);

    // Process any asynchronous direct I/O writes queued by compaction
    let processed = runtime.process_pending_io();
    assert_eq!(processed, report.io_writes_submitted);
}

#[test]
fn test_gap05_speculative_lookahead_triggers_async_io_on_ram_miss() {
    reset_hot_path_counters();
    let mut pipeline = SpeculativeLookaheadPipeline::new();
    let runtime = CoreShardRuntime::<16, 64, 64>::new(0, 16, HlcTimestamp(100));

    let mut current_page = TensorGraphPage::zeroed();
    current_page.header.active_vertex_count = 4;
    current_page.neighbor_ids[0] = 101; // RAM cache hit
    current_page.neighbor_ids[1] = 201; // RAM miss -> NVMe async direct I/O
    current_page.neighbor_ids[2] = 201; // Duplicate of 201 -> duplicate skip
    current_page.neighbor_ids[3] = 301; // RAM miss -> NVMe async direct I/O

    let mut cached_page = TensorGraphPage::zeroed();
    cached_page.header.page_id = 101;
    let cached_ptr = &cached_page as *const TensorGraphPage;

    // Execute speculative prefetch inside HotPathAllocGuard
    let report = {
        let _guard = HotPathAllocGuard::enter();
        pipeline
            .prefetch_neighbor_pages_with_io(
                &current_page,
                16,
                |id| if id == 101 { Some(cached_ptr) } else { None },
                |id| runtime.submit_page_prefetch(id, 0),
            )
            .unwrap()
    };

    assert_eq!(report.ram_cache_hits, 1);
    assert_eq!(report.async_io_submissions, 2);
    assert_eq!(report.duplicate_skips, 1);

    assert_eq!(
        runtime
            .total_io_submitted
            .load(core::sync::atomic::Ordering::Relaxed),
        2
    );
}

#[test]
fn test_gap10_thread_pinning_and_zero_alloc_prometheus_telemetry() {
    reset_hot_path_counters();

    // 1. Thread affinity pinning inside HotPathAllocGuard
    {
        let _guard = HotPathAllocGuard::enter();
        let pin_res = pin_current_thread_to_core(0);
        assert!(pin_res.is_ok());
    }

    // 2. Telemetry and Prometheus formatting inside HotPathAllocGuard
    let mut runtime = CoreShardRuntime::<16, 64, 64>::new(0, 8, HlcTimestamp(50));
    let s0 = runtime
        .arena
        .allocate_page(501, HlcTimestamp(60), 0)
        .unwrap();
    let s1 = runtime
        .arena
        .allocate_page(502, HlcTimestamp(70), 0)
        .unwrap();

    let config = ProductManifoldConfig {
        hyperbolic: vec![HyperbolicSpec {
            dim: 16,
            kappa: -1.0,
            weight: 1.0,
        }],
        spherical: vec![],
        euclidean: None,
    };
    let mut base = [0.0f32; 16];
    base[0] = 1.0;
    let mut commit = base;
    commit[1] = 0.02;
    commit[0] = (1.0 + 0.02 * 0.02f32).sqrt();

    runtime
        .l0_log
        .record_geodesic_delta(
            &config,
            999,
            HlcTimestamp(65),
            &base,
            HlcTimestamp(75),
            &commit,
        )
        .unwrap();

    runtime.submit_page_write(501, s0).unwrap();
    runtime.submit_page_prefetch(502, s1).unwrap();
    let _ = runtime.process_pending_io();

    let mut out_buf = [0u8; 2048];
    let bytes_written = {
        let _guard = HotPathAllocGuard::enter();
        runtime.export_prometheus_text(&mut out_buf)
    };
    assert!(bytes_written > 0);

    let text = core::str::from_utf8(&out_buf[..bytes_written]).unwrap();
    assert!(text.contains("aether_shard_allocated_pages{core=\"0\"} 2"));
    assert!(text.contains("aether_shard_io_submitted_total{core=\"0\"} 2"));
    assert!(text.contains("aether_shard_io_completed_total{core=\"0\"} 2"));
    assert!(text.contains("aether_shard_l0_entries{core=\"0\"} 1"));
}

#[test]
fn test_gap06_morphism_replicator_hlc_lru_eviction_beyond_2048_vertices() {
    reset_hot_path_counters();
    let mut replicator = MorphismReplicator::new(0);
    assert_eq!(replicator.occupied_count(), 0);
    assert_eq!(replicator.evicted_count(), 0);

    // Insert 2,100 distinct vertices with monotonically increasing HLCs inside HotPathAllocGuard
    {
        let _guard = HotPathAllocGuard::enter();

        for v in 1..=2100u64 {
            let mut payload = [0u8; 24];
            payload[0] = (v & 0xFF) as u8;
            payload[1] = ((v >> 8) & 0xFF) as u8;

            let mut msg = ReplicationMessage::new(
                v * 10,
                0,
                v,
                v,
                ReplicationMessageType::Commit,
                0,
                0,
                payload,
            );
            msg.update_crc();
            replicator.apply_commit(&msg).unwrap();
        }

        // Table capacity is MAX_TRACKED_VERTICES (2048). 2100 - 2048 = 52 evictions.
        assert_eq!(replicator.occupied_count(), 2048);
        assert_eq!(replicator.evicted_count(), 52);

        // The 52 oldest vertices (v = 1..=52) had the lowest HLCs and must be evicted
        for v in 1..=52u64 {
            assert!(
                replicator.get_vertex_state(v).is_none(),
                "Vertex {} should have been evicted",
                v
            );
        }

        // The remaining 2048 vertices (v = 53..=2100) must be present with exact states
        for v in 53..=2100u64 {
            let state = replicator.get_vertex_state(v);
            assert!(state.is_some(), "Vertex {} should be present", v);
            let (hlc, payload) = state.unwrap();
            assert_eq!(hlc.as_u64(), v * 10);
            assert_eq!(payload[0], (v & 0xFF) as u8);
            assert_eq!(payload[1], ((v >> 8) & 0xFF) as u8);
        }
    }
}

#[test]
fn test_gap08_consensus_256_node_quorum_bitmask_no_modulo_collision() {
    reset_hot_path_counters();

    // Configure cluster with 256 nodes, write quorum = 4, read quorum = 253 (W + R = 257 > 256)
    let config = QuorumConfig::new(0, 256, 4, 253).unwrap();
    let mut engine = LeaderlessQuorumEngine::new(config, 1_000_000);

    {
        let _guard = HotPathAllocGuard::enter();

        // 1. Propose mutation from node 0 (records self acknowledgment at word 0, bit 0)
        let propose_msg = engine.propose(42, [9u8; 24], 1_000_001).unwrap();
        assert_eq!(propose_msg.node_id, 0);

        // 2. Acknowledgment from node 64 (word 1, bit 0)
        // In the old implementation (node_id % 64), 64 % 64 == 0 would collide with node 0!
        let mut ack_64 = ReplicationMessage::new(
            1_000_002,
            64,
            propose_msg.sequence_number,
            42,
            ReplicationMessageType::Ack,
            0,
            0,
            [9u8; 24],
        );
        ack_64.update_crc();
        let commit_64 = engine.handle_ack(&ack_64).unwrap();
        assert!(
            commit_64.is_none(),
            "Quorum not yet reached (ack_count = 2, required = 4)"
        );

        // 3. Acknowledgment from node 128 (word 2, bit 0)
        // Previously collided with bit 0 as well (128 % 64 == 0)
        let mut ack_128 = ReplicationMessage::new(
            1_000_003,
            128,
            propose_msg.sequence_number,
            42,
            ReplicationMessageType::Ack,
            0,
            0,
            [9u8; 24],
        );
        ack_128.update_crc();
        let commit_128 = engine.handle_ack(&ack_128).unwrap();
        assert!(
            commit_128.is_none(),
            "Quorum not yet reached (ack_count = 3, required = 4)"
        );

        // 4. Acknowledgment from node 192 (word 3, bit 0)
        // Previously collided with bit 0 as well (192 % 64 == 0)
        let mut ack_192 = ReplicationMessage::new(
            1_000_004,
            192,
            propose_msg.sequence_number,
            42,
            ReplicationMessageType::Ack,
            0,
            0,
            [9u8; 24],
        );
        ack_192.update_crc();
        let commit_192 = engine.handle_ack(&ack_192).unwrap();
        // Now ack_count == 4, write quorum of 4 is reached without modulo collisions!
        assert!(
            commit_192.is_some(),
            "Quorum should be reached with 4 distinct words"
        );
        let commit_msg = commit_192.unwrap();
        assert_eq!(commit_msg.msg_type, ReplicationMessageType::Commit as u8);
        assert_eq!(commit_msg.vertex_id, 42);

        // 5. Out-of-bounds node_id (>= 256) is rejected cleanly with InvalidNodeId
        let mut ack_oob = ReplicationMessage::new(
            1_000_005,
            MAX_CLUSTER_NODES as u64,
            propose_msg.sequence_number,
            42,
            ReplicationMessageType::Ack,
            0,
            0,
            [9u8; 24],
        );
        ack_oob.update_crc();
        let oob_res = engine.handle_ack(&ack_oob);
        assert_eq!(
            oob_res,
            Err(AetherError::InvalidNodeId {
                node_id: MAX_CLUSTER_NODES as u64
            })
        );
    }
}

#[test]
fn test_gap09_dma_and_rdma_hardware_trait_backends_zero_alloc() {
    reset_hot_path_counters();

    // 1. DMA Hardware Trait Backend Verification
    let engine = DmaCompactionEngine::new();
    let mut backend = HostLoopbackDmaBackend::new();

    let mut src_page = TensorGraphPage::zeroed();
    src_page.header.page_id = 1001;
    src_page.neighbor_ids[0] = 777;
    src_page.at_rq_codebooks[0..4].copy_from_slice(&[1, 2, 3, 4]);
    src_page.header.update_checksum();

    let dst_page = TensorGraphPage::zeroed();

    let mut desc = DmaPageDescriptor::new(1001, 2002, 0, 0, 0x01);
    desc.update_crc();

    let src_slice = [src_page];
    let mut dst_slice = [dst_page];

    {
        let _guard = HotPathAllocGuard::enter();
        let transferred = engine
            .execute_batch_with_backend(&mut backend, &[desc], &src_slice, &mut dst_slice)
            .unwrap();
        assert_eq!(transferred, 1);
        assert_eq!(dst_slice[0].header.page_id, 2002);
        assert_eq!(dst_slice[0].neighbor_ids[0], 777);
        assert_eq!(&dst_slice[0].at_rq_codebooks[0..4], &[1, 2, 3, 4]);
        assert!(dst_slice[0].header.verify_checksum().is_ok());
    }

    // 2. RDMA Hardware Trait Backend Verification
    let mut fabric = RdmaFabric::new();
    let window = RdmaMemoryWindow::with_capacity(4);
    fabric.register_node(1, window).unwrap();

    let mut read_page = TensorGraphPage::zeroed();

    {
        let _guard = HotPathAllocGuard::enter();
        // Write to node 1, slot 0 via RdmaTransportBackend trait
        fabric.post_one_sided_write(1, 0, &src_page).unwrap();

        // Read back from node 1, slot 0 via RdmaTransportBackend trait
        fabric.post_one_sided_read(1, 0, &mut read_page).unwrap();

        assert_eq!(read_page.header.page_id, 1001);
        assert_eq!(read_page.neighbor_ids[0], 777);
        assert_eq!(&read_page.at_rq_codebooks[0..4], &[1, 2, 3, 4]);
        assert!(read_page.header.verify_checksum().is_ok());
    }
}

#[test]
fn test_gap01_aql_parser_and_spmv_execution_end_to_end() {
    reset_hot_path_counters();

    // 1. Parse CREATE MANIFOLD
    let create_sql = "CREATE MANIFOLD bio_graph (HYPERBOLIC(dim=4, kappa=-1.0), SPHERICAL(dim=2, tau=1.0), EUCLIDEAN(dim=2))";
    let stmt_create = parse_aql(create_sql).unwrap();
    match stmt_create {
        AqlStatement::CreateManifold {
            name,
            hyperbolic_dim,
            kappa,
            spherical_dim,
            tau,
            euclidean_dim,
        } => {
            assert_eq!(name, "bio_graph");
            assert_eq!(hyperbolic_dim, 4);
            assert_eq!(kappa, -1.0);
            assert_eq!(spherical_dim, 2);
            assert_eq!(tau, 1.0);
            assert_eq!(euclidean_dim, 2);
        }
        _ => panic!("Expected CreateManifold AST variant"),
    }

    // 2. Parse INSERT VERTEX
    let insert_sql = "INSERT VERTEX 42 VECTOR [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5] STALK [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] NEIGHBORS [1, 2]";
    let stmt_insert = parse_aql(insert_sql).unwrap();
    match stmt_insert {
        AqlStatement::InsertVertex {
            vertex_id,
            vector,
            stalk,
            neighbors,
        } => {
            assert_eq!(vertex_id, 42);
            assert_eq!(vector.len(), 8);
            assert_eq!(stalk[0], 0.1);
            assert_eq!(stalk[7], 0.8);
            assert_eq!(neighbors, vec![1, 2]);
        }
        _ => panic!("Expected InsertVertex AST variant"),
    }

    // 3. Set up in-memory topology (Nodes 0 -> [1, 2], 1 -> [2], 2 -> [])
    let mut pages = vec![TensorGraphPage::zeroed(); 3];
    let topology: [(u64, &[u64]); 3] = [(0, &[1, 2]), (1, &[2]), (2, &[])];

    for (idx, &(node_id, neighbors)) in topology.iter().enumerate() {
        if let Some(page) = pages.get_mut(idx) {
            page.header.page_id = node_id;
            page.header.active_vertex_count = neighbors.len() as u32;
            let stalk = [(node_id as f32 + 1.0) * 0.1; 8];
            page.set_primary_stalk(&stalk);

            for (n_idx, &n_id) in neighbors.iter().enumerate() {
                if let Some(slot) = page.neighbor_ids.get_mut(n_idx) {
                    *slot = n_id;
                }
            }

            let morphisms =
                [SheafRestrictionMap::<8>::from_reflector(&[0.353_553_38f32; 8], 1.0); 64];
            let edge_weights = [1.0f32; 64];
            let _ = pack_page_sheaf_morphisms(page, &morphisms, &edge_weights);
            page.header.update_checksum();
        }
    }

    let static_pages: &'static [TensorGraphPage] = Box::leak(pages.into_boxed_slice());
    let page_lookup = move |id: u64| -> Option<&'static TensorGraphPage> {
        static_pages.iter().find(|p| p.header.page_id == id)
    };
    let distance_oracle = move |_id: u64| -> f32 { 0.05f32 };

    // 4. Parse & Execute TRAVERSE under Tropical, Diffusion, and Bottleneck semirings
    let queries = [
        ("TRAVERSE FROM 0 HOPS 2 USING SEMIRING TROPICAL WHERE BETA = 0.5 AND GAMMA = 1.0 LIMIT 5", SemiringKind::Tropical),
        ("TRAVERSE FROM 0 HOPS 2 USING SEMIRING DIFFUSION WHERE BETA = 0.5 AND GAMMA = 1.0 LIMIT 5", SemiringKind::Diffusion),
        ("TRAVERSE FROM 0 HOPS 2 USING SEMIRING BOTTLENECK WHERE BETA = 0.5 AND GAMMA = 1.0 LIMIT 5", SemiringKind::Bottleneck),
    ];

    for (query_str, expected_kind) in queries {
        let stmt = parse_aql(query_str).unwrap();
        match &stmt {
            AqlStatement::Traverse {
                semiring,
                hops,
                limit,
                ..
            } => {
                assert_eq!(*semiring, expected_kind);
                assert_eq!(*hops, 2);
                assert_eq!(*limit, 5);
            }
            _ => panic!("Expected Traverse AST variant"),
        }

        let mut results = [FrontierNode::default(); MAX_QUERY_FRONTIER];
        let mut count = 0usize;

        // Traversal execution MUST execute with strictly 0 heap allocations on the hot path (Rule 3.1)
        {
            let _guard = HotPathAllocGuard::enter();
            AqlCompileHelper::execute_traverse(
                &stmt,
                page_lookup,
                distance_oracle,
                &mut results,
                &mut count,
            )
            .unwrap();
        }

        assert!(
            count > 0,
            "Traverse query should produce non-empty frontier for {:?}",
            expected_kind
        );
        assert!(count <= 5, "Result count should respect LIMIT 5");
    }
}

#[test]
fn test_aql_expanded_ast_statements() {
    let upsert_query = "UPSERT VERTEX 42 COORDINATES [0.1, 0.2, 0.3, 0.4] STALK [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0] NEIGHBORS [1, 2, 3]";
    let stmt = parse_aql(upsert_query).unwrap();
    match stmt {
        AqlStatement::UpsertVertex {
            vertex_id,
            vector,
            stalk,
            neighbors,
            ..
        } => {
            assert_eq!(vertex_id, 42);
            assert_eq!(vector.len(), 4);
            assert_eq!(stalk[0], 0.5);
            assert_eq!(neighbors, vec![1, 2, 3]);
        }
        _ => panic!("Expected UpsertVertex"),
    }

    let connect_query = "CONNECT VERTEX 10 TO 20 USING MORPHISM HOUSEHOLDER WEIGHT 0.95";
    let stmt2 = parse_aql(connect_query).unwrap();
    match stmt2 {
        AqlStatement::ConnectVertex {
            source,
            target,
            morphism_type,
            weight,
            ..
        } => {
            assert_eq!(source, 10);
            assert_eq!(target, 20);
            assert_eq!(morphism_type, "HOUSEHOLDER");
            assert_eq!(weight, 0.95);
        }
        _ => panic!("Expected ConnectVertex"),
    }

    let optimize_query = "OPTIMIZE MANIFOLD research_knowledge_graph FROM LEVEL 1 TO LEVEL 2 USING METHOD VORONOI_BOUNDARY_CERTIFICATES RETURN TELEMETRY";
    let stmt3 = parse_aql(optimize_query).unwrap();
    match stmt3 {
        AqlStatement::OptimizeManifold {
            manifold,
            from_level,
            to_level,
            method,
            return_telemetry,
        } => {
            assert_eq!(manifold, "research_knowledge_graph");
            assert_eq!(from_level, 1);
            assert_eq!(to_level, 2);
            assert_eq!(method, "VORONOI_BOUNDARY_CERTIFICATES");
            assert!(return_telemetry);
        }
        _ => panic!("Expected OptimizeManifold"),
    }
}

#[test]
fn test_aetherd_daemon_cli_and_tcp_server() {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::process::Command;

    // Locate compiled aetherd binary via CARGO_MANIFEST_DIR workspace root
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("Workspace root");
    let binary_name = if cfg!(windows) {
        "aetherd.exe"
    } else {
        "aetherd"
    };

    let mut exe_path = workspace_root
        .join("target")
        .join("debug")
        .join(binary_name);
    if !exe_path.exists() {
        let release_path = workspace_root
            .join("target")
            .join("release")
            .join(binary_name);
        if release_path.exists() {
            exe_path = release_path;
        } else {
            let status = Command::new("cargo")
                .args(["build", "-p", "aether-daemon", "--bin", "aetherd"])
                .status()
                .expect("Failed to build aetherd");
            assert!(status.success());
        }
    }

    // 1. Test `aetherd status`
    let status_out = Command::new(&exe_path)
        .arg("status")
        .output()
        .expect("Failed to run aetherd status");
    assert!(status_out.status.success());
    let status_str = String::from_utf8_lossy(&status_out.stdout);
    assert!(status_str.contains("Project Aether-Manifold: Engine Architecture"));
    assert!(status_str.contains("TensorGraphPage Size:      4096 bytes"));

    // 2. Test `aetherd verify-gates`
    let gates_out = Command::new(&exe_path)
        .arg("verify-gates")
        .output()
        .expect("Failed to run aetherd verify-gates");
    assert!(gates_out.status.success());
    let gates_str = String::from_utf8_lossy(&gates_out.stdout);
    assert!(gates_str.contains("All exit gates passed self-check successfully"));

    // 3. Test `aetherd exec`
    let exec_out = Command::new(&exe_path)
        .args([
            "exec",
            "TRAVERSE FROM 0 HOPS 2 USING SEMIRING TROPICAL WHERE BETA = 0.5 AND GAMMA = 1.0 LIMIT 5",
        ])
        .output()
        .expect("Failed to run aetherd exec");
    assert!(exec_out.status.success());
    let exec_str = String::from_utf8_lossy(&exec_out.stdout);
    assert!(exec_str.contains("Status: TRAVERSAL COMPLETE"));
    assert!(exec_str.contains("Frontier Candidates: 5"));

    // 4. Test `aetherd serve` over TCP
    let port = 9188;
    let listen_addr = format!("127.0.0.1:{}", port);
    let mut server_child = Command::new(&exe_path)
        .args(["serve", "--listen", &listen_addr, "--max-requests", "1"])
        .spawn()
        .expect("Failed to spawn aetherd serve");

    // Allow server to bind and enter accept loop
    std::thread::sleep(std::time::Duration::from_millis(250));

    let mut stream = TcpStream::connect(&listen_addr).expect("Failed to connect to aetherd server");
    let query =
        "TRAVERSE FROM 0 HOPS 2 USING SEMIRING TROPICAL WHERE BETA = 0.5 AND GAMMA = 1.0 LIMIT 5\n";
    stream.write_all(query.as_bytes()).expect("Write failed");
    stream.flush().expect("Flush failed");

    let mut response_buf = [0u8; 1024];
    let n = stream.read(&mut response_buf).expect("Read failed");
    let resp_str = String::from_utf8_lossy(&response_buf[..n]);
    assert!(
        resp_str.contains("STATUS OK"),
        "Expected STATUS OK in response, got: {}",
        resp_str
    );

    let exit_status = server_child.wait().expect("Server child failed");
    assert!(exit_status.success());
}
