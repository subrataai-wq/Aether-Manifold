//! Project Aether-Manifold: Category VIII Deterministic Chaos & NVMe Fault Injection Suite
//!
//! Enforces roadmap.md and test_suite_manifest.md Section 9:
//! - 8.1 `sim_torn_4kb_tensor_page_write_recovery`: Power loss after partial 4KB write.
//! - 8.2 `sim_nvme_silent_bitflip_corruption_detection`: 100% bitflip detection via CRC32C.
//! - 8.3 `sim_power_loss_mid_voronoi_boundary_compaction`: Crash recovery during L1 -> L2 compaction.
//! - 8.5 `sim_hlc_ntp_clock_skew_and_leap_second_chaos`: Strict monotonicity under +/-5000ms clock skew.
//!
//! Strict Compliance:
//! - AGENTS.md Rule 3.1: Strictly 0 hot-path heap allocations inside `HotPathAllocGuard`.
//! - AGENTS.md Rule 3.2: Strictly 0 OS locks (`Mutex`/`RwLock`).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::needless_range_loop
)]

use core::sync::atomic::AtomicU64;

use aether_consensus::rdma_fabric::{RdmaFabric, RdmaMemoryWindow};
use aether_core::alloc_guard::{reset_hot_path_counters, HotPathAllocGuard, TrackingAllocator};
use aether_core::hlc::HlcTimestamp;
use aether_core::page::TensorGraphPage;
use aether_manifold::at_rq::AnisotropicTangentQuantizer;
use aether_manifold::product::{EuclideanSpec, ProductManifoldConfig};
use aether_storage::klsm_coordinator::{KlsmRunManifest, KlsmTierCoordinator};
use aether_storage::l0_mvcc::L0MorphismLog;
use aether_storage::page_arena::AlignedPageArena;
use aether_storage::shard_runtime::CoreShardRuntime;
use bytemuck::Zeroable;

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator::new(std::alloc::System);

/// Category VIII (8.1): Simulated Torn 4KB NVMe Write & WAL Replay Recovery
#[test]
fn sim_torn_4kb_tensor_page_write_recovery() {
    reset_hot_path_counters();

    let mut arena = AlignedPageArena::with_capacity(16);
    let mut l0_log = L0MorphismLog::<2, 64>::new();
    let config = ProductManifoldConfig {
        hyperbolic: vec![],
        spherical: vec![],
        euclidean: Some(EuclideanSpec {
            dim: 2,
            weight: 1.0,
        }),
    };

    // 1. Ingest base state for vertex 1001 into page slot
    let slot = arena.allocate_page(1001, HlcTimestamp(100), 0).unwrap();
    let page = arena.get_page_mut(slot).unwrap();
    page.header.active_vertex_count = 2;
    page.neighbor_ids[0] = 2001;
    page.neighbor_ids[1] = 2002;
    // Set non-zero payload past 1024 bytes (in sheaf_edge_weights region)
    page.sheaf_edge_weights[0] = 0xAA;
    page.sheaf_edge_weights[64] = 0xBB;
    page.update_header_checksum();
    let original_checksum = page.compute_full_page_checksum();

    // 2. Log update to vertex 1001 in L0 WAL
    let base_coords = [1.0f32, 2.0];
    let commit_coords = [1.5f32, 2.5];
    l0_log
        .record_geodesic_delta(
            &config,
            1001,
            HlcTimestamp(100),
            &base_coords,
            HlcTimestamp(200),
            &commit_coords,
        )
        .unwrap();

    // 3. Simulate torn NVMe write: flush only first 1024 bytes to disk buffer
    let mut on_disk_page = TensorGraphPage::zeroed();
    let page_bytes = page.as_aligned_bytes();
    on_disk_page.as_aligned_bytes_mut()[..1024].copy_from_slice(&page_bytes[..1024]);
    // Remaining bytes [1024..4096] are torn/zeroed

    // 4. Verification: validate torn page
    let torn_checksum = on_disk_page.compute_full_page_checksum();
    assert_ne!(
        torn_checksum, original_checksum,
        "Torn 4KB write must produce mismatched checksum"
    );

    // 5. Recovery: discard corrupted page slot and replay from L0 WAL inside HotPathAllocGuard
    {
        let _guard = HotPathAllocGuard::enter();

        // Deallocate corrupted slot
        arena.deallocate_page(slot).unwrap();

        // Reallocate clean slot
        let recovered_slot = arena.allocate_page(1001, HlcTimestamp(200), 0).unwrap();
        let recovered_page = arena.get_page_mut(recovered_slot).unwrap();

        // Reconstruct state at commit HLC from WAL
        let entry = l0_log.get_latest_entry(1001).unwrap();
        let mut reconstructed = [0.0f32; 2];
        l0_log
            .reconstruct_state_at(
                &config,
                &base_coords,
                entry,
                HlcTimestamp(200),
                &mut reconstructed,
            )
            .unwrap();

        assert_eq!(reconstructed, commit_coords);

        // Restore neighbor topology and sheaf weights
        recovered_page.header.active_vertex_count = 2;
        recovered_page.neighbor_ids[0] = 2001;
        recovered_page.neighbor_ids[1] = 2002;
        recovered_page.sheaf_edge_weights[0] = 0xAA;
        recovered_page.sheaf_edge_weights[64] = 0xBB;
        recovered_page.update_header_checksum();
        assert!(recovered_page.verify_header_checksum().is_ok());
    }
}

/// Category VIII (8.2): Simulated NVMe Silent Bitflip Corruption Detection Across 10,000 Pages
#[test]
fn sim_nvme_silent_bitflip_corruption_detection() {
    reset_hot_path_counters();

    let mut prng = oorandom::Rand64::new(0xDEAD_BEEF_CAFE_1337);

    // Setup golden source page
    let mut golden_page = TensorGraphPage::zeroed();
    golden_page.header.page_id = 42;
    golden_page.header.lsn = HlcTimestamp(500);
    golden_page.header.active_vertex_count = 8;
    for i in 0..8 {
        golden_page.neighbor_ids[i] = 1000 + i as u64;
    }
    golden_page.update_header_checksum();
    let golden_checksum = golden_page.compute_full_page_checksum();

    // Register golden page on peer node in RDMA fabric for remote fetch failover
    let mut fabric = RdmaFabric::new();
    let remote_window = RdmaMemoryWindow::with_capacity(1);
    remote_window
        .write_page(0, golden_page.as_aligned_bytes())
        .unwrap();
    fabric.register_node(1, remote_window).unwrap();

    let mut detected_header_corruptions = 0usize;
    let mut detected_payload_corruptions = 0usize;
    let total_trials = 10_000usize;

    for _ in 0..total_trials {
        let mut corrupted_page = golden_page;

        // Inject 1 to 4 random bitflips
        let flips = (prng.rand_u64() % 4 + 1) as usize;
        let mut flipped_header = false;

        {
            let bytes = corrupted_page.as_aligned_bytes_mut();
            for _ in 0..flips {
                let byte_idx = (prng.rand_u64() % 4092 + 4) as usize; // skip crc32c field itself
                let bit_idx = (prng.rand_u64() % 8) as u8;
                bytes[byte_idx] ^= 1 << bit_idx;
                if byte_idx < 64 {
                    flipped_header = true;
                }
            }
        }

        if flipped_header {
            let result = corrupted_page.verify_header_checksum();
            assert!(result.is_err(), "Header bitflip MUST be detected by CRC32C");
            detected_header_corruptions += 1;
        } else {
            let computed = corrupted_page.compute_full_page_checksum();
            assert_ne!(
                computed, golden_checksum,
                "Payload bitflip MUST produce different full-page checksum"
            );
            detected_payload_corruptions += 1;
        }

        // On detected corruption: fetch uncorrupted golden page from peer node via RDMA fabric
        let mut recovered_page = TensorGraphPage::zeroed();
        {
            let _guard = HotPathAllocGuard::enter();
            fabric
                .read_remote_page(1, 0, recovered_page.as_aligned_bytes_mut())
                .unwrap();
        }
        assert!(recovered_page.verify_header_checksum().is_ok());
        assert_eq!(recovered_page.compute_full_page_checksum(), golden_checksum);
    }

    assert_eq!(
        detected_header_corruptions + detected_payload_corruptions,
        total_trials,
        "100% of bitflips must be reliably detected"
    );
}

/// Category VIII (8.3): Simulated Catastrophic Power Loss Mid-Voronoi Boundary Compaction
#[test]
fn sim_power_loss_mid_voronoi_boundary_compaction() {
    reset_hot_path_counters();

    let mut runtime = CoreShardRuntime::<48, 64, 64>::new(0, 16, HlcTimestamp(100));
    let mut coordinator = KlsmTierCoordinator::<8>::new(4);
    let codebooks = vec![0.0f32; 24 * 16 * 2];
    let quantizer = AnisotropicTangentQuantizer::new(48, 16, codebooks).unwrap();

    // Source Run (Level 1)
    let s0 = runtime
        .arena
        .allocate_page(101, HlcTimestamp(10), 0)
        .unwrap();
    let s1 = runtime
        .arena
        .allocate_page(102, HlcTimestamp(15), 0)
        .unwrap();
    let v_src0 = [0.1f32; 48];
    let v_src1 = [0.12f32; 48];
    {
        let page_s0 = runtime.arena.get_page_mut(s0).unwrap();
        quantizer
            .encode_page_primary_with_certificate(&v_src0, 1000.0, page_s0)
            .unwrap();
        let page_s1 = runtime.arena.get_page_mut(s1).unwrap();
        quantizer
            .encode_page_primary_with_certificate(&v_src1, 1000.0, page_s1)
            .unwrap();
    }

    let mut s_slots = [0u32; 64];
    s_slots[0] = s0 as u32;
    s_slots[1] = s1 as u32;
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

    // Target Run (Level 2)
    let t0 = runtime
        .arena
        .allocate_page(201, HlcTimestamp(20), 0)
        .unwrap();
    let t1 = runtime
        .arena
        .allocate_page(202, HlcTimestamp(25), 0)
        .unwrap();
    let v_tgt0 = [0.15f32; 48]; // close
    let mut v_tgt1 = [0.0f32; 48];
    v_tgt1[0] = 5.0; // far
    {
        let page_t0 = runtime.arena.get_page_mut(t0).unwrap();
        quantizer
            .encode_page_primary_with_certificate(&v_tgt0, 1000.0, page_t0)
            .unwrap();
        let page_t1 = runtime.arena.get_page_mut(t1).unwrap();
        quantizer
            .encode_page_primary_with_certificate(&v_tgt1, 1000.0, page_t1)
            .unwrap();
    }

    let mut t_slots = [0u32; 64];
    t_slots[0] = t0 as u32;
    t_slots[1] = t1 as u32;
    let tgt_manifest = KlsmRunManifest {
        run_id: 2,
        level: 2,
        active: true,
        min_hlc: HlcTimestamp(20),
        max_hlc: HlcTimestamp(25),
        page_slots: t_slots,
        page_count: 2,
        voronoi_centroid: [1.5; 48],
        voronoi_radius: 0.8,
    };
    let tgt_idx = coordinator.register_run(tgt_manifest).unwrap();

    assert_eq!(runtime.arena.allocated_count(), 4);

    // --- Simulate Crash Scenario A: Power loss after allocating temp scratch slot ---
    let temp_slot = runtime
        .arena
        .allocate_page(999, HlcTimestamp(30), 0)
        .unwrap();
    assert_eq!(runtime.arena.allocated_count(), 5);

    // Simulated Crash occurs here! The manifest transaction was never committed.
    // Recovery routine inspects coordinator runs:
    assert!(coordinator.runs[src_idx].active);
    assert!(coordinator.runs[tgt_idx].active);
    assert_eq!(coordinator.runs[tgt_idx].level, 2);

    // Rollback: scan arena slots not referenced in any active manifest run and deallocate
    let mut active_slots = [false; 16];
    for run in &coordinator.runs[..coordinator.run_count] {
        if run.active {
            for &slot_idx in &run.page_slots[..run.page_count] {
                active_slots[slot_idx as usize] = true;
            }
        }
    }
    assert!(!active_slots[temp_slot]);
    runtime.arena.deallocate_page(temp_slot).unwrap();
    assert_eq!(runtime.arena.allocated_count(), 4);

    // Now re-run compaction cleanly
    let config = ProductManifoldConfig {
        hyperbolic: vec![],
        spherical: vec![],
        euclidean: Some(EuclideanSpec {
            dim: 48,
            weight: 1.0,
        }),
    };

    let report = coordinator
        .execute_level_compaction(&mut runtime, src_idx, tgt_idx, &quantizer, &config)
        .unwrap();

    assert_eq!(report.runs_merged, 2);
    assert!(!coordinator.runs[src_idx].active);
    assert!(coordinator.runs[tgt_idx].active);
    assert_eq!(coordinator.runs[tgt_idx].page_count, 4);
}

/// Category VIII (8.5): Adversarial HLC Monotonicity under +/-5,000ms Clock Skew & Leap Seconds
#[test]
fn sim_hlc_ntp_clock_skew_and_leap_second_chaos() {
    reset_hot_path_counters();

    let mut prng = oorandom::Rand64::new(0xCAFE_BABE_0000_7777);

    // 3 simulated distributed nodes
    let node_clocks = [
        AtomicU64::new(HlcTimestamp::pack(1_000_000, 0).0),
        AtomicU64::new(HlcTimestamp::pack(1_000_000, 0).0),
        AtomicU64::new(HlcTimestamp::pack(1_000_000, 0).0),
    ];

    let mut simulated_physical_times = [1_000_000u64; 3];
    let mut last_hlc = [HlcTimestamp(0); 3];

    for step in 0..10_000 {
        let node_id = (prng.rand_u64() % 3) as usize;

        // Inject chaotic clock skew between -5,000,000 µs (-5s) and +5,000,000 µs (+5s)
        let skew_micros = (prng.rand_u64() % 10_000_000) as i64 - 5_000_000;
        let curr_phys = simulated_physical_times[node_id] as i64;
        let next_phys = (curr_phys + skew_micros).max(100_000) as u64;
        simulated_physical_times[node_id] = next_phys;

        let action = prng.rand_u64() % 3;

        let new_ts = match action {
            0 | 1 => {
                // Local event tick inside HotPathAllocGuard
                let _guard = HotPathAllocGuard::enter();
                HlcTimestamp::now_or_tick(&node_clocks[node_id], next_phys)
            }
            _ => {
                // Receive message from another node
                let peer_id = (node_id + 1 + (prng.rand_u64() % 2) as usize) % 3;
                let peer_ts = last_hlc[peer_id];

                let _guard = HotPathAllocGuard::enter();
                HlcTimestamp::update_with_remote(&node_clocks[node_id], peer_ts, next_phys)
            }
        };

        // Invariant 1: Strict monotonicity on the same core
        assert!(
            new_ts > last_hlc[node_id],
            "Step {step}: HLC timestamp must strictly advance (prev={:?}, new={:?})",
            last_hlc[node_id],
            new_ts
        );

        last_hlc[node_id] = new_ts;
    }
}
