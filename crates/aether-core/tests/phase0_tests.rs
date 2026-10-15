//! Project Aether-Manifold: Phase 0 Verification Test Suite
//!
//! Enforces:
//! - TEST-MEM-01 (`proof_tensor_page_layout_invariants`)
//! - TEST-MEM-02 (`proof_zero_copy_cast_soundness`)
//! - TEST-MEM-04 (`proof_pod_stack_scratchpad_bounds`)
//! - TEST-CONC-05 (`loom_hlc_timestamp_monotonicity`)
//! - TEST-PMU-01 (`gate_zero_hot_path_heap_allocations`)

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use aether_core::alloc_guard::{
    reset_hot_path_counters, HotPathAllocGuard, TrackingAllocator,
};
use aether_core::error::AetherError;
use aether_core::hlc::HlcTimestamp;
use aether_core::page::{PageHeader, TensorGraphPage};
use aether_core::scratchpad::Scratchpad;
use bytemuck::Zeroable;

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator::new(std::alloc::System);

/// TEST-MEM-01: Compile-time and runtime validation of 4KB page and 64B header layouts.
#[test]
fn proof_tensor_page_layout_invariants() {
    // Compile-time static assertions per AGENTS.md Rule 3.3 and test_suite_manifest.md Section 2
    static_assertions::const_assert_eq!(core::mem::size_of::<TensorGraphPage>(), 4096);
    static_assertions::const_assert_eq!(core::mem::align_of::<TensorGraphPage>(), 4096);
    static_assertions::const_assert_eq!(core::mem::size_of::<PageHeader>(), 64);
    static_assertions::const_assert_eq!(core::mem::align_of::<PageHeader>(), 64);

    // Runtime size and alignment invariants
    assert_eq!(core::mem::size_of::<TensorGraphPage>(), 4096);
    assert_eq!(core::mem::align_of::<TensorGraphPage>(), 4096);
    assert_eq!(core::mem::size_of::<PageHeader>(), 64);
    assert_eq!(core::mem::align_of::<PageHeader>(), 64);

    // PageHeader internal field offsets
    assert_eq!(core::mem::offset_of!(PageHeader, crc32c_checksum), 0);
    assert_eq!(core::mem::offset_of!(PageHeader, active_vertex_count), 4);
    assert_eq!(core::mem::offset_of!(PageHeader, lsn), 8);
    assert_eq!(core::mem::offset_of!(PageHeader, page_id), 16);
    assert_eq!(core::mem::offset_of!(PageHeader, manifold_curvature_bits), 24);
    assert_eq!(core::mem::offset_of!(PageHeader, csr_offset_table), 32);

    // TensorGraphPage field offsets per roadmap.md Section 3.2
    assert_eq!(core::mem::offset_of!(TensorGraphPage, header), 0);
    assert_eq!(core::mem::offset_of!(TensorGraphPage, at_rq_codebooks), 64);
    assert_eq!(core::mem::offset_of!(TensorGraphPage, neighbor_ids), 320);
    assert_eq!(core::mem::offset_of!(TensorGraphPage, quantized_neighbors), 832);
    assert_eq!(core::mem::offset_of!(TensorGraphPage, sheaf_edge_weights), 2880);
}

/// TEST-MEM-02: Zero-copy cast soundness, alignment enforcement, and 0-allocation casting.
#[test]
fn proof_zero_copy_cast_soundness() {
    // 1. Aligned page cast
    let mut page = TensorGraphPage::zeroed();
    page.header.page_id = 42;
    page.header.active_vertex_count = 5;
    page.header.update_checksum();

    let aligned_bytes = page.as_aligned_bytes();
    let cast_res = TensorGraphPage::from_aligned_bytes(aligned_bytes);
    assert!(cast_res.is_ok());
    let cast_page = cast_res.unwrap();
    assert_eq!(cast_page.header.page_id, 42);
    assert_eq!(cast_page.header.active_vertex_count, 5);
    assert!(cast_page.verify_header_checksum().is_ok());

    // 2. Unaligned buffer verification: must return typed AetherError::InvalidAlignment
    let unaligned_raw = vec![0u8; 8192];
    let addr = unaligned_raw.as_ptr() as usize;
    let offset = if addr.is_multiple_of(4096) { 1 } else { 0 };
    let unaligned_slice = &unaligned_raw[offset..offset + 4096];
    let unaligned_res = TensorGraphPage::from_aligned_bytes(unaligned_slice);
    match unaligned_res {
        Err(AetherError::InvalidAlignment { ptr, required_align }) => {
            assert_eq!(required_align, 4096);
            assert_eq!(ptr, unaligned_slice.as_ptr() as usize);
        }
        _ => panic!("Expected AetherError::InvalidAlignment for unaligned slice"),
    }

    // 3. Invalid length verification: must return typed AetherError::InvalidBufferSize
    let short_slice = &unaligned_raw[0..4095];
    match TensorGraphPage::from_aligned_bytes(short_slice) {
        Err(AetherError::InvalidBufferSize { actual, expected }) => {
            assert_eq!(actual, 4095);
            assert_eq!(expected, 4096);
        }
        _ => panic!("Expected AetherError::InvalidBufferSize for short slice"),
    }

    let long_slice = &unaligned_raw[0..4097];
    match TensorGraphPage::from_aligned_bytes(long_slice) {
        Err(AetherError::InvalidBufferSize { actual, expected }) => {
            assert_eq!(actual, 4097);
            assert_eq!(expected, 4096);
        }
        _ => panic!("Expected AetherError::InvalidBufferSize for long slice"),
    }

    // 4. Zero allocation verification on valid cast
    reset_hot_path_counters();
    let guard = HotPathAllocGuard::enter();
    let cast_page = TensorGraphPage::from_aligned_bytes(aligned_bytes).unwrap();
    assert_eq!(cast_page.header.page_id, 42);
    assert_eq!(guard.alloc_count(), 0);
    assert_eq!(guard.allocated_bytes(), 0);
}

/// TEST-MEM-04: Deterministic scratchpad bounds and graceful error handling on overflow.
#[test]
fn proof_pod_stack_scratchpad_bounds() {
    let mut memory = [0u8; 1024];
    let mut scratchpad = Scratchpad::new(&mut memory);

    assert_eq!(scratchpad.available_bytes(), 1024);

    // Allocate 100 f32s (400 bytes) - succeeds
    let slice1 = scratchpad.try_alloc_slice::<f32>(100);
    assert!(slice1.is_ok());
    let slice1 = slice1.unwrap();
    assert_eq!(slice1.len(), 100);

    // Initialize values deterministically
    for (i, val) in slice1.iter_mut().enumerate() {
        *val = i as f32;
    }
    assert_eq!(slice1[99], 99.0);

    // Allocate another 100 f32s (400 bytes) - succeeds
    let slice2 = scratchpad.try_alloc_slice::<f32>(100);
    assert!(slice2.is_ok());
    let slice2 = slice2.unwrap();
    assert_eq!(slice2.len(), 100);

    // Capacity overflow: attempting to allocate 200 f32s (800 bytes) when < 224 bytes remaining
    // Must return Err(AetherError::ScratchpadExhausted) without panicking
    let overflow_res = scratchpad.try_alloc_slice::<f32>(200);
    assert!(overflow_res.is_err());
    match overflow_res {
        Err(AetherError::ScratchpadExhausted { requested, available }) => {
            assert_eq!(requested, 200 * core::mem::size_of::<f32>());
            assert!(available < requested);
        }
        _ => panic!("Expected ScratchpadExhausted error on capacity overflow"),
    }
}

/// TEST-CONC-05: Strict monotonicity of HLC timestamps under concurrency and simulated clock drift.
#[test]
fn loom_hlc_timestamp_monotonicity() {
    #[cfg(not(loom))]
    {
        use std::sync::atomic::AtomicU64;
        use std::sync::Arc;
        use std::thread;

        let clock = Arc::new(AtomicU64::new(0));
        let num_threads = 4;
        let iters = 1000;

        let mut handles = Vec::new();
        for tid in 0..num_threads {
            let clock_clone = Arc::clone(&clock);
            handles.push(thread::spawn(move || {
                let mut timestamps = Vec::with_capacity(iters);
                let mut prev = HlcTimestamp::from_raw(0);

                for i in 0..iters {
                    let ts = match tid {
                        0 => HlcTimestamp::now_or_tick(&clock_clone, 10_000 + i as u64 * 10),
                        1 => HlcTimestamp::now_or_tick(
                            &clock_clone,
                            20_000 - (i as u64 % 50) * 10,
                        ), // backward jumps
                        2 => HlcTimestamp::now_or_tick(&clock_clone, 15_000), // stalled clock
                        _ => {
                            if i % 2 == 0 {
                                let remote =
                                    HlcTimestamp::pack(12_000 + i as u64, (i % 20) as u16);
                                let merged = HlcTimestamp::update_with_remote(
                                    &clock_clone,
                                    remote,
                                    11_000,
                                );
                                assert!(merged > remote);
                                merged
                            } else {
                                HlcTimestamp::now_or_tick(&clock_clone, 11_000 + i as u64)
                            }
                        }
                    };

                    assert!(ts > prev, "Monotonicity violation: {:?} <= {:?}", ts, prev);
                    prev = ts;
                    timestamps.push(ts);
                }
                timestamps
            }));
        }

        let mut all_timestamps = Vec::new();
        for h in handles {
            let ts_list = h.join().unwrap();
            all_timestamps.extend(ts_list);
        }

        let total_count = all_timestamps.len();
        all_timestamps.sort();
        all_timestamps.dedup();
        assert_eq!(
            all_timestamps.len(),
            total_count,
            "Found duplicate HLC timestamps!"
        );
    }

    #[cfg(loom)]
    {
        loom::model(|| {
            let clock = loom::sync::Arc::new(loom::sync::atomic::AtomicU64::new(0));
            let c1 = clock.clone();
            let c2 = clock.clone();

            let h1 = loom::thread::spawn(move || {
                let t1 = HlcTimestamp::now_or_tick(&c1, 100);
                let t2 = HlcTimestamp::now_or_tick(&c1, 90);
                assert!(t2 > t1);
                t2
            });

            let h2 = loom::thread::spawn(move || {
                let remote = HlcTimestamp::pack(95, 3);
                let t3 = HlcTimestamp::update_with_remote(&c2, remote, 80);
                assert!(t3 > remote);
                t3
            });

            let r1 = h1.join().unwrap();
            let r2 = h2.join().unwrap();
            assert_ne!(r1, r2);
        });
    }
}

/// TEST-PMU-01: Zero hot-path heap allocations during page casting, CRC32C, and HLC advancement.
#[test]
fn gate_zero_hot_path_heap_allocations() {
    // 1. Initialize aligned 4KB page
    let mut page = TensorGraphPage::zeroed();
    page.header.page_id = 1001;
    page.header.active_vertex_count = 16;
    page.header.lsn = HlcTimestamp::pack(500_000, 1);
    page.header.update_checksum();

    let atomic_clock = core::sync::atomic::AtomicU64::new(page.header.lsn.0);
    let page_bytes = page.as_aligned_bytes();

    // Reset tracking counters before entering hot-path scope
    reset_hot_path_counters();

    // Hot-path execution section
    {
        let guard = HotPathAllocGuard::enter();

        // 1. Zero-copy in-place casting from 4KB aligned bytes
        let cast_page = TensorGraphPage::from_aligned_bytes(page_bytes).unwrap();
        assert_eq!(cast_page.header.page_id, 1001);

        // 2. CRC32C checksum validation
        assert!(cast_page.verify_header_checksum().is_ok());

        // 3. Advancing the HLC timestamp via wait-free CAS
        let advanced_ts = HlcTimestamp::now_or_tick(&atomic_clock, 500_001);
        assert_eq!(advanced_ts.physical(), 500_001);
        assert_eq!(advanced_ts.logical(), 0);

        // Advance again under same microsecond
        let next_ts = HlcTimestamp::now_or_tick(&atomic_clock, 500_001);
        assert_eq!(next_ts.physical(), 500_001);
        assert_eq!(next_ts.logical(), 1);

        // 4. Assert ZERO heap allocations occurred inside the guard
        assert_eq!(
            guard.alloc_count(),
            0,
            "TEST-PMU-01 Violation: Non-zero allocations detected on hot path"
        );
        assert_eq!(
            guard.allocated_bytes(),
            0,
            "TEST-PMU-01 Violation: Non-zero allocated bytes on hot path"
        );
        assert!(guard.assert_zero_allocations().is_ok());
    }

    // Sanity check: outside the guard, allocating dynamic memory succeeds
    let dynamic_boxed = Box::new(100u32);
    assert_eq!(*dynamic_boxed, 100);
}

/// Verification that the custom allocation guard correctly detects heap allocations when they do occur.
#[test]
fn proof_alloc_guard_detects_violations() {
    reset_hot_path_counters();
    let guard = HotPathAllocGuard::enter();
    let boxed = Box::new(42u64);
    assert_eq!(*boxed, 42);
    assert!(
        guard.alloc_count() > 0,
        "Alloc guard must intercept heap allocation"
    );
    assert!(guard.allocated_bytes() >= core::mem::size_of::<u64>());
    assert!(guard.assert_zero_allocations().is_err());
}
