//! Project Aether-Manifold: Hot-Path Zero-Allocation Detection & Hardware PMU Gate
//!
//! Intercepts heap allocations during query evaluation, vector distance calculations,
//! and graph traversals, enforcing AGENTS.md Rule 3.1 (Zero Allocations on the Hot Path).

use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::error::AetherError;

/// Number of allocations intercepted while hot-path guard is active.
pub static HOT_PATH_ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Total number of bytes allocated while hot-path guard is active.
pub static HOT_PATH_ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Number of deallocations intercepted while hot-path guard is active.
pub static HOT_PATH_DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

std::thread_local! {
    static HOT_PATH_DEPTH: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// Check if the calling thread is currently executing inside an active `HotPathAllocGuard`.
#[inline]
pub fn is_hot_path_active() -> bool {
    HOT_PATH_DEPTH
        .try_with(|cell| cell.get() > 0)
        .unwrap_or_default()
}

/// Reset all hot-path allocation tracking counters to zero.
#[inline]
pub fn reset_hot_path_counters() {
    HOT_PATH_ALLOC_COUNT.store(0, Ordering::Release);
    HOT_PATH_ALLOC_BYTES.store(0, Ordering::Release);
    HOT_PATH_DEALLOC_COUNT.store(0, Ordering::Release);
}

/// Global allocator wrapper that tracks allocations on the hot path.
///
/// Use as `#[global_allocator]` in test and verification builds to enforce
/// zero hot-path allocation invariants.
pub struct TrackingAllocator<A = std::alloc::System> {
    inner: A,
}

impl<A> TrackingAllocator<A> {
    /// Wrap an underlying allocator.
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
}

impl Default for TrackingAllocator<std::alloc::System> {
    fn default() -> Self {
        Self {
            inner: std::alloc::System,
        }
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if is_hot_path_active() {
            HOT_PATH_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            HOT_PATH_ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { self.inner.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if is_hot_path_active() {
            HOT_PATH_DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { self.inner.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if is_hot_path_active() {
            HOT_PATH_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            HOT_PATH_ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        unsafe { self.inner.realloc(ptr, layout, new_size) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if is_hot_path_active() {
            HOT_PATH_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            HOT_PATH_ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { self.inner.alloc_zeroed(layout) }
    }
}

/// Scoped RAII guard activating hot-path allocation detection on the current thread.
#[derive(Debug)]
pub struct HotPathAllocGuard {
    start_alloc_count: usize,
    start_alloc_bytes: usize,
}

impl HotPathAllocGuard {
    /// Enter a zero-allocation hot-path section on the current thread.
    pub fn enter() -> Self {
        let start_alloc_count = HOT_PATH_ALLOC_COUNT.load(Ordering::Acquire);
        let start_alloc_bytes = HOT_PATH_ALLOC_BYTES.load(Ordering::Acquire);
        let _ = HOT_PATH_DEPTH.try_with(|cell| {
            cell.set(cell.get().saturating_add(1));
        });
        Self {
            start_alloc_count,
            start_alloc_bytes,
        }
    }

    /// Retrieve the number of heap allocations executed since entering this guard.
    #[inline]
    pub fn alloc_count(&self) -> usize {
        let current = HOT_PATH_ALLOC_COUNT.load(Ordering::Acquire);
        current.saturating_sub(self.start_alloc_count)
    }

    /// Retrieve the total number of bytes allocated since entering this guard.
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        let current = HOT_PATH_ALLOC_BYTES.load(Ordering::Acquire);
        current.saturating_sub(self.start_alloc_bytes)
    }

    /// Verify that zero allocations occurred; returns `Err(AetherError::HotPathAllocationViolation)`
    /// if any allocation was detected.
    pub fn assert_zero_allocations(&self) -> Result<(), AetherError> {
        let count = self.alloc_count();
        if count == 0 {
            Ok(())
        } else {
            Err(AetherError::HotPathAllocationViolation { count })
        }
    }
}

impl Drop for HotPathAllocGuard {
    fn drop(&mut self) {
        let _ = HOT_PATH_DEPTH.try_with(|cell| {
            cell.set(cell.get().saturating_sub(1));
        });
    }
}

/// Alias for `HotPathAllocGuard` per AGENTS.md.
pub type HotPathGuard = HotPathAllocGuard;

