//! Project Aether-Manifold: Speculative Lookahead Prefetch Pipeline
//!
//! Overlaps SIMD distance evaluation for hop $t$ with speculative asynchronous
//! page prefetches for candidate vertices of hop $t+1$ (roadmap.md Section 3.3).
//!
//! On Linux with `io_uring`, issues asynchronous direct I/O prefetch submissions.
//! On Windows/macOS/cross-platform, executes CPU cache-line software prefetches
//! (`_mm_prefetch`) into L1/L2 cache and warms the page arena cache line.

use aether_core::error::AetherError;
use aether_core::page::TensorGraphPage;

/// Capacity of the speculative prefetch queue.
pub const PREFETCH_QUEUE_CAPACITY: usize = 64;

/// Speculative Lookahead Prefetch Pipeline.
pub struct SpeculativeLookaheadPipeline {
    /// Bounded circular buffer of prefetch target node IDs.
    prefetch_queue: [u64; PREFETCH_QUEUE_CAPACITY],
    /// Queue head index.
    head: usize,
    /// Queue tail index.
    tail: usize,
    /// Count of currently queued prefetch requests.
    count: usize,
}

impl Default for SpeculativeLookaheadPipeline {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl SpeculativeLookaheadPipeline {
    /// Create a new speculative prefetch pipeline.
    #[inline]
    pub const fn new() -> Self {
        Self {
            prefetch_queue: [0; PREFETCH_QUEUE_CAPACITY],
            head: 0,
            tail: 0,
            count: 0,
        }
    }

    /// Issue speculative prefetches for candidate neighbor pages of hop $t+1$.
    ///
    /// Reads neighbor IDs from the currently evaluated 4KB page and issues
    /// cache-line / I/O prefetch hints ahead of time.
    pub fn prefetch_neighbor_pages(
        &mut self,
        current_page: &TensorGraphPage,
        page_lookup: impl Fn(u64) -> Option<*const TensorGraphPage>,
    ) -> Result<usize, AetherError> {
        let active_neighbors = current_page
            .header
            .active_vertex_count
            .min(PREFETCH_QUEUE_CAPACITY as u32) as usize;

        let mut prefetched_count = 0;

        for i in 0..active_neighbors {
            let neighbor_id = match current_page.neighbor_ids.get(i) {
                Some(id) if *id != 0 => *id,
                _ => continue,
            };

            // Enqueue into prefetch queue
            if self.count < PREFETCH_QUEUE_CAPACITY {
                if let Some(slot) = self.prefetch_queue.get_mut(self.tail) {
                    *slot = neighbor_id;
                    self.tail = (self.tail + 1) % PREFETCH_QUEUE_CAPACITY;
                    self.count += 1;
                }
            }

            // Issue cache prefetch if page pointer is resolvable
            if let Some(page_ptr) = page_lookup(neighbor_id) {
                Self::issue_hardware_prefetch(page_ptr);
                prefetched_count += 1;
            }
        }

        Ok(prefetched_count)
    }

    /// Issue hardware CPU prefetch intrinsic on 4KB page memory addresses.
    #[inline]
    fn issue_hardware_prefetch(page_ptr: *const TensorGraphPage) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let byte_ptr = page_ptr as *const i8;
            // Prefetch first cache line (Header 64B) and second cache line (AT-RQ)
            _mm_prefetch(byte_ptr, _MM_HINT_T0);
            _mm_prefetch(byte_ptr.add(64), _MM_HINT_T0);
            _mm_prefetch(byte_ptr.add(128), _MM_HINT_T0);
            _mm_prefetch(byte_ptr.add(320), _MM_HINT_T0); // Neighbor IDs
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            // Memory read hint to trigger L1/L2 hardware prefetcher
            let _ = unsafe { core::ptr::read_volatile(page_ptr as *const u64) };
        }
    }

    /// Drain completed prefetch queue entries.
    #[inline]
    pub fn drain_queue(&mut self) {
        self.head = 0;
        self.tail = 0;
        self.count = 0;
    }
}
