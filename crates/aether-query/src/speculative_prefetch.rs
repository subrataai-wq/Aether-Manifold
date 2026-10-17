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

/// Result report from speculative prefetch evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefetchReport {
    /// Number of candidate neighbor pages found in memory cache and hardware prefetched.
    pub ram_cache_hits: usize,
    /// Number of asynchronous direct I/O prefetch requests submitted to NVMe driver.
    pub async_io_submissions: usize,
    /// Number of duplicate page requests skipped because they are already queued.
    pub duplicate_skips: usize,
}

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

    /// Speculatively prefetch 1-hop neighbor pages, resolving in-memory cache hits via hardware
    /// CPU prefetch instructions and enqueuing asynchronous direct I/O requests for disk-resident pages.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    pub fn prefetch_neighbor_pages_with_io<F, I>(
        &mut self,
        page: &TensorGraphPage,
        max_lookahead: usize,
        page_lookup: F,
        mut io_submit: I,
    ) -> Result<PrefetchReport, AetherError>
    where
        F: Fn(u64) -> Option<*const TensorGraphPage>,
        I: FnMut(u64) -> Result<(), AetherError>,
    {
        let limit = (page.header.active_vertex_count as usize)
            .min(PREFETCH_QUEUE_CAPACITY)
            .min(max_lookahead);

        let mut report = PrefetchReport::default();

        for i in 0..limit {
            let neighbor_id = match page.neighbor_ids.get(i) {
                Some(&id) if id != 0 => id,
                _ => continue,
            };

            if let Some(page_ptr) = page_lookup(neighbor_id) {
                Self::issue_hardware_prefetch(page_ptr);
                report.ram_cache_hits += 1;
            } else {
                // Check deduplication in active prefetch queue
                let mut already_queued = false;
                for k in 0..self.count {
                    let idx = (self.head + k) % PREFETCH_QUEUE_CAPACITY;
                    if let Some(&queued_id) = self.prefetch_queue.get(idx) {
                        if queued_id == neighbor_id {
                            already_queued = true;
                            break;
                        }
                    }
                }

                if already_queued {
                    report.duplicate_skips += 1;
                } else {
                    // Record into deduplication ring buffer
                    if let Some(slot) = self.prefetch_queue.get_mut(self.tail) {
                        *slot = neighbor_id;
                        self.tail = (self.tail + 1) % PREFETCH_QUEUE_CAPACITY;
                        if self.count < PREFETCH_QUEUE_CAPACITY {
                            self.count += 1;
                        } else {
                            self.head = (self.head + 1) % PREFETCH_QUEUE_CAPACITY;
                        }
                    }

                    io_submit(neighbor_id)?;
                    report.async_io_submissions += 1;
                }
            }
        }

        Ok(report)
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
