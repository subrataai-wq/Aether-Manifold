//! Project Aether-Manifold: Shared-Nothing Thread-Per-Core Runtime (`shard_runtime.rs`)
//!
//! Assigns each physical CPU core a private Product Manifold shard, private `AlignedPageArena`,
//! private `L0MorphismLog`, and dedicated wait-free `SpscRingBuffer` queue pair for asynchronous
//! 4KB direct page I/O / speculative prefetch without any global locks or heap allocations.

use crate::l0_mvcc::L0MorphismLog;
use crate::page_arena::AlignedPageArena;
use crate::spsc::SpscRingBuffer;
use aether_core::error::AetherError;
use aether_core::hlc::{HlcClock, HlcTimestamp};

/// Operation opcode for asynchronous direct page I/O.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoOpcode {
    /// Asynchronously read a 4KB NVMe page directly into an arena slot.
    ReadPage = 0,
    /// Asynchronously write a 4KB NVMe page directly from an arena slot.
    WritePage = 1,
}

/// Asynchronous Direct Page I/O Submission Queue Entry (SQE).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IoSubmissionSqe {
    pub opcode: IoOpcode,
    pub page_id: u64,
    pub arena_slot: usize,
    pub user_data: u64,
}

/// Asynchronous Direct Page I/O Completion Queue Entry (CQE).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IoCompletionCqe {
    pub result: i32,
    pub user_data: u64,
}

/// Shared-Nothing Thread-Per-Core Runtime Shard.
///
/// Contains private memory, arena, clock, and lock-free I/O queue pair.
pub struct CoreShardRuntime<const D: usize, const L0_CAP: usize, const IO_DEPTH: usize = 64> {
    pub core_id: usize,
    pub clock: HlcClock,
    pub arena: AlignedPageArena,
    pub l0_log: L0MorphismLog<D, L0_CAP>,
    pub sq: SpscRingBuffer<IoSubmissionSqe, IO_DEPTH>,
    pub cq: SpscRingBuffer<IoCompletionCqe, IO_DEPTH>,
    pub io_driver: crate::io_driver::DirectPageIoDriver,
    pub total_io_submitted: core::sync::atomic::AtomicUsize,
    pub total_io_completed: core::sync::atomic::AtomicUsize,
}

impl<const D: usize, const L0_CAP: usize, const IO_DEPTH: usize>
    CoreShardRuntime<D, L0_CAP, IO_DEPTH>
{
    /// Cold-path initialization of a core shard runtime.
    pub fn new(core_id: usize, arena_capacity: usize, initial_ts: HlcTimestamp) -> Self {
        Self {
            core_id,
            clock: HlcClock::new(initial_ts),
            arena: AlignedPageArena::with_capacity(arena_capacity),
            l0_log: L0MorphismLog::new(),
            sq: SpscRingBuffer::new(),
            cq: SpscRingBuffer::new(),
            io_driver: crate::io_driver::DirectPageIoDriver::memory_only(),
            total_io_submitted: core::sync::atomic::AtomicUsize::new(0),
            total_io_completed: core::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Set an explicit physical page I/O driver (e.g. Linux io_uring or physical block driver).
    pub fn with_driver(mut self, driver: crate::io_driver::DirectPageIoDriver) -> Self {
        self.io_driver = driver;
        self
    }

    /// Submit a speculative 4KB page prefetch / read request to the lock-free submission queue.
    ///
    /// # Hard Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    #[inline]
    pub fn submit_page_prefetch(&self, page_id: u64, arena_slot: usize) -> Result<(), AetherError> {
        let sqe = IoSubmissionSqe {
            opcode: IoOpcode::ReadPage,
            page_id,
            arena_slot,
            user_data: page_id,
        };
        self.sq.try_push(sqe)?;
        self.total_io_submitted
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Submit an asynchronous 4KB page write request to the lock-free submission queue.
    ///
    /// # Hard Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    #[inline]
    pub fn submit_page_write(&self, page_id: u64, arena_slot: usize) -> Result<(), AetherError> {
        let sqe = IoSubmissionSqe {
            opcode: IoOpcode::WritePage,
            page_id,
            arena_slot,
            user_data: page_id,
        };
        self.sq.try_push(sqe)?;
        self.total_io_submitted
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Poll completed direct page I/O completions into `out_cqes`.
    ///
    /// Returns the number of completions dequeued.
    ///
    /// # Hard Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    #[inline]
    pub fn poll_completions(&self, max_batch: usize, out_cqes: &mut [IoCompletionCqe]) -> usize {
        let limit = max_batch.min(out_cqes.len());
        let mut count = 0;

        while count < limit {
            match self.cq.try_pop() {
                Some(cqe) => {
                    if let Some(dest) = out_cqes.get_mut(count) {
                        *dest = cqe;
                        count += 1;
                    } else {
                        break;
                    }
                }
                None => break,
            }
        }

        count
    }

    /// Process and advance pending direct I/O submissions.
    ///
    /// Routes requests through `io_driver` with strictly zero hot-path heap allocations.
    ///
    /// # Hard Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    pub fn process_pending_io(&mut self) -> usize {
        let mut processed = 0;

        while let Some(sqe) = self.sq.try_pop() {
            let result = if sqe.arena_slot < self.arena.capacity() {
                if let Ok(page) = self.arena.get_page_mut(sqe.arena_slot) {
                    match sqe.opcode {
                        IoOpcode::ReadPage => match self.io_driver.read_page(sqe.page_id, page) {
                            Ok(bytes) => bytes as i32,
                            Err(_) => -1i32,
                        },
                        IoOpcode::WritePage => match self.io_driver.write_page(sqe.page_id, page) {
                            Ok(bytes) => bytes as i32,
                            Err(_) => -1i32,
                        },
                    }
                } else {
                    -1i32
                }
            } else {
                -1i32
            };

            let cqe = IoCompletionCqe {
                result,
                user_data: sqe.user_data,
            };

            let _ = self.cq.try_push(cqe);
            self.total_io_completed
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            processed += 1;
        }

        processed
    }

    /// Capture a point-in-time telemetry snapshot of this shard runtime.
    #[inline]
    pub fn telemetry_snapshot(&self) -> ShardTelemetrySnapshot {
        ShardTelemetrySnapshot {
            core_id: self.core_id,
            allocated_pages: self.arena.allocated_count(),
            free_pages: self.arena.free_count(),
            io_submitted_total: self
                .total_io_submitted
                .load(core::sync::atomic::Ordering::Relaxed),
            io_completed_total: self
                .total_io_completed
                .load(core::sync::atomic::Ordering::Relaxed),
            l0_entries: self.l0_log.len(),
        }
    }

    /// Export Prometheus text-format metrics into a pre-allocated buffer with 0 heap allocations.
    ///
    /// Returns the number of bytes written into `out_buf`.
    ///
    /// # Hard Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    #[inline]
    pub fn export_prometheus_text(&self, out_buf: &mut [u8]) -> usize {
        self.telemetry_snapshot().export_prometheus_text(out_buf)
    }
}

/// Point-in-time telemetry snapshot for a core shard runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShardTelemetrySnapshot {
    pub core_id: usize,
    pub allocated_pages: usize,
    pub free_pages: usize,
    pub io_submitted_total: usize,
    pub io_completed_total: usize,
    pub l0_entries: usize,
}

impl ShardTelemetrySnapshot {
    /// Format Prometheus metrics into a pre-allocated byte buffer with 0 heap allocations.
    ///
    /// Returns the number of bytes written.
    pub fn export_prometheus_text(&self, out_buf: &mut [u8]) -> usize {
        use core::fmt::Write;
        let mut writer = BufferWriter::new(out_buf);
        let _ = writeln!(
            writer,
            "# HELP aether_shard_allocated_pages Number of active allocated 4KB pages.\n\
             # TYPE aether_shard_allocated_pages gauge\n\
             aether_shard_allocated_pages{{core=\"{}\"}} {}\n\
             # HELP aether_shard_io_submitted_total Cumulative asynchronous direct I/O requests submitted.\n\
             # TYPE aether_shard_io_submitted_total counter\n\
             aether_shard_io_submitted_total{{core=\"{}\"}} {}\n\
             # HELP aether_shard_io_completed_total Cumulative asynchronous direct I/O requests completed.\n\
             # TYPE aether_shard_io_completed_total counter\n\
             aether_shard_io_completed_total{{core=\"{}\"}} {}\n\
             # HELP aether_shard_l0_entries Number of entries currently stored in L0 morphism log.\n\
             # TYPE aether_shard_l0_entries gauge\n\
             aether_shard_l0_entries{{core=\"{}\"}} {}",
            self.core_id,
            self.allocated_pages,
            self.core_id,
            self.io_submitted_total,
            self.core_id,
            self.io_completed_total,
            self.core_id,
            self.l0_entries
        );
        writer.written()
    }
}

/// Stack-allocated zero-heap formatting sink for Prometheus telemetry.
struct BufferWriter<'a> {
    buf: &'a mut [u8],
    offset: usize,
}

impl<'a> BufferWriter<'a> {
    #[inline]
    fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, offset: 0 }
    }

    #[inline]
    fn written(&self) -> usize {
        self.offset
    }
}

impl core::fmt::Write for BufferWriter<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let remaining = self.buf.len().saturating_sub(self.offset);
        let to_copy = bytes.len().min(remaining);
        if let (Some(src), Some(dest)) = (
            bytes.get(..to_copy),
            self.buf.get_mut(self.offset..self.offset + to_copy),
        ) {
            dest.copy_from_slice(src);
            self.offset += to_copy;
        }
        if to_copy < bytes.len() {
            Err(core::fmt::Error)
        } else {
            Ok(())
        }
    }
}

/// Pin the calling OS thread to a specific hardware CPU core.
///
/// Guaranteed zero heap allocations on the hot path.
///
/// # Hard Invariants
/// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
#[inline]
pub fn pin_current_thread_to_core(core_id: usize) -> Result<(), AetherError> {
    #[cfg(target_os = "linux")]
    {
        unsafe {
            let mut cpuset: libc::cpu_set_t = core::mem::zeroed();
            libc::CPU_ZERO(&mut cpuset);
            libc::CPU_SET(core_id % 1024, &mut cpuset);
            let ret = libc::sched_setaffinity(0, core::mem::size_of::<libc::cpu_set_t>(), &cpuset);
            if ret == 0 {
                Ok(())
            } else {
                Err(AetherError::IoFault { code: ret })
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        extern "system" {
            fn GetCurrentThread() -> isize;
            fn SetThreadAffinityMask(hThread: isize, dwThreadAffinityMask: usize) -> usize;
        }
        unsafe {
            let mask = 1usize << (core_id % 64);
            let prev = SetThreadAffinityMask(GetCurrentThread(), mask);
            if prev != 0 {
                Ok(())
            } else {
                Err(AetherError::IoFault { code: 1 })
            }
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        let _ = core_id;
        Ok(())
    }
}
