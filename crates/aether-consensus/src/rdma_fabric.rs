//! Project Aether-Manifold: RDMA / CXL 3.0 Zero-Copy Remote Page Fetching Fabric
//!
//! Provides an asynchronous, one-sided zero-copy remote page transfer abstraction
//! simulating RDMA RoCEv2 and CXL 3.0 Type-3 shared memory fabrics with configurable
//! latency and deterministic fault injection. Strictly zero heap allocations and
//! zero OS locks on the transfer hot path (AGENTS.md Rule 3.1 & Rule 3.2).

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use aether_core::error::AetherError;
use aether_core::page::TensorGraphPage;

/// Maximum number of cluster nodes supported by the virtual RDMA fabric.
pub const MAX_RDMA_NODES: usize = 64;

/// 64-byte cache-line aligned lock-free page slot with atomic double-buffering (Rule 3.2).
#[repr(C, align(64))]
pub struct LockFreeRdmaSlot {
    active_idx: AtomicUsize,
    seq_version: AtomicU64,
    buffers: [UnsafeCell<TensorGraphPage>; 2],
}

unsafe impl Sync for LockFreeRdmaSlot {}
unsafe impl Send for LockFreeRdmaSlot {}

impl LockFreeRdmaSlot {
    /// Creates a new zeroed lock-free double-buffered page slot.
    pub fn new() -> Self {
        Self {
            active_idx: AtomicUsize::new(0),
            seq_version: AtomicU64::new(0),
            buffers: [
                UnsafeCell::new(bytemuck::Zeroable::zeroed()),
                UnsafeCell::new(bytemuck::Zeroable::zeroed()),
            ],
        }
    }

    /// Single-writer lock-free page write into the alternate buffer.
    #[inline]
    pub fn write_page(&self, src: &[u8; 4096]) {
        let v = self.seq_version.fetch_add(1, Ordering::Acquire);
        let curr = self.active_idx.load(Ordering::Acquire) & 1;
        let next = 1 - curr;
        if let Some(buf) = self.buffers.get(next) {
            unsafe {
                let page_ptr = buf.get() as *mut u8;
                core::ptr::copy_nonoverlapping(src.as_ptr(), page_ptr, 4096);
            }
            self.active_idx.store(next, Ordering::Release);
            self.seq_version.store(v.wrapping_add(2), Ordering::Release);
        }
    }

    /// Optimistic lock-free read into `dest` from the active buffer. Returns `true` if read was consistent.
    #[inline]
    pub fn read_page(&self, dest: &mut [u8; 4096]) -> bool {
        for _ in 0..64 {
            let v1 = self.seq_version.load(Ordering::Acquire);
            if (v1 & 1) != 0 {
                core::hint::spin_loop();
                continue;
            }

            let idx = self.active_idx.load(Ordering::Acquire) & 1;
            if let Some(buf) = self.buffers.get(idx) {
                unsafe {
                    let page_ptr = buf.get() as *const u8;
                    core::ptr::copy_nonoverlapping(page_ptr, dest.as_mut_ptr(), 4096);
                }

                core::sync::atomic::fence(Ordering::Acquire);
                let v2 = self.seq_version.load(Ordering::Acquire);
                if v1 == v2 {
                    return true;
                }
            }
        }
        false
    }
}

impl Default for LockFreeRdmaSlot {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory region representing an exported 4KB-page aligned memory window on a node.
pub struct RdmaMemoryWindow {
    slots: Vec<LockFreeRdmaSlot>,
}

impl RdmaMemoryWindow {
    /// Creates a pre-allocated memory window capable of hosting `num_pages` 4KB pages.
    pub fn with_capacity(num_pages: usize) -> Self {
        let mut slots = Vec::with_capacity(num_pages);
        for _ in 0..num_pages {
            slots.push(LockFreeRdmaSlot::new());
        }
        Self { slots }
    }

    /// Number of pages hosted in this memory window.
    #[inline]
    pub fn num_pages(&self) -> usize {
        self.slots.len()
    }

    /// Reads a 4KB page directly into `dest` with zero heap allocation and zero OS locks.
    #[inline]
    pub fn read_page(&self, page_index: usize, dest: &mut [u8; 4096]) -> Result<(), AetherError> {
        let slot = self.slots.get(page_index).ok_or(AetherError::OutOfBounds {
            index: page_index,
            limit: self.slots.len(),
        })?;
        if slot.read_page(dest) {
            Ok(())
        } else {
            Err(AetherError::IoFault { code: 12 })
        }
    }

    /// Writes a 4KB page directly from `src` with zero heap allocation and zero OS locks.
    #[inline]
    pub fn write_page(&self, page_index: usize, src: &[u8; 4096]) -> Result<(), AetherError> {
        let slot = self.slots.get(page_index).ok_or(AetherError::OutOfBounds {
            index: page_index,
            limit: self.slots.len(),
        })?;
        slot.write_page(src);
        Ok(())
    }
}

/// Node registration state in the RDMA fabric.
struct FabricNodeSlot {
    active: AtomicBool,
    online: AtomicBool,
    window: Option<Arc<RdmaMemoryWindow>>,
}

/// Virtual RDMA / CXL 3.0 Fabric coordinating zero-copy 4KB page transfers.
pub struct RdmaFabric {
    nodes: Vec<FabricNodeSlot>,
    synthetic_drop_rate_ppm: AtomicU32, // Parts per million (1e6 = 100%)
    prng_seed: AtomicU32,
}

impl RdmaFabric {
    /// Creates a new virtual RDMA fabric supporting up to `MAX_RDMA_NODES`.
    pub fn new() -> Self {
        let mut nodes = Vec::with_capacity(MAX_RDMA_NODES);
        for _ in 0..MAX_RDMA_NODES {
            nodes.push(FabricNodeSlot {
                active: AtomicBool::new(false),
                online: AtomicBool::new(false),
                window: None,
            });
        }
        Self {
            nodes,
            synthetic_drop_rate_ppm: AtomicU32::new(0),
            prng_seed: AtomicU32::new(0x1337_C0DE),
        }
    }

    /// Registers a node and its memory window into the fabric.
    pub fn register_node(
        &mut self,
        node_id: u64,
        window: RdmaMemoryWindow,
    ) -> Result<(), AetherError> {
        let idx = node_id as usize;
        if idx >= MAX_RDMA_NODES {
            return Err(AetherError::OutOfBounds {
                index: idx,
                limit: MAX_RDMA_NODES,
            });
        }
        if let Some(slot) = self.nodes.get_mut(idx) {
            slot.window = Some(Arc::new(window));
            slot.active.store(true, Ordering::Release);
            slot.online.store(true, Ordering::Release);
        }
        Ok(())
    }

    /// Sets node online or offline status to simulate node failures / network partitions.
    pub fn set_node_online(&self, node_id: u64, online: bool) -> Result<(), AetherError> {
        let idx = node_id as usize;
        if idx >= MAX_RDMA_NODES {
            return Err(AetherError::OutOfBounds {
                index: idx,
                limit: MAX_RDMA_NODES,
            });
        }
        let slot = self
            .nodes
            .get(idx)
            .ok_or(AetherError::NodeUnreachable { node_id })?;
        slot.online.store(online, Ordering::Release);
        Ok(())
    }

    /// Checks if a node is currently online.
    pub fn is_node_online(&self, node_id: u64) -> bool {
        let idx = node_id as usize;
        if idx >= MAX_RDMA_NODES {
            return false;
        }
        match self.nodes.get(idx) {
            Some(slot) => {
                slot.active.load(Ordering::Acquire) && slot.online.load(Ordering::Acquire)
            }
            None => false,
        }
    }

    /// Sets a synthetic packet drop rate in parts-per-million (e.g. 0.05 = 50,000 ppm = 5%).
    pub fn set_drop_rate(&self, drop_rate: f32) {
        let clamped = drop_rate.clamp(0.0, 1.0);
        let ppm = (clamped * 1_000_000.0) as u32;
        self.synthetic_drop_rate_ppm.store(ppm, Ordering::Release);
    }

    /// Simple fast deterministic pseudo-random check for packet drop.
    #[inline]
    fn should_drop_packet(&self) -> bool {
        let ppm = self.synthetic_drop_rate_ppm.load(Ordering::Relaxed);
        if ppm == 0 {
            return false;
        }
        let prev = self.prng_seed.fetch_add(0x6D2B_79F5, Ordering::Relaxed);
        let mut x = prev;
        x ^= x >> 15;
        x = x.wrapping_mul(0x2C1B_3C6D);
        x ^= x >> 12;
        (x % 1_000_000) < ppm
    }

    /// Executes a one-sided zero-copy RDMA / CXL 3.0 page read.
    ///
    /// Copies 4096 bytes directly from the remote node's registered memory window
    /// into `dest` with strictly zero hot-path heap allocations and zero OS locks.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    /// - Zero OS-level locks on the hot path (AGENTS.md Rule 3.2).
    pub fn read_remote_page(
        &self,
        target_node_id: u64,
        page_index: usize,
        dest: &mut [u8; 4096],
    ) -> Result<(), AetherError> {
        let idx = target_node_id as usize;
        if idx >= MAX_RDMA_NODES {
            return Err(AetherError::OutOfBounds {
                index: idx,
                limit: MAX_RDMA_NODES,
            });
        }

        let slot = self.nodes.get(idx).ok_or(AetherError::NodeUnreachable {
            node_id: target_node_id,
        })?;

        if !slot.active.load(Ordering::Acquire) || !slot.online.load(Ordering::Acquire) {
            return Err(AetherError::NodeUnreachable {
                node_id: target_node_id,
            });
        }

        if self.should_drop_packet() {
            return Err(AetherError::NetworkFault { reason_code: 1 });
        }

        let window_ref = slot.window.as_ref().ok_or(AetherError::NodeUnreachable {
            node_id: target_node_id,
        })?;

        window_ref.read_page(page_index, dest)
    }

    /// Executes a one-sided zero-copy RDMA / CXL 3.0 page write.
    ///
    /// Copies 4096 bytes directly from `src` into the remote node's registered memory window
    /// with strictly zero hot-path heap allocations and zero OS locks.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    /// - Zero OS-level locks on the hot path (AGENTS.md Rule 3.2).
    pub fn write_remote_page(
        &self,
        target_node_id: u64,
        page_index: usize,
        src: &[u8; 4096],
    ) -> Result<(), AetherError> {
        let idx = target_node_id as usize;
        if idx >= MAX_RDMA_NODES {
            return Err(AetherError::OutOfBounds {
                index: idx,
                limit: MAX_RDMA_NODES,
            });
        }

        let slot = self.nodes.get(idx).ok_or(AetherError::NodeUnreachable {
            node_id: target_node_id,
        })?;

        if !slot.active.load(Ordering::Acquire) || !slot.online.load(Ordering::Acquire) {
            return Err(AetherError::NodeUnreachable {
                node_id: target_node_id,
            });
        }

        if self.should_drop_packet() {
            return Err(AetherError::NetworkFault { reason_code: 1 });
        }

        let window_ref = slot.window.as_ref().ok_or(AetherError::NodeUnreachable {
            node_id: target_node_id,
        })?;

        window_ref.write_page(page_index, src)
    }
}

impl Default for RdmaFabric {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-allocation RDMA transport trait for one-sided remote memory access.
pub trait RdmaTransportBackend {
    /// Posts an asynchronous one-sided RDMA read into local `dst_page`.
    fn post_one_sided_read(
        &self,
        target_node: u16,
        page_slot: usize,
        dst_page: &mut TensorGraphPage,
    ) -> Result<(), AetherError>;

    /// Posts an asynchronous one-sided RDMA write from local `src_page`.
    fn post_one_sided_write(
        &self,
        target_node: u16,
        page_slot: usize,
        src_page: &TensorGraphPage,
    ) -> Result<(), AetherError>;
}

impl RdmaTransportBackend for RdmaFabric {
    fn post_one_sided_read(
        &self,
        target_node: u16,
        page_slot: usize,
        dst_page: &mut TensorGraphPage,
    ) -> Result<(), AetherError> {
        self.read_remote_page(
            target_node as u64,
            page_slot,
            dst_page.as_aligned_bytes_mut(),
        )
    }

    fn post_one_sided_write(
        &self,
        target_node: u16,
        page_slot: usize,
        src_page: &TensorGraphPage,
    ) -> Result<(), AetherError> {
        self.write_remote_page(target_node as u64, page_slot, src_page.as_aligned_bytes())
    }
}
