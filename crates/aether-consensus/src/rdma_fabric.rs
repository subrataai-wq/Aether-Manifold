//! Project Aether-Manifold: RDMA / CXL 3.0 Zero-Copy Remote Page Fetching Fabric
//!
//! Provides an asynchronous, one-sided zero-copy remote page transfer abstraction
//! simulating RDMA RoCEv2 and CXL 3.0 Type-3 shared memory fabrics with configurable
//! latency and deterministic fault injection. Strictly zero heap allocations on the
//! transfer hot path (AGENTS.md Rule 3.1).

use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use aether_core::error::AetherError;
use aether_core::page::TensorGraphPage;

/// Maximum number of cluster nodes supported by the virtual RDMA fabric.
pub const MAX_RDMA_NODES: usize = 64;

/// Memory region representing an exported 4KB-page aligned memory window on a node.
pub struct RdmaMemoryWindow {
    pages: Vec<TensorGraphPage>,
}

impl RdmaMemoryWindow {
    /// Creates a pre-allocated memory window capable of hosting `num_pages` 4KB pages.
    pub fn with_capacity(num_pages: usize) -> Self {
        let pages = vec![bytemuck::Zeroable::zeroed(); num_pages];
        Self { pages }
    }

    /// Number of pages hosted in this memory window.
    #[inline]
    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    /// Reads a 4KB page directly into `dest` with zero heap allocation.
    #[inline]
    pub fn read_page(&self, page_index: usize, dest: &mut [u8; 4096]) -> Result<(), AetherError> {
        let page = self.pages.get(page_index).ok_or(AetherError::OutOfBounds {
            index: page_index,
            limit: self.pages.len(),
        })?;
        let bytes: &[u8; 4096] = page.as_aligned_bytes();
        dest.copy_from_slice(bytes);
        Ok(())
    }

    /// Writes a 4KB page directly from `src` with zero heap allocation.
    #[inline]
    pub fn write_page(&mut self, page_index: usize, src: &[u8; 4096]) -> Result<(), AetherError> {
        let limit = self.pages.len();
        let page = self
            .pages
            .get_mut(page_index)
            .ok_or(AetherError::OutOfBounds {
                index: page_index,
                limit,
            })?;
        let bytes: &mut [u8; 4096] = page.as_aligned_bytes_mut();
        bytes.copy_from_slice(src);
        Ok(())
    }
}

/// Node registration state in the RDMA fabric.
struct FabricNodeSlot {
    active: AtomicBool,
    online: AtomicBool,
    window: Option<Arc<SharedWindow>>,
}

// Internal wrapper to allow lock-free/interior mutable page updates across the fabric simulation
struct SharedWindow {
    raw_window: std::sync::RwLock<RdmaMemoryWindow>,
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
            slot.window = Some(Arc::new(SharedWindow {
                raw_window: std::sync::RwLock::new(window),
            }));
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
        if let Some(slot) = self.nodes.get(idx) {
            slot.online.store(online, Ordering::Release);
        }
        Ok(())
    }

    /// Configures synthetic packet drop rate (0.0 = 0%, 1.0 = 100%).
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
    /// into `dest` with strictly zero hot-path heap allocations.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
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

        let window = window_ref
            .raw_window
            .read()
            .map_err(|_| AetherError::IoFault { code: 11 })?;

        window.read_page(page_index, dest)
    }

    /// Executes a one-sided zero-copy RDMA / CXL 3.0 page write.
    ///
    /// Copies 4096 bytes directly from `src` into the remote node's registered memory window
    /// with strictly zero hot-path heap allocations.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
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

        let mut window = window_ref
            .raw_window
            .write()
            .map_err(|_| AetherError::IoFault { code: 11 })?;

        window.write_page(page_index, src)
    }
}

impl Default for RdmaFabric {
    fn default() -> Self {
        Self::new()
    }
}
