//! Project Aether-Manifold: 4KB Co-Located Tensor-Graph Page Allocator (`page_arena.rs`)
//!
//! Manages pre-allocated slabs of 4096-byte aligned `TensorGraphPage` instances.
//! Guarantees zero heap allocations on page allocation, mutation, and 1-hop traversal packing.

use aether_core::error::AetherError;
use aether_core::hlc::HlcTimestamp;
use aether_core::page::TensorGraphPage;
use aether_manifold::at_rq::CompactNeighborCode;
use bytemuck::Zeroable;

/// Pre-allocated aligned page arena managing a contiguous slab of 4KB `TensorGraphPage`s.
pub struct AlignedPageArena {
    pages: Vec<TensorGraphPage>,
    free_slots: Vec<usize>,
    allocated_count: usize,
}

impl AlignedPageArena {
    /// Cold-path initialization: pre-allocates `num_pages` 4KB-aligned `TensorGraphPage`s.
    pub fn with_capacity(num_pages: usize) -> Self {
        let mut pages = Vec::with_capacity(num_pages);
        pages.resize_with(num_pages, TensorGraphPage::zeroed);
        let mut free_slots = Vec::with_capacity(num_pages);
        for i in (0..num_pages).rev() {
            free_slots.push(i);
        }

        Self {
            pages,
            free_slots,
            allocated_count: 0,
        }
    }

    /// Total capacity in 4KB pages.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.pages.len()
    }

    /// Number of currently allocated active page slots.
    #[inline]
    pub fn allocated_count(&self) -> usize {
        self.allocated_count
    }

    /// Number of available free page slots.
    #[inline]
    pub fn free_count(&self) -> usize {
        self.free_slots.len()
    }

    /// Allocate a 4KB `TensorGraphPage` slot inside the pre-allocated arena.
    ///
    /// # Hard Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    pub fn allocate_page(
        &mut self,
        page_id: u64,
        lsn: HlcTimestamp,
        curvature_bits: u64,
    ) -> Result<usize, AetherError> {
        let slot = match self.free_slots.pop() {
            Some(s) => s,
            None => {
                return Err(AetherError::ScratchpadExhausted {
                    requested: 1,
                    available: 0,
                });
            }
        };

        let page = match self.pages.get_mut(slot) {
            Some(p) => p,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: slot,
                    limit: self.pages.len(),
                });
            }
        };

        *page = TensorGraphPage::zeroed();
        page.header.page_id = page_id;
        page.header.lsn = lsn;
        page.header.manifold_curvature_bits = curvature_bits;
        page.header.update_checksum();

        self.allocated_count += 1;
        Ok(slot)
    }

    /// Return a page slot back to the arena.
    ///
    /// # Hard Invariants
    /// - Strictly 0 heap allocations on the hot path.
    pub fn deallocate_page(&mut self, slot: usize) -> Result<(), AetherError> {
        if slot >= self.pages.len() {
            return Err(AetherError::OutOfBounds {
                index: slot,
                limit: self.pages.len(),
            });
        }

        let page = match self.pages.get_mut(slot) {
            Some(p) => p,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: slot,
                    limit: self.pages.len(),
                });
            }
        };

        *page = TensorGraphPage::zeroed();
        self.free_slots.push(slot);
        self.allocated_count = self.allocated_count.saturating_sub(1);
        Ok(())
    }

    /// Retrieve an immutable reference to an allocated page slot.
    #[inline]
    pub fn get_page(&self, slot: usize) -> Result<&TensorGraphPage, AetherError> {
        match self.pages.get(slot) {
            Some(p) => Ok(p),
            None => Err(AetherError::OutOfBounds {
                index: slot,
                limit: self.pages.len(),
            }),
        }
    }

    /// Retrieve a mutable reference to an allocated page slot.
    #[inline]
    pub fn get_page_mut(&mut self, slot: usize) -> Result<&mut TensorGraphPage, AetherError> {
        let limit = self.pages.len();
        match self.pages.get_mut(slot) {
            Some(p) => Ok(p),
            None => Err(AetherError::OutOfBounds { index: slot, limit }),
        }
    }

    /// Populate a 4KB `TensorGraphPage` slot with primary vertex AT-RQ compressed codebooks,
    /// active neighbor node IDs, 32-byte `CompactNeighborCode`s, and Cellular Sheaf restriction
    /// maps, updating `active_vertex_count` and CRC32C checksum in-place with 0 heap allocations.
    ///
    /// # Hard Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    pub fn pack_1hop_neighborhood(
        &mut self,
        slot: usize,
        primary_codebook: &[u8; 256],
        neighbor_ids: &[u64],
        quantized_neighbors: &[CompactNeighborCode],
        sheaf_edge_weights: &[u8],
    ) -> Result<(), AetherError> {
        let neighbor_count = neighbor_ids.len();
        let quantized_count = quantized_neighbors.len();

        if neighbor_count > 64 {
            return Err(AetherError::InvalidBufferSize {
                actual: neighbor_count,
                expected: 64,
            });
        }
        if quantized_count > 64 {
            return Err(AetherError::InvalidBufferSize {
                actual: quantized_count,
                expected: 64,
            });
        }
        if neighbor_count != quantized_count {
            return Err(AetherError::InvalidBufferSize {
                actual: quantized_count,
                expected: neighbor_count,
            });
        }
        if sheaf_edge_weights.len() > 1216 {
            return Err(AetherError::InvalidBufferSize {
                actual: sheaf_edge_weights.len(),
                expected: 1216,
            });
        }

        let limit = self.pages.len();
        let page = match self.pages.get_mut(slot) {
            Some(p) => p,
            None => return Err(AetherError::OutOfBounds { index: slot, limit }),
        };

        // 1. Primary vertex AT-RQ codebooks (256B)
        page.at_rq_codebooks.copy_from_slice(primary_codebook);

        // 2. Neighbor node IDs (512B: up to 64 x u64)
        page.neighbor_ids.fill(0);
        if let Some(dest) = page.neighbor_ids.get_mut(..neighbor_count) {
            dest.copy_from_slice(neighbor_ids);
        }

        // 3. Quantized neighbor representations (2048B: 64 x 32B)
        page.quantized_neighbors.fill(0);
        for (i, code) in quantized_neighbors.iter().enumerate() {
            let start = i * 32;
            let end = start + 32;
            if let Some(dest) = page.quantized_neighbors.get_mut(start..end) {
                dest.copy_from_slice(code.as_bytes());
            }
        }

        // 4. Cellular Sheaf restriction maps / BF16 edge weights (1216B)
        page.sheaf_edge_weights.fill(0);
        if let Some(dest) = page.sheaf_edge_weights.get_mut(..sheaf_edge_weights.len()) {
            dest.copy_from_slice(sheaf_edge_weights);
        }

        // 5. Update active count and page header CRC32C checksum
        page.header.active_vertex_count = neighbor_count as u32;
        page.header.update_checksum();

        Ok(())
    }
}

