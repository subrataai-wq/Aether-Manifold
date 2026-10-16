//! Project Aether-Manifold: 4KB Co-Located Tensor-Graph Page & 64B Header Layout
//!
//! Enforces exact 4096-byte NVMe page and 64-byte L1 cache-line layouts with zero
//! implicit compiler padding and zero-copy byte casting sound under bytemuck and zerocopy.

use bytemuck::{Pod, Zeroable};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::error::AetherError;
use crate::hlc::HlcTimestamp;

/// 64-byte cache-line aligned page header.
///
/// Layout:
/// - 00..04 (4B): CRC32C Checksum
/// - 04..08 (4B): Active Vertex Count
/// - 08..16 (8B): LSN (Hybrid Logical Clock Timestamp)
/// - 16..24 (8B): Physical Page Identifier
/// - 24..32 (8B): Riemannian Manifold Curvature Parameters (packed bits)
/// - 32..64 (32B): Intra-page CSR Offset Table (16 x u16)
#[repr(C, align(64))]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable, FromBytes, IntoBytes, KnownLayout, Immutable,
)]
pub struct PageHeader {
    /// CRC32C checksum of the page header payload (bytes 4..64).
    pub crc32c_checksum: u32,
    /// Number of active vertices stored within this 4KB page.
    pub active_vertex_count: u32,
    /// Causal MVCC timestamp / Log Sequence Number.
    pub lsn: HlcTimestamp,
    /// Page identifier within the KLSM run directory.
    pub page_id: u64,
    /// Curvature parameters for product manifold components (e.g. kappa, tau).
    pub manifold_curvature_bits: u64,
    /// Compressed Sparse Row offset table for local intra-page adjacency list.
    pub csr_offset_table: [u16; 16],
}

static_assertions::const_assert_eq!(core::mem::size_of::<PageHeader>(), 64);
static_assertions::const_assert_eq!(core::mem::align_of::<PageHeader>(), 64);

impl PageHeader {
    /// Compute CRC32C checksum across header bytes 4..64 (excluding checksum field).
    #[inline]
    pub fn compute_checksum(&self) -> u32 {
        let bytes = bytemuck::bytes_of(self);
        match bytes.get(4..64) {
            Some(payload) => {
                let mut hasher = crc32fast::Hasher::new();
                hasher.update(payload);
                hasher.finalize()
            }
            None => 0,
        }
    }

    /// Recompute and update the `crc32c_checksum` field in-place.
    #[inline]
    pub fn update_checksum(&mut self) {
        self.crc32c_checksum = self.compute_checksum();
    }

    /// Verify the integrity of the header against its stored CRC32C checksum.
    #[inline]
    pub fn verify_checksum(&self) -> Result<(), AetherError> {
        let calculated = self.compute_checksum();
        if self.crc32c_checksum == calculated {
            Ok(())
        } else {
            Err(AetherError::PageCorruption {
                expected: self.crc32c_checksum,
                calculated,
            })
        }
    }
}

/// 4096-byte NVMe physical page layout co-locating node vectors, topology, and sheaf morphisms.
///
/// Layout per roadmap.md Section 3.2:
/// - 0000..0064 (64B): Cache-line aligned PageHeader
/// - 0064..0320 (256B): Primary vertex AT-RQ compressed codebooks
/// - 0320..0832 (512B): 64 neighbor vertex IDs (u64 x 64)
/// - 0832..2880 (2048B): Quantized neighbor vectors (64 neighbors x 32B)
/// - 2880..4096 (1216B): Cellular Sheaf restriction maps / BF16 edge weights
#[repr(C, align(4096))]
#[derive(Clone, Copy, Pod, Zeroable, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct TensorGraphPage {
    /// Cache-line aligned page header (64 bytes).
    pub header: PageHeader,
    /// AT-RQ compressed sub-vector codebooks / centroid residual code (256 bytes).
    pub at_rq_codebooks: [u8; 256],
    /// 64 immediate neighbor node identifiers (512 bytes).
    pub neighbor_ids: [u64; 64],
    /// Quantized neighbor vector representations for 1-hop traversal (2048 bytes).
    pub quantized_neighbors: [u8; 2048],
    /// Cellular Sheaf restriction morphism edge weights / BF16 maps (1216 bytes).
    pub sheaf_edge_weights: [u8; 1216],
}

static_assertions::const_assert_eq!(core::mem::size_of::<TensorGraphPage>(), 4096);
static_assertions::const_assert_eq!(core::mem::align_of::<TensorGraphPage>(), 4096);

impl TensorGraphPage {
    /// Return the page as a 4096-byte aligned byte slice reference.
    #[inline]
    pub fn as_aligned_bytes(&self) -> &[u8; 4096] {
        bytemuck::cast_ref(self)
    }

    /// Return the page as a mutable 4096-byte aligned byte slice reference.
    #[inline]
    pub fn as_aligned_bytes_mut(&mut self) -> &mut [u8; 4096] {
        bytemuck::cast_mut(self)
    }

    /// Compute CRC32C over the full 4096-byte page payload (bytes 4..4096).
    #[inline]
    pub fn compute_full_page_checksum(&self) -> u32 {
        let bytes = self.as_aligned_bytes();
        match bytes.get(4..4096) {
            Some(payload) => {
                let mut hasher = crc32fast::Hasher::new();
                hasher.update(payload);
                hasher.finalize()
            }
            None => 0,
        }
    }

    /// Update page checksum from header.
    #[inline]
    pub fn update_header_checksum(&mut self) {
        self.header.update_checksum();
    }

    /// Verify header CRC32C checksum integrity.
    #[inline]
    pub fn verify_header_checksum(&self) -> Result<(), AetherError> {
        self.header.verify_checksum()
    }

    /// In-place zero-copy cast from an aligned byte slice into `&TensorGraphPage`.
    ///
    /// Returns `Err(AetherError::InvalidBufferSize)` if `slice.len() != 4096`.
    /// Returns `Err(AetherError::InvalidAlignment)` if `slice.as_ptr()` is not aligned to 4096 bytes.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)]
    pub fn from_aligned_bytes(slice: &[u8]) -> Result<&Self, AetherError> {
        let required_size = core::mem::size_of::<Self>();
        if slice.len() != required_size {
            return Err(AetherError::InvalidBufferSize {
                actual: slice.len(),
                expected: required_size,
            });
        }
        let ptr = slice.as_ptr();
        let addr = ptr as usize;
        let required_align = core::mem::align_of::<Self>();
        if !addr.is_multiple_of(required_align) {
            return Err(AetherError::InvalidAlignment {
                ptr: addr,
                required_align,
            });
        }
        // Safety:
        // 1. Length is verified to equal size_of::<TensorGraphPage>() (4096 bytes).
        // 2. Alignment is verified to be 4096 bytes.
        // 3. TensorGraphPage derives Pod / FromBytes, so all bit patterns are sound.
        let page_ptr = ptr.cast::<Self>();
        let page_ref = unsafe { &*page_ptr };
        Ok(page_ref)
    }

    /// In-place zero-copy mutable cast from an aligned byte slice into `&mut TensorGraphPage`.
    ///
    /// Returns `Err(AetherError::InvalidBufferSize)` if `slice.len() != 4096`.
    /// Returns `Err(AetherError::InvalidAlignment)` if `slice.as_ptr()` is not aligned to 4096 bytes.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)]
    pub fn from_aligned_bytes_mut(slice: &mut [u8]) -> Result<&mut Self, AetherError> {
        let required_size = core::mem::size_of::<Self>();
        if slice.len() != required_size {
            return Err(AetherError::InvalidBufferSize {
                actual: slice.len(),
                expected: required_size,
            });
        }
        let ptr = slice.as_mut_ptr();
        let addr = ptr as usize;
        let required_align = core::mem::align_of::<Self>();
        if !addr.is_multiple_of(required_align) {
            return Err(AetherError::InvalidAlignment {
                ptr: addr,
                required_align,
            });
        }
        // Safety:
        // 1. Length is verified to equal size_of::<TensorGraphPage>() (4096 bytes).
        // 2. Alignment is verified to be 4096 bytes.
        // 3. TensorGraphPage derives Pod / FromBytes, so all bit patterns are sound.
        let page_ptr = ptr.cast::<Self>();
        let page_mut = unsafe { &mut *page_ptr };
        Ok(page_mut)
    }
}
