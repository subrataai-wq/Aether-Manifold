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
/// - 32..62 (30B): Intra-page CSR Offset Table (15 x u16)
/// - 62..64 (2B): Page Control Flags (e.g. FP16 spillover active)
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
    /// 30-byte intra-page Compressed Sparse Row (CSR) offset table (15 x u16).
    ///
    /// Stores the index boundary offsets into `neighbor_ids`, `quantized_neighbors`, and
    /// `sheaf_edge_weights` when multiple small sub-entities / sub-vertices (up to 16) are
    /// packed into a single 4KB `TensorGraphPage`.
    ///
    /// Memory Layout & Alignment:
    /// - Occupies bytes 32..62 of `PageHeader`.
    /// - Followed immediately by 2 bytes of `flags` (bytes 62..64).
    /// - Preserves the exact 64-byte `PageHeader` footprint (`#[repr(C, align(64))]`)
    ///   with strictly zero implicit compiler padding bytes.
    pub csr_offset_table: [u16; 15],
    /// Page control flags bitmask.
    pub flags: u16,
}

/// Flag indicating that this page contains FP16 residual spillover in bytes 1184..1216.
pub const PAGE_FLAG_HAS_FP16_SPILLOVER: u16 = 1 << 0;

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

    /// Extract primary vertex stalk $x_u \in \mathbb{R}^8$ from bytes 1024..1056 of `sheaf_edge_weights`.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    #[inline]
    pub fn primary_stalk(&self) -> [f32; 8] {
        let mut stalk = [0.0f32; 8];
        if let Some(bytes) = self.sheaf_edge_weights.get(1024..1056) {
            let (chunks, _) = bytes.as_chunks::<4>();
            for (i, chunk) in chunks.iter().enumerate() {
                if let Some(slot) = stalk.get_mut(i) {
                    *slot = f32::from_le_bytes(*chunk);
                }
            }
        }
        stalk
    }

    /// Write primary vertex stalk $x_u \in \mathbb{R}^8$ into bytes 1024..1056 of `sheaf_edge_weights`.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    #[inline]
    pub fn set_primary_stalk(&mut self, stalk: &[f32; 8]) {
        if let Some(bytes) = self.sheaf_edge_weights.get_mut(1024..1056) {
            for (i, val) in stalk.iter().enumerate() {
                let le = val.to_le_bytes();
                let offset = i * 4;
                if let Some(slot) = bytes.get_mut(offset..offset + 4) {
                    slot.copy_from_slice(&le);
                }
            }
        }
    }

    /// Write up to 16 high-curvature FP16 residual components into bytes 1184..1216 of `sheaf_edge_weights`
    /// (u16 slots 592..608) and set `PAGE_FLAG_HAS_FP16_SPILLOVER` in header flags.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    #[inline]
    pub fn write_fp16_residual_spillover(&mut self, residual: &[f32]) {
        let count = residual.len().min(16);
        if let Some(target_bytes) = self.sheaf_edge_weights.get_mut(1184..1216) {
            for i in 0..16 {
                let bits = if i < count {
                    f32_to_f16_bits(residual.get(i).copied().unwrap_or(0.0))
                } else {
                    0u16
                };
                let le = bits.to_le_bytes();
                let offset = i * 2;
                if let Some(slot) = target_bytes.get_mut(offset..offset + 2) {
                    slot.copy_from_slice(&le);
                }
            }
            self.header.flags |= PAGE_FLAG_HAS_FP16_SPILLOVER;
            self.header.update_checksum();
        }
    }

    /// Read up to 16 high-curvature FP16 residual components from bytes 1184..1216 of `sheaf_edge_weights`
    /// into `out`.
    ///
    /// Returns `false` if `PAGE_FLAG_HAS_FP16_SPILLOVER` is not set; otherwise decodes and returns `true`.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    #[inline]
    pub fn read_fp16_residual_spillover(&self, out: &mut [f32; 16]) -> bool {
        if (self.header.flags & PAGE_FLAG_HAS_FP16_SPILLOVER) == 0 {
            return false;
        }

        if let Some(src_bytes) = self.sheaf_edge_weights.get(1184..1216) {
            for i in 0..16 {
                let offset = i * 2;
                if let Some(b) = src_bytes.get(offset..offset + 2) {
                    if let Ok(arr) = <[u8; 2]>::try_from(b) {
                        let bits = u16::from_le_bytes(arr);
                        if let Some(slot) = out.get_mut(i) {
                            *slot = f16_bits_to_f32(bits);
                        }
                    }
                }
            }
            true
        } else {
            false
        }
    }
}

/// Convert an f32 into IEEE-754 binary16 (half-precision) bits.
///
/// Clamps overflow to finite max/min without generating unwanted NaNs,
/// and handles zero, subnormals, and normal floats cleanly.
#[inline]
pub fn f32_to_f16_bits(val: f32) -> u16 {
    let fbits = val.to_bits();
    let sign = ((fbits >> 16) & 0x8000) as u16;
    let exp = ((fbits >> 23) & 0xFF) as i32;
    let frac = fbits & 0x007F_FFFF;

    if exp == 0xFF {
        if frac != 0 {
            return sign | 0x7E00;
        } else {
            return sign | 0x7C00;
        }
    }

    let new_exp = exp - 127 + 15;

    if new_exp >= 31 {
        // Clamp to max finite f16 (65504.0)
        return sign | 0x7BFF;
    }

    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let full_mantissa = frac | 0x0080_0000;
        let shift = (14 - new_exp) as u32;
        let sub_frac = (full_mantissa >> shift) as u16;
        return sign | sub_frac;
    }

    let exp_bits = ((new_exp as u16) & 0x1F) << 10;
    let frac_bits = ((frac >> 13) & 0x03FF) as u16;
    sign | exp_bits | frac_bits
}

/// Convert IEEE-754 binary16 (half-precision) bits into an f32.
#[inline]
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x03FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign);
        }
        let mut f = frac;
        let mut shift = 0;
        while (f & 0x0400) == 0 {
            f <<= 1;
            shift += 1;
        }
        f &= 0x03FF;
        let new_exp = (127 - 15 + 1 - shift) << 23;
        let new_frac = f << 13;
        return f32::from_bits(sign | new_exp | new_frac);
    }

    if exp == 31 {
        if frac != 0 {
            return f32::from_bits(sign | 0x7F80_0000 | (frac << 13));
        } else {
            return f32::from_bits(sign | 0x7F80_0000);
        }
    }

    let new_exp = (exp + 112) << 23;
    let new_frac = frac << 13;
    f32::from_bits(sign | new_exp | new_frac)
}
