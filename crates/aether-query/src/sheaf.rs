//! Project Aether-Manifold: Cellular Sheaf Stalks & Orthogonal Edge Morphisms
//!
//! Implements discrete Cellular Sheaf theory over the graph topology:
//! - Vertex Stalks $\mathcal{F}(v) \cong \mathbb{R}^D$ and Edge Stalks $\mathcal{F}(e) \cong \mathbb{R}^D$.
//! - Orthogonal restriction maps $\mathcal{F}_{v \trianglelefteq e} : \mathcal{F}(v) \to \mathcal{F}(e)$
//!   constructed via Householder reflectors $H = I - 2 w w^T$ to guarantee spectral
//!   norm bound $\|\mathcal{F}_{v \trianglelefteq e}\|_2 = 1$ identically (Rule 4.2).
//! - Zero-copy serialization into 4KB `TensorGraphPage::sheaf_edge_weights` using BF16 packing.
//! - Strictly 0 heap allocations and no unwrap/panic primitives.

use aether_core::error::AetherError;
use aether_core::page::TensorGraphPage;
use half::bf16;

/// Default stalk dimensionality for compact cellular sheaf maps.
pub const DEFAULT_STALK_DIM: usize = 8;
/// Maximum number of neighbors supported per 4KB tensor-graph page.
pub const MAX_NEIGHBORS_PER_PAGE: usize = 64;
/// Byte size of a single packed 8-dim Householder reflector in BF16 (8 x 2B = 16B).
pub const PACKED_RESTRICTION_BYTES: usize = DEFAULT_STALK_DIM * core::mem::size_of::<bf16>();

/// Vertex Stalk $\mathcal{F}(v) \in \mathbb{R}^D$: local feature/tangent vector space.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VertexStalk<const D: usize> {
    /// Coordinate values of the stalk vector.
    pub components: [f32; D],
}

impl<const D: usize> Default for VertexStalk<D> {
    #[inline]
    fn default() -> Self {
        Self {
            components: [0.0; D],
        }
    }
}

impl<const D: usize> VertexStalk<D> {
    /// Create a new vertex stalk from coordinate array.
    #[inline]
    pub const fn new(components: [f32; D]) -> Self {
        Self { components }
    }

    /// Compute Euclidean L2 norm of the stalk vector.
    #[inline]
    pub fn norm(&self) -> f32 {
        let mut sum = 0.0f32;
        for c in &self.components {
            sum += c * c;
        }
        libm::sqrtf(sum)
    }
}

/// Edge Stalk $\mathcal{F}(e) \in \mathbb{R}^D$: interaction space along relation $e = (u, v)$.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeStalk<const D: usize> {
    /// Coordinate values of the edge stalk vector.
    pub components: [f32; D],
}

impl<const D: usize> Default for EdgeStalk<D> {
    #[inline]
    fn default() -> Self {
        Self {
            components: [0.0; D],
        }
    }
}

impl<const D: usize> EdgeStalk<D> {
    /// Create a new edge stalk from coordinate array.
    #[inline]
    pub const fn new(components: [f32; D]) -> Self {
        Self { components }
    }
}

/// Orthogonal Sheaf Restriction Map $\mathcal{F}_{v \trianglelefteq e} : \mathcal{F}(v) \to \mathcal{F}(e)$.
///
/// Encoded via a unit Householder reflection vector $w \in \mathbb{S}^{D-1}$ such that:
/// $$H_w x = x - 2 \langle w, x \rangle w$$
///
/// Guarantees:
/// 1. $H_w^T = H_w$ (Symmetric)
/// 2. $H_w^T H_w = I$ (Orthogonal / Isometric: $\|H_w x\|_2 = \|x\|_2$)
/// 3. Spectral norm $\|H_w\|_2 = 1$ identically, satisfying Rule 4.2 without numerical drift.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SheafRestrictionMap<const D: usize> {
    /// Unit Householder reflection vector $w$.
    pub reflector_w: [f32; D],
    /// Optional edge attenuation / capacity factor $\alpha \in [0.0, 1.0]$.
    pub attenuation: f32,
}

impl<const D: usize> Default for SheafRestrictionMap<D> {
    #[inline]
    fn default() -> Self {
        // Default is identity: zero reflector vector yields H x = x
        Self {
            reflector_w: [0.0; D],
            attenuation: 1.0,
        }
    }
}

impl<const D: usize> SheafRestrictionMap<D> {
    /// Construct a normalized Householder restriction map from a direction vector.
    ///
    /// If $\|w\|_2 < 10^{-7}$, returns the identity map ($w = \mathbf{0}$).
    pub fn from_reflector(reflector: &[f32; D], attenuation: f32) -> Self {
        let mut sum_sq = 0.0f32;
        for val in reflector {
            sum_sq += val * val;
        }
        let norm = libm::sqrtf(sum_sq);
        let mut normalized = [0.0f32; D];

        if norm > 1e-7 {
            let inv_norm = 1.0f32 / norm;
            for (dst, src) in normalized.iter_mut().zip(reflector.iter()) {
                *dst = *src * inv_norm;
            }
        }

        Self {
            reflector_w: normalized,
            attenuation: attenuation.clamp(0.0, 1.0),
        }
    }

    /// Construct an identity restriction map ($\mathcal{F}_{v \trianglelefteq e} = I$).
    #[inline]
    pub fn identity() -> Self {
        Self {
            reflector_w: [0.0; D],
            attenuation: 1.0,
        }
    }

    /// Apply the restriction map $\mathcal{F}_{v \trianglelefteq e} x$ in $O(D)$ operations.
    ///
    /// $$y = \alpha \cdot \left( x - 2 \langle w, x \rangle w \right)$$
    #[inline]
    pub fn apply(&self, x: &VertexStalk<D>, out: &mut EdgeStalk<D>) {
        // Inner product alpha = <w, x>
        let mut dot = 0.0f32;
        for (w_i, x_i) in self.reflector_w.iter().zip(x.components.iter()) {
            dot += w_i * x_i;
        }

        let two_dot = 2.0f32 * dot;
        let scale = self.attenuation;

        // y = scale * (x - 2 * dot * w)
        for ((y_i, x_i), w_i) in out
            .components
            .iter_mut()
            .zip(x.components.iter())
            .zip(self.reflector_w.iter())
        {
            *y_i = scale * (*x_i - two_dot * *w_i);
        }
    }

    /// Apply the adjoint (transpose) restriction map $\mathcal{F}_{v \trianglelefteq e}^T y$.
    ///
    /// For Householder reflectors, $H_w^T = H_w$, so the adjoint is identical to apply.
    #[inline]
    pub fn apply_adjoint(&self, y: &EdgeStalk<D>, out: &mut VertexStalk<D>) {
        let mut dot = 0.0f32;
        for (w_i, y_i) in self.reflector_w.iter().zip(y.components.iter()) {
            dot += w_i * y_i;
        }

        let two_dot = 2.0f32 * dot;
        let scale = self.attenuation;

        for ((x_i, y_i), w_i) in out
            .components
            .iter_mut()
            .zip(y.components.iter())
            .zip(self.reflector_w.iter())
        {
            *x_i = scale * (*y_i - two_dot * *w_i);
        }
    }

    /// Pack the 8-dim Householder reflector into 16 bytes of BF16.
    pub fn pack_bf16(&self, dest: &mut [u8]) -> Result<(), AetherError> {
        if dest.len() < PACKED_RESTRICTION_BYTES {
            return Err(AetherError::InvalidBufferSize {
                actual: dest.len(),
                expected: PACKED_RESTRICTION_BYTES,
            });
        }

        for (i, val) in self.reflector_w.iter().enumerate() {
            let bf = bf16::from_f32(*val);
            let bytes = bf.to_le_bytes();
            let offset = i * 2;
            if let Some(target) = dest.get_mut(offset..offset + 2) {
                target.copy_from_slice(&bytes);
            }
        }

        Ok(())
    }

    /// Unpack an 8-dim Householder reflector from 16 bytes of BF16.
    pub fn unpack_bf16(src: &[u8], attenuation: f32) -> Result<Self, AetherError> {
        if src.len() < PACKED_RESTRICTION_BYTES {
            return Err(AetherError::InvalidBufferSize {
                actual: src.len(),
                expected: PACKED_RESTRICTION_BYTES,
            });
        }

        let mut reflector = [0.0f32; D];
        for (i, target) in reflector.iter_mut().enumerate() {
            let offset = i * 2;
            let bytes: [u8; 2] = match src.get(offset..offset + 2) {
                Some(slice) => slice.try_into().unwrap_or_default(),
                None => [0u8; 2],
            };
            let bf = bf16::from_le_bytes(bytes);
            *target = bf.to_f32();
        }

        Ok(Self::from_reflector(&reflector, attenuation))
    }
}

impl VertexStalk<DEFAULT_STALK_DIM> {
    /// Reconstruct an 8-dim neighbor stalk from co-located 32B `CompactNeighborCode` bytes.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    #[inline]
    pub fn from_neighbor_bytes(bytes: &[u8; 32], primary_stalk: &Self) -> Self {
        let mut components = [0.0f32; DEFAULT_STALK_DIM];
        let scale_bytes: [u8; 4] = match bytes.get(4..8) {
            Some(s) => s.try_into().unwrap_or([0u8; 4]),
            None => [0u8; 4],
        };
        let scale = f32::from_le_bytes(scale_bytes);
        let safe_scale = if scale.is_finite() && scale > 1e-6 {
            scale
        } else {
            1.0
        };

        for (i, comp) in components.iter_mut().enumerate() {
            let code_byte = bytes.get(8 + i).copied().unwrap_or(0);
            let delta = ((code_byte as f32) / 127.5 - 1.0) * safe_scale * 0.1;
            let base = primary_stalk.components.get(i).copied().unwrap_or(0.0);
            *comp = base + delta;
        }
        Self { components }
    }
}

/// Byte offset where primary vertex stalk starts in `TensorGraphPage::sheaf_edge_weights` (1024 bytes).
pub const SHEAF_PRIMARY_STALK_OFFSET: usize = MAX_NEIGHBORS_PER_PAGE * PACKED_RESTRICTION_BYTES;
/// Byte offset where scalar attenuation edge weights start in `TensorGraphPage::sheaf_edge_weights` (1056 bytes).
pub const SHEAF_WEIGHTS_OFFSET: usize = 1056;

/// Helper to serialize all 64 neighbor restriction maps into `TensorGraphPage::sheaf_edge_weights`.
pub fn pack_page_sheaf_morphisms(
    page: &mut TensorGraphPage,
    morphisms: &[SheafRestrictionMap<DEFAULT_STALK_DIM>],
    scalar_weights: &[f32],
) -> Result<(), AetherError> {
    let count = morphisms.len().min(MAX_NEIGHBORS_PER_PAGE);

    for (idx, morphism) in morphisms.iter().take(count).enumerate() {
        let offset = idx * PACKED_RESTRICTION_BYTES;
        let end = offset + PACKED_RESTRICTION_BYTES;
        if let Some(slot) = page.sheaf_edge_weights.get_mut(offset..end) {
            morphism.pack_bf16(slot)?;
        }
    }

    // Pack scalar edge weights into bytes 1056..1184 (64 x 2B BF16 = 128B),
    // reserving bytes 1024..1056 (32B) for the primary vertex stalk.
    for (idx, weight) in scalar_weights
        .iter()
        .take(MAX_NEIGHBORS_PER_PAGE)
        .enumerate()
    {
        let offset = SHEAF_WEIGHTS_OFFSET + idx * 2;
        let end = offset + 2;
        if let Some(slot) = page.sheaf_edge_weights.get_mut(offset..end) {
            let bf = bf16::from_f32(*weight);
            slot.copy_from_slice(&bf.to_le_bytes());
        }
    }

    Ok(())
}

/// Helper to read neighbor restriction map `idx` from a `TensorGraphPage`.
pub fn unpack_page_sheaf_morphism(
    page: &TensorGraphPage,
    neighbor_idx: usize,
) -> Result<SheafRestrictionMap<DEFAULT_STALK_DIM>, AetherError> {
    if neighbor_idx >= MAX_NEIGHBORS_PER_PAGE {
        return Err(AetherError::OutOfBounds {
            index: neighbor_idx,
            limit: MAX_NEIGHBORS_PER_PAGE,
        });
    }

    let offset = neighbor_idx * PACKED_RESTRICTION_BYTES;
    let end = offset + PACKED_RESTRICTION_BYTES;
    let slice = match page.sheaf_edge_weights.get(offset..end) {
        Some(s) => s,
        None => {
            return Err(AetherError::InvalidBufferSize {
                actual: 0,
                expected: PACKED_RESTRICTION_BYTES,
            });
        }
    };

    // Read scalar weight for attenuation from SHEAF_WEIGHTS_OFFSET (1056)
    let w_offset = SHEAF_WEIGHTS_OFFSET + neighbor_idx * 2;
    let attenuation = match page.sheaf_edge_weights.get(w_offset..w_offset + 2) {
        Some(bytes) => {
            let arr: [u8; 2] = bytes.try_into().unwrap_or_default();
            bf16::from_le_bytes(arr).to_f32()
        }
        None => 1.0f32,
    };

    SheafRestrictionMap::unpack_bf16(slice, attenuation)
}
