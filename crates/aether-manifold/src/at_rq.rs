//! Project Aether-Manifold: Anisotropic Tangent-Space Residual Quantizer (AT-RQ)
//!
//! Implements the 32-byte Compact Neighbor Code packed layout for `TensorGraphPage`,
//! anisotropic parallel/orthogonal residual penalty loss, and curvature-corrected
//! Asymmetric Distance Computation (ADC) via Second-Order Rauch/Jacobi expansion.
//!
//! In strict compliance with roadmap.md Sections 2.4, 3.2, 3.4, and AGENTS.md Rule 3.1
//! (strictly 0 hot-path heap allocations).

use aether_core::error::AetherError;
use aether_core::page::TensorGraphPage;
use crate::simd::{simd_dot_product, simd_euclidean_distance_squared};

/// Exactly 32-byte packed representation of a quantized neighbor vector for TensorGraphPage.
///
/// Layout (32 bytes = 256 bits, 64 neighbors x 32B = 2048B in `page.quantized_neighbors`):
/// - 00..04 (4B): Precomputed radial norm squared $\|\hat{v}_u\|_2^2$ for fast ADC Rauch expansion.
/// - 04..08 (4B): Quantization scale factor $\gamma > 0$.
/// - 08..32 (24B): 24 sub-quantizer centroid codes ($u_8 \in [0, 255]$).
#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    bytemuck::Pod,
    bytemuck::Zeroable,
    zerocopy::FromBytes,
    zerocopy::IntoBytes,
    zerocopy::KnownLayout,
    zerocopy::Immutable,
)]
pub struct CompactNeighborCode {
    /// Precomputed Euclidean norm squared $\|\hat{v}_u\|_2^2$ of the quantized vector in $T_{\mu_c}\mathcal{M}$.
    pub radial_norm_sq: f32,
    /// Quantization scale factor.
    pub scale: f32,
    /// 24 sub-space quantization indices for Product Quantization codebooks.
    pub subspace_codes: [u8; 24],
}

static_assertions::const_assert_eq!(core::mem::size_of::<CompactNeighborCode>(), 32);
static_assertions::const_assert_eq!(core::mem::align_of::<CompactNeighborCode>(), 4);

impl CompactNeighborCode {
    /// Zeroed default neighbor code.
    pub const ZERO: Self = Self {
        radial_norm_sq: 0.0,
        scale: 1.0,
        subspace_codes: [0u8; 24],
    };

    /// View as 32-byte array reference.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; 32] {
        bytemuck::cast_ref(self)
    }

    /// View as mutable 32-byte array reference.
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        bytemuck::cast_mut(self)
    }

    /// Safe zero-copy cast from a 32-byte array.
    #[inline]
    pub fn from_bytes(bytes: &[u8; 32]) -> &Self {
        bytemuck::cast_ref(bytes)
    }

    /// Safe zero-copy mutable cast from a 32-byte array.
    #[inline]
    pub fn from_bytes_mut(bytes: &mut [u8; 32]) -> &mut Self {
        bytemuck::cast_mut(bytes)
    }
}

/// Compute Anisotropic Residual Quantization Loss:
///
/// $$\mathcal{L}_{\text{quant}}(v, \hat{v}) = \eta_{\parallel} \|r_{\parallel}\|_2^2 + \eta_{\perp} \|r_{\perp}\|_2^2$$
///
/// Decomposing quantization error $r = v - \hat{v}$ into:
/// - Parallel residual along the radial direction $v$:
///   $$r_{\parallel} = \frac{\langle r, v \rangle}{\|v\|_2^2} v, \quad \|r_{\parallel}\|_2^2 = \frac{\langle r, v \rangle^2}{\|v\|_2^2}$$
/// - Orthogonal residual perpendicular to $v$:
///   $$r_{\perp} = r - r_{\parallel}, \quad \|r_{\perp}\|_2^2 = \|r\|_2^2 - \|r_{\parallel}\|_2^2$$
///
/// When $\eta_{\parallel} \gg \eta_{\perp}$, penalizes radial distortion heavily to preserve
/// geodesic distance ordering in negative and positive curvature tangent spaces.
#[inline]
pub fn anisotropic_residual_loss(
    v_tangent: &[f32],
    v_reconstructed: &[f32],
    eta_parallel: f32,
    eta_ortho: f32,
) -> Result<f32, AetherError> {
    if v_tangent.len() != v_reconstructed.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: v_reconstructed.len(),
            expected: v_tangent.len(),
        });
    }
    if v_tangent.is_empty() {
        return Err(AetherError::InvalidBufferSize {
            actual: 0,
            expected: 1,
        });
    }

    let r_norm_sq = simd_euclidean_distance_squared(v_tangent, v_reconstructed);
    let v_norm_sq = simd_dot_product(v_tangent, v_tangent);

    if v_norm_sq <= 1e-12 {
        return Ok(eta_ortho * r_norm_sq);
    }

    // <r, v> = <v - v_rec, v> = ||v||^2 - <v_rec, v>
    let dot_rec_v = simd_dot_product(v_reconstructed, v_tangent);
    let dot_rv = v_norm_sq - dot_rec_v;

    let r_parallel_norm_sq = (dot_rv * dot_rv) / v_norm_sq;
    let r_ortho_norm_sq = (r_norm_sq - r_parallel_norm_sq).max(0.0);

    Ok(eta_parallel * r_parallel_norm_sq + eta_ortho * r_ortho_norm_sq)
}

/// Encode a tangent vector $v \in T_{\mu_c}\mathcal{M}$ into a 32-byte `CompactNeighborCode`
/// using 24-subspace Product Quantization codebooks.
///
/// # Arguments
/// - `v_tangent`: Tangent vector of dimension $D$ (where $D$ must be a multiple of 24).
/// - `codebooks`: Flat slice of $24 \times K \times d_{\text{sub}}$ codebook centroids.
/// - `centroids_per_subspace`: Number of centroids $K$ per subspace ($1 \le K \le 256$).
/// - `out_code`: Output 32-byte neighbor code struct.
pub fn at_rq_encode_neighbor(
    v_tangent: &[f32],
    codebooks: &[f32],
    centroids_per_subspace: usize,
    out_code: &mut CompactNeighborCode,
) -> Result<(), AetherError> {
    let d = v_tangent.len();
    if d == 0 || !d.is_multiple_of(24) {
        return Err(AetherError::InvalidBufferSize {
            actual: d,
            expected: 24,
        });
    }
    if centroids_per_subspace == 0 || centroids_per_subspace > 256 {
        return Err(AetherError::InvalidBufferSize {
            actual: centroids_per_subspace,
            expected: 256,
        });
    }

    let d_sub = d / 24;
    let required_codebook_len = 24 * centroids_per_subspace * d_sub;
    if codebooks.len() < required_codebook_len {
        return Err(AetherError::InvalidBufferSize {
            actual: codebooks.len(),
            expected: required_codebook_len,
        });
    }

    let mut total_recon_norm_sq = 0.0f32;

    for s in 0..24 {
        let sub_v = match v_tangent.get(s * d_sub..(s + 1) * d_sub) {
            Some(slice) => slice,
            None => return Err(AetherError::OutOfBounds { index: (s + 1) * d_sub, limit: d }),
        };

        let mut best_dist = f32::INFINITY;
        let mut best_c = 0u8;

        for c in 0..centroids_per_subspace {
            let offset = (s * centroids_per_subspace + c) * d_sub;
            let centroid = match codebooks.get(offset..offset + d_sub) {
                Some(slice) => slice,
                None => return Err(AetherError::OutOfBounds { index: offset + d_sub, limit: codebooks.len() }),
            };

            let dist = simd_euclidean_distance_squared(sub_v, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_c = c as u8;
            }
        }

        if let Some(code_ref) = out_code.subspace_codes.get_mut(s) {
            *code_ref = best_c;
        }

        // Add centroid norm squared to radial norm
        let chosen_offset = (s * centroids_per_subspace + (best_c as usize)) * d_sub;
        if let Some(chosen_centroid) = codebooks.get(chosen_offset..chosen_offset + d_sub) {
            total_recon_norm_sq += simd_dot_product(chosen_centroid, chosen_centroid);
        }
    }

    out_code.radial_norm_sq = total_recon_norm_sq;
    out_code.scale = 1.0;
    Ok(())
}

/// Decode a 32-byte `CompactNeighborCode` back into a reconstructed tangent vector $\hat{v}_u$.
pub fn at_rq_decode_neighbor(
    code: &CompactNeighborCode,
    codebooks: &[f32],
    centroids_per_subspace: usize,
    out_tangent: &mut [f32],
) -> Result<(), AetherError> {
    let d = out_tangent.len();
    if d == 0 || !d.is_multiple_of(24) {
        return Err(AetherError::InvalidBufferSize {
            actual: d,
            expected: 24,
        });
    }
    let d_sub = d / 24;
    let required_codebook_len = 24 * centroids_per_subspace * d_sub;
    if codebooks.len() < required_codebook_len {
        return Err(AetherError::InvalidBufferSize {
            actual: codebooks.len(),
            expected: required_codebook_len,
        });
    }

    let scale = code.scale;

    for s in 0..24 {
        let code_val = match code.subspace_codes.get(s) {
            Some(&c) => c as usize,
            None => 0,
        };
        let offset = (s * centroids_per_subspace + code_val) * d_sub;
        let centroid = match codebooks.get(offset..offset + d_sub) {
            Some(slice) => slice,
            None => return Err(AetherError::OutOfBounds { index: offset + d_sub, limit: codebooks.len() }),
        };
        let out_sub = match out_tangent.get_mut(s * d_sub..(s + 1) * d_sub) {
            Some(slice) => slice,
            None => return Err(AetherError::OutOfBounds { index: (s + 1) * d_sub, limit: d }),
        };

        for (out_val, &c_val) in out_sub.iter_mut().zip(centroid.iter()) {
            *out_val = scale * c_val;
        }
    }

    Ok(())
}

/// Precompute the Query Look-Up Table (LUT) for Asymmetric Distance Computation:
///
/// $$\text{LUT}[s, c] = \langle q_c^{(s)}, c^{(s)} \rangle$$
///
/// across all 24 subspaces $s \in [0, 23]$ and centroid codes $c \in [0, K-1]$.
pub fn at_rq_build_query_lut(
    qc: &[f32],
    codebooks: &[f32],
    centroids_per_subspace: usize,
    out_lut: &mut [f32],
) -> Result<(), AetherError> {
    let d = qc.len();
    if d == 0 || !d.is_multiple_of(24) {
        return Err(AetherError::InvalidBufferSize {
            actual: d,
            expected: 24,
        });
    }
    let d_sub = d / 24;
    let lut_len = 24 * centroids_per_subspace;
    if out_lut.len() < lut_len {
        return Err(AetherError::InvalidBufferSize {
            actual: out_lut.len(),
            expected: lut_len,
        });
    }

    for s in 0..24 {
        let q_sub = match qc.get(s * d_sub..(s + 1) * d_sub) {
            Some(slice) => slice,
            None => return Err(AetherError::OutOfBounds { index: (s + 1) * d_sub, limit: d }),
        };

        for c in 0..centroids_per_subspace {
            let offset = (s * centroids_per_subspace + c) * d_sub;
            let centroid = match codebooks.get(offset..offset + d_sub) {
                Some(slice) => slice,
                None => return Err(AetherError::OutOfBounds { index: offset + d_sub, limit: codebooks.len() }),
            };

            let dot = simd_dot_product(q_sub, centroid);
            let lut_idx = s * centroids_per_subspace + c;
            if let Some(lut_ref) = out_lut.get_mut(lut_idx) {
                *lut_ref = dot;
            }
        }
    }

    Ok(())
}

/// Evaluate Curvature-Corrected Geodesic Distance Approximation:
///
/// $$d_{\mathcal{M}}^2(q, x_u) \approx \|q_c\|_2^2 + \|\hat{v}_u\|_2^2 - 2\langle q_c, \hat{v}_u \rangle - \frac{\kappa}{3} \left( \|q_c\|_2^2 \|\hat{v}_u\|_2^2 - \langle q_c, \hat{v}_u \rangle^2 \right)$$
///
/// Derived from the Second-Order Rauch/Jacobi Comparison Theorem in normal coordinates.
#[inline]
pub fn curvature_corrected_tangent_distance_squared(
    qc_norm_sq: f32,
    vu_norm_sq: f32,
    dot_product: f32,
    kappa: f32,
) -> f32 {
    let euclidean_d2 = qc_norm_sq + vu_norm_sq - 2.0 * dot_product;
    let cross_term = (qc_norm_sq * vu_norm_sq - dot_product * dot_product).max(0.0);
    let correction = (kappa / 3.0) * cross_term;
    let d2 = euclidean_d2 - correction;
    if d2 < 0.0 { 0.0 } else { d2 }
}

/// Curvature-Corrected Page Asymmetric Distance Computation (ADC) Kernel:
///
/// Scores all active neighbor vectors ($\le 64$) stored in `page.quantized_neighbors` against
/// a query tangent vector $q_c = \log_{\mu_c}(q)$ using precomputed query look-up tables
/// and Second-Order Rauch Comparison curvature correction.
///
/// Writes exact squared geodesic distance estimates into `out_distances: &mut [f32; 64]`.
/// Unused slots ($i \ge \text{active\_vertex\_count}$) are populated with `f32::INFINITY`.
///
/// # Hard Invariants
/// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
/// - Tested inside `HotPathAllocGuard`.
#[inline]
pub fn at_rq_score_page_neighbors(
    qc_norm_sq: f32,
    query_lut: &[f32],
    centroids_per_subspace: usize,
    kappa: f32,
    page: &TensorGraphPage,
    out_distances: &mut [f32; 64],
) -> Result<(), AetherError> {
    if centroids_per_subspace == 0 {
        return Err(AetherError::InvalidBufferSize {
            actual: 0,
            expected: 1,
        });
    }

    let active_count = (page.header.active_vertex_count as usize).min(64);

    for i in 0..64 {
        if i < active_count {
            let offset = i * 32;
            let code_bytes = match page.quantized_neighbors.get(offset..offset + 32) {
                Some(slice) => slice,
                None => return Err(AetherError::OutOfBounds { index: offset + 32, limit: page.quantized_neighbors.len() }),
            };

            let code = match bytemuck::try_from_bytes::<CompactNeighborCode>(code_bytes) {
                Ok(c) => c,
                Err(_) => return Err(AetherError::InvalidBufferSize { actual: code_bytes.len(), expected: 32 }),
            };

            let mut dot_sum = 0.0f32;
            for s in 0..24 {
                let c_val = match code.subspace_codes.get(s) {
                    Some(&c) => c as usize,
                    None => 0,
                };
                let lut_idx = s * centroids_per_subspace + c_val;
                if let Some(&val) = query_lut.get(lut_idx) {
                    dot_sum += val;
                }
            }

            let dot = code.scale * dot_sum;
            let vu_norm_sq = code.radial_norm_sq;
            let d2 = curvature_corrected_tangent_distance_squared(qc_norm_sq, vu_norm_sq, dot, kappa);

            if let Some(out_ref) = out_distances.get_mut(i) {
                *out_ref = d2;
            }
        } else if let Some(out_ref) = out_distances.get_mut(i) {
            *out_ref = f32::INFINITY;
        }
    }

    Ok(())
}
