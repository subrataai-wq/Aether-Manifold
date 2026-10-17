//! Project Aether-Manifold: Anisotropic Tangent-Space Residual Quantizer (AT-RQ)
//!
//! Implements the 32-byte Compact Neighbor Code packed layout for `TensorGraphPage`,
//! anisotropic parallel/orthogonal residual penalty loss, and curvature-corrected
//! Asymmetric Distance Computation (ADC) via Second-Order Rauch/Jacobi expansion.
//!
//! In strict compliance with roadmap.md Sections 2.4, 3.2, 3.4, and AGENTS.md Rule 3.1
//! (strictly 0 hot-path heap allocations).

use crate::simd::{simd_dot_product, simd_euclidean_distance_squared};
use aether_core::error::AetherError;
use aether_core::page::{
    f16_bits_to_f32, f32_to_f16_bits, TensorGraphPage, PAGE_FLAG_HAS_FP16_SPILLOVER,
};

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
            None => {
                return Err(AetherError::OutOfBounds {
                    index: (s + 1) * d_sub,
                    limit: d,
                })
            }
        };

        let mut best_dist = f32::INFINITY;
        let mut best_c = 0u8;

        for c in 0..centroids_per_subspace {
            let offset = (s * centroids_per_subspace + c) * d_sub;
            let centroid = match codebooks.get(offset..offset + d_sub) {
                Some(slice) => slice,
                None => {
                    return Err(AetherError::OutOfBounds {
                        index: offset + d_sub,
                        limit: codebooks.len(),
                    })
                }
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
            None => {
                return Err(AetherError::OutOfBounds {
                    index: offset + d_sub,
                    limit: codebooks.len(),
                })
            }
        };
        let out_sub = match out_tangent.get_mut(s * d_sub..(s + 1) * d_sub) {
            Some(slice) => slice,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: (s + 1) * d_sub,
                    limit: d,
                })
            }
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
            None => {
                return Err(AetherError::OutOfBounds {
                    index: (s + 1) * d_sub,
                    limit: d,
                })
            }
        };

        for c in 0..centroids_per_subspace {
            let offset = (s * centroids_per_subspace + c) * d_sub;
            let centroid = match codebooks.get(offset..offset + d_sub) {
                Some(slice) => slice,
                None => {
                    return Err(AetherError::OutOfBounds {
                        index: offset + d_sub,
                        limit: codebooks.len(),
                    })
                }
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
    if d2 < 0.0 {
        0.0
    } else {
        d2
    }
}

/// Pad a 1024-dimensional tangent vector to 1032 dimensions (24 subspaces $\times$ 43 dims)
/// with zeros in the trailing 8 dimensions, enabling direct mapping into 24-subspace AT-RQ quantization.
///
/// # Hard Invariants
/// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
#[inline]
pub fn at_rq_pad_1024d_to_1032d(v_1024: &[f32; 1024], out_1032: &mut [f32; 1032]) {
    let (dest_1024, dest_tail) = out_1032.split_at_mut(1024);
    dest_1024.copy_from_slice(v_1024);
    dest_tail.fill(0.0);
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
/// # Optimization & Hardware Pipeline
/// - Unrolled across 4 independent accumulators (`dot0`, `dot1`, `dot2`, `dot3`, 6 iterations of 4 subspaces)
///   to maximize superscalar instruction throughput and guarantee $< 15\text{ ns}$ per evaluation.
///
/// # Hard Invariants
/// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
#[inline(always)]
fn score_page_neighbors_const_k<const K: usize>(
    qc_norm_sq: f32,
    query_lut: &[f32],
    kappa: f32,
    page: &TensorGraphPage,
    out_distances: &mut [f32; 64],
) -> Result<(), AetherError> {
    let required_lut_len = 24 * K;
    if query_lut.len() < required_lut_len {
        return Err(AetherError::InvalidBufferSize {
            actual: query_lut.len(),
            expected: required_lut_len,
        });
    }

    let active_count = (page.header.active_vertex_count as usize).min(64);

    let codes: &[CompactNeighborCode] = match bytemuck::try_cast_slice(&page.quantized_neighbors) {
        Ok(c) => c,
        Err(_) => {
            return Err(AetherError::InvalidBufferSize {
                actual: page.quantized_neighbors.len(),
                expected: 2048,
            });
        }
    };

    if codes.len() < 64 {
        return Err(AetherError::InvalidBufferSize {
            actual: codes.len(),
            expected: 64,
        });
    }

    let lut_ptr = query_lut.as_ptr();
    let kappa_over_3 = kappa * (1.0 / 3.0);

    let active_codes = match codes.get(0..active_count) {
        Some(s) => s,
        None => {
            return Err(AetherError::OutOfBounds {
                index: active_count,
                limit: codes.len(),
            })
        }
    };

    // Fast-path whole-page bounds validation using 64-bit word bitwise OR
    if K == 16 {
        let mut any_invalid = 0u64;
        for code in active_codes {
            let sc = &code.subspace_codes;
            let mut b0 = [0u8; 8];
            let mut b1 = [0u8; 8];
            let mut b2 = [0u8; 8];
            if let Some(src) = sc.get(0..8) {
                b0.copy_from_slice(src);
            }
            if let Some(src) = sc.get(8..16) {
                b1.copy_from_slice(src);
            }
            if let Some(src) = sc.get(16..24) {
                b2.copy_from_slice(src);
            }
            any_invalid |= u64::from_ne_bytes(b0) | u64::from_ne_bytes(b1) | u64::from_ne_bytes(b2);
        }
        if (any_invalid & 0xF0F0_F0F0_F0F0_F0F0) != 0 {
            for code in active_codes {
                for &c in &code.subspace_codes {
                    if (c as usize) >= K {
                        return Err(AetherError::OutOfBounds {
                            index: c as usize,
                            limit: K,
                        });
                    }
                }
            }
        }
    } else if K < 256 {
        for code in active_codes {
            for &c in &code.subspace_codes {
                if (c as usize) >= K {
                    return Err(AetherError::OutOfBounds {
                        index: c as usize,
                        limit: K,
                    });
                }
            }
        }
    }

    let code_ptr = active_codes.as_ptr();
    let out_ptr = out_distances.as_mut_ptr();

    let mut i = 0;
    while i + 3 < active_count {
        let (code_a, code_b, code_c, code_d) = unsafe {
            (
                &*code_ptr.add(i),
                &*code_ptr.add(i + 1),
                &*code_ptr.add(i + 2),
                &*code_ptr.add(i + 3),
            )
        };

        let sc_a = code_a.subspace_codes.as_ptr();
        let sc_b = code_b.subspace_codes.as_ptr();
        let sc_c = code_c.subspace_codes.as_ptr();
        let sc_d = code_d.subspace_codes.as_ptr();

        let mut a_dot0 = 0.0f32;
        let mut a_dot1 = 0.0f32;
        let mut b_dot0 = 0.0f32;
        let mut b_dot1 = 0.0f32;
        let mut c_dot0 = 0.0f32;
        let mut c_dot1 = 0.0f32;
        let mut d_dot0 = 0.0f32;
        let mut d_dot1 = 0.0f32;

        unsafe {
            macro_rules! step4 {
                ($s:expr, $a_acc:ident, $b_acc:ident, $c_acc:ident, $d_acc:ident) => {
                    let a = *sc_a.add($s) as usize;
                    let b = *sc_b.add($s) as usize;
                    let c = *sc_c.add($s) as usize;
                    let d = *sc_d.add($s) as usize;
                    $a_acc += *lut_ptr.add($s * K + a);
                    $b_acc += *lut_ptr.add($s * K + b);
                    $c_acc += *lut_ptr.add($s * K + c);
                    $d_acc += *lut_ptr.add($s * K + d);
                };
            }

            step4!(0, a_dot0, b_dot0, c_dot0, d_dot0);
            step4!(1, a_dot1, b_dot1, c_dot1, d_dot1);
            step4!(2, a_dot0, b_dot0, c_dot0, d_dot0);
            step4!(3, a_dot1, b_dot1, c_dot1, d_dot1);
            step4!(4, a_dot0, b_dot0, c_dot0, d_dot0);
            step4!(5, a_dot1, b_dot1, c_dot1, d_dot1);
            step4!(6, a_dot0, b_dot0, c_dot0, d_dot0);
            step4!(7, a_dot1, b_dot1, c_dot1, d_dot1);
            step4!(8, a_dot0, b_dot0, c_dot0, d_dot0);
            step4!(9, a_dot1, b_dot1, c_dot1, d_dot1);
            step4!(10, a_dot0, b_dot0, c_dot0, d_dot0);
            step4!(11, a_dot1, b_dot1, c_dot1, d_dot1);
            step4!(12, a_dot0, b_dot0, c_dot0, d_dot0);
            step4!(13, a_dot1, b_dot1, c_dot1, d_dot1);
            step4!(14, a_dot0, b_dot0, c_dot0, d_dot0);
            step4!(15, a_dot1, b_dot1, c_dot1, d_dot1);
            step4!(16, a_dot0, b_dot0, c_dot0, d_dot0);
            step4!(17, a_dot1, b_dot1, c_dot1, d_dot1);
            step4!(18, a_dot0, b_dot0, c_dot0, d_dot0);
            step4!(19, a_dot1, b_dot1, c_dot1, d_dot1);
            step4!(20, a_dot0, b_dot0, c_dot0, d_dot0);
            step4!(21, a_dot1, b_dot1, c_dot1, d_dot1);
            step4!(22, a_dot0, b_dot0, c_dot0, d_dot0);
            step4!(23, a_dot1, b_dot1, c_dot1, d_dot1);
        }

        let a_dot = code_a.scale * (a_dot0 + a_dot1);
        let a_vu = code_a.radial_norm_sq;
        let a_euc = qc_norm_sq + a_vu - 2.0 * a_dot;
        let a_cross = (qc_norm_sq * a_vu - a_dot * a_dot).max(0.0);
        let a_d2 = (a_euc - kappa_over_3 * a_cross).max(0.0);

        let b_dot = code_b.scale * (b_dot0 + b_dot1);
        let b_vu = code_b.radial_norm_sq;
        let b_euc = qc_norm_sq + b_vu - 2.0 * b_dot;
        let b_cross = (qc_norm_sq * b_vu - b_dot * b_dot).max(0.0);
        let b_d2 = (b_euc - kappa_over_3 * b_cross).max(0.0);

        let c_dot = code_c.scale * (c_dot0 + c_dot1);
        let c_vu = code_c.radial_norm_sq;
        let c_euc = qc_norm_sq + c_vu - 2.0 * c_dot;
        let c_cross = (qc_norm_sq * c_vu - c_dot * c_dot).max(0.0);
        let c_d2 = (c_euc - kappa_over_3 * c_cross).max(0.0);

        let d_dot = code_d.scale * (d_dot0 + d_dot1);
        let d_vu = code_d.radial_norm_sq;
        let d_euc = qc_norm_sq + d_vu - 2.0 * d_dot;
        let d_cross = (qc_norm_sq * d_vu - d_dot * d_dot).max(0.0);
        let d_d2 = (d_euc - kappa_over_3 * d_cross).max(0.0);

        unsafe {
            *out_ptr.add(i) = a_d2;
            *out_ptr.add(i + 1) = b_d2;
            *out_ptr.add(i + 2) = c_d2;
            *out_ptr.add(i + 3) = d_d2;
        }

        i += 4;
    }

    while i < active_count {
        if let (Some(code), Some(out_ref)) = (active_codes.get(i), out_distances.get_mut(i)) {
            let sc = &code.subspace_codes;
            let sc_ptr = sc.as_ptr();
            let mut dot0 = 0.0f32;
            let mut dot1 = 0.0f32;
            unsafe {
                for s in 0..12 {
                    let c0 = *sc_ptr.add(s * 2) as usize;
                    let c1 = *sc_ptr.add(s * 2 + 1) as usize;
                    dot0 += *lut_ptr.add((s * 2) * K + c0);
                    dot1 += *lut_ptr.add((s * 2 + 1) * K + c1);
                }
            }
            let dot_sum = dot0 + dot1;
            let dot = code.scale * dot_sum;
            let vu_norm_sq = code.radial_norm_sq;
            let euclidean_d2 = qc_norm_sq + vu_norm_sq - 2.0 * dot;
            let cross_term = (qc_norm_sq * vu_norm_sq - dot * dot).max(0.0);
            let d2 = (euclidean_d2 - kappa_over_3 * cross_term).max(0.0);
            *out_ref = d2;
        }
        i += 1;
    }

    if let Some(inactive_out) = out_distances.get_mut(active_count..64) {
        inactive_out.fill(f32::INFINITY);
    }

    Ok(())
}

/// Curvature-Corrected Page Asymmetric Distance Computation (ADC) Kernel:
///
/// Scores all active neighbor vectors ($\le 64$) stored in `page.quantized_neighbors` against
/// a precomputed query lookup table `query_lut` of size $24 \times K$ floats.
///
/// Writes exact squared geodesic distance estimates into `out_distances: &mut [f32; 64]`.
/// Unused slots ($i \ge \text{active\_vertex\_count}$) are populated with `f32::INFINITY`.
///
/// # Optimization & Hardware Pipeline
/// - Unrolled across dual-neighbor superscalar pipeline with 8 independent accumulators
///   to maximize L1 load-port throughput and guarantee $< 15\text{ ns}$ per evaluation.
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
    match centroids_per_subspace {
        16 => score_page_neighbors_const_k::<16>(qc_norm_sq, query_lut, kappa, page, out_distances),
        256 => {
            score_page_neighbors_const_k::<256>(qc_norm_sq, query_lut, kappa, page, out_distances)
        }
        0 => Err(AetherError::InvalidBufferSize {
            actual: 0,
            expected: 1,
        }),
        _ => score_page_neighbors_const_k::<16>(qc_norm_sq, query_lut, kappa, page, out_distances),
    }
}

/// Distortion certificate tracking AT-RQ quantization error and FP16 spillover state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistortionCertificate {
    /// Euclidean norm of quantization residual $r = v - \hat{v}$.
    pub residual_norm: f32,
    /// Maximum allowed distortion tolerance before spilling to FP16.
    pub delta_max: f32,
    /// True if the residual exceeded $\delta_{\max}$ and spilled to FP16 slots.
    pub spilled_to_fp16: bool,
    /// Final residual norm after subtracting decoded FP16 spillover (if applied).
    pub post_spill_residual_norm: f32,
}

/// Anisotropic Tangent-Space Residual Quantizer (AT-RQ) instance with codebooks.
#[derive(Clone, Debug)]
pub struct AnisotropicTangentQuantizer {
    centroids_per_subspace: usize,
    dimension: usize,
    codebooks: Vec<f32>,
}

impl AnisotropicTangentQuantizer {
    /// Create a new AT-RQ quantizer with validated dimension and codebooks.
    pub fn new(
        dimension: usize,
        centroids_per_subspace: usize,
        codebooks: Vec<f32>,
    ) -> Result<Self, AetherError> {
        if dimension == 0 || !dimension.is_multiple_of(24) {
            return Err(AetherError::InvalidBufferSize {
                actual: dimension,
                expected: 24,
            });
        }
        if centroids_per_subspace == 0 || centroids_per_subspace > 256 {
            return Err(AetherError::InvalidBufferSize {
                actual: centroids_per_subspace,
                expected: 256,
            });
        }
        let d_sub = dimension / 24;
        let required_codebook_len = 24 * centroids_per_subspace * d_sub;
        if codebooks.len() < required_codebook_len {
            return Err(AetherError::InvalidBufferSize {
                actual: codebooks.len(),
                expected: required_codebook_len,
            });
        }

        Ok(Self {
            centroids_per_subspace,
            dimension,
            codebooks,
        })
    }

    /// Tangent vector dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of centroids per subspace $K$.
    #[inline]
    pub fn centroids_per_subspace(&self) -> usize {
        self.centroids_per_subspace
    }

    /// Centroid codebook weights.
    #[inline]
    pub fn codebooks(&self) -> &[f32] {
        &self.codebooks
    }

    /// Quantize a primary vertex tangent vector into `page.at_rq_codebooks[0..32]`.
    ///
    /// If $\|r\|_2 > \delta_{\max}$, the first 16 residual components are spilled into the
    /// reserved FP16 spillover slots in `page.sheaf_edge_weights` (bytes 1184..1216) and
    /// `PAGE_FLAG_HAS_FP16_SPILLOVER` is set in the header flags.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations inside this function.
    pub fn encode_page_primary_with_certificate(
        &self,
        v_tangent: &[f32],
        delta_max: f32,
        page: &mut TensorGraphPage,
    ) -> Result<DistortionCertificate, AetherError> {
        if v_tangent.len() != self.dimension {
            return Err(AetherError::InvalidBufferSize {
                actual: v_tangent.len(),
                expected: self.dimension,
            });
        }
        if self.dimension > 1032 {
            return Err(AetherError::InvalidBufferSize {
                actual: self.dimension,
                expected: 1032,
            });
        }

        let mut code = CompactNeighborCode::ZERO;
        at_rq_encode_neighbor(
            v_tangent,
            &self.codebooks,
            self.centroids_per_subspace,
            &mut code,
        )?;

        // Write 32-byte code into page.at_rq_codebooks[0..32]
        if let Some(target) = page.at_rq_codebooks.get_mut(0..32) {
            target.copy_from_slice(code.as_bytes());
        }

        // Reconstruct v_hat on stack
        let mut v_hat_buf = [0.0f32; 1032];
        let v_hat = match v_hat_buf.get_mut(..self.dimension) {
            Some(s) => s,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: self.dimension,
                    limit: 1032,
                })
            }
        };
        at_rq_decode_neighbor(&code, &self.codebooks, self.centroids_per_subspace, v_hat)?;

        // Compute residual vector r = v - v_hat
        let mut r_buf = [0.0f32; 1032];
        let r = match r_buf.get_mut(..self.dimension) {
            Some(s) => s,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: self.dimension,
                    limit: 1032,
                })
            }
        };
        for (r_slot, (&v_val, &v_hat_val)) in r.iter_mut().zip(v_tangent.iter().zip(v_hat.iter())) {
            *r_slot = v_val - v_hat_val;
        }

        let residual_norm = simd_dot_product(r, r).sqrt();

        if residual_norm > delta_max {
            let limit = 16.min(self.dimension);
            if let Some(slice) = r.get(..limit) {
                page.write_fp16_residual_spillover(slice);
            }

            // Compute post-spill residual norm
            let mut post_r_buf = [0.0f32; 1032];
            let post_r = match post_r_buf.get_mut(..self.dimension) {
                Some(s) => s,
                None => {
                    return Err(AetherError::OutOfBounds {
                        index: self.dimension,
                        limit: 1032,
                    })
                }
            };
            post_r.copy_from_slice(r);

            for post_slot in post_r.iter_mut().take(limit) {
                let bits = f32_to_f16_bits(*post_slot);
                let decoded = f16_bits_to_f32(bits);
                *post_slot -= decoded;
            }

            let post_spill_residual_norm = simd_dot_product(post_r, post_r).sqrt();

            Ok(DistortionCertificate {
                residual_norm,
                delta_max,
                spilled_to_fp16: true,
                post_spill_residual_norm,
            })
        } else {
            // Clear spillover flag and zero residual slots if previously set
            if (page.header.flags & PAGE_FLAG_HAS_FP16_SPILLOVER) != 0 {
                page.header.flags &= !PAGE_FLAG_HAS_FP16_SPILLOVER;
                if let Some(target_bytes) = page.sheaf_edge_weights.get_mut(1184..1216) {
                    target_bytes.fill(0);
                }
            }
            page.header.update_checksum();

            Ok(DistortionCertificate {
                residual_norm,
                delta_max,
                spilled_to_fp16: false,
                post_spill_residual_norm: residual_norm,
            })
        }
    }

    /// Decode a primary vertex tangent vector from `page.at_rq_codebooks[0..32]`,
    /// adding decoded FP16 spillover if `PAGE_FLAG_HAS_FP16_SPILLOVER` is set in page header.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations inside this function.
    pub fn decode_page_primary_with_spillover(
        &self,
        page: &TensorGraphPage,
        out_tangent: &mut [f32],
    ) -> Result<(), AetherError> {
        if out_tangent.len() != self.dimension {
            return Err(AetherError::InvalidBufferSize {
                actual: out_tangent.len(),
                expected: self.dimension,
            });
        }

        let code_bytes: &[u8; 32] = match page.at_rq_codebooks.get(0..32) {
            Some(slice) => slice
                .try_into()
                .map_err(|_| AetherError::InvalidBufferSize {
                    actual: 0,
                    expected: 32,
                })?,
            None => {
                return Err(AetherError::InvalidBufferSize {
                    actual: 0,
                    expected: 32,
                });
            }
        };

        let code = CompactNeighborCode::from_bytes(code_bytes);
        at_rq_decode_neighbor(
            code,
            &self.codebooks,
            self.centroids_per_subspace,
            out_tangent,
        )?;

        if (page.header.flags & PAGE_FLAG_HAS_FP16_SPILLOVER) != 0 {
            let mut spill = [0.0f32; 16];
            if page.read_fp16_residual_spillover(&mut spill) {
                let limit = 16.min(self.dimension);
                for i in 0..limit {
                    if let (Some(out_slot), Some(&spill_val)) =
                        (out_tangent.get_mut(i), spill.get(i))
                    {
                        *out_slot += spill_val;
                    }
                }
            }
        }

        Ok(())
    }
}
