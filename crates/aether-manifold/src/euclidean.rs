//! Project Aether-Manifold: Flat Euclidean Space ($\mathbb{R}^{d_E}$) Metric Kernels
//!
//! Implements flat Euclidean distance and inner product metrics over continuous features
//! and spatio-temporal coordinates with 0 curvature ($R = 0$) in strict compliance
//! with roadmap.md Section 2.1 and AGENTS.md Rule 3.1.
//!
//! Mathematical formulation:
//! $$d_{\mathbb{R}}^2(u, v) = \|u - v\|_2^2 = \sum_{l=0}^{d_E-1} (u_l - v_l)^2$$
//! $$d_{\mathbb{R}}(u, v) = \sqrt{d_{\mathbb{R}}^2(u, v)}$$

use crate::simd::{
    scalar_dot_product, scalar_euclidean_distance_squared, simd_dot_product,
    simd_euclidean_distance_squared,
};
use aether_core::error::AetherError;

/// Compute the squared Euclidean distance $d_{\mathbb{R}}^2(u, v) = \|u - v\|_2^2$
/// using runtime-dispatched SIMD with 4x unrolled accumulators.
///
/// Returns `Err(AetherError::InvalidBufferSize)` if slice dimensions differ.
#[inline]
pub fn euclidean_distance_squared(u: &[f32], v: &[f32]) -> Result<f32, AetherError> {
    if u.len() != v.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: v.len(),
            expected: u.len(),
        });
    }
    Ok(simd_euclidean_distance_squared(u, v))
}

/// Compute the Euclidean geodesic distance $d_{\mathbb{R}}(u, v) = \|u - v\|_2$
/// using runtime-dispatched SIMD with 4x unrolled accumulators.
///
/// Returns `Err(AetherError::InvalidBufferSize)` if slice dimensions differ.
#[inline]
pub fn euclidean_distance(u: &[f32], v: &[f32]) -> Result<f32, AetherError> {
    let d2 = euclidean_distance_squared(u, v)?;
    Ok(libm::sqrtf(d2))
}

/// Compute the Euclidean inner product $\langle u, v \rangle = \sum_{l=0}^{d_E-1} u_l v_l$
/// using runtime-dispatched SIMD with 4x unrolled accumulators.
///
/// Returns `Err(AetherError::InvalidBufferSize)` if slice dimensions differ.
#[inline]
pub fn euclidean_dot_product(u: &[f32], v: &[f32]) -> Result<f32, AetherError> {
    if u.len() != v.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: v.len(),
            expected: u.len(),
        });
    }
    Ok(simd_dot_product(u, v))
}

/// Pure scalar reference implementation of squared Euclidean distance for verification.
#[inline]
pub fn scalar_euclidean_distance_squared_checked(u: &[f32], v: &[f32]) -> Result<f32, AetherError> {
    if u.len() != v.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: v.len(),
            expected: u.len(),
        });
    }
    Ok(scalar_euclidean_distance_squared(u, v))
}

/// Pure scalar reference implementation of Euclidean distance for verification.
#[inline]
pub fn scalar_euclidean_distance_checked(u: &[f32], v: &[f32]) -> Result<f32, AetherError> {
    let d2 = scalar_euclidean_distance_squared_checked(u, v)?;
    Ok(libm::sqrtf(d2))
}

/// Pure scalar reference implementation of dot product for verification.
#[inline]
pub fn scalar_euclidean_dot_product_checked(u: &[f32], v: &[f32]) -> Result<f32, AetherError> {
    if u.len() != v.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: v.len(),
            expected: u.len(),
        });
    }
    Ok(scalar_dot_product(u, v))
}
