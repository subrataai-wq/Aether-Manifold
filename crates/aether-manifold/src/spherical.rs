//! Project Aether-Manifold: Hyperspherical Space ($\mathbb{S}_{\tau}^{d_S}$) Metric Kernels
//!
//! Implements exact geodesic distance, antipodal numerical stability, and normalization
//! on the hypersphere with positive curvature $\tau > 0$, in strict compliance with
//! roadmap.md Section 2.1, AGENTS.md Rule 3.1, and test_suite_manifest.md Category III.
//!
//! Mathematical formulation:
//! $$d_{\mathbb{S}_{\tau}}(u, v) = \frac{1}{\sqrt{\tau}} \arccos\left( \operatorname{clamp}\left( \langle u, v \rangle, -1.0, 1.0 \right) \right)$$
//! $$d_{\mathbb{S}_{\tau}}^2(u, v) = \left( d_{\mathbb{S}_{\tau}}(u, v) \right)^2$$

use crate::hyperbolic::HORIZON_EPSILON;
use crate::simd::{
    scalar_dot_product, scalar_sum_of_squares, simd_dot_product, simd_sum_of_squares,
};
use aether_core::error::AetherError;

/// Compute the geodesic distance $d_{\mathbb{S}_\tau}(u, v)$ on a hypersphere with positive curvature $\tau > 0$.
///
/// Clamps the inner product strictly to $[-1.0, 1.0]$ to eliminate floating-point `NaN`
/// generation near antipodal points ($\theta \to \pi$) or identical points ($\theta \to 0$).
///
/// Returns `Err(AetherError::LorentzHorizonViolation)` if $\tau \le 0$.
/// Returns `Err(AetherError::InvalidBufferSize)` if vector dimensions differ or are empty.
#[inline]
pub fn spherical_distance(u: &[f32], v: &[f32], tau: f32) -> Result<f32, AetherError> {
    if tau <= 0.0 {
        return Err(AetherError::LorentzHorizonViolation);
    }
    if u.len() != v.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: v.len(),
            expected: u.len(),
        });
    }
    if u.is_empty() {
        return Err(AetherError::InvalidBufferSize {
            actual: 0,
            expected: 1,
        });
    }
    if core::ptr::eq(u, v) {
        return Ok(0.0);
    }

    let dot = simd_dot_product(u, v);
    // Explicit projection clamp preventing NaN in acos:
    // If dot >= 1.0 - HORIZON_EPSILON, the vectors are identical within float precision,
    // clamping cos to 1.0 so acos(1.0) = 0.0.
    let cos_clamped = if dot >= 1.0 - HORIZON_EPSILON {
        1.0
    } else {
        dot.clamp(-1.0, 1.0)
    };

    let theta = libm::acosf(cos_clamped);
    let scale = 1.0 / libm::sqrtf(tau);
    Ok(scale * theta)
}

/// Compute the squared spherical geodesic distance $d_{\mathbb{S}_\tau}^2(u, v)$.
#[inline]
pub fn spherical_distance_squared(u: &[f32], v: &[f32], tau: f32) -> Result<f32, AetherError> {
    let d = spherical_distance(u, v, tau)?;
    Ok(d * d)
}

/// Normalize a vector to lie on the unit hypersphere ($\|x\|_2 = 1.0$) in-place.
///
/// Returns `Err(AetherError::InvalidBufferSize)` if vector is empty.
#[inline]
pub fn spherical_normalize(x: &mut [f32]) -> Result<(), AetherError> {
    if x.is_empty() {
        return Err(AetherError::InvalidBufferSize {
            actual: 0,
            expected: 1,
        });
    }
    let sum_sq = simd_sum_of_squares(x);
    let norm = libm::sqrtf(sum_sq);
    if norm > 0.0 {
        let inv_norm = 1.0 / norm;
        for val in x.iter_mut() {
            *val *= inv_norm;
        }
    }
    Ok(())
}

/// Pure scalar reference implementation of spherical distance for verification.
#[inline]
pub fn scalar_spherical_distance(u: &[f32], v: &[f32], tau: f32) -> Result<f32, AetherError> {
    if tau <= 0.0 {
        return Err(AetherError::LorentzHorizonViolation);
    }
    if u.len() != v.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: v.len(),
            expected: u.len(),
        });
    }
    if u.is_empty() {
        return Err(AetherError::InvalidBufferSize {
            actual: 0,
            expected: 1,
        });
    }
    if core::ptr::eq(u, v) {
        return Ok(0.0);
    }

    let dot = scalar_dot_product(u, v);
    let cos_clamped = if dot >= 1.0 - HORIZON_EPSILON {
        1.0
    } else {
        dot.clamp(-1.0, 1.0)
    };
    let theta = libm::acosf(cos_clamped);
    let scale = 1.0 / libm::sqrtf(tau);
    Ok(scale * theta)
}

/// Pure scalar reference implementation of spherical normalization for verification.
#[inline]
pub fn scalar_spherical_normalize(x: &mut [f32]) -> Result<(), AetherError> {
    if x.is_empty() {
        return Err(AetherError::InvalidBufferSize {
            actual: 0,
            expected: 1,
        });
    }
    let sum_sq = scalar_sum_of_squares(x);
    let norm = libm::sqrtf(sum_sq);
    if norm > 0.0 {
        let inv_norm = 1.0 / norm;
        for val in x.iter_mut() {
            *val *= inv_norm;
        }
    }
    Ok(())
}
