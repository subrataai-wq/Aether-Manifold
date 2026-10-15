//! Project Aether-Manifold: Lorentz Hyperbolic Space ($\mathbb{H}_{\kappa}^{d_H}$) Metric Kernels
//!
//! Implements exact geodesic distance, Lorentzian inner products, horizon clamping, and
//! surface re-projection in the Lorentz (Hyperboloid) model with negative curvature $\kappa < 0$,
//! in strict compliance with roadmap.md Section 2.1, AGENTS.md Rule 4.1, and AGENTS.md Rule 3.1.
//!
//! Mathematical formulation:
//! - Lorentzian inner product:
//!   $$\langle x, y \rangle_{\mathcal{L}} = -x_0 y_0 + \sum_{l=1}^{d_H} x_l y_l$$
//! - Surface normalization invariant:
//!   $$\langle x, x \rangle_{\mathcal{L}} = \frac{1}{\kappa} \implies x_0 = \sqrt{\frac{1}{-\kappa} + \sum_{l=1}^{d_H} x_l^2}$$
//! - Boundary-stabilized geodesic distance with horizon clamping:
//!   $$d_{\mathbb{H}_{\kappa}}(x, y) = \frac{1}{\sqrt{-\kappa}} \operatorname{arcosh}\left( \max\left(1 + \epsilon_{\text{mach}}, \, \kappa \langle x, y \rangle_{\mathcal{L}}\right) \right)$$

use aether_core::error::AetherError;
use crate::simd::{scalar_dot_product, scalar_sum_of_squares, simd_dot_product, simd_sum_of_squares};

/// Machine epsilon horizon stabilization threshold per AGENTS.md Rule 4.1.
pub const HORIZON_EPSILON: f32 = 1e-7;

/// Compute the Lorentzian inner product $\langle x, y \rangle_{\mathcal{L}} = -x_0 y_0 + \sum_{l=1}^{d} x_l y_l$
/// using runtime-dispatched SIMD with 4x unrolled accumulators.
///
/// Returns `Err(AetherError::InvalidBufferSize)` if lengths do not match or if length $< 2$.
#[inline]
pub fn lorentz_inner_product(x: &[f32], y: &[f32]) -> Result<f32, AetherError> {
    if x.len() != y.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: y.len(),
            expected: x.len(),
        });
    }
    if x.len() < 2 {
        return Err(AetherError::InvalidBufferSize {
            actual: x.len(),
            expected: 2,
        });
    }

    let x0 = match x.first() {
        Some(&val) => val,
        None => return Err(AetherError::OutOfBounds { index: 0, limit: x.len() }),
    };
    let y0 = match y.first() {
        Some(&val) => val,
        None => return Err(AetherError::OutOfBounds { index: 0, limit: y.len() }),
    };

    let spatial_x = match x.get(1..) {
        Some(s) => s,
        None => &[],
    };
    let spatial_y = match y.get(1..) {
        Some(s) => s,
        None => &[],
    };

    let spatial_dot = simd_dot_product(spatial_x, spatial_y);
    Ok(-x0 * y0 + spatial_dot)
}

/// Compute the exact geodesic distance $d_{\mathbb{H}_\kappa}(x, y)$ in Lorentz hyperbolic space
/// with hardware horizon clamping to eliminate any possibility of `NaN` generation.
///
/// $$d_{\mathbb{H}_{\kappa}}(x, y) = \frac{1}{\sqrt{-\kappa}} \operatorname{arcosh}\left( \max\left(1 + \epsilon_{\text{mach}}, \, \kappa \langle x, y \rangle_{\mathcal{L}}\right) \right)$$
///
/// Returns `Err(AetherError::LorentzHorizonViolation)` if $\kappa \ge 0$.
/// Returns `Err(AetherError::InvalidBufferSize)` if vector dimensions differ or are $< 2$.
#[inline]
pub fn lorentz_distance(x: &[f32], y: &[f32], kappa: f32) -> Result<f32, AetherError> {
    if kappa >= 0.0 {
        return Err(AetherError::LorentzHorizonViolation);
    }
    if core::ptr::eq(x, y) {
        return Ok(0.0);
    }

    let inner_prod = lorentz_inner_product(x, y)?;
    let raw_z = kappa * inner_prod;

    // Horizon clamp per Rule 4.1:
    // If raw_z <= 1.0 + HORIZON_EPSILON, the points are identical or within machine float precision
    // of the identity horizon, clamping z strictly to 1.0 so arcosh(1.0) = 0.0.
    // If raw_z > 1.0 + HORIZON_EPSILON, z is used directly and is strictly >= 1.0.
    let z = if raw_z <= 1.0 + HORIZON_EPSILON {
        1.0
    } else {
        raw_z
    };

    let acosh_z = libm::acoshf(z);
    let scale = 1.0 / libm::sqrtf(-kappa);
    Ok(scale * acosh_z)
}

/// Compute the squared Lorentz geodesic distance $d_{\mathbb{H}_\kappa}^2(x, y)$.
#[inline]
pub fn lorentz_distance_squared(x: &[f32], y: &[f32], kappa: f32) -> Result<f32, AetherError> {
    let d = lorentz_distance(x, y, kappa)?;
    Ok(d * d)
}

/// Compute the time-like coordinate $x_0 = \sqrt{\frac{1}{-\kappa} + \sum_{l=1}^{d} x_l^2}$
/// given spatial coordinates, enforcing the hyperboloid surface invariant $\langle x, x \rangle_{\mathcal{L}} = \frac{1}{\kappa}$.
///
/// Returns `Err(AetherError::LorentzHorizonViolation)` if $\kappa \ge 0$.
#[inline]
pub fn lorentz_compute_x0(spatial: &[f32], kappa: f32) -> Result<f32, AetherError> {
    if kappa >= 0.0 {
        return Err(AetherError::LorentzHorizonViolation);
    }
    let sum_sq = simd_sum_of_squares(spatial);
    let radius_sq = 1.0 / (-kappa);
    Ok(libm::sqrtf(radius_sq + sum_sq))
}

/// Re-project a point $(x_0, x_1, \dots, x_d)$ onto the hyperboloid sheet in-place,
/// correcting for numerical drift per AGENTS.md Rule 4.1:
/// $$x_0 = \sqrt{\frac{1}{-\kappa} + \sum_{l=1}^{d} x_l^2}$$
///
/// Returns `Err(AetherError::InvalidBufferSize)` if slice length $< 2$.
/// Returns `Err(AetherError::LorentzHorizonViolation)` if $\kappa \ge 0$.
#[inline]
pub fn lorentz_reproject(point: &mut [f32], kappa: f32) -> Result<(), AetherError> {
    if point.len() < 2 {
        return Err(AetherError::InvalidBufferSize {
            actual: point.len(),
            expected: 2,
        });
    }
    let spatial = match point.get(1..) {
        Some(s) => s,
        None => &[],
    };
    let new_x0 = lorentz_compute_x0(spatial, kappa)?;
    if let Some(x0_ref) = point.first_mut() {
        *x0_ref = new_x0;
    }
    Ok(())
}

/// Pure scalar reference implementation of Lorentzian inner product for verification.
#[inline]
pub fn scalar_lorentz_inner_product(x: &[f32], y: &[f32]) -> Result<f32, AetherError> {
    if x.len() != y.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: y.len(),
            expected: x.len(),
        });
    }
    if x.len() < 2 {
        return Err(AetherError::InvalidBufferSize {
            actual: x.len(),
            expected: 2,
        });
    }
    let x0 = match x.first() {
        Some(&v) => v,
        None => return Err(AetherError::OutOfBounds { index: 0, limit: x.len() }),
    };
    let y0 = match y.first() {
        Some(&v) => v,
        None => return Err(AetherError::OutOfBounds { index: 0, limit: y.len() }),
    };
    let spatial_x = match x.get(1..) {
        Some(s) => s,
        None => &[],
    };
    let spatial_y = match y.get(1..) {
        Some(s) => s,
        None => &[],
    };
    let spatial_dot = scalar_dot_product(spatial_x, spatial_y);
    Ok(-x0 * y0 + spatial_dot)
}

/// Pure scalar reference implementation of Lorentz distance for verification.
#[inline]
pub fn scalar_lorentz_distance(x: &[f32], y: &[f32], kappa: f32) -> Result<f32, AetherError> {
    if kappa >= 0.0 {
        return Err(AetherError::LorentzHorizonViolation);
    }
    if core::ptr::eq(x, y) {
        return Ok(0.0);
    }
    let inner_prod = scalar_lorentz_inner_product(x, y)?;
    let raw_z = kappa * inner_prod;
    let z = if raw_z <= 1.0 + HORIZON_EPSILON {
        1.0
    } else {
        raw_z
    };
    let acosh_z = libm::acoshf(z);
    let scale = 1.0 / libm::sqrtf(-kappa);
    Ok(scale * acosh_z)
}

/// Pure scalar reference implementation of $x_0$ computation for verification.
#[inline]
pub fn scalar_lorentz_compute_x0(spatial: &[f32], kappa: f32) -> Result<f32, AetherError> {
    if kappa >= 0.0 {
        return Err(AetherError::LorentzHorizonViolation);
    }
    let sum_sq = scalar_sum_of_squares(spatial);
    let radius_sq = 1.0 / (-kappa);
    Ok(libm::sqrtf(radius_sq + sum_sq))
}
