//! Project Aether-Manifold: Riemannian Product Manifold ($\mathcal{M}$) Metric Kernels
//!
//! Implements the unified geodesic distance metric over heterogeneous product geometries:
//! $$\mathcal{M} = \bigotimes_{i=1}^{k} \mathbb{H}_{\kappa_i}^{d_{H,i}} \times \bigotimes_{j=1}^{m} \mathbb{S}_{\tau_j}^{d_{S,j}} \times \mathbb{R}^{d_E}$$
//! $$d_{\mathcal{M}}^2(u, v) = \sum_{i=1}^{k} w_{H,i} d_{\mathbb{H}_{\kappa_i}}^2\left(u^{(H,i)}, v^{(H,i)}\right) + \sum_{j=1}^{m} w_{S,j} d_{\mathbb{S}_{\tau_j}}^2\left(u^{(S,j)}, v^{(S,j)}\right) + w_E \left\| u^{(E)} - v^{(E)} \right\|_2^2$$
//! $$d_{\mathcal{M}}(u, v) = \sqrt{d_{\mathcal{M}}^2(u, v)}$$
//!
//! In strict compliance with roadmap.md Section 2.1, AGENTS.md Rule 3.1, and AGENTS.md Rule 4.1.

extern crate alloc;
use alloc::vec::Vec;

use crate::euclidean::{euclidean_distance_squared, scalar_euclidean_distance_squared_checked};
use crate::hyperbolic::{lorentz_distance_squared, scalar_lorentz_distance};
use crate::spherical::{scalar_spherical_distance, spherical_distance_squared};
use aether_core::error::AetherError;

/// Specification for a hyperbolic Lorentz component $\mathbb{H}_{\kappa}^{d_H}$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HyperbolicSpec {
    /// Dimension of the component (including the time-like coordinate $x_0$, so $d_H + 1$).
    pub dim: usize,
    /// Negative curvature parameter $\kappa < 0$.
    pub kappa: f32,
    /// Metric weighting coefficient $w_H > 0$.
    pub weight: f32,
}

/// Specification for a hyperspherical component $\mathbb{S}_{\tau}^{d_S}$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphericalSpec {
    /// Dimension of the component.
    pub dim: usize,
    /// Positive curvature parameter $\tau > 0$.
    pub tau: f32,
    /// Metric weighting coefficient $w_S > 0$.
    pub weight: f32,
}

/// Specification for a flat Euclidean component $\mathbb{R}^{d_E}$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EuclideanSpec {
    /// Dimension of the Euclidean component.
    pub dim: usize,
    /// Metric weighting coefficient $w_E > 0$.
    pub weight: f32,
}

/// Product Manifold Configuration defining components and weights.
#[derive(Clone, Debug, PartialEq)]
pub struct ProductManifoldConfig {
    /// Hyperbolic space components.
    pub hyperbolic: Vec<HyperbolicSpec>,
    /// Spherical space components.
    pub spherical: Vec<SphericalSpec>,
    /// Optional Euclidean space component.
    pub euclidean: Option<EuclideanSpec>,
}

impl ProductManifoldConfig {
    /// Compute the total combined dimension of all manifold components.
    pub fn total_dim(&self) -> usize {
        let mut total = 0;
        for h in &self.hyperbolic {
            total += h.dim;
        }
        for s in &self.spherical {
            total += s.dim;
        }
        if let Some(e) = self.euclidean {
            total += e.dim;
        }
        total
    }
}

/// Compute the squared unified geodesic distance $d_{\mathcal{M}}^2(u, v)$ across the product manifold
/// using runtime-dispatched SIMD with 4x unrolled accumulators.
///
/// Returns `Err(AetherError::InvalidBufferSize)` if vector lengths do not match or do not equal `config.total_dim()`.
pub fn product_manifold_distance_squared(
    config: &ProductManifoldConfig,
    u: &[f32],
    v: &[f32],
) -> Result<f32, AetherError> {
    if core::ptr::eq(u, v) {
        let expected_dim = config.total_dim();
        if u.len() != expected_dim {
            return Err(AetherError::InvalidBufferSize {
                actual: u.len(),
                expected: expected_dim,
            });
        }
        return Ok(0.0);
    }
    if u.len() != v.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: v.len(),
            expected: u.len(),
        });
    }
    let expected_dim = config.total_dim();
    if u.len() != expected_dim {
        return Err(AetherError::InvalidBufferSize {
            actual: u.len(),
            expected: expected_dim,
        });
    }

    let mut offset = 0;
    let mut total_d2 = 0.0f32;

    for h in &config.hyperbolic {
        let next_offset = offset + h.dim;
        let sub_u = match u.get(offset..next_offset) {
            Some(s) => s,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: next_offset,
                    limit: u.len(),
                })
            }
        };
        let sub_v = match v.get(offset..next_offset) {
            Some(s) => s,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: next_offset,
                    limit: v.len(),
                })
            }
        };
        let d2 = lorentz_distance_squared(sub_u, sub_v, h.kappa)?;
        total_d2 += h.weight * d2;
        offset = next_offset;
    }

    for s in &config.spherical {
        let next_offset = offset + s.dim;
        let sub_u = match u.get(offset..next_offset) {
            Some(s) => s,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: next_offset,
                    limit: u.len(),
                })
            }
        };
        let sub_v = match v.get(offset..next_offset) {
            Some(s) => s,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: next_offset,
                    limit: v.len(),
                })
            }
        };
        let d2 = spherical_distance_squared(sub_u, sub_v, s.tau)?;
        total_d2 += s.weight * d2;
        offset = next_offset;
    }

    if let Some(e) = config.euclidean {
        let next_offset = offset + e.dim;
        let sub_u = match u.get(offset..next_offset) {
            Some(s) => s,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: next_offset,
                    limit: u.len(),
                })
            }
        };
        let sub_v = match v.get(offset..next_offset) {
            Some(s) => s,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: next_offset,
                    limit: v.len(),
                })
            }
        };
        let d2 = euclidean_distance_squared(sub_u, sub_v)?;
        total_d2 += e.weight * d2;
    }

    Ok(total_d2)
}

/// Compute the unified geodesic distance $d_{\mathcal{M}}(u, v) = \sqrt{d_{\mathcal{M}}^2(u, v)}$.
pub fn product_manifold_distance(
    config: &ProductManifoldConfig,
    u: &[f32],
    v: &[f32],
) -> Result<f32, AetherError> {
    let d2 = product_manifold_distance_squared(config, u, v)?;
    Ok(libm::sqrtf(d2))
}

/// Pure scalar reference implementation of product manifold squared distance for verification.
pub fn scalar_product_manifold_distance_squared(
    config: &ProductManifoldConfig,
    u: &[f32],
    v: &[f32],
) -> Result<f32, AetherError> {
    if u.len() != v.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: v.len(),
            expected: u.len(),
        });
    }
    let expected_dim = config.total_dim();
    if u.len() != expected_dim {
        return Err(AetherError::InvalidBufferSize {
            actual: u.len(),
            expected: expected_dim,
        });
    }

    let mut offset = 0;
    let mut total_d2 = 0.0f32;

    for h in &config.hyperbolic {
        let next_offset = offset + h.dim;
        let sub_u = match u.get(offset..next_offset) {
            Some(s) => s,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: next_offset,
                    limit: u.len(),
                })
            }
        };
        let sub_v = match v.get(offset..next_offset) {
            Some(s) => s,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: next_offset,
                    limit: v.len(),
                })
            }
        };
        let d = scalar_lorentz_distance(sub_u, sub_v, h.kappa)?;
        total_d2 += h.weight * (d * d);
        offset = next_offset;
    }

    for s in &config.spherical {
        let next_offset = offset + s.dim;
        let sub_u = match u.get(offset..next_offset) {
            Some(s) => s,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: next_offset,
                    limit: u.len(),
                })
            }
        };
        let sub_v = match v.get(offset..next_offset) {
            Some(s) => s,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: next_offset,
                    limit: v.len(),
                })
            }
        };
        let d = scalar_spherical_distance(sub_u, sub_v, s.tau)?;
        total_d2 += s.weight * (d * d);
        offset = next_offset;
    }

    if let Some(e) = config.euclidean {
        let next_offset = offset + e.dim;
        let sub_u = match u.get(offset..next_offset) {
            Some(s) => s,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: next_offset,
                    limit: u.len(),
                })
            }
        };
        let sub_v = match v.get(offset..next_offset) {
            Some(s) => s,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: next_offset,
                    limit: v.len(),
                })
            }
        };
        let d2 = scalar_euclidean_distance_squared_checked(sub_u, sub_v)?;
        total_d2 += e.weight * d2;
    }

    Ok(total_d2)
}

/// Pure scalar reference implementation of product manifold distance for verification.
pub fn scalar_product_manifold_distance(
    config: &ProductManifoldConfig,
    u: &[f32],
    v: &[f32],
) -> Result<f32, AetherError> {
    let d2 = scalar_product_manifold_distance_squared(config, u, v)?;
    Ok(libm::sqrtf(d2))
}
