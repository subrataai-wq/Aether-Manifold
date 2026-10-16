//! Project Aether-Manifold: Tangent-Space Differential Geometry Operators
//!
//! Implements stabilized Exponential Map $\exp_x(v)$, Logarithmic Map $\log_x(y)$,
//! and Parallel Transport $P_{x \to y}(v)$ across:
//! - Lorentz Hyperboloid ($\mathbb{H}_{\kappa}^{d_H}$, $\kappa < 0$)
//! - Hyperspherical Space ($\mathbb{S}_{\tau}^{d_S}$, $\tau > 0$)
//! - Flat Euclidean Space ($\mathbb{R}^{d_E}$)
//! - Riemannian Product Manifolds ($\mathcal{M}$)
//!
//! In strict compliance with roadmap.md Sections 2.1, 2.4, 3.2, AGENTS.md Rule 3.1
//! (zero heap allocations), and AGENTS.md Rule 4.1 (numerical horizon stabilization).

use aether_core::error::AetherError;
use crate::hyperbolic::{lorentz_inner_product, lorentz_reproject};
use crate::product::ProductManifoldConfig;
use crate::simd::simd_dot_product;
use crate::spherical::spherical_normalize;

// ============================================================================
// 1. Lorentz Hyperbolic Space (\mathbb{H}_{\kappa}^{d_H}, \kappa < 0)
// ============================================================================

/// Project an ambient vector $v \in \mathbb{R}^{d_H+1}$ onto the tangent space
/// $T_x \mathbb{H}_{\kappa}^{d_H} = \{ v \mid \langle x, v \rangle_{\mathcal{L}} = 0 \}$
/// at base point $x$:
///
/// $$v_{\perp} = v - \kappa \langle x, v \rangle_{\mathcal{L}} x$$
///
/// # Invariants
/// - Surface invariant: $\langle x, x \rangle_{\mathcal{L}} = \frac{1}{\kappa} \implies \langle x, v_{\perp} \rangle_{\mathcal{L}} = 0$.
/// - Zero hot-path heap allocations (AGENTS.md Rule 3.1).
#[inline]
pub fn lorentz_project_tangent(
    base: &[f32],
    vec: &[f32],
    kappa: f32,
    out: &mut [f32],
) -> Result<(), AetherError> {
    if kappa >= 0.0 {
        return Err(AetherError::LorentzHorizonViolation);
    }
    if base.len() != vec.len() || base.len() != out.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: vec.len(),
            expected: base.len(),
        });
    }
    if base.len() < 2 {
        return Err(AetherError::InvalidBufferSize {
            actual: base.len(),
            expected: 2,
        });
    }

    let inner = lorentz_inner_product(base, vec)?;
    let factor = kappa * inner;

    for i in 0..base.len() {
        if let (Some(out_ref), Some(&v_val), Some(&b_val)) = (
            out.get_mut(i),
            vec.get(i),
            base.get(i),
        ) {
            *out_ref = v_val - factor * b_val;
        }
    }
    Ok(())
}

/// Compute the Lorentz Exponential Map $\exp_x(v) \in \mathbb{H}_{\kappa}^{d_H}$
/// mapping tangent vector $v \in T_x \mathbb{H}_{\kappa}^{d_H}$ to the hyperboloid surface:
///
/// $$\exp_x(v) = \cosh(\beta \|v\|_{\mathcal{L}}) x + \frac{\sinh(\beta \|v\|_{\mathcal{L}})}{\beta \|v\|_{\mathcal{L}}} v, \quad \beta = \sqrt{-\kappa}$$
///
/// Followed by in-place re-projection to guarantee exact hyperboloid surface membership.
///
/// # Numerical Safeguard (AGENTS.md Rule 4.1)
/// For micro-angles $\theta = \beta \|v\|_{\mathcal{L}} < 10^{-4}$, evaluates the asymptotic
/// Taylor expansion to prevent $0/0$ `NaN`:
/// $$\frac{\sinh(\theta)}{\theta} \approx 1 + \frac{\theta^2}{6} + \frac{\theta^4}{120}$$
/// $$\cosh(\theta) \approx 1 + \frac{\theta^2}{2} + \frac{\theta^4}{24}$$
#[inline]
pub fn lorentz_exp_map(
    base: &[f32],
    tangent: &[f32],
    kappa: f32,
    out: &mut [f32],
) -> Result<(), AetherError> {
    if kappa >= 0.0 {
        return Err(AetherError::LorentzHorizonViolation);
    }
    if base.len() != tangent.len() || base.len() != out.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: tangent.len(),
            expected: base.len(),
        });
    }
    if base.len() < 2 {
        return Err(AetherError::InvalidBufferSize {
            actual: base.len(),
            expected: 2,
        });
    }

    let beta = libm::sqrtf(-kappa);
    let v_norm_sq = lorentz_inner_product(tangent, tangent)?;
    let v_norm = libm::sqrtf(if v_norm_sq > 0.0 { v_norm_sq } else { 0.0 });
    let theta = beta * v_norm;

    let (coef_x, coef_v) = if theta < 1e-4 {
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let c_x = 1.0 + 0.5 * t2 + (1.0 / 24.0) * t4;
        let c_v = 1.0 + (1.0 / 6.0) * t2 + (1.0 / 120.0) * t4;
        (c_x, c_v)
    } else {
        (libm::coshf(theta), libm::sinhf(theta) / theta)
    };

    for i in 0..base.len() {
        if let (Some(out_ref), Some(&b_val), Some(&t_val)) = (
            out.get_mut(i),
            base.get(i),
            tangent.get(i),
        ) {
            *out_ref = coef_x * b_val + coef_v * t_val;
        }
    }

    lorentz_reproject(out, kappa)?;
    Ok(())
}

/// Compute the Lorentz Logarithmic Map $\log_x(y) \in T_x \mathbb{H}_{\kappa}^{d_H}$
/// mapping target hyperboloid point $y$ to the tangent space at base point $x$:
///
/// $$\log_x(y) = \frac{\operatorname{arcosh}(z)}{\sqrt{z^2 - 1}} (y - z x), \quad z = \max(1.0, \kappa \langle x, y \rangle_{\mathcal{L}})$$
///
/// # Numerical Safeguard (AGENTS.md Rule 4.1)
/// For near-identical points with $\delta = z - 1.0 < 10^{-5}$, evaluates the asymptotic
/// Taylor expansion to prevent catastrophic cancellation:
/// $$\frac{\operatorname{arcosh}(z)}{\sqrt{z^2 - 1}} \approx 1 - \frac{\delta}{3} + \frac{2\delta^2}{15} - \frac{3\delta^3}{35}$$
#[inline]
pub fn lorentz_log_map(
    base: &[f32],
    target: &[f32],
    kappa: f32,
    out: &mut [f32],
) -> Result<(), AetherError> {
    if kappa >= 0.0 {
        return Err(AetherError::LorentzHorizonViolation);
    }
    if base.len() != target.len() || base.len() != out.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: target.len(),
            expected: base.len(),
        });
    }
    if base.len() < 2 {
        return Err(AetherError::InvalidBufferSize {
            actual: base.len(),
            expected: 2,
        });
    }

    if core::ptr::eq(base, target) {
        for x in out.iter_mut() {
            *x = 0.0;
        }
        return Ok(());
    }

    let inner = lorentz_inner_product(base, target)?;
    let z_raw = kappa * inner;
    let z = if z_raw < 1.0 { 1.0 } else { z_raw };
    let delta = z - 1.0;

    let coef = if delta < 1e-5 {
        let d2 = delta * delta;
        let d3 = d2 * delta;
        1.0 - (1.0 / 3.0) * delta + (2.0 / 15.0) * d2 - (3.0 / 35.0) * d3
    } else {
        let acosh_z = libm::acoshf(z);
        let denom = libm::sqrtf(z * z - 1.0);
        if denom > 0.0 {
            acosh_z / denom
        } else {
            1.0
        }
    };

    for i in 0..base.len() {
        if let (Some(out_ref), Some(&t_val), Some(&b_val)) = (
            out.get_mut(i),
            target.get(i),
            base.get(i),
        ) {
            *out_ref = coef * (t_val - z * b_val);
        }
    }
    Ok(())
}

/// Compute Parallel Transport $P_{x \to y}(v) \in T_y \mathbb{H}_{\kappa}^{d_H}$ of tangent vector
/// $v \in T_x \mathbb{H}_{\kappa}^{d_H}$ along the unique geodesic connecting $x$ and $y$:
///
/// $$P_{x \to y}(v) = v - \frac{\kappa \langle y, v \rangle_{\mathcal{L}}}{1 + \kappa \langle x, y \rangle_{\mathcal{L}}} (x + y)$$
///
/// Guaranteed Riemannian isometry: $\langle P_{x \to y}(u), P_{x \to y}(v) \rangle_{\mathcal{L}} = \langle u, v \rangle_{\mathcal{L}}$.
#[inline]
pub fn lorentz_parallel_transport(
    base: &[f32],
    target: &[f32],
    tangent: &[f32],
    kappa: f32,
    out: &mut [f32],
) -> Result<(), AetherError> {
    if kappa >= 0.0 {
        return Err(AetherError::LorentzHorizonViolation);
    }
    if base.len() != target.len() || base.len() != tangent.len() || base.len() != out.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: target.len(),
            expected: base.len(),
        });
    }
    if base.len() < 2 {
        return Err(AetherError::InvalidBufferSize {
            actual: base.len(),
            expected: 2,
        });
    }

    if core::ptr::eq(base, target) {
        out.copy_from_slice(tangent);
        return Ok(());
    }

    let inner_xy = lorentz_inner_product(base, target)?;
    let inner_yv = lorentz_inner_product(target, tangent)?;

    let denom = 1.0 + kappa * inner_xy;
    if denom.abs() < 1e-7 {
        out.copy_from_slice(tangent);
        return Ok(());
    }

    let factor = (kappa * inner_yv) / denom;

    for i in 0..base.len() {
        if let (Some(out_ref), Some(&v_val), Some(&x_val), Some(&y_val)) = (
            out.get_mut(i),
            tangent.get(i),
            base.get(i),
            target.get(i),
        ) {
            *out_ref = v_val - factor * (x_val + y_val);
        }
    }
    Ok(())
}

// ============================================================================
// 2. Hyperspherical Space (\mathbb{S}_{\tau}^{d_S}, \tau > 0)
// ============================================================================

/// Project an ambient vector $v \in \mathbb{R}^{d_S}$ onto the tangent space
/// $T_x \mathbb{S}_{\tau}^{d_S} = \{ v \mid \langle x, v \rangle = 0 \}$
/// at unit base point $x$:
///
/// $$v_{\perp} = v - \langle x, v \rangle x$$
#[inline]
pub fn spherical_project_tangent(
    base: &[f32],
    vec: &[f32],
    out: &mut [f32],
) -> Result<(), AetherError> {
    if base.len() != vec.len() || base.len() != out.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: vec.len(),
            expected: base.len(),
        });
    }
    if base.is_empty() {
        return Err(AetherError::InvalidBufferSize {
            actual: 0,
            expected: 1,
        });
    }

    let dot = simd_dot_product(base, vec);
    for i in 0..base.len() {
        if let (Some(out_ref), Some(&v_val), Some(&b_val)) = (
            out.get_mut(i),
            vec.get(i),
            base.get(i),
        ) {
            *out_ref = v_val - dot * b_val;
        }
    }
    Ok(())
}

/// Compute the Spherical Exponential Map $\exp_x(v) \in \mathbb{S}_{\tau}^{d_S}$
/// mapping tangent vector $v \in T_x \mathbb{S}_{\tau}^{d_S}$ to the unit hypersphere:
///
/// $$\exp_x(v) = \cos(\alpha \|v\|_2) x + \alpha \frac{\sin(\alpha \|v\|_2)}{\alpha \|v\|_2} v, \quad \alpha = \sqrt{\tau}$$
///
/// Followed by in-place normalization to guarantee exact unit sphere surface membership.
///
/// # Numerical Safeguard (AGENTS.md Rule 4.1)
/// For micro-angles $\theta = \alpha \|v\|_2 < 10^{-4}$, evaluates the asymptotic
/// Taylor expansion to prevent $0/0$ `NaN`:
/// $$\frac{\sin(\theta)}{\theta} \approx 1 - \frac{\theta^2}{6} + \frac{\theta^4}{120}$$
/// $$\cos(\theta) \approx 1 - \frac{\theta^2}{2} + \frac{\theta^4}{24}$$
#[inline]
pub fn spherical_exp_map(
    base: &[f32],
    tangent: &[f32],
    tau: f32,
    out: &mut [f32],
) -> Result<(), AetherError> {
    if tau <= 0.0 {
        return Err(AetherError::LorentzHorizonViolation);
    }
    if base.len() != tangent.len() || base.len() != out.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: tangent.len(),
            expected: base.len(),
        });
    }
    if base.is_empty() {
        return Err(AetherError::InvalidBufferSize {
            actual: 0,
            expected: 1,
        });
    }

    let alpha = libm::sqrtf(tau);
    let v_norm_sq = simd_dot_product(tangent, tangent);
    let v_norm = libm::sqrtf(if v_norm_sq > 0.0 { v_norm_sq } else { 0.0 });
    let theta = alpha * v_norm;

    let (coef_x, coef_v) = if theta < 1e-4 {
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let c_x = 1.0 - 0.5 * t2 + (1.0 / 24.0) * t4;
        let c_v = alpha * (1.0 - (1.0 / 6.0) * t2 + (1.0 / 120.0) * t4);
        (c_x, c_v)
    } else {
        (libm::cosf(theta), alpha * (libm::sinf(theta) / theta))
    };

    for i in 0..base.len() {
        if let (Some(out_ref), Some(&b_val), Some(&t_val)) = (
            out.get_mut(i),
            base.get(i),
            tangent.get(i),
        ) {
            *out_ref = coef_x * b_val + coef_v * t_val;
        }
    }

    spherical_normalize(out)?;
    Ok(())
}

/// Compute the Spherical Logarithmic Map $\log_x(y) \in T_x \mathbb{S}_{\tau}^{d_S}$
/// mapping target unit sphere point $y$ to the tangent space at base point $x$:
///
/// $$\log_x(y) = \frac{\arccos(z)}{\alpha \sqrt{1 - z^2}} (y - z x), \quad z = \operatorname{clamp}(\langle x, y \rangle, -1.0, 1.0), \quad \alpha = \sqrt{\tau}$$
///
/// # Numerical Safeguard (AGENTS.md Rule 4.1)
/// For near-identical points with $\delta = 1.0 - z < 10^{-5}$, evaluates the asymptotic
/// Taylor expansion to prevent catastrophic cancellation:
/// $$\frac{\arccos(z)}{\sqrt{1 - z^2}} \approx 1 + \frac{\delta}{3} + \frac{2\delta^2}{15} + \frac{3\delta^3}{35}$$
#[inline]
pub fn spherical_log_map(
    base: &[f32],
    target: &[f32],
    tau: f32,
    out: &mut [f32],
) -> Result<(), AetherError> {
    if tau <= 0.0 {
        return Err(AetherError::LorentzHorizonViolation);
    }
    if base.len() != target.len() || base.len() != out.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: target.len(),
            expected: base.len(),
        });
    }
    if base.is_empty() {
        return Err(AetherError::InvalidBufferSize {
            actual: 0,
            expected: 1,
        });
    }

    if core::ptr::eq(base, target) {
        for x in out.iter_mut() {
            *x = 0.0;
        }
        return Ok(());
    }

    let alpha = libm::sqrtf(tau);
    let dot = simd_dot_product(base, target);
    let z = dot.clamp(-1.0, 1.0);
    let delta = (1.0 - z).max(0.0);

    let factor = if delta < 1e-5 {
        let d2 = delta * delta;
        let d3 = d2 * delta;
        let geom_series = 1.0 + (1.0 / 3.0) * delta + (2.0 / 15.0) * d2 + (3.0 / 35.0) * d3;
        (1.0 / alpha) * geom_series
    } else {
        let acos_z = libm::acosf(z);
        let denom = libm::sqrtf((1.0 - z * z).max(0.0));
        if denom > 0.0 {
            acos_z / (alpha * denom)
        } else {
            1.0 / alpha
        }
    };

    for i in 0..base.len() {
        if let (Some(out_ref), Some(&t_val), Some(&b_val)) = (
            out.get_mut(i),
            target.get(i),
            base.get(i),
        ) {
            *out_ref = factor * (t_val - z * b_val);
        }
    }
    Ok(())
}

/// Compute Parallel Transport $P_{x \to y}(v) \in T_y \mathbb{S}_{\tau}^{d_S}$ of tangent vector
/// $v \in T_x \mathbb{S}_{\tau}^{d_S}$ along the great circle connecting $x$ and $y$:
///
/// $$P_{x \to y}(v) = v - \frac{\langle y, v \rangle}{1 + \langle x, y \rangle} (x + y)$$
///
/// Guaranteed Riemannian isometry: $\langle P_{x \to y}(u), P_{x \to y}(v) \rangle = \langle u, v \rangle$.
#[inline]
pub fn spherical_parallel_transport(
    base: &[f32],
    target: &[f32],
    tangent: &[f32],
    out: &mut [f32],
) -> Result<(), AetherError> {
    if base.len() != target.len() || base.len() != tangent.len() || base.len() != out.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: target.len(),
            expected: base.len(),
        });
    }
    if base.is_empty() {
        return Err(AetherError::InvalidBufferSize {
            actual: 0,
            expected: 1,
        });
    }

    if core::ptr::eq(base, target) {
        out.copy_from_slice(tangent);
        return Ok(());
    }

    let inner_xy = simd_dot_product(base, target);
    let inner_yv = simd_dot_product(target, tangent);

    let denom = 1.0 + inner_xy;
    if denom.abs() < 1e-7 {
        out.copy_from_slice(tangent);
        return Ok(());
    }

    let factor = inner_yv / denom;

    for i in 0..base.len() {
        if let (Some(out_ref), Some(&v_val), Some(&x_val), Some(&y_val)) = (
            out.get_mut(i),
            tangent.get(i),
            base.get(i),
            target.get(i),
        ) {
            *out_ref = v_val - factor * (x_val + y_val);
        }
    }
    Ok(())
}

// ============================================================================
// 3. Flat Euclidean Space (\mathbb{R}^{d_E})
// ============================================================================

/// Project an ambient vector onto Euclidean tangent space (identity operation $v_{\perp} = v$).
#[inline]
pub fn euclidean_project_tangent(vec: &[f32], out: &mut [f32]) -> Result<(), AetherError> {
    if vec.len() != out.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: out.len(),
            expected: vec.len(),
        });
    }
    out.copy_from_slice(vec);
    Ok(())
}

/// Compute the Euclidean Exponential Map $\exp_x(v) = x + v$.
#[inline]
pub fn euclidean_exp_map(base: &[f32], tangent: &[f32], out: &mut [f32]) -> Result<(), AetherError> {
    if base.len() != tangent.len() || base.len() != out.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: tangent.len(),
            expected: base.len(),
        });
    }
    for i in 0..base.len() {
        if let (Some(out_ref), Some(&b_val), Some(&t_val)) = (
            out.get_mut(i),
            base.get(i),
            tangent.get(i),
        ) {
            *out_ref = b_val + t_val;
        }
    }
    Ok(())
}

/// Compute the Euclidean Logarithmic Map $\log_x(y) = y - x$.
#[inline]
pub fn euclidean_log_map(base: &[f32], target: &[f32], out: &mut [f32]) -> Result<(), AetherError> {
    if base.len() != target.len() || base.len() != out.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: target.len(),
            expected: base.len(),
        });
    }
    for i in 0..base.len() {
        if let (Some(out_ref), Some(&t_val), Some(&b_val)) = (
            out.get_mut(i),
            target.get(i),
            base.get(i),
        ) {
            *out_ref = t_val - b_val;
        }
    }
    Ok(())
}

/// Compute Euclidean Parallel Transport $P_{x \to y}(v) = v$ (identity translation in flat space).
#[inline]
pub fn euclidean_parallel_transport(tangent: &[f32], out: &mut [f32]) -> Result<(), AetherError> {
    if tangent.len() != out.len() {
        return Err(AetherError::InvalidBufferSize {
            actual: out.len(),
            expected: tangent.len(),
        });
    }
    out.copy_from_slice(tangent);
    Ok(())
}

// ============================================================================
// 4. Riemannian Product Manifold (\mathcal{M})
// ============================================================================

/// Compute the Product Manifold Exponential Map $\exp_x(v) \in \mathcal{M}$:
///
/// $$\exp_x(v) = \left( \exp_{x^{(H, 1)}}(v^{(H, 1)}), \dots, \exp_{x^{(S, 1)}}(v^{(S, 1)}), \dots, x^{(E)} + v^{(E)} \right)$$
///
/// Executes component-wise in-place with strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
#[inline]
pub fn product_exp_map(
    config: &ProductManifoldConfig,
    base: &[f32],
    tangent: &[f32],
    out: &mut [f32],
) -> Result<(), AetherError> {
    let total_dim = config.total_dim();
    if base.len() != total_dim || tangent.len() != total_dim || out.len() != total_dim {
        return Err(AetherError::InvalidBufferSize {
            actual: base.len(),
            expected: total_dim,
        });
    }

    let mut offset = 0;

    for h in &config.hyperbolic {
        let next_offset = offset + h.dim;
        let sub_base = match base.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: base.len() }),
        };
        let sub_tangent = match tangent.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: tangent.len() }),
        };
        let sub_out = match out.get_mut(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: out.len() }),
        };

        lorentz_exp_map(sub_base, sub_tangent, h.kappa, sub_out)?;
        offset = next_offset;
    }

    for s in &config.spherical {
        let next_offset = offset + s.dim;
        let sub_base = match base.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: base.len() }),
        };
        let sub_tangent = match tangent.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: tangent.len() }),
        };
        let sub_out = match out.get_mut(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: out.len() }),
        };

        spherical_exp_map(sub_base, sub_tangent, s.tau, sub_out)?;
        offset = next_offset;
    }

    if let Some(e) = config.euclidean {
        let next_offset = offset + e.dim;
        let sub_base = match base.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: base.len() }),
        };
        let sub_tangent = match tangent.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: tangent.len() }),
        };
        let sub_out = match out.get_mut(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: out.len() }),
        };

        euclidean_exp_map(sub_base, sub_tangent, sub_out)?;
    }

    Ok(())
}

/// Compute the Product Manifold Logarithmic Map $\log_x(y) \in T_x \mathcal{M}$:
///
/// $$\log_x(y) = \left( \log_{x^{(H, 1)}}(y^{(H, 1)}), \dots, \log_{x^{(S, 1)}}(y^{(S, 1)}), \dots, y^{(E)} - x^{(E)} \right)$$
///
/// Executes component-wise in-place with strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
#[inline]
pub fn product_log_map(
    config: &ProductManifoldConfig,
    base: &[f32],
    target: &[f32],
    out: &mut [f32],
) -> Result<(), AetherError> {
    let total_dim = config.total_dim();
    if base.len() != total_dim || target.len() != total_dim || out.len() != total_dim {
        return Err(AetherError::InvalidBufferSize {
            actual: base.len(),
            expected: total_dim,
        });
    }

    let mut offset = 0;

    for h in &config.hyperbolic {
        let next_offset = offset + h.dim;
        let sub_base = match base.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: base.len() }),
        };
        let sub_target = match target.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: target.len() }),
        };
        let sub_out = match out.get_mut(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: out.len() }),
        };

        lorentz_log_map(sub_base, sub_target, h.kappa, sub_out)?;
        offset = next_offset;
    }

    for s in &config.spherical {
        let next_offset = offset + s.dim;
        let sub_base = match base.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: base.len() }),
        };
        let sub_target = match target.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: target.len() }),
        };
        let sub_out = match out.get_mut(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: out.len() }),
        };

        spherical_log_map(sub_base, sub_target, s.tau, sub_out)?;
        offset = next_offset;
    }

    if let Some(e) = config.euclidean {
        let next_offset = offset + e.dim;
        let sub_base = match base.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: base.len() }),
        };
        let sub_target = match target.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: target.len() }),
        };
        let sub_out = match out.get_mut(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: out.len() }),
        };

        euclidean_log_map(sub_base, sub_target, sub_out)?;
    }

    Ok(())
}

/// Compute Product Manifold Parallel Transport $P_{x \to y}(v) \in T_y \mathcal{M}$
/// across all component sub-manifolds:
///
/// $$P_{x \to y}(v) = \left( P_{x^{(H, 1)} \to y^{(H, 1)}}(v^{(H, 1)}), \dots, P_{x^{(S, 1)} \to y^{(S, 1)}}(v^{(S, 1)}), \dots, v^{(E)} \right)$$
///
/// Executes component-wise in-place with strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
#[inline]
pub fn product_parallel_transport(
    config: &ProductManifoldConfig,
    base: &[f32],
    target: &[f32],
    tangent: &[f32],
    out: &mut [f32],
) -> Result<(), AetherError> {
    let total_dim = config.total_dim();
    if base.len() != total_dim || target.len() != total_dim || tangent.len() != total_dim || out.len() != total_dim {
        return Err(AetherError::InvalidBufferSize {
            actual: base.len(),
            expected: total_dim,
        });
    }

    let mut offset = 0;

    for h in &config.hyperbolic {
        let next_offset = offset + h.dim;
        let sub_base = match base.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: base.len() }),
        };
        let sub_target = match target.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: target.len() }),
        };
        let sub_tangent = match tangent.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: tangent.len() }),
        };
        let sub_out = match out.get_mut(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: out.len() }),
        };

        lorentz_parallel_transport(sub_base, sub_target, sub_tangent, h.kappa, sub_out)?;
        offset = next_offset;
    }

    for s in &config.spherical {
        let next_offset = offset + s.dim;
        let sub_base = match base.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: base.len() }),
        };
        let sub_target = match target.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: target.len() }),
        };
        let sub_tangent = match tangent.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: tangent.len() }),
        };
        let sub_out = match out.get_mut(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: out.len() }),
        };

        spherical_parallel_transport(sub_base, sub_target, sub_tangent, sub_out)?;
        offset = next_offset;
    }

    if let Some(e) = config.euclidean {
        let next_offset = offset + e.dim;
        let sub_tangent = match tangent.get(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: tangent.len() }),
        };
        let sub_out = match out.get_mut(offset..next_offset) {
            Some(s) => s,
            None => return Err(AetherError::OutOfBounds { index: next_offset, limit: out.len() }),
        };

        euclidean_parallel_transport(sub_tangent, sub_out)?;
    }

    Ok(())
}
