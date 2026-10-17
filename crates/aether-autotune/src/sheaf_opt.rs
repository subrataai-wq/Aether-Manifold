//! Project Aether-Manifold: Continuous Householder Sheaf Morphism Optimizer
//!
//! Optimizes Cellular Sheaf restriction morphisms $\mathcal{F}_{v \trianglelefteq e}$ to minimize
//! discrete Dirichlet energy while preserving exact isometry $\|\mathcal{F}_{v \trianglelefteq e}\|_2 = 1$
//! identically via Householder reflection manifolds (AGENTS.md Rule 4.2).

use aether_core::error::AetherError;
use aether_query::sheaf::{EdgeStalk, SheafRestrictionMap, VertexStalk, DEFAULT_STALK_DIM};

/// Continuous gradient descent optimizer for Cellular Sheaf restriction morphisms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SheafMorphismOptimizer<const D: usize = DEFAULT_STALK_DIM> {
    /// Learning rate / smoothing step size eta in (0.0, 1.0].
    pub learning_rate: f32,
    /// Minimum gradient norm threshold for convergence.
    pub convergence_epsilon: f32,
}

impl<const D: usize> Default for SheafMorphismOptimizer<D> {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            convergence_epsilon: 1e-5,
        }
    }
}

impl<const D: usize> SheafMorphismOptimizer<D> {
    /// Instantiates a new `SheafMorphismOptimizer`.
    pub fn new(learning_rate: f32, convergence_epsilon: f32) -> Result<Self, AetherError> {
        if learning_rate <= 0.0 || learning_rate > 1.0 {
            return Err(AetherError::OutOfBounds { index: 0, limit: 1 });
        }
        Ok(Self {
            learning_rate,
            convergence_epsilon,
        })
    }

    /// Evaluates discrete Dirichlet energy for an edge:
    /// E_e = 0.5 * || F_{u <= e} x_u - F_{v <= e} x_v ||_2^2
    #[inline]
    pub fn compute_dirichlet_energy(
        &self,
        map_u: &SheafRestrictionMap<D>,
        map_v: &SheafRestrictionMap<D>,
        stalk_u: &VertexStalk<D>,
        stalk_v: &VertexStalk<D>,
    ) -> f32 {
        let mut edge_u = EdgeStalk::<D>::default();
        let mut edge_v = EdgeStalk::<D>::default();
        map_u.apply(stalk_u, &mut edge_u);
        map_v.apply(stalk_v, &mut edge_v);

        let mut diff_sq = 0.0f32;
        for i in 0..D {
            let du = edge_u.components.get(i).copied().unwrap_or(0.0);
            let dv = edge_v.components.get(i).copied().unwrap_or(0.0);
            let diff = du - dv;
            diff_sq += diff * diff;
        }
        0.5 * diff_sq
    }

    /// Performs one step of Riemannian gradient descent on the Householder reflector
    /// for F_{u <= e} towards F_{v <= e} x_v, reducing Dirichlet mismatch energy while
    /// preserving exact isometry ||F_{u <= e}||_2 = 1.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1 & 4.2).
    pub fn step_optimize(
        &self,
        current_map_u: &SheafRestrictionMap<D>,
        current_map_v: &SheafRestrictionMap<D>,
        stalk_u: &VertexStalk<D>,
        stalk_v: &VertexStalk<D>,
    ) -> SheafRestrictionMap<D> {
        let mut current_image_u = EdgeStalk::<D>::default();
        let mut target_image_v = EdgeStalk::<D>::default();
        current_map_u.apply(stalk_u, &mut current_image_u);
        current_map_v.apply(stalk_v, &mut target_image_v);

        let u_norm = stalk_u.norm();
        if u_norm < 1e-7 {
            return *current_map_u;
        }

        // Interpolated target in edge stalk space:
        // z = (1 - eta) * current_image_u + eta * target_image_v
        let eta = self.learning_rate;
        let mut target_dir = [0.0f32; D];
        let mut target_norm_sq = 0.0f32;

        for i in 0..D {
            let cu = current_image_u.components.get(i).copied().unwrap_or(0.0);
            let tv = target_image_v.components.get(i).copied().unwrap_or(0.0);
            let z = (1.0 - eta) * cu + eta * tv;
            if let Some(slot) = target_dir.get_mut(i) {
                *slot = z;
            }
            target_norm_sq += z * z;
        }

        let target_norm = libm::sqrtf(target_norm_sq);
        if target_norm < 1e-7 {
            return *current_map_u;
        }

        // Normalize source x_u and target z to unit vectors
        let mut x_hat = [0.0f32; D];
        let mut z_hat = [0.0f32; D];
        let inv_u = 1.0 / u_norm;
        let inv_z = 1.0 / target_norm;

        for i in 0..D {
            if let (Some(x_slot), Some(z_slot)) = (x_hat.get_mut(i), z_hat.get_mut(i)) {
                *x_slot = stalk_u.components.get(i).copied().unwrap_or(0.0) * inv_u;
                *z_slot = target_dir.get(i).copied().unwrap_or(0.0) * inv_z;
            }
        }

        // Optimal Householder reflection vector: w = normalize(x_hat - z_hat)
        let mut diff_w = [0.0f32; D];
        let mut w_norm_sq = 0.0f32;

        for i in 0..D {
            let xi = x_hat.get(i).copied().unwrap_or(0.0);
            let zi = z_hat.get(i).copied().unwrap_or(0.0);
            let diff = xi - zi;
            if let Some(slot) = diff_w.get_mut(i) {
                *slot = diff;
            }
            w_norm_sq += diff * diff;
        }

        if w_norm_sq < self.convergence_epsilon {
            // Already aligned
            return *current_map_u;
        }

        // New orthogonal Householder map with ||H_w||_2 = 1
        SheafRestrictionMap::from_reflector(&diff_w, current_map_u.attenuation)
    }
}
