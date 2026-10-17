//! Project Aether-Manifold: Sheaf Dirichlet Energy Solver
//!
//! Solves the joint topological-geometric objective:
//! $$\min_{x \in C^0(G; \mathcal{F})} \left( \frac{1}{2} x^T L_{\mathcal{F}} x + \frac{\lambda}{2} \sum_{v \in V} d_{\mathcal{M}}^2(\exp_v(x_v), q) \right)$$
//!
//! Executes with strictly zero heap allocations on caller-provided scratchpads (Rule 3.1).

use aether_core::error::AetherError;

use crate::laplacian::SheafCoboundary;
use crate::sheaf::{EdgeStalk, SheafRestrictionMap, VertexStalk};

/// Configuration parameters for the Sheaf Dirichlet Energy Solver.
#[derive(Clone, Copy, Debug)]
pub struct DirichletSolverConfig {
    /// Geodesic anchor fidelity weight lambda >= 0.
    pub lambda: f32,
    /// Gradient step size eta.
    pub step_size: f32,
    /// Maximum optimization iterations.
    pub max_iterations: usize,
    /// Gradient norm convergence threshold.
    pub tolerance: f32,
}

impl Default for DirichletSolverConfig {
    #[inline]
    fn default() -> Self {
        Self {
            lambda: 0.5,
            step_size: 0.05,
            max_iterations: 20,
            tolerance: 1e-4,
        }
    }
}

/// Incident edge descriptor linking vertex $u$ to a neighbor vertex by cluster index.
#[derive(Clone, Copy, Debug)]
pub struct ClusterEdgeDescriptor<const D: usize> {
    /// Index of neighbor vertex in the cluster slice.
    pub neighbor_index: usize,
    /// Source restriction map $\mathcal{F}_{u \trianglelefteq e}$.
    pub map_u: SheafRestrictionMap<D>,
    /// Target restriction map $\mathcal{F}_{v \trianglelefteq e}$.
    pub map_v: SheafRestrictionMap<D>,
}

/// Sheaf Dirichlet Energy Solver.
pub struct SheafDirichletSolver;

impl SheafDirichletSolver {
    /// Compute the gradient of the Dirichlet objective at vertex $u$:
    /// $$\nabla_{x_u} E(x) = (L_{\mathcal{F}} x)_u + \lambda (x_u - v_{\text{target}})$$
    #[inline]
    pub fn compute_vertex_gradient<const D: usize>(
        x_u: &VertexStalk<D>,
        target: &VertexStalk<D>,
        cluster_stalks: &[VertexStalk<D>],
        incident_edges: &[ClusterEdgeDescriptor<D>],
        lambda: f32,
        out_grad: &mut VertexStalk<D>,
    ) {
        // Clear accumulator
        for c in out_grad.components.iter_mut() {
            *c = 0.0f32;
        }

        let mut delta_e = EdgeStalk::<D>::default();
        let mut back_proj = VertexStalk::<D>::default();

        for edge in incident_edges {
            let x_v = match cluster_stalks.get(edge.neighbor_index) {
                Some(v) => v,
                None => continue,
            };

            // 1. (delta x)_e = F_{u <= e} x_u - F_{v <= e} x_v
            SheafCoboundary::evaluate_edge(&edge.map_u, x_u, &edge.map_v, x_v, &mut delta_e);

            // 2. F_{u <= e}^T (delta x)_e
            edge.map_u.apply_adjoint(&delta_e, &mut back_proj);

            // 3. Accumulate (L_F x)_u
            for (accum, val) in out_grad
                .components
                .iter_mut()
                .zip(back_proj.components.iter())
            {
                *accum += *val;
            }
        }

        // Add anchor attraction: + lambda * (x_u - target)
        for ((grad_comp, x_comp), t_comp) in out_grad
            .components
            .iter_mut()
            .zip(x_u.components.iter())
            .zip(target.components.iter())
        {
            *grad_comp += lambda * (*x_comp - *t_comp);
        }
    }

    /// Solve the Dirichlet energy objective over a local active cluster of vertices.
    ///
    /// Iterates until convergence or `max_iterations`, allocating zero heap bytes.
    pub fn solve_local_cluster<const D: usize>(
        cluster_stalks: &mut [VertexStalk<D>],
        target_attractions: &[VertexStalk<D>],
        adjacency_edges: &[&[ClusterEdgeDescriptor<D>]],
        config: &DirichletSolverConfig,
    ) -> Result<usize, AetherError> {
        let num_vertices = cluster_stalks.len();
        if target_attractions.len() < num_vertices || adjacency_edges.len() < num_vertices {
            return Err(AetherError::InvalidBufferSize {
                actual: target_attractions.len().min(adjacency_edges.len()),
                expected: num_vertices,
            });
        }

        let mut iterations_run = 0;
        let mut grad = VertexStalk::<D>::default();

        for iter in 0..config.max_iterations {
            iterations_run = iter + 1;
            let mut max_grad_norm: f32 = 0.0;

            for i in 0..num_vertices {
                let target = match target_attractions.get(i) {
                    Some(t) => t,
                    None => continue,
                };
                let edges = match adjacency_edges.get(i) {
                    Some(e) => *e,
                    None => continue,
                };

                let x_u_curr = match cluster_stalks.get(i) {
                    Some(s) => *s,
                    None => continue,
                };

                Self::compute_vertex_gradient(
                    &x_u_curr,
                    target,
                    cluster_stalks,
                    edges,
                    config.lambda,
                    &mut grad,
                );

                let mut grad_norm_sq = 0.0f32;
                for g in &grad.components {
                    grad_norm_sq += *g * *g;
                }
                let grad_norm = libm::sqrtf(grad_norm_sq);
                if grad_norm > max_grad_norm {
                    max_grad_norm = grad_norm;
                }

                // Update x_u in-place: x_u -= step_size * grad
                if let Some(x_u_mut) = cluster_stalks.get_mut(i) {
                    for (x_comp, g_comp) in
                        x_u_mut.components.iter_mut().zip(grad.components.iter())
                    {
                        *x_comp -= config.step_size * *g_comp;
                    }
                }
            }

            if max_grad_norm < config.tolerance {
                break;
            }
        }

        Ok(iterations_run)
    }
}
