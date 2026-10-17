//! Project Aether-Manifold: Background Discrete Ricci Flow Curvature Optimizer
//!
//! Evaluates discrete Ollivier-Ricci and Forman-Ricci curvatures across graph topologies
//! and continuously optimizes the Riemannian Product Manifold curvatures $\kappa < 0$ and $\tau > 0$.
//! Operates with strictly zero heap allocations on the hot path (AGENTS.md Rule 3.1 & Rule 3.4).

use aether_core::error::AetherError;

/// Discrete curvature evaluation for a topological edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeCurvature {
    /// Source vertex identifier.
    pub u: usize,
    /// Destination vertex identifier.
    pub v: usize,
    /// Discrete Forman-Ricci curvature: R_F(u, v) = 4 - deg(u) - deg(v).
    pub forman_curvature: f32,
    /// Discrete Ollivier-Ricci curvature: 1 - W_1(m_u, m_v) / d_M(u, v).
    pub ollivier_curvature: f32,
}

impl EdgeCurvature {
    /// Creates a new `EdgeCurvature` record.
    #[inline]
    pub const fn new(u: usize, v: usize, forman_curvature: f32, ollivier_curvature: f32) -> Self {
        Self {
            u,
            v,
            forman_curvature,
            ollivier_curvature,
        }
    }
}

/// Configuration parameters for continuous Ricci Flow optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RicciFlowConfig {
    /// Learning rate for negative hyperbolic curvature adjustment: eta_kappa.
    pub learning_rate_kappa: f32,
    /// Learning rate for positive spherical curvature adjustment: eta_tau.
    pub learning_rate_tau: f32,
    /// Bounds on hyperbolic curvature kappa (min_kappa, max_kappa) in (-infty, 0).
    pub kappa_bounds: (f32, f32),
    /// Bounds on spherical curvature tau (min_tau, max_tau) in (0, +infty).
    pub tau_bounds: (f32, f32),
    /// Maximum execution slice in microseconds before yielding cooperatively (Rule 3.4).
    pub cooperative_yield_micros: u64,
}

impl Default for RicciFlowConfig {
    fn default() -> Self {
        Self {
            learning_rate_kappa: 0.05,
            learning_rate_tau: 0.05,
            kappa_bounds: (-10.0, -0.05),
            tau_bounds: (0.05, 10.0),
            cooperative_yield_micros: 250,
        }
    }
}

/// Background discrete Ricci Flow curvature optimizer.
pub struct RicciFlowOptimizer {
    config: RicciFlowConfig,
}

impl RicciFlowOptimizer {
    /// Instantiates a new `RicciFlowOptimizer` with validated configuration.
    pub fn new(config: RicciFlowConfig) -> Result<Self, AetherError> {
        if config.kappa_bounds.0 >= config.kappa_bounds.1 || config.kappa_bounds.1 >= 0.0 {
            return Err(AetherError::OutOfBounds { index: 0, limit: 1 });
        }
        if config.tau_bounds.0 >= config.tau_bounds.1 || config.tau_bounds.0 <= 0.0 {
            return Err(AetherError::OutOfBounds { index: 0, limit: 1 });
        }
        Ok(Self { config })
    }

    /// Evaluates discrete Forman-Ricci curvature across an edge list given vertex degrees:
    /// R_F(e) = 4 - deg(u) - deg(v).
    pub fn compute_forman_curvatures(
        &self,
        num_vertices: usize,
        edges: &[(usize, usize)],
    ) -> Result<Vec<EdgeCurvature>, AetherError> {
        let mut degrees = vec![0usize; num_vertices];
        for &(u, v) in edges {
            if u >= num_vertices {
                return Err(AetherError::OutOfBounds {
                    index: u,
                    limit: num_vertices,
                });
            }
            if v >= num_vertices {
                return Err(AetherError::OutOfBounds {
                    index: v,
                    limit: num_vertices,
                });
            }
            if u == v {
                continue;
            }
            if let Some(deg) = degrees.get_mut(u) {
                *deg = deg.saturating_add(1);
            }
            if let Some(deg) = degrees.get_mut(v) {
                *deg = deg.saturating_add(1);
            }
        }

        let mut results = Vec::with_capacity(edges.len());
        for &(u, v) in edges {
            if u == v {
                continue;
            }
            let du = degrees.get(u).copied().unwrap_or(1);
            let dv = degrees.get(v).copied().unwrap_or(1);

            // Forman-Ricci curvature: R_F = 4 - deg(u) - deg(v)
            let forman = 4.0 - (du as f32) - (dv as f32);

            // Normalized Ollivier proxy based on Forman relation:
            // High degree hubs => negative Ollivier curvature (hyperbolic expansion)
            // Low degree paths / dense cliques => positive Ollivier curvature
            let max_d = (du.max(dv) as f32).max(1.0);
            let ollivier = forman / max_d;

            results.push(EdgeCurvature::new(u, v, forman, ollivier));
        }

        Ok(results)
    }

    /// Evaluates discrete Ollivier-Ricci curvatures across an edge list using Sinkhorn-Knopp
    /// Optimal Transport between 1-hop neighborhood coordinate distributions on the manifold:
    ///
    /// $$\kappa_{\text{OR}}(u, v) = 1.0 - \frac{\mathcal{W}_1(m_u, m_v)}{d_{\mathcal{M}}(x_u, x_v)}$$
    ///
    /// Neighborhood measures $m_u$ and $m_v$ are formed over the closed 1-hop neighborhoods $N[u]$ and $N[v]$.
    pub fn compute_ollivier_curvatures(
        &self,
        num_vertices: usize,
        vertex_coords: &[f32],
        coord_dim: usize,
        edges: &[(usize, usize)],
    ) -> Result<Vec<EdgeCurvature>, AetherError> {
        let mut ws = vec![0.0f32; 32768];
        let mut out = vec![EdgeCurvature::new(0, 0, 0.0, 0.0); edges.len()];
        let count = self.compute_ollivier_curvatures_with_workspace(
            num_vertices,
            vertex_coords,
            coord_dim,
            edges,
            &mut out,
            &mut ws,
        )?;
        out.truncate(count);
        Ok(out)
    }

    /// Zero-allocation hot-path variant of `compute_ollivier_curvatures` executing inside `HotPathAllocGuard`
    /// (Rule 3.1 & Rule 3.4). Uses pre-allocated stack arrays and caller-provided `workspace` slice.
    pub fn compute_ollivier_curvatures_with_workspace(
        &self,
        num_vertices: usize,
        vertex_coords: &[f32],
        coord_dim: usize,
        edges: &[(usize, usize)],
        out_curvatures: &mut [EdgeCurvature],
        workspace: &mut [f32],
    ) -> Result<usize, AetherError> {
        if coord_dim == 0 {
            return Err(AetherError::InvalidBufferSize {
                actual: 0,
                expected: 1,
            });
        }
        if vertex_coords.len() < num_vertices * coord_dim {
            return Err(AetherError::InvalidBufferSize {
                actual: vertex_coords.len(),
                expected: num_vertices * coord_dim,
            });
        }

        let max_results = out_curvatures.len();
        let mut count = 0usize;

        let sinkhorn_config = aether_manifold::optimal_transport::SinkhornConfig {
            epsilon: 0.1,
            max_iter: 100,
            tolerance: 1e-4,
        };

        for &(u, v) in edges {
            if u >= num_vertices {
                return Err(AetherError::OutOfBounds {
                    index: u,
                    limit: num_vertices,
                });
            }
            if v >= num_vertices {
                return Err(AetherError::OutOfBounds {
                    index: v,
                    limit: num_vertices,
                });
            }
            if u == v {
                continue;
            }
            if count >= max_results {
                break;
            }

            // Gather closed 1-hop neighborhoods N[u] and N[v] on stack
            let mut n_u = [0usize; 64];
            if let Some(slot) = n_u.first_mut() {
                *slot = u;
            }
            let mut count_u = 1usize;
            for &(eu, ev) in edges {
                let neighbor = if eu == u && ev != u {
                    Some(ev)
                } else if ev == u && eu != u {
                    Some(eu)
                } else {
                    None
                };
                if let Some(nbr) = neighbor {
                    if let Some(slice) = n_u.get(..count_u) {
                        if !slice.contains(&nbr) {
                            if let Some(slot) = n_u.get_mut(count_u) {
                                *slot = nbr;
                                count_u += 1;
                            }
                        }
                    }
                }
            }

            let mut n_v = [0usize; 64];
            if let Some(slot) = n_v.first_mut() {
                *slot = v;
            }
            let mut count_v = 1usize;
            for &(eu, ev) in edges {
                let neighbor = if eu == v && ev != v {
                    Some(ev)
                } else if ev == v && eu != v {
                    Some(eu)
                } else {
                    None
                };
                if let Some(nbr) = neighbor {
                    if let Some(slice) = n_v.get(..count_v) {
                        if !slice.contains(&nbr) {
                            if let Some(slot) = n_v.get_mut(count_v) {
                                *slot = nbr;
                                count_v += 1;
                            }
                        }
                    }
                }
            }

            let deg_u = count_u.saturating_sub(1);
            let deg_v = count_v.saturating_sub(1);
            let forman = 4.0 - (deg_u as f32) - (deg_v as f32);

            let a_len = count_u * coord_dim;
            let b_len = count_v * coord_dim;
            let ot_req =
                aether_manifold::optimal_transport::sinkhorn_workspace_size(count_u, count_v);
            let total_req = a_len + b_len + ot_req;

            if workspace.len() < total_req {
                return Err(AetherError::ScratchpadExhausted {
                    requested: total_req * core::mem::size_of::<f32>(),
                    available: core::mem::size_of_val(workspace),
                });
            }

            let (a_bag, rest) = workspace.split_at_mut(a_len);
            let (b_bag, ot_ws) = rest.split_at_mut(b_len);

            if let Some(u_slice) = n_u.get(..count_u) {
                for (i, &idx) in u_slice.iter().enumerate() {
                    if let (Some(dst), Some(src)) = (
                        a_bag.get_mut(i * coord_dim..(i + 1) * coord_dim),
                        vertex_coords.get(idx * coord_dim..(idx + 1) * coord_dim),
                    ) {
                        dst.copy_from_slice(src);
                    }
                }
            }

            if let Some(v_slice) = n_v.get(..count_v) {
                for (j, &idx) in v_slice.iter().enumerate() {
                    if let (Some(dst), Some(src)) = (
                        b_bag.get_mut(j * coord_dim..(j + 1) * coord_dim),
                        vertex_coords.get(idx * coord_dim..(idx + 1) * coord_dim),
                    ) {
                        dst.copy_from_slice(src);
                    }
                }
            }

            let u_slice = match vertex_coords.get(u * coord_dim..(u + 1) * coord_dim) {
                Some(s) => s,
                None => {
                    return Err(AetherError::OutOfBounds {
                        index: (u + 1) * coord_dim,
                        limit: vertex_coords.len(),
                    })
                }
            };
            let v_slice = match vertex_coords.get(v * coord_dim..(v + 1) * coord_dim) {
                Some(s) => s,
                None => {
                    return Err(AetherError::OutOfBounds {
                        index: (v + 1) * coord_dim,
                        limit: vertex_coords.len(),
                    })
                }
            };
            let d_sq = aether_manifold::simd::simd_euclidean_distance_squared(u_slice, v_slice);
            let d_uv = libm::sqrtf(d_sq).max(1e-5);

            let ot_res = aether_manifold::optimal_transport::sinkhorn_knopp_with_workspace(
                a_bag,
                count_u,
                b_bag,
                count_v,
                coord_dim,
                &sinkhorn_config,
                ot_ws,
            )?;
            let w1 = libm::sqrtf(ot_res.cost.max(0.0));
            let ollivier = 1.0 - (w1 / d_uv);

            if let Some(slot) = out_curvatures.get_mut(count) {
                *slot = EdgeCurvature::new(u, v, forman, ollivier);
                count += 1;
            }
        }

        Ok(count)
    }

    /// Dynamically evolves product manifold curvatures (kappa, tau) based on discrete edge curvatures.
    ///
    /// - If tree-like negative curvature dominates: kappa shifts more negative to accommodate hierarchical branching.
    /// - If cluster-like positive curvature dominates: tau increases to accommodate spherical angular density.
    #[inline]
    pub fn evolve_curvatures(
        &self,
        current_kappa: f32,
        current_tau: f32,
        curvatures: &[EdgeCurvature],
    ) -> (f32, f32) {
        if curvatures.is_empty() {
            return (current_kappa, current_tau);
        }

        let mut sum_negative = 0.0f32;
        let mut count_negative = 0usize;
        let mut sum_positive = 0.0f32;
        let mut count_positive = 0usize;

        for edge in curvatures {
            if edge.ollivier_curvature < 0.0 {
                sum_negative += edge.ollivier_curvature;
                count_negative = count_negative.saturating_add(1);
            } else {
                sum_positive += edge.ollivier_curvature;
                count_positive = count_positive.saturating_add(1);
            }
        }

        // Negative Ricci curvature drives hyperbolic curvature kappa more negative
        let delta_kappa = if count_negative > 0 {
            let avg_neg = sum_negative / (count_negative as f32);
            self.config.learning_rate_kappa * avg_neg
        } else {
            0.0
        };

        // Positive Ricci curvature drives spherical curvature tau more positive
        let delta_tau = if count_positive > 0 {
            let avg_pos = sum_positive / (count_positive as f32);
            self.config.learning_rate_tau * avg_pos
        } else {
            0.0
        };

        // Gradient flow update:
        // kappa(t+1) = kappa(t) + delta_kappa (where delta_kappa < 0)
        // tau(t+1) = tau(t) + delta_tau (where delta_tau > 0)
        let next_kappa = current_kappa + delta_kappa;
        let next_tau = current_tau + delta_tau;

        let clamped_kappa =
            next_kappa.clamp(self.config.kappa_bounds.0, self.config.kappa_bounds.1);
        let clamped_tau = next_tau.clamp(self.config.tau_bounds.0, self.config.tau_bounds.1);

        (clamped_kappa, clamped_tau)
    }
}
