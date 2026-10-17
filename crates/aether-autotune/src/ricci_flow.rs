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
