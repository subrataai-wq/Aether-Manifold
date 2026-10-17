//! Project Aether-Manifold: Semiring-Sheaf Algebraic Query Compiler
//!
//! Compiles hybrid graph traversal and vector similarity into Generalized SpMV
//! over Tropical, Diffusion, and Bottleneck semirings:
//! $$\psi_u^{(t+1)} = \bigoplus_{v \in \mathcal{N}(u)} \left( \psi_v^{(t)} \otimes \exp\left( -\beta \|\mathcal{F}_{u \trianglelefteq e} x_u - \mathcal{F}_{v \trianglelefteq e} x_v\|_2^2 - \gamma d_{\mathcal{M}}^2(u, q) \right) \right)$$
//!
//! Operates with strictly zero heap allocations on pre-allocated scratchpad buffers (Rule 3.1).

use aether_core::error::AetherError;
use aether_core::page::TensorGraphPage;

use crate::semiring::{semiring_eval, semiring_one, SemiringMode};
use crate::sheaf::{unpack_page_sheaf_morphism, DEFAULT_STALK_DIM, MAX_NEIGHBORS_PER_PAGE};

/// Maximum frontier capacity for zero-allocation query execution.
pub const MAX_QUERY_FRONTIER: usize = 256;

/// Candidate entry in the traversal frontier $\psi^{(t)}$.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontierNode {
    /// Node identifier in the graph.
    pub node_id: u64,
    /// Accumulated semiring score $\psi_v^{(t)}$.
    pub score: f32,
    /// Traversal hop depth.
    pub hop: u32,
    /// Reserved alignment padding.
    pub _pad: u32,
}

impl Default for FrontierNode {
    #[inline]
    fn default() -> Self {
        Self {
            node_id: 0,
            score: 0.0,
            hop: 0,
            _pad: 0,
        }
    }
}

/// Query compilation specification.
#[derive(Clone, Copy, Debug)]
pub struct QueryPlan {
    /// Semiring operational mode: Tropical, Diffusion, or Bottleneck.
    pub semiring_mode: SemiringMode,
    /// Traversal hop limit (e.g., 3 hops).
    pub max_hops: usize,
    /// Top-K results limit.
    pub top_k: usize,
    /// Beam search exploration width.
    pub beam_width: usize,
    /// Sheaf Dirichlet energy sensitivity weight beta >= 0.
    pub beta: f32,
    /// Riemannian manifold distance sensitivity weight gamma >= 0.
    pub gamma: f32,
    /// Threshold to prune low-affinity frontier candidates.
    pub prune_threshold: f32,
}

impl Default for QueryPlan {
    #[inline]
    fn default() -> Self {
        Self {
            semiring_mode: SemiringMode::Tropical,
            max_hops: 3,
            top_k: 10,
            beam_width: 64,
            beta: 1.0,
            gamma: 1.0,
            prune_threshold: 1e-4,
        }
    }
}

/// Semiring SpMV Algebraic Query Engine.
pub struct SemiringSpMvEngine;

impl SemiringSpMvEngine {
    /// Calculate composite Sheaf-Riemannian affinity weight $W(u, v, q)$.
    ///
    /// For Tropical mode: $W = \beta \cdot E_{\mathcal{F}} + \gamma \cdot d_{\mathcal{M}}^2$.
    /// For Diffusion/Bottleneck: $W = \exp(-\beta \cdot E_{\mathcal{F}} - \gamma \cdot d_{\mathcal{M}}^2)$.
    #[inline]
    pub fn compute_affinity_weight(
        mode: SemiringMode,
        sheaf_energy: f32,
        geodesic_distance_sq: f32,
        beta: f32,
        gamma: f32,
    ) -> f32 {
        let linear_cost = beta * sheaf_energy + gamma * geodesic_distance_sq;

        match mode {
            SemiringMode::Tropical => linear_cost,
            SemiringMode::Diffusion | SemiringMode::Bottleneck => libm::expf(-linear_cost),
        }
    }

    /// Execute a single traversal hop SpMV step across active frontier nodes.
    ///
    /// Evaluates 1-hop neighbor candidates directly from co-located 4KB `TensorGraphPage`s,
    /// accumulating scores using the selected semiring algebra with 0 heap allocations.
    #[allow(clippy::too_many_arguments)]
    pub fn step_spmv(
        plan: &QueryPlan,
        current_frontier: &[FrontierNode],
        page_lookup: impl Fn(u64) -> Option<&'static TensorGraphPage>,
        distance_oracle: impl Fn(u64) -> f32,
        next_frontier: &mut [FrontierNode; MAX_QUERY_FRONTIER],
        next_count: &mut usize,
        all_visited: &mut [FrontierNode; MAX_QUERY_FRONTIER],
        all_count: &mut usize,
    ) -> Result<(), AetherError> {
        let mode = plan.semiring_mode;
        *next_count = 0;

        for current in current_frontier {
            let page = match page_lookup(current.node_id) {
                Some(p) => p,
                None => continue,
            };

            let active_neighbors = page
                .header
                .active_vertex_count
                .min(MAX_NEIGHBORS_PER_PAGE as u32) as usize;

            for neighbor_idx in 0..active_neighbors {
                let neighbor_id = match page.neighbor_ids.get(neighbor_idx) {
                    Some(id) if *id != 0 => *id,
                    _ => continue,
                };

                // Read primary vertex stalk x_u from page
                let primary_stalk =
                    crate::sheaf::VertexStalk::<DEFAULT_STALK_DIM>::new(page.primary_stalk());

                // Read restriction map from 4KB page
                let morphism_u = unpack_page_sheaf_morphism(page, neighbor_idx)?;

                // Reconstruct neighbor stalk x_v from neighbor's page if available, or from co-located 32B code
                let neighbor_stalk = if let Some(n_page) = page_lookup(neighbor_id) {
                    crate::sheaf::VertexStalk::<DEFAULT_STALK_DIM>::new(n_page.primary_stalk())
                } else {
                    let offset = neighbor_idx * 32;
                    let code_bytes: [u8; 32] =
                        match page.quantized_neighbors.get(offset..offset + 32) {
                            Some(slice) => slice.try_into().unwrap_or([0u8; 32]),
                            None => [0u8; 32],
                        };
                    crate::sheaf::VertexStalk::<DEFAULT_STALK_DIM>::from_neighbor_bytes(
                        &code_bytes,
                        &primary_stalk,
                    )
                };

                // Dual morphism across edge relation
                let morphism_v = morphism_u;

                // Evaluate real Sheaf Dirichlet energy: || F_{u<=e} x_u - F_{v<=e} x_v ||_2^2
                let sheaf_energy = crate::laplacian::SheafCoboundary::edge_dirichlet_energy(
                    &morphism_u,
                    &primary_stalk,
                    &morphism_v,
                    &neighbor_stalk,
                );

                // Geodesic distance from neighbor to query target
                let geo_dist_sq = distance_oracle(neighbor_id);

                let affinity = Self::compute_affinity_weight(
                    mode,
                    sheaf_energy,
                    geo_dist_sq,
                    plan.beta,
                    plan.gamma,
                );

                // 1. Record direct candidate affinity into the global visited pool
                Self::aggregate_frontier(
                    all_visited,
                    all_count,
                    neighbor_id,
                    affinity,
                    current.hop + 1,
                    mode,
                );

                // 2. Semiring multiply: psi_v (x) W(u, v) for the forward propagation frontier
                let candidate_score = semiring_eval(mode, current.score, affinity, true);

                Self::aggregate_frontier(
                    next_frontier,
                    next_count,
                    neighbor_id,
                    candidate_score,
                    current.hop + 1,
                    mode,
                );
            }
        }

        // Sort frontier according to semiring order (min for Tropical, max for Diffusion/Bottleneck)
        let active = match next_frontier.get_mut(0..*next_count) {
            Some(s) => s,
            None => return Ok(()),
        };

        if mode == SemiringMode::Tropical {
            active.sort_unstable_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
        } else {
            active.sort_unstable_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
        }

        // Truncate intermediate frontier to beam_width
        let max_frontier = plan.beam_width.min(MAX_QUERY_FRONTIER);
        if *next_count > max_frontier {
            *next_count = max_frontier;
        }

        Ok(())
    }

    /// Aggregate a candidate score into the frontier using the semiring's $\oplus$ operator.
    #[inline]
    fn aggregate_frontier(
        frontier: &mut [FrontierNode; MAX_QUERY_FRONTIER],
        count: &mut usize,
        node_id: u64,
        score: f32,
        hop: u32,
        mode: SemiringMode,
    ) {
        // Check if node already exists in frontier
        for i in 0..*count {
            if let Some(entry) = frontier.get_mut(i) {
                if entry.node_id == node_id {
                    entry.score = semiring_eval(mode, entry.score, score, false);
                    return;
                }
            }
        }

        // If not found and capacity permits, append
        if *count < MAX_QUERY_FRONTIER {
            if let Some(slot) = frontier.get_mut(*count) {
                *slot = FrontierNode {
                    node_id,
                    score,
                    hop,
                    _pad: 0,
                };
                *count += 1;
            }
        }
    }

    /// Execute a multi-hop hybrid query from seed nodes up to `max_hops`.
    pub fn execute_query(
        plan: &QueryPlan,
        seed_nodes: &[u64],
        page_lookup: impl Fn(u64) -> Option<&'static TensorGraphPage>,
        distance_oracle: impl Fn(u64) -> f32,
        results: &mut [FrontierNode; MAX_QUERY_FRONTIER],
        result_count: &mut usize,
    ) -> Result<(), AetherError> {
        let mut frontier_a = [FrontierNode::default(); MAX_QUERY_FRONTIER];
        let mut frontier_b = [FrontierNode::default(); MAX_QUERY_FRONTIER];
        let mut count_a = 0;
        let mut count_b = 0;

        let mut all_visited = [FrontierNode::default(); MAX_QUERY_FRONTIER];
        let mut all_count = 0;

        let initial_score = semiring_one(plan.semiring_mode);

        for seed in seed_nodes.iter().take(MAX_QUERY_FRONTIER) {
            let seed_dist = distance_oracle(*seed);
            let seed_aff = Self::compute_affinity_weight(
                plan.semiring_mode,
                0.0,
                seed_dist,
                plan.beta,
                plan.gamma,
            );

            let node = FrontierNode {
                node_id: *seed,
                score: initial_score,
                hop: 0,
                _pad: 0,
            };
            if let Some(slot) = frontier_a.get_mut(count_a) {
                *slot = node;
                count_a += 1;
            }

            Self::aggregate_frontier(
                &mut all_visited,
                &mut all_count,
                *seed,
                seed_aff,
                0,
                plan.semiring_mode,
            );
        }

        let mut use_a = true;

        for _hop in 0..plan.max_hops {
            if use_a {
                Self::step_spmv(
                    plan,
                    match frontier_a.get(0..count_a) {
                        Some(s) => s,
                        None => &[],
                    },
                    &page_lookup,
                    &distance_oracle,
                    &mut frontier_b,
                    &mut count_b,
                    &mut all_visited,
                    &mut all_count,
                )?;
                if count_b == 0 {
                    break;
                }
                use_a = false;
            } else {
                Self::step_spmv(
                    plan,
                    match frontier_b.get(0..count_b) {
                        Some(s) => s,
                        None => &[],
                    },
                    &page_lookup,
                    &distance_oracle,
                    &mut frontier_a,
                    &mut count_a,
                    &mut all_visited,
                    &mut all_count,
                )?;
                if count_a == 0 {
                    break;
                }
                use_a = true;
            }
        }

        // Sort all accumulated candidates according to semiring order
        let active = match all_visited.get_mut(0..all_count) {
            Some(s) => s,
            None => return Ok(()),
        };

        if plan.semiring_mode == SemiringMode::Tropical {
            active.sort_unstable_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
        } else {
            active.sort_unstable_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
        }

        *result_count = all_count.min(plan.top_k);
        for i in 0..*result_count {
            if let (Some(dst), Some(src)) = (results.get_mut(i), all_visited.get(i)) {
                *dst = *src;
            }
        }

        Ok(())
    }
}
