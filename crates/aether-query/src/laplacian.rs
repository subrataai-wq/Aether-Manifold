//! Project Aether-Manifold: Sheaf Coboundary Operator & Block Sheaf Laplacian
//!
//! Evaluates the discrete Sheaf Coboundary $\delta: C^0(G; \mathcal{F}) \to C^1(G; \mathcal{F})$
//! and the block Sheaf Laplacian $L_{\mathcal{F}} = \delta^T \delta$ without materializing
//! dense matrices or invoking the global heap (Rule 3.1).

use aether_core::error::AetherError;

use crate::sheaf::{EdgeStalk, SheafRestrictionMap, VertexStalk};

/// Sheaf Coboundary Operator $\delta : C^0(G; \mathcal{F}) \to C^1(G; \mathcal{F})$.
///
/// For edge $e = (u, v)$ connecting vertex $u$ and vertex $v$:
/// $$(\delta x)_e = \mathcal{F}_{u \trianglelefteq e} x_u - \mathcal{F}_{v \trianglelefteq e} x_v$$
pub struct SheafCoboundary;

impl SheafCoboundary {
    /// Compute the coboundary difference across directed edge $e = (u, v)$ in $O(D)$ FLOPs.
    #[inline]
    pub fn evaluate_edge<const D: usize>(
        map_u: &SheafRestrictionMap<D>,
        x_u: &VertexStalk<D>,
        map_v: &SheafRestrictionMap<D>,
        x_v: &VertexStalk<D>,
        out_e: &mut EdgeStalk<D>,
    ) {
        let mut u_proj = EdgeStalk::<D>::default();
        let mut v_proj = EdgeStalk::<D>::default();

        map_u.apply(x_u, &mut u_proj);
        map_v.apply(x_v, &mut v_proj);

        for ((e_val, u_val), v_val) in out_e
            .components
            .iter_mut()
            .zip(u_proj.components.iter())
            .zip(v_proj.components.iter())
        {
            *e_val = *u_val - *v_val;
        }
    }

    /// Compute the squared Dirichlet energy across edge $e$: $\|(\delta x)_e\|_2^2$.
    #[inline]
    pub fn edge_dirichlet_energy<const D: usize>(
        map_u: &SheafRestrictionMap<D>,
        x_u: &VertexStalk<D>,
        map_v: &SheafRestrictionMap<D>,
        x_v: &VertexStalk<D>,
    ) -> f32 {
        let mut delta_e = EdgeStalk::<D>::default();
        Self::evaluate_edge(map_u, x_u, map_v, x_v, &mut delta_e);

        let mut energy = 0.0f32;
        for c in &delta_e.components {
            energy += *c * *c;
        }
        energy
    }
}

/// Block Sheaf Laplacian $L_{\mathcal{F}} = \delta^T \delta$.
///
/// Evaluates action $(L_{\mathcal{F}} x)_u$ at vertex $u$ by aggregating across its incident edges:
/// $$(L_{\mathcal{F}} x)_u = \sum_{e = (u, v) \in \mathcal{E}(u)} \mathcal{F}_{u \trianglelefteq e}^T \left( \mathcal{F}_{u \trianglelefteq e} x_u - \mathcal{F}_{v \trianglelefteq e} x_v \right)$$
pub struct BlockSheafLaplacian;

impl BlockSheafLaplacian {
    /// Multiply the block Sheaf Laplacian row for vertex $u$ with the global 0-cochain $x$.
    ///
    /// Executes with 0 heap allocations, streaming incident edges from a local slice.
    pub fn evaluate_vertex_action<const D: usize>(
        x_u: &VertexStalk<D>,
        incident_edges: &[(
            SheafRestrictionMap<D>,
            SheafRestrictionMap<D>,
            VertexStalk<D>,
        )],
        out_laplacian_u: &mut VertexStalk<D>,
    ) -> Result<(), AetherError> {
        // Clear accumulator
        for c in out_laplacian_u.components.iter_mut() {
            *c = 0.0f32;
        }

        let mut delta_e = EdgeStalk::<D>::default();
        let mut back_projected = VertexStalk::<D>::default();

        for (map_u, map_v, x_v) in incident_edges {
            // 1. (delta x)_e = F_{u <= e} x_u - F_{v <= e} x_v
            SheafCoboundary::evaluate_edge(map_u, x_u, map_v, x_v, &mut delta_e);

            // 2. F_{u <= e}^T (delta x)_e
            map_u.apply_adjoint(&delta_e, &mut back_projected);

            // 3. Accumulate: (L_F x)_u += back_projected
            for (accum, val) in out_laplacian_u
                .components
                .iter_mut()
                .zip(back_projected.components.iter())
            {
                *accum += *val;
            }
        }

        Ok(())
    }

    /// Compute total discrete Sheaf Dirichlet energy across all edges in the cochain:
    /// $$E_{\mathcal{F}}(x) = \frac{1}{2} x^T L_{\mathcal{F}} x = \frac{1}{2} \sum_{e = (u, v)} \|(\delta x)_e\|_2^2 \ge 0$$
    ///
    /// Because each term $\|(\delta x)_e\|_2^2 \ge 0$, the Sheaf Laplacian is symmetric positive semi-definite (PSD).
    pub fn total_dirichlet_energy<const D: usize>(
        edges: &[(
            SheafRestrictionMap<D>,
            VertexStalk<D>,
            SheafRestrictionMap<D>,
            VertexStalk<D>,
        )],
    ) -> f32 {
        let mut total = 0.0f32;
        for (map_u, x_u, map_v, x_v) in edges {
            total += SheafCoboundary::edge_dirichlet_energy(map_u, x_u, map_v, x_v);
        }
        0.5f32 * total
    }
}
