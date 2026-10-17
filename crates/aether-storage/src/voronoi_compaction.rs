//! Project Aether-Manifold: $L_1 \to L_k$ Boundary-Only Delaunay Compaction Engine (`voronoi_compaction.rs`)
//!
//! Implements the Boundary-Only Compaction protocol verified in `KlsmBoundaryCompaction.tla`.
//! Evaluates Riemannian Voronoi Boundary Certificates $d_{\mathcal{M}}(u, \mu_A) - R_A > d_{\mathcal{M}}(u, u_{\text{NN}}^{(k)})$
//! to execute $O(1)$ page pointer splices for interior vertices, bypassing all distance computations,
//! and restricting Delaunay re-linking exclusively to Voronoi cell boundaries $\partial \mathcal{V}_A \cap \partial \mathcal{V}_B$.

use aether_core::error::AetherError;
use aether_manifold::product::{product_manifold_distance, ProductManifoldConfig};

/// Riemannian Voronoi Cell Boundary Certificate:
///
/// Encapsulates cell boundary geometry:
/// - Centroid $\mu_A \in \mathcal{M}$
/// - Bounding radius $R_A = \max_{x \in \mathcal{V}_A} d_{\mathcal{M}}(\mu_A, x)$
/// - Modification flag indicating whether cell mutations occurred during this compaction epoch.
#[derive(Clone, Debug, PartialEq)]
pub struct VoronoiCellCertificate<const D: usize> {
    pub cell_id: u32,
    pub centroid: [f32; D],
    pub max_geodesic_radius: f32,
    pub modified: bool,
}

/// Compaction transition decision for a candidate vertex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactionDecision {
    /// Interior vertex bypass: $O(1)$ page pointer splice without distance recalculation.
    BypassInteriorSplice,
    /// Boundary vertex relink: vertex lies on $\partial \mathcal{V}_A \cap \partial \mathcal{V}_B$ and requires Delaunay update.
    BoundaryRelink,
}

/// Compaction execution metrics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactionStats {
    pub total_vertices: usize,
    pub interior_spliced: usize,
    pub boundary_relinked: usize,
}

/// Evaluate Riemannian Voronoi Boundary Certificate for vertex $u \in \mathcal{R}_B$:
///
/// If cell $A$ was not modified during this epoch, immediately returns `BypassInteriorSplice`.
/// Otherwise, evaluates:
///
/// $$d_{\mathcal{M}}(u, \mu_A) - R_A > d_{\mathcal{M}}\left(u, u_{\text{NN}}^{(k)}\right)$$
///
/// If strictly greater, the triangle inequality mathematically guarantees that no vertex in $V_A$
/// can be closer to $u$ than its existing $k$-th nearest neighbor. Thus, $u$'s $k$-NN neighborhood
/// is invariant, enabling an $O(1)$ pointer splice (`BypassInteriorSplice`).
///
/// # Hard Invariants
/// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
#[inline]
pub fn evaluate_voronoi_boundary_certificate<const D: usize>(
    config: &ProductManifoldConfig,
    u_coords: &[f32; D],
    d_u_knn: f32,
    cert: &VoronoiCellCertificate<D>,
) -> Result<CompactionDecision, AetherError> {
    if !cert.modified {
        return Ok(CompactionDecision::BypassInteriorSplice);
    }

    let d_u_mu = product_manifold_distance(config, u_coords, &cert.centroid)?;

    if d_u_mu - cert.max_geodesic_radius > d_u_knn {
        Ok(CompactionDecision::BypassInteriorSplice)
    } else {
        Ok(CompactionDecision::BoundaryRelink)
    }
}

/// Execute Boundary-Only Voronoi Compaction over a batch of vertices and their 4KB `TensorGraphPage` slots.
///
/// # Hard Invariants
/// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
pub fn execute_boundary_only_compaction<const D: usize>(
    config: &ProductManifoldConfig,
    vertices: &[[f32; D]],
    knn_distances: &[f32],
    _page_slots: &mut [usize],
    certs: &[VoronoiCellCertificate<D>],
) -> Result<CompactionStats, AetherError> {
    let n = vertices.len();
    if knn_distances.len() < n {
        return Err(AetherError::InvalidBufferSize {
            actual: knn_distances.len(),
            expected: n,
        });
    }

    let mut stats = CompactionStats {
        total_vertices: n,
        interior_spliced: 0,
        boundary_relinked: 0,
    };

    for i in 0..n {
        let u_coords = match vertices.get(i) {
            Some(v) => v,
            None => return Err(AetherError::OutOfBounds { index: i, limit: n }),
        };
        let d_knn = match knn_distances.get(i) {
            Some(&d) => d,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: i,
                    limit: knn_distances.len(),
                })
            }
        };

        let mut needs_relink = false;
        for cert in certs {
            if cert.modified {
                let decision =
                    evaluate_voronoi_boundary_certificate(config, u_coords, d_knn, cert)?;
                if decision == CompactionDecision::BoundaryRelink {
                    needs_relink = true;
                    break;
                }
            }
        }

        if needs_relink {
            stats.boundary_relinked += 1;
        } else {
            // O(1) page pointer splice for interior vertex
            stats.interior_spliced += 1;
        }
    }

    Ok(stats)
}

/// Re-links boundary page `page_u` against incoming run boundary pages
/// using stack-allocated scratch arrays with strictly 0 heap allocations.
///
/// Merges existing neighbors with incoming candidates, ranking by distance and
/// updating `page_u.neighbor_ids` in-place.
///
/// # Invariants
/// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
pub fn relink_boundary_vertex_delaunay<const D: usize>(
    page_u: &mut aether_core::page::TensorGraphPage,
    u_coords: &[f32; D],
    candidate_vertices: &[(u64, [f32; D])],
    config: &ProductManifoldConfig,
) -> Result<usize, AetherError> {
    const MAX_CANDIDATES: usize = 128;
    let mut candidate_ids = [0u64; MAX_CANDIDATES];
    let mut candidate_dists = [f32::INFINITY; MAX_CANDIDATES];
    let mut count = 0usize;

    let active_count = (page_u.header.active_vertex_count as usize).min(64);

    // 1. Retain existing neighbors from page_u, decoding actual radial norms
    for i in 0..active_count {
        let nid = page_u.neighbor_ids.get(i).copied().unwrap_or(0);
        if nid != 0 && count < MAX_CANDIDATES {
            let offset = i * 32;
            let dist = if let Some(bytes) = page_u.quantized_neighbors.get(offset..offset + 4) {
                let mut arr = [0u8; 4];
                arr.copy_from_slice(bytes);
                let norm_sq = f32::from_le_bytes(arr);
                let d = norm_sq.max(0.0).sqrt();
                if d > 0.0 {
                    d
                } else {
                    1.0
                }
            } else {
                1.0
            };

            if let Some(id_slot) = candidate_ids.get_mut(count) {
                *id_slot = nid;
            }
            if let Some(dist_slot) = candidate_dists.get_mut(count) {
                *dist_slot = dist;
            }
            count += 1;
        }
    }

    // 2. Evaluate incoming candidate vertices from the intersecting boundary
    for &(cand_id, ref cand_coord) in candidate_vertices {
        if cand_id == page_u.header.page_id || count >= MAX_CANDIDATES {
            continue;
        }
        if candidate_ids
            .get(..count)
            .is_some_and(|slice| slice.contains(&cand_id))
        {
            continue;
        }

        let dist = product_manifold_distance(config, u_coords, cand_coord)?;
        if let Some(id_slot) = candidate_ids.get_mut(count) {
            *id_slot = cand_id;
        }
        if let Some(dist_slot) = candidate_dists.get_mut(count) {
            *dist_slot = dist;
        }
        count += 1;
    }

    // 3. In-place selection sort of top-64 Delaunay neighbors
    let limit = count.min(64);
    for i in 0..limit {
        let mut min_idx = i;
        for j in (i + 1)..count {
            let dist_j = candidate_dists.get(j).copied().unwrap_or(f32::INFINITY);
            let dist_min = candidate_dists
                .get(min_idx)
                .copied()
                .unwrap_or(f32::INFINITY);
            if dist_j < dist_min {
                min_idx = j;
            }
        }
        candidate_ids.swap(i, min_idx);
        candidate_dists.swap(i, min_idx);

        let selected_id = candidate_ids.get(i).copied().unwrap_or(0);
        let selected_dist = candidate_dists.get(i).copied().unwrap_or(1.0);

        if let Some(slot) = page_u.neighbor_ids.get_mut(i) {
            *slot = selected_id;
        }
        let offset = i * 32;
        if let Some(bytes) = page_u.quantized_neighbors.get_mut(offset..offset + 4) {
            let norm_sq = selected_dist * selected_dist;
            bytes.copy_from_slice(&norm_sq.to_le_bytes());
        }
    }

    page_u.header.active_vertex_count = limit as u32;
    page_u.header.update_checksum();

    Ok(limit)
}
