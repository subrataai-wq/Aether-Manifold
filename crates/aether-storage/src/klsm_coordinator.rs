//! Project Aether-Manifold: Autonomous KLSM Tier Compaction Coordinator & Manifest Manager
//!
//! Manages active runs across KLSM tiers (L0 -> L1 -> Lk), performs Voronoi boundary
//! certificate evaluation, splices interior page slots in $O(1)$ time, relinks boundary
//! pages, and dispatches asynchronous direct I/O write operations with zero hot-path heap allocations.

use aether_core::error::AetherError;
use aether_core::hlc::HlcTimestamp;
use aether_manifold::at_rq::AnisotropicTangentQuantizer;
use aether_manifold::product::ProductManifoldConfig;

use crate::shard_runtime::CoreShardRuntime;
use crate::voronoi_compaction::{
    evaluate_voronoi_boundary_certificate, relink_boundary_vertex_delaunay, CompactionDecision,
    VoronoiCellCertificate,
};

/// Maximum number of pages per KLSM run manifest.
pub const MAX_PAGES_PER_RUN: usize = 64;

/// Maximum dimension stored for Voronoi cell centroids in KLSM run manifests.
pub const MAX_VORONOI_CENTROID_DIM: usize = 48;

/// Fixed-size zero-allocation manifest descriptor for a single KLSM run.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KlsmRunManifest {
    pub run_id: u64,
    pub level: u8,
    pub active: bool,
    pub min_hlc: HlcTimestamp,
    pub max_hlc: HlcTimestamp,
    pub page_slots: [u32; MAX_PAGES_PER_RUN],
    pub page_count: usize,
    pub voronoi_centroid: [f32; MAX_VORONOI_CENTROID_DIM],
    pub voronoi_radius: f32,
}

impl Default for KlsmRunManifest {
    fn default() -> Self {
        Self {
            run_id: 0,
            level: 0,
            active: false,
            min_hlc: HlcTimestamp(0),
            max_hlc: HlcTimestamp(0),
            page_slots: [0; MAX_PAGES_PER_RUN],
            page_count: 0,
            voronoi_centroid: [0.0; MAX_VORONOI_CENTROID_DIM],
            voronoi_radius: 0.0,
        }
    }
}

/// Report produced upon executing a tier compaction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactionReport {
    pub runs_merged: usize,
    pub interior_pages_spliced: usize,
    pub boundary_pages_relinked: usize,
    pub io_writes_submitted: usize,
}

/// Autonomous KLSM Tier Compaction Coordinator.
pub struct KlsmTierCoordinator<const MAX_RUNS: usize = 16> {
    pub runs: [KlsmRunManifest; MAX_RUNS],
    pub run_count: usize,
    pub next_run_id: u64,
    pub l0_flush_watermark: usize,
    pub total_flushes: usize,
    pub total_spliced_interior: usize,
    pub total_relinked_boundary: usize,
    pub total_io_writes: usize,
}

impl<const MAX_RUNS: usize> Default for KlsmTierCoordinator<MAX_RUNS> {
    fn default() -> Self {
        Self::new(64)
    }
}

impl<const MAX_RUNS: usize> KlsmTierCoordinator<MAX_RUNS> {
    /// Create a new KLSM tier coordinator.
    pub const fn new(l0_flush_watermark: usize) -> Self {
        Self {
            runs: [KlsmRunManifest {
                run_id: 0,
                level: 0,
                active: false,
                min_hlc: HlcTimestamp(0),
                max_hlc: HlcTimestamp(0),
                page_slots: [0; MAX_PAGES_PER_RUN],
                page_count: 0,
                voronoi_centroid: [0.0; MAX_VORONOI_CENTROID_DIM],
                voronoi_radius: 0.0,
            }; MAX_RUNS],
            run_count: 0,
            next_run_id: 1,
            l0_flush_watermark,
            total_flushes: 0,
            total_spliced_interior: 0,
            total_relinked_boundary: 0,
            total_io_writes: 0,
        }
    }

    /// Register a newly flushed or imported run manifest.
    pub fn register_run(&mut self, mut manifest: KlsmRunManifest) -> Result<usize, AetherError> {
        for (idx, slot) in self.runs.iter_mut().enumerate() {
            if !slot.active {
                if manifest.run_id == 0 {
                    manifest.run_id = self.next_run_id;
                    self.next_run_id += 1;
                }
                manifest.active = true;
                *slot = manifest;
                self.run_count += 1;
                return Ok(idx);
            }
        }
        Err(AetherError::CapacityExhausted { capacity: MAX_RUNS })
    }

    /// Check if L0 flush watermark is exceeded.
    #[inline]
    pub fn should_flush_l0<const D: usize, const L0_CAP: usize, const IO_DEPTH: usize>(
        &self,
        runtime: &CoreShardRuntime<D, L0_CAP, IO_DEPTH>,
    ) -> bool {
        runtime.l0_log.len() >= self.l0_flush_watermark
    }

    /// Check whether compaction criteria are met (e.g. two runs at the same level)
    /// and return `Some((source_idx, target_idx))` of the candidate runs.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    #[inline]
    pub fn maybe_trigger_compaction(&self) -> Option<(usize, usize)> {
        for i in 0..MAX_RUNS {
            let run_a = match self.runs.get(i) {
                Some(r) if r.active => r,
                _ => continue,
            };
            for j in (i + 1)..MAX_RUNS {
                let run_b = match self.runs.get(j) {
                    Some(r) if r.active => r,
                    _ => continue,
                };
                if run_a.level == run_b.level {
                    return Some((i, j));
                }
            }
        }
        None
    }

    /// Execute tier compaction merging `source_run_idx` into `target_run_idx`.
    ///
    /// # Protocol
    /// 1. Evaluates Riemannian Voronoi boundary certificates for target pages.
    /// 2. Executes $O(1)$ page slot splices for interior vertices.
    /// 3. Executes Delaunay re-linking only on intersecting boundary vertices.
    /// 4. Dispatches asynchronous direct I/O page writes for modified pages.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    pub fn execute_level_compaction<const D: usize, const CAP: usize, const IO_DEPTH: usize>(
        &mut self,
        runtime: &mut CoreShardRuntime<D, CAP, IO_DEPTH>,
        source_run_idx: usize,
        target_run_idx: usize,
        quantizer: &AnisotropicTangentQuantizer,
        config: &ProductManifoldConfig,
    ) -> Result<CompactionReport, AetherError> {
        if source_run_idx >= MAX_RUNS
            || target_run_idx >= MAX_RUNS
            || source_run_idx == target_run_idx
        {
            return Err(AetherError::OutOfBounds {
                index: source_run_idx.max(target_run_idx),
                limit: MAX_RUNS,
            });
        }

        let source_run = match self.runs.get(source_run_idx) {
            Some(r) if r.active => *r,
            _ => return Err(AetherError::CapacityExhausted { capacity: MAX_RUNS }),
        };
        let target_run = match self.runs.get(target_run_idx) {
            Some(r) if r.active => *r,
            _ => return Err(AetherError::CapacityExhausted { capacity: MAX_RUNS }),
        };

        let source_page_count = source_run.page_count.min(MAX_PAGES_PER_RUN);
        let mut candidate_vertices = [(0u64, [0.0f32; D]); MAX_PAGES_PER_RUN];
        let mut cand_count = 0;

        if let Some(src_slots) = source_run.page_slots.get(..source_page_count) {
            for &slot_u32 in src_slots {
                let slot = slot_u32 as usize;
                if let Ok(src_page) = runtime.arena.get_page(slot) {
                    let mut coords = [0.0f32; D];
                    if quantizer
                        .decode_page_primary_with_spillover(src_page, &mut coords)
                        .is_ok()
                    {
                        if let Some(cand_slot) = candidate_vertices.get_mut(cand_count) {
                            *cand_slot = (src_page.header.page_id, coords);
                            cand_count += 1;
                        }
                    }
                }
            }
        }

        let mut source_centroid = [0.0f32; D];
        let copy_len = MAX_VORONOI_CENTROID_DIM.min(D);
        for i in 0..copy_len {
            if let (Some(dest), Some(&src)) = (
                source_centroid.get_mut(i),
                source_run.voronoi_centroid.get(i),
            ) {
                *dest = src;
            }
        }

        let cert = VoronoiCellCertificate {
            cell_id: source_run.run_id as u32,
            centroid: source_centroid,
            max_geodesic_radius: source_run.voronoi_radius,
            modified: true,
        };

        let target_page_count = target_run.page_count.min(MAX_PAGES_PER_RUN);
        let mut report = CompactionReport {
            runs_merged: 2,
            interior_pages_spliced: 0,
            boundary_pages_relinked: 0,
            io_writes_submitted: 0,
        };

        let cand_slice = match candidate_vertices.get(..cand_count) {
            Some(s) => s,
            None => &candidate_vertices[..0],
        };

        for i in 0..target_page_count {
            let slot = match target_run.page_slots.get(i) {
                Some(&s) => s as usize,
                None => continue,
            };

            let page = runtime.arena.get_page(slot)?;
            let mut u_coords = [0.0f32; D];
            quantizer.decode_page_primary_with_spillover(page, &mut u_coords)?;

            // Evaluate actual distance to vertex's k-th neighbor from quantized_neighbors
            let mut d_knn = 0.0f32;
            let active_cnt = (page.header.active_vertex_count as usize).min(64);
            if active_cnt > 0 {
                for j in 0..active_cnt {
                    let offset = j * 32;
                    if let Some(bytes) = page.quantized_neighbors.get(offset..offset + 4) {
                        let mut arr = [0u8; 4];
                        arr.copy_from_slice(bytes);
                        let norm_sq = f32::from_le_bytes(arr);
                        let d = norm_sq.max(0.0).sqrt();
                        if d > d_knn {
                            d_knn = d;
                        }
                    }
                }
            }
            if d_knn <= 1e-6 {
                d_knn = 1.0;
            }

            let decision = evaluate_voronoi_boundary_certificate(config, &u_coords, d_knn, &cert)?;

            match decision {
                CompactionDecision::BypassInteriorSplice => {
                    report.interior_pages_spliced += 1;
                }
                CompactionDecision::BoundaryRelink => {
                    report.boundary_pages_relinked += 1;
                    let page_u = runtime.arena.get_page_mut(slot)?;
                    relink_boundary_vertex_delaunay(page_u, &u_coords, cand_slice, config)?;
                    let page_id = page_u.header.page_id;
                    runtime.submit_page_write(page_id, slot)?;
                    report.io_writes_submitted += 1;
                }
            }
        }

        // Merge slot lists into target manifest
        let mut merged_slots = [0u32; MAX_PAGES_PER_RUN];
        let mut merged_count = 0;

        if let Some(target_slots) = target_run.page_slots.get(..target_page_count) {
            for &s in target_slots {
                if merged_count < MAX_PAGES_PER_RUN {
                    if let Some(dest) = merged_slots.get_mut(merged_count) {
                        *dest = s;
                        merged_count += 1;
                    }
                }
            }
        }

        if let Some(src_slots) = source_run.page_slots.get(..source_page_count) {
            for &s in src_slots {
                if merged_count < MAX_PAGES_PER_RUN {
                    if let Some(dest) = merged_slots.get_mut(merged_count) {
                        *dest = s;
                        merged_count += 1;
                    }
                }
            }
        }

        let merged_level = target_run.level.max(source_run.level).saturating_add(1);
        let merged_min_hlc = source_run.min_hlc.min(target_run.min_hlc);
        let merged_max_hlc = source_run.max_hlc.max(target_run.max_hlc);
        let merged_radius = target_run.voronoi_radius.max(source_run.voronoi_radius);

        let merged_manifest = KlsmRunManifest {
            run_id: self.next_run_id,
            level: merged_level,
            active: true,
            min_hlc: merged_min_hlc,
            max_hlc: merged_max_hlc,
            page_slots: merged_slots,
            page_count: merged_count,
            voronoi_centroid: target_run.voronoi_centroid,
            voronoi_radius: merged_radius,
        };
        self.next_run_id += 1;

        if let Some(src_slot) = self.runs.get_mut(source_run_idx) {
            *src_slot = KlsmRunManifest::default();
        }
        if let Some(tgt_slot) = self.runs.get_mut(target_run_idx) {
            *tgt_slot = merged_manifest;
        }
        self.run_count = self.run_count.saturating_sub(1);

        self.total_spliced_interior += report.interior_pages_spliced;
        self.total_relinked_boundary += report.boundary_pages_relinked;
        self.total_io_writes += report.io_writes_submitted;

        Ok(report)
    }
}
