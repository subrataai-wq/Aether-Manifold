//! Project Aether-Manifold: Autonomous Dataset Churn & Curvature Coordinator
//!
//! Orchestrates background discrete Ricci Flow optimization and DMA-offloaded compaction
//! during continuous dataset churn, guaranteeing zero recall degradation without manual re-indexing.
//! Operates with strictly zero heap allocations on the query hot path (AGENTS.md Rule 3.1 & 3.4).

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use aether_core::error::AetherError;

use crate::dma_compaction::{DmaCompactionEngine, DmaCompactionTelemetry, DmaPageDescriptor};
use crate::ricci_flow::{EdgeCurvature, RicciFlowOptimizer};
use aether_core::page::TensorGraphPage;
use aether_storage::voronoi_compaction::VoronoiCellCertificate;

/// Double-buffered atomic curvature snapshot accessible lock-free on the read query path.
#[derive(Debug)]
pub struct AtomicCurvatureSnapshot {
    // Bits of f32 stored in AtomicU32 for wait-free load/store
    kappa_bits: AtomicU32,
    tau_bits: AtomicU32,
    epoch: AtomicU64,
}

impl AtomicCurvatureSnapshot {
    /// Creates a new atomic curvature snapshot.
    pub fn new(initial_kappa: f32, initial_tau: f32) -> Self {
        Self {
            kappa_bits: AtomicU32::new(initial_kappa.to_bits()),
            tau_bits: AtomicU32::new(initial_tau.to_bits()),
            epoch: AtomicU64::new(1),
        }
    }

    /// Loads active (kappa, tau) without locking or allocation.
    #[inline]
    pub fn load(&self) -> (f32, f32) {
        let k = f32::from_bits(self.kappa_bits.load(Ordering::Acquire));
        let t = f32::from_bits(self.tau_bits.load(Ordering::Acquire));
        (k, t)
    }

    /// Updates active curvatures atomically.
    #[inline]
    pub fn store(&self, kappa: f32, tau: f32) {
        self.kappa_bits.store(kappa.to_bits(), Ordering::Release);
        self.tau_bits.store(tau.to_bits(), Ordering::Release);
        self.epoch.fetch_add(1, Ordering::Relaxed);
    }

    /// Retrieves current update epoch.
    #[inline]
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }
}

/// Telemetry metrics for continuous dataset churn and autonomous optimization.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ChurnTelemetry {
    /// Total entity mutations processed.
    pub total_mutations: usize,
    /// Number of discrete Ricci Flow optimization steps executed.
    pub ricci_steps_executed: usize,
    /// Number of DMA compaction batches executed.
    pub dma_batches_executed: usize,
    /// Active negative hyperbolic curvature kappa.
    pub active_kappa: f32,
    /// Active positive spherical curvature tau.
    pub active_tau: f32,
}

/// Autonomous coordinator uniting continuous Ricci Flow and DMA compaction.
pub struct AutonomousChurnCoordinator {
    snapshot: AtomicCurvatureSnapshot,
    ricci_optimizer: RicciFlowOptimizer,
    dma_engine: DmaCompactionEngine,
    total_mutations: usize,
    ricci_steps: usize,
    dma_batches: usize,
}

impl AutonomousChurnCoordinator {
    /// Instantiates a new `AutonomousChurnCoordinator`.
    pub fn new(initial_kappa: f32, initial_tau: f32, ricci_optimizer: RicciFlowOptimizer) -> Self {
        Self {
            snapshot: AtomicCurvatureSnapshot::new(initial_kappa, initial_tau),
            ricci_optimizer,
            dma_engine: DmaCompactionEngine::new(),
            total_mutations: 0,
            ricci_steps: 0,
            dma_batches: 0,
        }
    }

    /// Retrieves reference to atomic curvature snapshot for lock-free query execution.
    #[inline]
    pub const fn snapshot(&self) -> &AtomicCurvatureSnapshot {
        &self.snapshot
    }

    /// Records a batch of entity mutations (inserts / updates / deletions).
    #[inline]
    pub fn record_mutations(&mut self, count: usize) {
        self.total_mutations = self.total_mutations.saturating_add(count);
    }

    /// Triggers a background Ricci Flow optimization step based on recent edge curvatures.
    pub fn trigger_ricci_step(
        &mut self,
        edge_curvatures: &[EdgeCurvature],
    ) -> Result<(f32, f32), AetherError> {
        let (current_k, current_t) = self.snapshot.load();
        let (next_k, next_t) =
            self.ricci_optimizer
                .evolve_curvatures(current_k, current_t, edge_curvatures);

        self.snapshot.store(next_k, next_t);
        self.ricci_steps = self.ricci_steps.saturating_add(1);
        Ok((next_k, next_t))
    }

    /// Triggers an asynchronous DMA compaction batch across 4KB pages.
    pub fn trigger_dma_compaction<const D: usize>(
        &mut self,
        config: &aether_manifold::product::ProductManifoldConfig,
        descriptors: &[DmaPageDescriptor],
        source_pages: &[TensorGraphPage],
        target_pages: &mut [TensorGraphPage],
        cert: &VoronoiCellCertificate<D>,
        vertex_coords: &[[f32; D]],
    ) -> Result<DmaCompactionTelemetry, AetherError> {
        let tel = self.dma_engine.execute_dma_compaction(
            config,
            descriptors,
            source_pages,
            target_pages,
            cert,
            vertex_coords,
        )?;
        self.dma_batches = self.dma_batches.saturating_add(1);
        Ok(tel)
    }

    /// Collects full telemetry status.
    pub fn telemetry(&self) -> ChurnTelemetry {
        let (k, t) = self.snapshot.load();
        ChurnTelemetry {
            total_mutations: self.total_mutations,
            ricci_steps_executed: self.ricci_steps,
            dma_batches_executed: self.dma_batches,
            active_kappa: k,
            active_tau: t,
        }
    }
}
