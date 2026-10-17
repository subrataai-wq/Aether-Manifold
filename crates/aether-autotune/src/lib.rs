//! Project Aether-Manifold: Autonomous Self-Tuning & Continuous Curvature Optimization
//!
//! Provides:
//! - Background Discrete Ricci Flow Curvature Optimizer (`ricci_flow`)
//! - Continuous Householder Sheaf Restriction Morphism Optimizer (`sheaf_opt`)
//! - GPU/NPU DMA-Offloaded Compaction Kernels (`dma_compaction`)
//! - Autonomous Dataset Churn Coordinator (`churn_controller`)

#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

pub mod churn_controller;
pub mod dma_compaction;
pub mod ricci_flow;
pub mod sheaf_opt;

pub use churn_controller::{AtomicCurvatureSnapshot, AutonomousChurnCoordinator, ChurnTelemetry};
pub use dma_compaction::{DmaCompactionEngine, DmaCompactionTelemetry, DmaPageDescriptor};
pub use ricci_flow::{EdgeCurvature, RicciFlowConfig, RicciFlowOptimizer};
pub use sheaf_opt::SheafMorphismOptimizer;
