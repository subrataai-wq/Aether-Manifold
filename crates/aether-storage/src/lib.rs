//! Project Aether-Manifold: Single-Node Storage Engine & KLSM-Tree (`crates/aether-storage`)
//!
//! Provides the core shared-nothing thread-per-core storage engine, wait-free SPSC ring buffers,
//! 4KB co-located `TensorGraphPage` arena allocators, $L_0$ geodesic MVCC temporal logs,
//! and the lock-free Boundary-Only Voronoi Delaunay compaction protocol per roadmap.md Section 3.
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod spsc;
pub mod page_arena;
pub mod l0_mvcc;
pub mod shard_runtime;
pub mod voronoi_compaction;

pub use spsc::{CachePaddedAtomicUsize, SpscRingBuffer};
pub use page_arena::AlignedPageArena;
pub use l0_mvcc::{L0MorphismLog, TangentDeltaEntry};
pub use shard_runtime::{CoreShardRuntime, IoCompletionCqe, IoOpcode, IoSubmissionSqe};
pub use voronoi_compaction::{
    evaluate_voronoi_boundary_certificate, execute_boundary_only_compaction, CompactionDecision,
    CompactionStats, VoronoiCellCertificate,
};
