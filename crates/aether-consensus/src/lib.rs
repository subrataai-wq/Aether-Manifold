//! Project Aether-Manifold: Distributed Consensus & Dynamic Topology Partitioning
//!
//! Provides:
//! - Balanced Spectral-Geodesic Sheaf Graph Partitioner (`partitioner`)
//! - Leaderless HLC Quorum Consensus Engine with 64-byte aligned messages (`consensus`)
//! - $L_0$ Morphism Log Replication and sub-10ms Fast Replay (`morphism_replicator`)
//! - RDMA / CXL 3.0 Zero-Copy Remote Page Fetching Fabric (`rdma_fabric`)

#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

pub mod consensus;
pub mod morphism_replicator;
pub mod partitioner;
pub mod rdma_fabric;

pub use consensus::{
    LeaderlessQuorumEngine, QuorumConfig, ReplicationMessage, ReplicationMessageType,
    ReplicationRingBuffer, MAX_IN_FLIGHT_PROPOSALS,
};
pub use morphism_replicator::{
    CommittedLogRecord, MorphismReplicator, MAX_TRACKED_VERTICES, REPLICATOR_LOG_CAPACITY,
};
pub use partitioner::{GraphEdge, PartitionPlan, PartitionerConfig, SpectralGeodesicPartitioner};
pub use rdma_fabric::{RdmaFabric, RdmaMemoryWindow, MAX_RDMA_NODES};
