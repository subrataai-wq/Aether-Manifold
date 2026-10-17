//! Unit and integration test suite for `aether-consensus`.
//!
//! Verifies:
//! - 64-byte message alignment and CRC32C integrity.
//! - Leaderless HLC quorum consensus lifecycle with 0 hot-path heap allocations.
//! - Causal total ordering and read quorum resolution.
//! - Sub-10ms $L_0$ morphism replay and Last-Write-Wins causal resolution.
//! - Balanced spectral-geodesic sheaf graph partitioner (+/-5% balance, >90% local traversals).
//! - RDMA / CXL 3.0 zero-copy remote page reads and fault injection.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::assertions_on_constants,
    clippy::needless_range_loop
)]

use aether_consensus::consensus::{
    LeaderlessQuorumEngine, QuorumConfig, ReplicationMessage, ReplicationMessageType,
};
use aether_consensus::morphism_replicator::{CommittedLogRecord, MorphismReplicator};
use aether_consensus::partitioner::{GraphEdge, PartitionerConfig, SpectralGeodesicPartitioner};
use aether_consensus::rdma_fabric::{RdmaFabric, RdmaMemoryWindow};
use aether_core::alloc_guard::{reset_hot_path_counters, HotPathAllocGuard};
use aether_core::hlc::HlcTimestamp;

#[test]
fn test_replication_message_layout_and_crc() {
    assert_eq!(core::mem::size_of::<ReplicationMessage>(), 64);
    assert_eq!(core::mem::align_of::<ReplicationMessage>(), 64);

    let mut msg = ReplicationMessage::new(
        0x0001_0000_0000_0001,
        42,
        100,
        999,
        ReplicationMessageType::Propose,
        0,
        0,
        [7u8; 24],
    );
    msg.update_crc();

    assert!(msg.verify_crc().is_ok());

    // Corrupt one byte in payload and verify error
    msg.payload[0] ^= 0xFF;
    assert!(msg.verify_crc().is_err());
}

#[test]
fn test_causal_dominance() {
    let msg1 = ReplicationMessage::new(
        100,
        1,
        1,
        10,
        ReplicationMessageType::Propose,
        0,
        0,
        [0u8; 24],
    );
    let msg2 = ReplicationMessage::new(
        100,
        2,
        1,
        10,
        ReplicationMessageType::Propose,
        0,
        0,
        [0u8; 24],
    );
    let msg3 = ReplicationMessage::new(
        101,
        1,
        1,
        10,
        ReplicationMessageType::Propose,
        0,
        0,
        [0u8; 24],
    );

    // msg2 dominates msg1 due to higher node_id tie-breaker
    assert!(msg2.causally_dominates(&msg1));
    assert!(!msg1.causally_dominates(&msg2));

    // msg3 dominates msg2 due to strictly higher HLC
    assert!(msg3.causally_dominates(&msg2));
}

#[test]
fn test_leaderless_quorum_lifecycle_zero_alloc() {
    reset_hot_path_counters();

    // 3-node cluster: N=3, W=2, R=2
    let config0 = match QuorumConfig::new(0, 3, 2, 2) {
        Ok(c) => c,
        Err(_) => return assert!(false, "config failed"),
    };
    let config1 = match QuorumConfig::new(1, 3, 2, 2) {
        Ok(c) => c,
        Err(_) => return assert!(false, "config failed"),
    };

    let mut node0 = LeaderlessQuorumEngine::new(config0, 1_000_000);
    let mut node1 = LeaderlessQuorumEngine::new(config1, 1_000_000);

    // Enter HotPathAllocGuard to verify strictly 0 heap allocations on the consensus path
    let guard = HotPathAllocGuard::enter();

    // Node 0 proposes state update
    let proposal = match node0.propose(1234, [0xAA; 24], 1_000_010) {
        Ok(p) => p,
        Err(_) => return assert!(false, "propose failed"),
    };

    // Node 1 receives remote propose and generates Ack
    let ack = match node1.handle_remote_propose(&proposal, 1_000_015) {
        Ok(a) => a,
        Err(_) => return assert!(false, "handle_remote_propose failed"),
    };

    // Node 0 receives Ack from Node 1, achieving write quorum (2 acks: self + node1)
    let commit_opt = match node0.handle_ack(&ack) {
        Ok(c) => c,
        Err(_) => return assert!(false, "handle_ack failed"),
    };

    core::mem::drop(guard);

    assert!(commit_opt.is_some());
    if let Some(commit) = commit_opt {
        assert_eq!(commit.msg_type, ReplicationMessageType::Commit as u8);
        assert_eq!(commit.vertex_id, 1234);
        assert_eq!(commit.payload, [0xAA; 24]);
        assert!(commit.verify_crc().is_ok());
    }
}

#[test]
fn test_read_quorum_resolution() {
    let config = match QuorumConfig::new(0, 3, 2, 2) {
        Ok(c) => c,
        Err(_) => return assert!(false, "config failed"),
    };
    let engine = LeaderlessQuorumEngine::new(config, 1_000_000);

    let mut msg_v1 = ReplicationMessage::new(
        1000,
        1,
        1,
        42,
        ReplicationMessageType::ReadResponse,
        0,
        0,
        [1u8; 24],
    );
    msg_v1.update_crc();

    let mut msg_v2 = ReplicationMessage::new(
        1005,
        2,
        1,
        42,
        ReplicationMessageType::ReadResponse,
        0,
        0,
        [2u8; 24],
    );
    msg_v2.update_crc();

    let responses = [msg_v1, msg_v2];
    let winning = match engine.evaluate_read_quorum(&responses) {
        Ok(w) => w,
        Err(_) => return assert!(false, "evaluate_read_quorum failed"),
    };

    assert_eq!(winning.hlc_timestamp, 1005);
    assert_eq!(winning.payload, [2u8; 24]);
}

#[test]
fn test_morphism_replicator_replay_and_lww() {
    let mut replicator = MorphismReplicator::new(0);

    let mut commit1 = ReplicationMessage::new(
        100,
        1,
        1,
        77,
        ReplicationMessageType::Commit,
        0x02,
        0,
        [10u8; 24],
    );
    commit1.update_crc();

    let mut commit2 = ReplicationMessage::new(
        120,
        2,
        2,
        77,
        ReplicationMessageType::Commit,
        0x02,
        0,
        [20u8; 24],
    );
    commit2.update_crc();

    // Older out-of-order commit (should be ignored by LWW)
    let mut commit_old = ReplicationMessage::new(
        90,
        0,
        3,
        77,
        ReplicationMessageType::Commit,
        0x02,
        0,
        [5u8; 24],
    );
    commit_old.update_crc();

    assert!(replicator.apply_commit(&commit1).is_ok());
    assert!(replicator.apply_commit(&commit2).is_ok());
    assert!(replicator.apply_commit(&commit_old).is_ok());

    // Verify LWW holds latest state
    let state = replicator.get_vertex_state(77);
    assert!(state.is_some());
    if let Some((ts, payload)) = state {
        assert_eq!(ts, HlcTimestamp(120));
        assert_eq!(payload, [20u8; 24]);
    }

    // Test catchup buffer querying
    let mut catchup_records = [CommittedLogRecord::empty(); 16];
    let count = replicator.get_records_since(95, &mut catchup_records);
    assert_eq!(count, 2); // 100 and 120

    // Test fast replay on a new replica
    let mut new_replica = MorphismReplicator::new(1);
    let start = std::time::Instant::now();
    let replayed = match new_replica.replay_batch(&catchup_records[0..count]) {
        Ok(r) => r,
        Err(_) => return assert!(false, "replay failed"),
    };
    let elapsed = start.elapsed();

    assert_eq!(replayed, 2);
    assert!(
        elapsed.as_millis() < 10,
        "Replay must complete in < 10ms, took {:?}",
        elapsed
    );
    assert_eq!(
        new_replica.get_vertex_state(77),
        Some((HlcTimestamp(120), [20u8; 24]))
    );
}

#[test]
fn test_spectral_geodesic_partitioner() {
    let config = PartitionerConfig {
        num_shards: 4,
        geodesic_weight_mu: 1.0,
        balance_tolerance: 0.05,
        max_power_iterations: 100,
    };
    let partitioner = match SpectralGeodesicPartitioner::new(config) {
        Ok(p) => p,
        Err(_) => return assert!(false, "partitioner init failed"),
    };

    // Construct 4 tightly clustered cliques connected by sparse boundary edges
    let num_vertices = 100;
    let mut edges = Vec::new();
    let mut intra_edges = Vec::new();

    // 4 clusters of 25 nodes each: [0..25], [25..50], [50..75], [75..100]
    for cluster in 0..4 {
        let base = cluster * 25;
        for i in 0..25 {
            for j in (i + 1)..25 {
                let u = base + i;
                let v = base + j;
                // Low energy / distance inside cluster
                let edge = GraphEdge::new(u, v, 0.01, 0.01);
                edges.push(edge);
                intra_edges.push((u, v));
            }
        }
    }

    // Sparse cross-cluster edges with high energy
    edges.push(GraphEdge::new(24, 25, 5.0, 5.0));
    edges.push(GraphEdge::new(49, 50, 5.0, 5.0));
    edges.push(GraphEdge::new(74, 75, 5.0, 5.0));

    let plan = match partitioner.partition(num_vertices, &edges) {
        Ok(p) => p,
        Err(_) => return assert!(false, "partition failed"),
    };

    // Verify balance: 100 nodes / 4 shards = 25 nodes per shard exactly
    assert!(plan.is_balanced(0.05));
    for &size in plan.shard_sizes() {
        assert_eq!(size, 25);
    }

    // Verify high locality: > 90% of intra-cluster edges stay local
    let ratio = match plan.local_traversal_ratio(&intra_edges) {
        Ok(r) => r,
        Err(_) => return assert!(false, "local_traversal_ratio failed"),
    };
    assert!(
        ratio > 0.90,
        "Expected local traversal ratio > 0.90, got {}",
        ratio
    );
}

#[test]
fn test_rdma_fabric_zero_copy_and_faults() {
    let mut fabric = RdmaFabric::new();

    let window0 = RdmaMemoryWindow::with_capacity(4);
    let mut test_page = [0u8; 4096];
    test_page[0..4].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
    assert!(window0.write_page(0, &test_page).is_ok());

    assert!(fabric.register_node(0, window0).is_ok());

    // Hot-path zero copy remote read
    let mut read_dest = [0u8; 4096];
    let guard = HotPathAllocGuard::enter();
    let res = fabric.read_remote_page(0, 0, &mut read_dest);
    core::mem::drop(guard);

    assert!(res.is_ok());
    assert_eq!(&read_dest[0..4], &[0xDE, 0xAD, 0xBE, 0xEF]);

    // Test offline node failure
    assert!(fabric.set_node_online(0, false).is_ok());
    assert!(fabric.read_remote_page(0, 0, &mut read_dest).is_err());

    // Recover node
    assert!(fabric.set_node_online(0, true).is_ok());
    assert!(fabric.read_remote_page(0, 0, &mut read_dest).is_ok());
}
