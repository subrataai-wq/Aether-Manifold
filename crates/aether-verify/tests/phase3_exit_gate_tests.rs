//! Project Aether-Manifold: Phase 3 Exit Gate Verification Suite
//!
//! Validates:
//! 1. 64-node distributed cluster simulation with zero consensus deadlocks.
//! 2. Minority node failure recovery in < 10ms with zero data loss.
//! 3. Strictly 0 heap allocations on the consensus and replication hot path inside `HotPathAllocGuard`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::assertions_on_constants,
    clippy::needless_range_loop
)]

use std::time::Instant;

use aether_consensus::consensus::{
    LeaderlessQuorumEngine, QuorumConfig, ReplicationMessage, ReplicationMessageType,
};
use aether_consensus::morphism_replicator::{CommittedLogRecord, MorphismReplicator};
use aether_consensus::rdma_fabric::{RdmaFabric, RdmaMemoryWindow};
use aether_core::alloc_guard::{reset_hot_path_counters, HotPathAllocGuard, TrackingAllocator};

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator::new(std::alloc::System);

/// Step 3.4 Exit Gate 1: 64-node distributed cluster simulation.
///
/// Verifies that a full 64-node topology coordinates causal commits without deadlocks.
#[test]
fn gate_64_node_distributed_cluster_simulation() {
    const NUM_NODES: usize = 64;
    // 3-node replication groups across 64 shards: W=2, R=2
    let mut engines = Vec::with_capacity(NUM_NODES);
    let mut replicators = Vec::with_capacity(NUM_NODES);

    for node_id in 0..NUM_NODES {
        let config = match QuorumConfig::new(node_id as u64, 3, 2, 2) {
            Ok(c) => c,
            Err(_) => return assert!(false, "Failed to initialize QuorumConfig"),
        };
        engines.push(LeaderlessQuorumEngine::new(config, 1_000_000));
        replicators.push(MorphismReplicator::new(node_id as u64));
    }

    let mut fabric = RdmaFabric::new();
    for node_id in 0..NUM_NODES {
        let window = RdmaMemoryWindow::with_capacity(16);
        assert!(fabric.register_node(node_id as u64, window).is_ok());
    }

    // Ingest 1,000 distributed mutation proposals across shards
    let num_proposals = 1000;
    for i in 0..num_proposals {
        let primary_id = i % NUM_NODES;
        let peer1_id = (primary_id + 1) % NUM_NODES;
        let vertex_id = (i as u64) * 7;
        let payload = [(i & 0xFF) as u8; 24];

        // 1. Primary node proposes
        let proposal = {
            let primary_engine = match engines.get_mut(primary_id) {
                Some(e) => e,
                None => return assert!(false, "primary engine missing"),
            };
            match primary_engine.propose(vertex_id, payload, 1_000_000 + (i as u64) * 10) {
                Ok(p) => p,
                Err(_) => return assert!(false, "propose failed"),
            }
        };

        // 2. Peer node receives and acks
        let ack = {
            let peer_engine = match engines.get_mut(peer1_id) {
                Some(e) => e,
                None => return assert!(false, "peer engine missing"),
            };
            match peer_engine.handle_remote_propose(&proposal, 1_000_000 + (i as u64) * 10 + 2) {
                Ok(a) => a,
                Err(_) => return assert!(false, "handle_remote_propose failed"),
            }
        };

        // 3. Primary node processes ack and reaches quorum W=2 (self + peer1)
        let commit_opt = {
            let primary_engine = match engines.get_mut(primary_id) {
                Some(e) => e,
                None => return assert!(false, "primary engine missing"),
            };
            match primary_engine.handle_ack(&ack) {
                Ok(c) => c,
                Err(_) => return assert!(false, "handle_ack failed"),
            }
        };

        assert!(
            commit_opt.is_some(),
            "Commit must trigger upon reaching W=2"
        );
        if let Some(commit) = commit_opt {
            // Apply commit to both nodes' morphism logs
            if let Some(rep_primary) = replicators.get_mut(primary_id) {
                assert!(rep_primary.apply_commit(&commit).is_ok());
            }
            if let Some(rep_peer) = replicators.get_mut(peer1_id) {
                assert!(rep_peer.apply_commit(&commit).is_ok());
            }
        }
    }

    // Verify all 1,000 commits were processed across the cluster
    let total_replicated: usize = replicators.iter().map(|r| r.total_committed()).sum();
    assert_eq!(
        total_replicated,
        num_proposals * 2,
        "Every commit must replicate to both primary and peer nodes"
    );
}

/// Step 3.4 Exit Gate 2: Minority node failure recovery in < 10ms with zero data loss.
#[test]
fn gate_minority_node_failure_recovery_under_10ms() {
    // 3-node shard: Node 0, Node 1, Node 2 (N=3, W=2, R=2)
    let cfg0 = match QuorumConfig::new(0, 3, 2, 2) {
        Ok(c) => c,
        Err(_) => return assert!(false, "cfg0"),
    };
    let cfg1 = match QuorumConfig::new(1, 3, 2, 2) {
        Ok(c) => c,
        Err(_) => return assert!(false, "cfg1"),
    };
    let cfg2 = match QuorumConfig::new(2, 3, 2, 2) {
        Ok(c) => c,
        Err(_) => return assert!(false, "cfg2"),
    };

    let mut engine0 = LeaderlessQuorumEngine::new(cfg0, 1_000_000);
    let mut engine1 = LeaderlessQuorumEngine::new(cfg1, 1_000_000);
    let mut _engine2 = LeaderlessQuorumEngine::new(cfg2, 1_000_000);

    let mut rep0 = MorphismReplicator::new(0);
    let mut rep1 = MorphismReplicator::new(1);
    let mut rep2 = MorphismReplicator::new(2);

    // Initial baseline sync
    let mut base_commit = ReplicationMessage::new(
        1_000_050,
        0,
        1,
        500,
        ReplicationMessageType::Commit,
        0x02,
        0,
        [0x11; 24],
    );
    base_commit.update_crc();
    assert!(rep0.apply_commit(&base_commit).is_ok());
    assert!(rep1.apply_commit(&base_commit).is_ok());
    assert!(rep2.apply_commit(&base_commit).is_ok());

    let node2_last_hlc = 1_000_050;

    // Simulate Node 2 failure:
    // Node 0 and Node 1 continue committing 500 mutations while Node 2 is offline
    for i in 2..502 {
        let v_id = 500 + i;
        let payload = [(i & 0xFF) as u8; 24];
        let p = match engine0.propose(v_id, payload, 1_000_050 + i * 10) {
            Ok(p) => p,
            Err(_) => return assert!(false, "propose"),
        };
        let ack = match engine1.handle_remote_propose(&p, 1_000_050 + i * 10 + 2) {
            Ok(a) => a,
            Err(_) => return assert!(false, "ack"),
        };
        let commit = match engine0.handle_ack(&ack) {
            Ok(Some(c)) => c,
            _ => return assert!(false, "commit expected"),
        };
        assert!(rep0.apply_commit(&commit).is_ok());
        assert!(rep1.apply_commit(&commit).is_ok());
    }

    // Verify Node 0 and Node 1 have 501 commits, while Node 2 has only 1
    assert_eq!(rep0.total_committed(), 501);
    assert_eq!(rep1.total_committed(), 501);
    assert_eq!(rep2.total_committed(), 1);

    // Node 2 recovers and requests catchup from Node 0 since `node2_last_hlc`
    let mut catchup_buffer = vec![CommittedLogRecord::empty(); 1024];

    let start_recovery = Instant::now();
    let records_to_catchup = rep0.get_records_since(node2_last_hlc, &mut catchup_buffer);
    assert_eq!(records_to_catchup, 500);

    let replayed = match rep2.replay_batch(&catchup_buffer[0..records_to_catchup]) {
        Ok(r) => r,
        Err(_) => return assert!(false, "replay_batch failed"),
    };
    let recovery_elapsed = start_recovery.elapsed();

    // Verify hard gate: recovery elapsed < 10ms
    assert!(
        recovery_elapsed.as_millis() < 10,
        "Recovery must take < 10ms, took {:?}",
        recovery_elapsed
    );
    assert_eq!(replayed, 500);

    // Verify Zero Data Loss: Node 2 has caught up completely
    assert_eq!(rep2.total_committed(), 501);
    for i in 2..502 {
        let v_id = 500 + i;
        let expected_payload = [(i & 0xFF) as u8; 24];
        let state2 = rep2.get_vertex_state(v_id);
        assert!(state2.is_some());
        if let Some((_, payload)) = state2 {
            assert_eq!(payload, expected_payload);
        }
    }
}

/// Step 3.4 Exit Gate 3: Strictly zero heap allocations inside HotPathAllocGuard.
#[test]
fn gate_consensus_zero_hot_path_allocations() {
    reset_hot_path_counters();

    let cfg0 = match QuorumConfig::new(0, 3, 2, 2) {
        Ok(c) => c,
        Err(_) => return assert!(false, "cfg0"),
    };
    let cfg1 = match QuorumConfig::new(1, 3, 2, 2) {
        Ok(c) => c,
        Err(_) => return assert!(false, "cfg1"),
    };

    let mut engine0 = LeaderlessQuorumEngine::new(cfg0, 1_000_000);
    let mut engine1 = LeaderlessQuorumEngine::new(cfg1, 1_000_000);
    let mut rep0 = MorphismReplicator::new(0);

    let mut fabric = RdmaFabric::new();
    let window0 = RdmaMemoryWindow::with_capacity(4);
    assert!(fabric.register_node(0, window0).is_ok());

    let mut remote_page_dest = [0u8; 4096];

    // Enter HotPathAllocGuard to guarantee zero heap allocations across 1,000 operations
    let guard = HotPathAllocGuard::enter();

    for i in 0..1000 {
        // 1. Propose (0 allocations)
        let proposal = match engine0.propose(i, [0xFE; 24], 1_000_000 + i) {
            Ok(p) => p,
            Err(_) => return assert!(false, "propose"),
        };

        // 2. Peer Ack (0 allocations)
        let ack = match engine1.handle_remote_propose(&proposal, 1_000_000 + i + 1) {
            Ok(a) => a,
            Err(_) => return assert!(false, "ack"),
        };

        // 3. Commit check (0 allocations)
        let commit = match engine0.handle_ack(&ack) {
            Ok(Some(c)) => c,
            _ => return assert!(false, "commit"),
        };

        // 4. Morphism log replication apply (0 allocations)
        if rep0.apply_commit(&commit).is_err() {
            return assert!(false, "apply_commit");
        }

        // 5. Read quorum evaluation (0 allocations)
        let responses = [proposal, ack];
        if engine0.evaluate_read_quorum(&responses).is_err() {
            return assert!(false, "evaluate_read_quorum");
        }

        // 6. Zero-copy RDMA page fetch (0 allocations)
        if fabric
            .read_remote_page(0, 0, &mut remote_page_dest)
            .is_err()
        {
            return assert!(false, "read_remote_page");
        }
    }

    core::mem::drop(guard);

    assert_eq!(rep0.total_committed(), 1000);
}
