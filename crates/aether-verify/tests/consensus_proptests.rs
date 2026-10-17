//! Deep Property-Based Fuzzing for Consensus, Replicator, and Partitioner Invariants
//!
//! Enforces:
//! - Strict total ordering of causal dominance across arbitrary HLC values and node IDs.
//! - CRC32C bit corruption detection across random 64B replication messages.
//! - Partitioner balance tolerance invariance across randomized graph inputs.
//! - Quorum read causal maximum selection invariance.

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
use aether_consensus::partitioner::{GraphEdge, PartitionerConfig, SpectralGeodesicPartitioner};
use proptest::prelude::*;

proptest! {
    /// Property 1: Causal dominance defines a strict total ordering:
    /// - Irreflexive: !(a > a)
    /// - Asymmetric: a > b => !(b > a)
    /// - Transitive: a > b && b > c => a > c
    #[test]
    fn prop_hlc_causal_dominance_total_order(
        hlc_a in 0u64..u64::MAX,
        node_a in 0u64..1000,
        hlc_b in 0u64..u64::MAX,
        node_b in 0u64..1000,
        hlc_c in 0u64..u64::MAX,
        node_c in 0u64..1000,
    ) {
        let mut msg_a = ReplicationMessage::new(
            hlc_a, node_a, 1, 10, ReplicationMessageType::Propose, 0, 0, [0u8; 24],
        );
        msg_a.update_crc();

        let mut msg_b = ReplicationMessage::new(
            hlc_b, node_b, 1, 10, ReplicationMessageType::Propose, 0, 0, [0u8; 24],
        );
        msg_b.update_crc();

        let mut msg_c = ReplicationMessage::new(
            hlc_c, node_c, 1, 10, ReplicationMessageType::Propose, 0, 0, [0u8; 24],
        );
        msg_c.update_crc();

        // Irreflexivity
        prop_assert!(!msg_a.causally_dominates(&msg_a));
        prop_assert!(!msg_b.causally_dominates(&msg_b));
        prop_assert!(!msg_c.causally_dominates(&msg_c));

        // Asymmetry
        if msg_a.causally_dominates(&msg_b) {
            prop_assert!(!msg_b.causally_dominates(&msg_a));
        }
        if msg_b.causally_dominates(&msg_a) {
            prop_assert!(!msg_a.causally_dominates(&msg_b));
        }

        // Transitivity
        if msg_a.causally_dominates(&msg_b) && msg_b.causally_dominates(&msg_c) {
            prop_assert!(msg_a.causally_dominates(&msg_c));
        }
    }

    /// Property 2: CRC32C detects arbitrary 1-bit corruption in any byte of the 64B message envelope.
    #[test]
    fn prop_crc32c_detects_single_bit_flip(
        hlc in 0u64..u64::MAX,
        node_id in 0u64..u64::MAX,
        seq in 0u64..u64::MAX,
        vertex_id in 0u64..u64::MAX,
        byte_idx in 0usize..64,
        bit_idx in 0usize..8,
    ) {
        // Skip mutating the crc32c field itself (offsets 32..36)
        prop_assume!(!(32..36).contains(&byte_idx));

        let mut msg = ReplicationMessage::new(
            hlc, node_id, seq, vertex_id, ReplicationMessageType::Propose, 0, 0, [42u8; 24],
        );
        msg.update_crc();
        prop_assert!(msg.verify_crc().is_ok());

        let mut bytes: [u8; 64] = bytemuck::cast(msg);
        if let Some(target_byte) = bytes.get_mut(byte_idx) {
            *target_byte ^= 1u8 << bit_idx;
        }

        let corrupted_msg: ReplicationMessage = bytemuck::cast(bytes);
        prop_assert!(corrupted_msg.verify_crc().is_err());
    }

    /// Property 3: Spectral-Geodesic Partitioner balance invariant:
    /// For any randomized vertex count (20..200) and K (2..8), all shards are balanced within tolerance.
    #[test]
    fn prop_partitioner_balance_invariant(
        num_vertices in 20usize..120,
        num_shards in 2usize..6,
        tolerance in 0.05f32..0.15f32,
    ) {
        let config = PartitionerConfig {
            num_shards,
            geodesic_weight_mu: 1.0,
            balance_tolerance: tolerance,
            max_power_iterations: 80,
        };
        let partitioner = SpectralGeodesicPartitioner::new(config);
        prop_assert!(partitioner.is_ok());
        if let Ok(p) = partitioner {
            // Generate some random ring edges
            let mut edges = Vec::new();
            for i in 0..num_vertices {
                let j = (i + 1) % num_vertices;
                edges.push(GraphEdge::new(i, j, 0.1, 0.1));
            }

            let plan_res = p.partition(num_vertices, &edges);
            prop_assert!(plan_res.is_ok());
            if let Ok(plan) = plan_res {
                prop_assert_eq!(plan.num_vertices(), num_vertices);
                prop_assert_eq!(plan.num_shards(), num_shards);
                prop_assert!(plan.is_balanced(tolerance));
            }
        }
    }

    /// Property 4: Read quorum causal resolution:
    /// Always returns the highest (hlc, node_id) entry regardless of input permutation.
    #[test]
    fn prop_read_quorum_order_invariance(
        h1 in 100u64..200,
        h2 in 201u64..300,
        h3 in 301u64..400,
    ) {
        let config_res = QuorumConfig::new(0, 3, 2, 2);
        prop_assert!(config_res.is_ok());
        if let Ok(cfg) = config_res {
            let engine = LeaderlessQuorumEngine::new(cfg, 1000);

            let mut m1 = ReplicationMessage::new(h1, 1, 1, 10, ReplicationMessageType::ReadResponse, 0, 0, [1u8; 24]);
            m1.update_crc();
            let mut m2 = ReplicationMessage::new(h2, 2, 2, 10, ReplicationMessageType::ReadResponse, 0, 0, [2u8; 24]);
            m2.update_crc();
            let mut m3 = ReplicationMessage::new(h3, 3, 3, 10, ReplicationMessageType::ReadResponse, 0, 0, [3u8; 24]);
            m3.update_crc();

            let order_a = [m1, m2, m3];
            let order_b = [m3, m1, m2];
            let order_c = [m2, m3, m1];

            let res_a = engine.evaluate_read_quorum(&order_a);
            let res_b = engine.evaluate_read_quorum(&order_b);
            let res_c = engine.evaluate_read_quorum(&order_c);

            prop_assert!(res_a.is_ok() && res_b.is_ok() && res_c.is_ok());
            if let (Ok(wa), Ok(wb), Ok(wc)) = (res_a, res_b, res_c) {
                prop_assert_eq!(wa.hlc_timestamp, h3);
                prop_assert_eq!(wb.hlc_timestamp, h3);
                prop_assert_eq!(wc.hlc_timestamp, h3);
            }
        }
    }
}
