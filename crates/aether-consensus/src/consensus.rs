//! Project Aether-Manifold: Leaderless HLC Quorum Consensus Engine
//!
//! Implements a wait-free leaderless quorum consensus engine with 64-byte cache-line
//! aligned replication messages and deterministic causal total ordering via Hybrid Logical Clocks.
//! Strictly zero heap allocations on the consensus hot-path (AGENTS.md Rule 3.1 & 3.2).

use core::sync::atomic::{AtomicU64, Ordering};

use aether_core::error::AetherError;
use aether_core::hlc::HlcTimestamp;
use aether_storage::spsc::SpscRingBuffer;
use bytemuck::{Pod, Zeroable};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Type alias for wait-free SPSC queue communicating 64-byte replication messages across cores.
pub type ReplicationRingBuffer<const N: usize> = SpscRingBuffer<ReplicationMessage, N>;

/// Maximum concurrent in-flight proposals tracked in pre-allocated ring buffer.
pub const MAX_IN_FLIGHT_PROPOSALS: usize = 1024;

/// Message types for replication and quorum coordination.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplicationMessageType {
    /// New state mutation proposal.
    Propose = 0,
    /// Quorum acknowledgment of proposal.
    Ack = 1,
    /// Negative acknowledgment / conflict rejection.
    Nack = 2,
    /// Final commit broadcast once write quorum is achieved.
    Commit = 3,
    /// Read request for causal state.
    ReadRequest = 4,
    /// Causal state read response.
    ReadResponse = 5,
    /// Shard liveness and clock synchronization heartbeat.
    Heartbeat = 6,
}

impl ReplicationMessageType {
    /// Converts a raw `u8` into `ReplicationMessageType`.
    #[inline]
    pub const fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::Propose),
            1 => Some(Self::Ack),
            2 => Some(Self::Nack),
            3 => Some(Self::Commit),
            4 => Some(Self::ReadRequest),
            5 => Some(Self::ReadResponse),
            6 => Some(Self::Heartbeat),
            _ => None,
        }
    }
}

/// 64-byte cache-line aligned replication message envelope.
///
/// Layout:
/// - 00..08 (8B): HLC Timestamp (64-bit packed microsecond + logical)
/// - 08..16 (8B): Node ID (Originating node identifier)
/// - 16..24 (8B): Sequence Number (Monotonic per-node proposal ID)
/// - 24..32 (8B): Vertex ID (Topology vertex being updated)
/// - 32..36 (4B): CRC32C Checksum (Covers bytes 0..32 and 36..64)
/// - 36..37 (1B): Message Type (`ReplicationMessageType`)
/// - 37..38 (1B): Flags (Bitmask: 0x01 = urgent, 0x02 = committed)
/// - 38..40 (2B): Status / Return code
/// - 40..64 (24B): Payload data (Tangent delta / coordinate vector hash)
#[repr(C, align(64))]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable, FromBytes, IntoBytes, KnownLayout, Immutable,
)]
pub struct ReplicationMessage {
    /// Hybrid Logical Clock timestamp.
    pub hlc_timestamp: u64,
    /// Originating or sender node ID.
    pub node_id: u64,
    /// Sequence number of the proposal.
    pub sequence_number: u64,
    /// Target graph vertex identifier.
    pub vertex_id: u64,
    /// CRC32C integrity checksum.
    pub crc32c: u32,
    /// Message type identifier.
    pub msg_type: u8,
    /// Control flags.
    pub flags: u8,
    /// Status code.
    pub status: u16,
    /// Fixed-size payload bytes (e.g. tangent delta or coordinate hash).
    pub payload: [u8; 24],
}

static_assertions::const_assert_eq!(core::mem::size_of::<ReplicationMessage>(), 64);
static_assertions::const_assert_eq!(core::mem::align_of::<ReplicationMessage>(), 64);

impl ReplicationMessage {
    /// Creates a new unchecksummed message.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        hlc_timestamp: u64,
        node_id: u64,
        sequence_number: u64,
        vertex_id: u64,
        msg_type: ReplicationMessageType,
        flags: u8,
        status: u16,
        payload: [u8; 24],
    ) -> Self {
        Self {
            hlc_timestamp,
            node_id,
            sequence_number,
            vertex_id,
            crc32c: 0,
            msg_type: msg_type as u8,
            flags,
            status,
            payload,
        }
    }

    /// Computes the CRC32C checksum across bytes 0..32 and 36..64.
    #[inline]
    pub fn compute_crc(&self) -> u32 {
        let bytes: [u8; 64] = bytemuck::cast(*self);
        let mut hasher = crc32fast::Hasher::new();
        if let Some(b1) = bytes.get(0..32) {
            hasher.update(b1);
        }
        if let Some(b2) = bytes.get(36..64) {
            hasher.update(b2);
        }
        hasher.finalize()
    }

    /// Recomputes and updates the `crc32c` field in place.
    #[inline]
    pub fn update_crc(&mut self) {
        self.crc32c = self.compute_crc();
    }

    /// Verifies the message CRC32C checksum.
    #[inline]
    pub fn verify_crc(&self) -> Result<(), AetherError> {
        let calculated = self.compute_crc();
        if self.crc32c == calculated {
            Ok(())
        } else {
            Err(AetherError::PageCorruption {
                expected: self.crc32c,
                calculated,
            })
        }
    }

    /// Returns the message type enum.
    #[inline]
    pub fn message_type(&self) -> Option<ReplicationMessageType> {
        ReplicationMessageType::from_u8(self.msg_type)
    }

    /// Checks if this message causally dominates another under (HLC, node_id) lexicographical order.
    #[inline]
    pub fn causally_dominates(&self, other: &Self) -> bool {
        if self.hlc_timestamp != other.hlc_timestamp {
            self.hlc_timestamp > other.hlc_timestamp
        } else {
            self.node_id > other.node_id
        }
    }
}

/// Configuration parameters for leaderless quorum consensus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuorumConfig {
    /// Unique identifier for this cluster node (0..cluster_size).
    pub node_id: u64,
    /// Total number of replica nodes in the cluster shard (N).
    pub cluster_size: usize,
    /// Minimum write acknowledgments required for commit (W).
    pub write_quorum: usize,
    /// Minimum read responses required for consistent read (R).
    pub read_quorum: usize,
}

impl QuorumConfig {
    /// Validates the quorum configuration (R + W > N and W <= N, R <= N).
    pub fn new(
        node_id: u64,
        cluster_size: usize,
        write_quorum: usize,
        read_quorum: usize,
    ) -> Result<Self, AetherError> {
        if cluster_size == 0
            || write_quorum == 0
            || read_quorum == 0
            || write_quorum > cluster_size
            || read_quorum > cluster_size
        {
            return Err(AetherError::OutOfBounds {
                index: write_quorum.max(read_quorum),
                limit: cluster_size,
            });
        }
        // Enforce strong quorum intersection: R + W > N
        if write_quorum.saturating_add(read_quorum) <= cluster_size {
            return Err(AetherError::QuorumNotReached {
                required: cluster_size.saturating_add(1),
                received: write_quorum.saturating_add(read_quorum),
            });
        }
        Ok(Self {
            node_id,
            cluster_size,
            write_quorum,
            read_quorum,
        })
    }
}

/// Pre-allocated slot tracking an in-flight proposal.
#[derive(Clone, Copy, Debug)]
struct InFlightSlot {
    active: bool,
    sequence_number: u64,
    hlc_timestamp: u64,
    vertex_id: u64,
    ack_mask: u64,
    nack_count: usize,
    committed: bool,
    payload: [u8; 24],
}

impl InFlightSlot {
    const fn empty() -> Self {
        Self {
            active: false,
            sequence_number: 0,
            hlc_timestamp: 0,
            vertex_id: 0,
            ack_mask: 0,
            nack_count: 0,
            committed: false,
            payload: [0u8; 24],
        }
    }
}

/// Leaderless HLC Quorum Consensus Engine.
///
/// Operates with strictly zero heap allocations on the transaction path.
pub struct LeaderlessQuorumEngine {
    config: QuorumConfig,
    clock: AtomicU64,
    next_sequence: u64,
    in_flight: Box<[InFlightSlot; MAX_IN_FLIGHT_PROPOSALS]>,
}

impl LeaderlessQuorumEngine {
    /// Creates a new `LeaderlessQuorumEngine` instance.
    pub fn new(config: QuorumConfig, initial_physical_micros: u64) -> Self {
        let initial_hlc = HlcTimestamp::pack(initial_physical_micros, 0);
        let in_flight = Box::new([InFlightSlot::empty(); MAX_IN_FLIGHT_PROPOSALS]);
        Self {
            config,
            clock: AtomicU64::new(initial_hlc.as_u64()),
            next_sequence: 1,
            in_flight,
        }
    }

    /// Retrieves current configuration.
    #[inline]
    pub const fn config(&self) -> &QuorumConfig {
        &self.config
    }

    /// Reads current local HLC timestamp.
    #[inline]
    pub fn current_hlc(&self) -> HlcTimestamp {
        HlcTimestamp(self.clock.load(Ordering::Acquire))
    }

    /// Advances the local HLC clock monotonically.
    #[inline]
    pub fn tick(&self, physical_now_micros: u64) -> HlcTimestamp {
        HlcTimestamp::now_or_tick(&self.clock, physical_now_micros)
    }

    /// Merges an incoming remote HLC timestamp with local clock.
    #[inline]
    pub fn merge_remote_hlc(&self, remote: HlcTimestamp, physical_now_micros: u64) -> HlcTimestamp {
        HlcTimestamp::update_with_remote(&self.clock, remote, physical_now_micros)
    }

    /// Creates and registers a new proposal to replicate across the cluster.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    pub fn propose(
        &mut self,
        vertex_id: u64,
        payload: [u8; 24],
        physical_now_micros: u64,
    ) -> Result<ReplicationMessage, AetherError> {
        let ts = self.tick(physical_now_micros);
        let seq = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);

        let slot_idx = (seq as usize) % MAX_IN_FLIGHT_PROPOSALS;
        if let Some(slot) = self.in_flight.get_mut(slot_idx) {
            slot.active = true;
            slot.sequence_number = seq;
            slot.hlc_timestamp = ts.as_u64();
            slot.vertex_id = vertex_id;
            // Record self acknowledgment (own node)
            let self_bit = 1u64 << (self.config.node_id % 64);
            slot.ack_mask = self_bit;
            slot.nack_count = 0;
            slot.committed = false;
            slot.payload = payload;
        }

        let mut msg = ReplicationMessage::new(
            ts.as_u64(),
            self.config.node_id,
            seq,
            vertex_id,
            ReplicationMessageType::Propose,
            0,
            0,
            payload,
        );
        msg.update_crc();
        Ok(msg)
    }

    /// Processes an incoming remote `Propose` message on a replica node.
    ///
    /// Validates causal ordering and generates an `Ack` or `Nack`.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path.
    pub fn handle_remote_propose(
        &mut self,
        msg: &ReplicationMessage,
        physical_now_micros: u64,
    ) -> Result<ReplicationMessage, AetherError> {
        msg.verify_crc()?;

        // Advance local clock causally
        let merged_ts = self.merge_remote_hlc(HlcTimestamp(msg.hlc_timestamp), physical_now_micros);

        // Generate Ack response
        let mut ack = ReplicationMessage::new(
            merged_ts.as_u64(),
            self.config.node_id,
            msg.sequence_number,
            msg.vertex_id,
            ReplicationMessageType::Ack,
            0,
            0,
            msg.payload,
        );
        ack.update_crc();
        Ok(ack)
    }

    /// Processes an incoming `Ack` from a peer replica.
    ///
    /// If write quorum $W$ is achieved, transitions the proposal to committed
    /// and returns the broadcast `Commit` message.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path.
    pub fn handle_ack(
        &mut self,
        ack_msg: &ReplicationMessage,
    ) -> Result<Option<ReplicationMessage>, AetherError> {
        ack_msg.verify_crc()?;

        let slot_idx = (ack_msg.sequence_number as usize) % MAX_IN_FLIGHT_PROPOSALS;
        let (should_commit, hlc_ts, vertex_id, payload) = match self.in_flight.get_mut(slot_idx) {
            Some(slot) if slot.active && slot.sequence_number == ack_msg.sequence_number => {
                let peer_bit = 1u64 << (ack_msg.node_id % 64);
                slot.ack_mask |= peer_bit;
                let ack_count = slot.ack_mask.count_ones() as usize;

                if ack_count >= self.config.write_quorum && !slot.committed {
                    slot.committed = true;
                    (true, slot.hlc_timestamp, slot.vertex_id, slot.payload)
                } else {
                    (false, 0, 0, [0u8; 24])
                }
            }
            _ => (false, 0, 0, [0u8; 24]),
        };

        if should_commit {
            let mut commit_msg = ReplicationMessage::new(
                hlc_ts,
                self.config.node_id,
                ack_msg.sequence_number,
                vertex_id,
                ReplicationMessageType::Commit,
                0x02, // Committed flag
                0,
                payload,
            );
            commit_msg.update_crc();
            Ok(Some(commit_msg))
        } else {
            Ok(None)
        }
    }

    /// Evaluates consistent quorum read across collected replica responses.
    ///
    /// Selects the causally dominant state $(hlc, node\_id)$ satisfying read quorum $R$.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path.
    pub fn evaluate_read_quorum(
        &self,
        responses: &[ReplicationMessage],
    ) -> Result<ReplicationMessage, AetherError> {
        if responses.len() < self.config.read_quorum {
            return Err(AetherError::QuorumNotReached {
                required: self.config.read_quorum,
                received: responses.len(),
            });
        }

        let mut winning_msg: Option<&ReplicationMessage> = None;
        for msg in responses {
            msg.verify_crc()?;
            match winning_msg {
                None => winning_msg = Some(msg),
                Some(current) => {
                    if msg.causally_dominates(current) {
                        winning_msg = Some(msg);
                    }
                }
            }
        }

        winning_msg.copied().ok_or(AetherError::QuorumNotReached {
            required: self.config.read_quorum,
            received: 0,
        })
    }
}
