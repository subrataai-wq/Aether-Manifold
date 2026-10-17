//! Project Aether-Manifold: $L_0$ Morphism Log Replication & Fast Replay Engine
//!
//! Replicates topological morphism logs and tangent updates across cluster shards.
//! Provides sub-10ms fast HLC log replay for minority node recovery with zero hot-path heap allocations.

use aether_core::error::AetherError;
use aether_core::hlc::HlcTimestamp;

use crate::consensus::{ReplicationMessage, ReplicationMessageType};

/// Maximum capacity of pre-allocated committed replication log.
pub const REPLICATOR_LOG_CAPACITY: usize = 4096;

/// Maximum number of tracked vertices in the fast state index.
pub const MAX_TRACKED_VERTICES: usize = 2048;

/// A committed log record storing causal HLC, vertex identifier, and payload.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommittedLogRecord {
    /// Causal HLC timestamp at commit.
    pub hlc_timestamp: u64,
    /// Originating node identifier.
    pub node_id: u64,
    /// Proposal sequence number.
    pub sequence_number: u64,
    /// Target graph vertex identifier.
    pub vertex_id: u64,
    /// 24-byte coordinate delta or tangent payload.
    pub payload: [u8; 24],
}

impl CommittedLogRecord {
    /// Creates a zeroed default record.
    pub const fn empty() -> Self {
        Self {
            hlc_timestamp: 0,
            node_id: 0,
            sequence_number: 0,
            vertex_id: 0,
            payload: [0u8; 24],
        }
    }

    /// Converts a committed `ReplicationMessage` into a `CommittedLogRecord`.
    pub const fn from_message(msg: &ReplicationMessage) -> Self {
        Self {
            hlc_timestamp: msg.hlc_timestamp,
            node_id: msg.node_id,
            sequence_number: msg.sequence_number,
            vertex_id: msg.vertex_id,
            payload: msg.payload,
        }
    }
}

/// Slot in the open-addressed vertex state table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VertexSlot {
    occupied: bool,
    vertex_id: u64,
    latest_hlc: u64,
    latest_payload: [u8; 24],
}

impl VertexSlot {
    const fn empty() -> Self {
        Self {
            occupied: false,
            vertex_id: 0,
            latest_hlc: 0,
            latest_payload: [0u8; 24],
        }
    }
}

/// $L_0$ Morphism Log Replicator and fast recovery replay engine.
pub struct MorphismReplicator {
    node_id: u64,
    log_ring: Box<[CommittedLogRecord; REPLICATOR_LOG_CAPACITY]>,
    log_head: usize,
    total_committed: usize,
    vertex_table: Box<[VertexSlot; MAX_TRACKED_VERTICES]>,
}

impl MorphismReplicator {
    /// Creates a new pre-allocated `MorphismReplicator`.
    pub fn new(node_id: u64) -> Self {
        let log_ring = Box::new([CommittedLogRecord::empty(); REPLICATOR_LOG_CAPACITY]);
        let vertex_table = Box::new([VertexSlot::empty(); MAX_TRACKED_VERTICES]);
        Self {
            node_id,
            log_ring,
            log_head: 0,
            total_committed: 0,
            vertex_table,
        }
    }

    /// Returns this node identifier.
    #[inline]
    pub const fn node_id(&self) -> u64 {
        self.node_id
    }

    /// Returns the total number of committed records recorded since startup.
    #[inline]
    pub const fn total_committed(&self) -> usize {
        self.total_committed
    }

    /// Applies a validated commit message to the local morphism log and vertex state.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    pub fn apply_commit(&mut self, commit_msg: &ReplicationMessage) -> Result<(), AetherError> {
        commit_msg.verify_crc()?;

        if commit_msg.msg_type != ReplicationMessageType::Commit as u8 {
            return Err(AetherError::ReplicationConflict {
                expected_hlc: commit_msg.hlc_timestamp,
                actual_hlc: 0,
            });
        }

        let record = CommittedLogRecord::from_message(commit_msg);
        self.apply_record(record);
        Ok(())
    }

    /// Internal record application with Last-Write-Wins causal resolution.
    fn apply_record(&mut self, record: CommittedLogRecord) {
        // Append to ring buffer
        let slot = self.log_head;
        if let Some(entry) = self.log_ring.get_mut(slot) {
            *entry = record;
        }
        self.log_head = (self.log_head.saturating_add(1)) % REPLICATOR_LOG_CAPACITY;
        self.total_committed = self.total_committed.saturating_add(1);

        // Update vertex state with Last-Write-Wins (HLC)
        self.upsert_vertex(record.vertex_id, record.hlc_timestamp, record.payload);
    }

    /// Open-addressed linear probing to update latest vertex state.
    fn upsert_vertex(&mut self, vertex_id: u64, hlc: u64, payload: [u8; 24]) {
        let mask = MAX_TRACKED_VERTICES.saturating_sub(1);
        let mut idx = ((vertex_id as usize) ^ ((vertex_id >> 16) as usize)) & mask;

        for _ in 0..MAX_TRACKED_VERTICES {
            if let Some(slot) = self.vertex_table.get_mut(idx) {
                if !slot.occupied {
                    slot.occupied = true;
                    slot.vertex_id = vertex_id;
                    slot.latest_hlc = hlc;
                    slot.latest_payload = payload;
                    return;
                } else if slot.vertex_id == vertex_id {
                    // Last-Write-Wins: only update if incoming HLC is strictly newer
                    if hlc > slot.latest_hlc {
                        slot.latest_hlc = hlc;
                        slot.latest_payload = payload;
                    }
                    return;
                }
            }
            idx = (idx.saturating_add(1)) & mask;
        }
    }

    /// Retrieves latest state for a given vertex.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path.
    pub fn get_vertex_state(&self, vertex_id: u64) -> Option<(HlcTimestamp, [u8; 24])> {
        let mask = MAX_TRACKED_VERTICES.saturating_sub(1);
        let mut idx = ((vertex_id as usize) ^ ((vertex_id >> 16) as usize)) & mask;

        for _ in 0..MAX_TRACKED_VERTICES {
            if let Some(slot) = self.vertex_table.get(idx) {
                if !slot.occupied {
                    return None;
                }
                if slot.vertex_id == vertex_id {
                    return Some((HlcTimestamp(slot.latest_hlc), slot.latest_payload));
                }
            }
            idx = (idx.saturating_add(1)) & mask;
        }
        None
    }

    /// Collects committed log records since a specified HLC timestamp into `out_buf`.
    ///
    /// Returns the number of records copied.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path.
    pub fn get_records_since(&self, since_hlc: u64, out_buf: &mut [CommittedLogRecord]) -> usize {
        let available = self.total_committed.min(REPLICATOR_LOG_CAPACITY);
        let mut copied = 0usize;

        let start_pos = if self.total_committed > REPLICATOR_LOG_CAPACITY {
            self.log_head
        } else {
            0
        };

        for i in 0..available {
            if copied >= out_buf.len() {
                break;
            }
            let pos = (start_pos.saturating_add(i)) % REPLICATOR_LOG_CAPACITY;
            if let Some(entry) = self.log_ring.get(pos) {
                if entry.hlc_timestamp > since_hlc {
                    if let Some(dest) = out_buf.get_mut(copied) {
                        *dest = *entry;
                        copied = copied.saturating_add(1);
                    }
                }
            }
        }

        copied
    }

    /// Replays a batch of replication records into local state.
    ///
    /// Used for fast failure recovery (<10ms).
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path.
    pub fn replay_batch(&mut self, records: &[CommittedLogRecord]) -> Result<usize, AetherError> {
        let mut applied = 0usize;
        for &record in records {
            self.apply_record(record);
            applied = applied.saturating_add(1);
        }
        Ok(applied)
    }
}
