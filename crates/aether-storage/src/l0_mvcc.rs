//! Project Aether-Manifold: $L_0$ Ephemeral Ring Buffer & Geodesic MVCC Versioning (`l0_mvcc.rs`)
//!
//! Implements causal spatio-temporal versioning per roadmap.md Section 3.4.
//! Appends geodesic velocity vectors $\Delta v \in T_{x_v(\tau_1)}\mathcal{M}$ and evaluates
//! $O(1)$ continuous state reconstruction $x_v(\tau_q) = \exp_{x_v(\tau_1)}(\alpha \cdot \Delta v)$
//! with strictly zero heap allocations on the hot path.

use aether_core::error::AetherError;
use aether_core::hlc::HlcTimestamp;
use aether_manifold::product::ProductManifoldConfig;
use aether_manifold::tangent::{product_exp_map, product_log_map};

/// Tangent-space geodesic velocity update entry:
///
/// Records the transition from base state $x_v(\tau_1)$ to committed state $x_v(\tau_2)$:
///
/// $$\Delta v_{(\tau_1 \to \tau_2)} = \log_{x_v(\tau_1)}\left( x_v(\tau_2) \right) \in T_{x_v(\tau_1)}\mathcal{M}$$
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TangentDeltaEntry<const D: usize> {
    pub vertex_id: u64,
    pub tau_base: HlcTimestamp,
    pub tau_commit: HlcTimestamp,
    pub velocity_delta: [f32; D],
}

impl<const D: usize> TangentDeltaEntry<D> {
    /// Zeroed default entry.
    pub const fn zeroed() -> Self {
        Self {
            vertex_id: 0,
            tau_base: HlcTimestamp(0),
            tau_commit: HlcTimestamp(0),
            velocity_delta: [0.0; D],
        }
    }
}

/// Open-addressed spectral hash table slot for $O(1)$ vertex lookup.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct L0HashSlot {
    vertex_id: u64,
    log_slot: u32,
    occupied: bool,
}

/// $L_0$ Ephemeral Ring Buffer & Open-Addressed Morphism Log.
pub struct L0MorphismLog<const D: usize, const CAP: usize> {
    entries: Box<[TangentDeltaEntry<D>]>,
    hash_buckets: Box<[L0HashSlot]>,
    head: usize,
    count: usize,
}

impl<const D: usize, const CAP: usize> L0MorphismLog<D, CAP> {
    /// Cold-path initialization: pre-allocates $L_0$ ring buffer and hash index.
    pub fn new() -> Self {
        let entries = vec![TangentDeltaEntry::zeroed(); CAP].into_boxed_slice();
        let hash_cap = (CAP.saturating_mul(2)).max(16);
        let hash_buckets = vec![L0HashSlot::default(); hash_cap].into_boxed_slice();

        Self {
            entries,
            hash_buckets,
            head: 0,
            count: 0,
        }
    }

    /// Capacity of the log in entries.
    #[inline]
    pub const fn capacity(&self) -> usize {
        CAP
    }

    /// Current number of valid entries recorded.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if no entries have been recorded yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Record a geodesic tangent velocity vector $\Delta v_{(\tau_1 \to \tau_2)} = \log_{x_v(\tau_1)}(x_v(\tau_2))$
    /// into the $L_0$ log and update the open-addressed index.
    ///
    /// # Hard Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    pub fn record_geodesic_delta(
        &mut self,
        config: &ProductManifoldConfig,
        vertex_id: u64,
        tau_base: HlcTimestamp,
        base_state: &[f32; D],
        tau_commit: HlcTimestamp,
        commit_state: &[f32; D],
    ) -> Result<usize, AetherError> {
        let expected_dim = config.total_dim();
        if D != expected_dim {
            return Err(AetherError::InvalidBufferSize {
                actual: D,
                expected: expected_dim,
            });
        }
        if CAP == 0 {
            return Err(AetherError::ScratchpadExhausted {
                requested: 1,
                available: 0,
            });
        }

        let mut velocity_delta = [0.0f32; D];
        product_log_map(config, base_state, commit_state, &mut velocity_delta)?;

        let slot = self.head % CAP;
        let entry = match self.entries.get_mut(slot) {
            Some(e) => e,
            None => return Err(AetherError::OutOfBounds { index: slot, limit: CAP }),
        };

        *entry = TangentDeltaEntry {
            vertex_id,
            tau_base,
            tau_commit,
            velocity_delta,
        };

        // Update open-addressed spectral hash index
        self.insert_hash_index(vertex_id, slot as u32);

        self.head = self.head.wrapping_add(1);
        self.count = (self.count + 1).min(CAP);

        Ok(slot)
    }

    /// Find the latest `TangentDeltaEntry` for a given `vertex_id`.
    ///
    /// # Hard Invariants
    /// - Strictly 0 heap allocations on the hot path.
    pub fn get_latest_entry(&self, vertex_id: u64) -> Option<&TangentDeltaEntry<D>> {
        let hash_cap = self.hash_buckets.len();
        if hash_cap == 0 {
            return None;
        }

        let h = xxhash_rust::xxh3::xxh3_64(&vertex_id.to_le_bytes()) as usize;
        let max_probes = hash_cap.min(64);

        for step in 0..max_probes {
            let idx = (h + step) % hash_cap;
            let slot = self.hash_buckets.get(idx)?;

            if !slot.occupied {
                return None;
            }

            if slot.vertex_id == vertex_id {
                let log_idx = slot.log_slot as usize;
                return self.entries.get(log_idx);
            }
        }

        None
    }

    /// Continuous Geodesic MVCC State Reconstruction:
    ///
    /// Given base state $x_v(\tau_1)$, entry $\Delta v_{(\tau_1 \to \tau_2)}$, and query timestamp $\tau_q$,
    /// computes the interpolation ratio $\alpha \in [0.0, 1.0]$ and evaluates:
    ///
    /// $$x_v(\tau_q) = \exp_{x_v(\tau_1)}\left( \alpha \cdot \Delta v_{(\tau_1 \to \tau_2)} \right)$$
    ///
    /// into `out_state: &mut [f32; D]` in $O(1)$ time and strictly 0 heap allocations.
    ///
    /// # Hard Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    pub fn reconstruct_state_at(
        &self,
        config: &ProductManifoldConfig,
        base_state: &[f32; D],
        entry: &TangentDeltaEntry<D>,
        tau_q: HlcTimestamp,
        out_state: &mut [f32; D],
    ) -> Result<(), AetherError> {
        let expected_dim = config.total_dim();
        if D != expected_dim {
            return Err(AetherError::InvalidBufferSize {
                actual: D,
                expected: expected_dim,
            });
        }

        let alpha = if tau_q <= entry.tau_base {
            0.0f32
        } else if tau_q >= entry.tau_commit {
            1.0f32
        } else {
            let total_span = entry.tau_commit.as_u64().saturating_sub(entry.tau_base.as_u64());
            let q_span = tau_q.as_u64().saturating_sub(entry.tau_base.as_u64());
            if total_span > 0 {
                (q_span as f32 / total_span as f32).clamp(0.0, 1.0)
            } else {
                1.0f32
            }
        };

        let mut scaled_v = [0.0f32; D];
        for i in 0..D {
            if let (Some(dest), Some(&v)) = (scaled_v.get_mut(i), entry.velocity_delta.get(i)) {
                *dest = alpha * v;
            }
        }

        product_exp_map(config, base_state, &scaled_v, out_state)?;
        Ok(())
    }

    #[inline]
    fn insert_hash_index(&mut self, vertex_id: u64, log_slot: u32) {
        let hash_cap = self.hash_buckets.len();
        if hash_cap == 0 {
            return;
        }

        let h = xxhash_rust::xxh3::xxh3_64(&vertex_id.to_le_bytes()) as usize;
        let max_probes = hash_cap.min(64);

        for step in 0..max_probes {
            let idx = (h + step) % hash_cap;
            let slot = match self.hash_buckets.get_mut(idx) {
                Some(s) => s,
                None => return,
            };

            if !slot.occupied || slot.vertex_id == vertex_id {
                slot.vertex_id = vertex_id;
                slot.log_slot = log_slot;
                slot.occupied = true;
                return;
            }
        }
    }
}

impl<const D: usize, const CAP: usize> Default for L0MorphismLog<D, CAP> {
    fn default() -> Self {
        Self::new()
    }
}

