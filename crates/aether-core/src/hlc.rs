//! Project Aether-Manifold: Hybrid Logical Clock (HLC)
//!
//! Implements a 64-bit Hybrid Logical Clock packing a 48-bit physical microsecond
//! wall clock and a 16-bit logical counter with wait-free/lock-free CAS monotonic
//! tick and receive-merge operations, per AGENTS.md and roadmap.md Section 3.4.

#[cfg(not(loom))]
use core::sync::atomic::{AtomicU64, Ordering};
#[cfg(loom)]
use loom::sync::atomic::{AtomicU64, Ordering};

use bytemuck::{Pod, Zeroable};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Number of bits dedicated to the physical microsecond timestamp.
pub const PHYSICAL_BITS: u32 = 48;

/// Number of bits dedicated to the logical counter.
pub const LOGICAL_BITS: u32 = 16;

/// Bitmask for the 48-bit physical timestamp.
pub const MAX_PHYSICAL: u64 = (1u64 << PHYSICAL_BITS) - 1;

/// Maximum value of the 16-bit logical counter.
pub const MAX_LOGICAL: u16 = 0xFFFF;

/// 64-bit Hybrid Logical Clock timestamp.
///
/// Encodes:
/// - Bits [16..64]: 48-bit physical microsecond wall clock ($t_{\text{physical}}$)
/// - Bits [0..16]: 16-bit logical counter ($c_{\text{logical}}$)
///
/// Since the physical component occupies the most significant bits, numerical
/// comparison (`u64::cmp`) coincides identically with causal lexicographical ordering.
#[repr(transparent)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Pod,
    Zeroable,
    FromBytes,
    IntoBytes,
    KnownLayout,
    Immutable,
)]
pub struct HlcTimestamp(pub u64);

impl HlcTimestamp {
    /// Construct an `HlcTimestamp` directly from its 64-bit raw packed value.
    #[inline]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Retrieve the underlying 64-bit raw representation.
    #[inline]
    pub const fn as_u64(&self) -> u64 {
        self.0
    }

    /// Pack a 48-bit physical microsecond clock and 16-bit logical counter into an `HlcTimestamp`.
    #[inline]
    pub const fn pack(physical_micros: u64, logical_counter: u16) -> Self {
        let phys_clamped = physical_micros & MAX_PHYSICAL;
        let raw = (phys_clamped << LOGICAL_BITS) | (logical_counter as u64);
        Self(raw)
    }

    /// Unpack into `(physical_micros, logical_counter)`.
    #[inline]
    pub const fn unpack(&self) -> (u64, u16) {
        let phys = (self.0 >> LOGICAL_BITS) & MAX_PHYSICAL;
        let log = (self.0 & (MAX_LOGICAL as u64)) as u16;
        (phys, log)
    }

    /// Extract the physical microsecond timestamp component (bits 16..64).
    #[inline]
    pub const fn physical(&self) -> u64 {
        (self.0 >> LOGICAL_BITS) & MAX_PHYSICAL
    }

    /// Extract the logical counter component (bits 0..16).
    #[inline]
    pub const fn logical(&self) -> u16 {
        (self.0 & (MAX_LOGICAL as u64)) as u16
    }

    /// Advance `self` monotonically in a single-threaded/mutable context.
    #[inline]
    pub fn tick(&mut self, physical_now_micros: u64) -> Self {
        let phys_clamped = physical_now_micros & MAX_PHYSICAL;
        let curr_phys = self.physical();
        let curr_log = self.logical();

        let (next_phys, next_log) = if phys_clamped > curr_phys {
            (phys_clamped, 0u16)
        } else if curr_log < MAX_LOGICAL {
            (curr_phys, curr_log.saturating_add(1))
        } else {
            // Logical counter exhausted in same microsecond: advance physical component
            (curr_phys.saturating_add(1) & MAX_PHYSICAL, 0u16)
        };

        *self = Self::pack(next_phys, next_log);
        *self
    }

    /// Wait-free CAS operation advancing an atomic HLC clock state monotonically.
    ///
    /// Preserves strict causality $\tau_{\text{next}} > \tau_{\text{curr}}$ even in
    /// the presence of backward physical clock drift (NTP jumps).
    pub fn now_or_tick(state: &AtomicU64, physical_now_micros: u64) -> Self {
        let phys_clamped = physical_now_micros & MAX_PHYSICAL;
        let mut current = state.load(Ordering::Acquire);
        loop {
            let curr_ts = HlcTimestamp(current);
            let curr_phys = curr_ts.physical();
            let curr_log = curr_ts.logical();

            let (next_phys, next_log) = if phys_clamped > curr_phys {
                (phys_clamped, 0u16)
            } else if curr_log < MAX_LOGICAL {
                (curr_phys, curr_log.saturating_add(1))
            } else {
                (curr_phys.saturating_add(1) & MAX_PHYSICAL, 0u16)
            };

            let next_ts = Self::pack(next_phys, next_log);
            match state.compare_exchange_weak(
                current,
                next_ts.0,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return next_ts,
                Err(actual) => current = actual,
            }
        }
    }

    /// Wait-free CAS operation merging a received causal timestamp with the local atomic state.
    ///
    /// Ensures $\tau_{\text{next}} > \tau_{\text{local}}$ and $\tau_{\text{next}} > \tau_{\text{remote}}$.
    pub fn update_with_remote(state: &AtomicU64, remote: Self, physical_now_micros: u64) -> Self {
        let phys_clamped = physical_now_micros & MAX_PHYSICAL;
        let remote_phys = remote.physical();
        let remote_log = remote.logical();

        let mut current = state.load(Ordering::Acquire);
        loop {
            let curr_ts = HlcTimestamp(current);
            let curr_phys = curr_ts.physical();
            let curr_log = curr_ts.logical();

            let max_phys = curr_phys.max(remote_phys).max(phys_clamped);

            let (next_phys, next_log) = if max_phys == curr_phys && max_phys == remote_phys {
                let max_log = curr_log.max(remote_log);
                if max_log < MAX_LOGICAL {
                    (max_phys, max_log.saturating_add(1))
                } else {
                    (max_phys.saturating_add(1) & MAX_PHYSICAL, 0u16)
                }
            } else if max_phys == curr_phys {
                if curr_log < MAX_LOGICAL {
                    (max_phys, curr_log.saturating_add(1))
                } else {
                    (max_phys.saturating_add(1) & MAX_PHYSICAL, 0u16)
                }
            } else if max_phys == remote_phys {
                if remote_log < MAX_LOGICAL {
                    (max_phys, remote_log.saturating_add(1))
                } else {
                    (max_phys.saturating_add(1) & MAX_PHYSICAL, 0u16)
                }
            } else {
                (max_phys, 0u16)
            };

            let next_ts = Self::pack(next_phys, next_log);
            match state.compare_exchange_weak(
                current,
                next_ts.0,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return next_ts,
                Err(actual) => current = actual,
            }
        }
    }
}

/// Atomic Hybrid Logical Clock wrapper.
#[repr(transparent)]
#[derive(Debug)]
pub struct HlcClock {
    state: AtomicU64,
}

impl HlcClock {
    /// Initialize a new atomic HLC clock.
    pub const fn new(initial: HlcTimestamp) -> Self {
        Self {
            state: AtomicU64::new(initial.0),
        }
    }

    /// Advance or tick the clock monotonically against the physical microsecond reading.
    #[inline]
    pub fn now_or_tick(&self, physical_now_micros: u64) -> HlcTimestamp {
        HlcTimestamp::now_or_tick(&self.state, physical_now_micros)
    }

    /// Merge with a remote timestamp monotonically.
    #[inline]
    pub fn update_with_remote(
        &self,
        remote: HlcTimestamp,
        physical_now_micros: u64,
    ) -> HlcTimestamp {
        HlcTimestamp::update_with_remote(&self.state, remote, physical_now_micros)
    }

    /// Read the current timestamp snapshot.
    #[inline]
    pub fn read(&self) -> HlcTimestamp {
        HlcTimestamp(self.state.load(Ordering::Acquire))
    }
}
