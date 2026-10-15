//! Project Aether-Manifold: Core Error Primitives
//!
//! Enforces zero-allocation, Copy error taxonomy covering numerical,
//! memory, storage, and concurrency invariants in strict compliance with
//! AGENTS.md Rule 3.1.

use thiserror::Error;

/// Core error enumeration for Project Aether-Manifold.
///
/// Every variant is `Copy` to guarantee zero heap allocation on any error path.
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AetherError {
    /// Numerical horizon violation in hyperbolic or spherical space.
    /// Triggered when the clamped Lorentz inner product z < 1.0 or NaN prevention threshold is violated.
    #[error("Lorentz horizon violation: inner product clamp boundary exceeded")]
    LorentzHorizonViolation,

    /// Page corruption detected via CRC32C checksum mismatch.
    #[error("Page corruption: CRC32C checksum mismatch (expected {expected:#010x}, calculated {calculated:#010x})")]
    PageCorruption {
        /// Checksum value recorded in the page header.
        expected: u32,
        /// Checksum value computed from the payload.
        calculated: u32,
    },

    /// Pre-allocated scratchpad memory exhausted.
    #[error("Scratchpad exhausted: requested {requested} bytes, but only {available} bytes available")]
    ScratchpadExhausted {
        /// Requested byte count.
        requested: usize,
        /// Currently available byte count.
        available: usize,
    },

    /// Memory alignment invariant violated.
    #[error("Invalid alignment: address {ptr:#x} is not aligned to {required_align} bytes")]
    InvalidAlignment {
        /// Pointer address that failed the alignment test.
        ptr: usize,
        /// Required byte alignment.
        required_align: usize,
    },

    /// Buffer size does not match exact page or tensor requirement.
    #[error("Invalid buffer size: buffer length {actual} does not match required {expected} bytes")]
    InvalidBufferSize {
        /// Observed buffer length in bytes.
        actual: usize,
        /// Required buffer length in bytes.
        expected: usize,
    },

    /// Hardware or operating system I/O fault.
    #[error("I/O fault: OS or hardware error code {code}")]
    IoFault {
        /// Platform error code (e.g. errno or OS error).
        code: i32,
    },

    /// Boundary or indexing limit exceeded.
    #[error("Index out of bounds: index {index} exceeds limit {limit}")]
    OutOfBounds {
        /// Attempted access index.
        index: usize,
        /// Maximum allowed boundary.
        limit: usize,
    },

    /// Hot path allocation violation detected by PMU / allocation guard.
    #[error("Hot path allocation violation: {count} heap allocations detected on zero-alloc path")]
    HotPathAllocationViolation {
        /// Number of heap allocations intercepted while hot-path guard was active.
        count: usize,
    },
}

