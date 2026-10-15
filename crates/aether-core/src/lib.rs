//! Project Aether-Manifold: Core Systems & Memory Architecture
//!
//! Provides the foundational 4KB page memory layout, 64B cache-line header,
//! zero-allocation tracking guard, Hybrid Logical Clock (HLC), scratchpad allocator,
//! and typed error primitives.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod alloc_guard;
pub mod error;
pub mod hlc;
pub mod page;
pub mod scratchpad;

pub use alloc_guard::{
    is_hot_path_active, reset_hot_path_counters, HotPathAllocGuard, HotPathGuard, TrackingAllocator,
};
pub use error::AetherError;
pub use hlc::{HlcClock, HlcTimestamp};
pub use page::{PageHeader, TensorGraphPage};
pub use scratchpad::Scratchpad;

