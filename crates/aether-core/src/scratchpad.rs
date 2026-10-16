//! Project Aether-Manifold: Bounded Zero-Allocation Scratchpad Allocator
//!
//! Provides deterministic scratchpad memory via `dyn_stack::PodStack` with
//! explicit bounds checking to prevent heap allocations or panics on overflow.

use crate::error::AetherError;
use dyn_stack::{PodStack, StackReq};

/// Bounded scratchpad memory workspace wrapping a pre-allocated byte buffer.
pub struct Scratchpad<'a> {
    stack: PodStack<'a>,
}

impl<'a> Scratchpad<'a> {
    /// Create a new scratchpad over an existing pre-allocated byte slice.
    #[inline]
    pub fn new(buffer: &'a mut [u8]) -> Self {
        Self {
            stack: PodStack::new(buffer),
        }
    }

    /// Available capacity in bytes.
    #[inline]
    pub fn available_bytes(&self) -> usize {
        self.stack.len_bytes()
    }

    /// Check if the scratchpad has sufficient capacity for `count` instances of `T`.
    #[inline]
    pub fn can_hold<T>(&self, count: usize) -> bool {
        let req = StackReq::new::<T>(count);
        self.stack.can_hold(req)
    }

    /// Allocate a slice of uninitialized POD elements from the scratchpad.
    ///
    /// Returns `Err(AetherError::ScratchpadExhausted)` if available capacity is exceeded,
    /// avoiding panics in compliance with AGENTS.md.
    pub fn try_alloc_slice<T: bytemuck::Pod>(
        &mut self,
        count: usize,
    ) -> Result<&'a mut [T], AetherError> {
        let req = StackReq::new::<T>(count);
        if !self.stack.can_hold(req) {
            return Err(AetherError::ScratchpadExhausted {
                requested: req.size_bytes(),
                available: self.stack.len_bytes(),
            });
        }
        let dummy = PodStack::new(&mut []);
        let current_stack = core::mem::replace(&mut self.stack, dummy);
        let (slice, remaining) =
            current_stack.make_aligned_raw::<T>(count, core::mem::align_of::<T>());
        self.stack = remaining;
        Ok(slice)
    }
}
