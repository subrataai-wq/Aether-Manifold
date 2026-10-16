//! Project Aether-Manifold: Wait-Free SPSC Ring Buffer (`spsc.rs`)
//!
//! Implements a cache-line-aligned (64-byte), wait-free Single-Producer Single-Consumer
//! ring buffer with zero global locks, zero allocations on the hot path, and Acquire/Release
//! memory ordering to prevent false sharing and eliminate pipeline stalls.

use aether_core::error::AetherError;

#[cfg(loom)]
use loom::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(loom))]
use core::sync::atomic::{AtomicUsize, Ordering};

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;

/// 64-byte cache-line aligned wrapper around an `AtomicUsize`.
///
/// Prevents false sharing between producer and consumer cores executing on
/// adjacent physical threads/cores per AGENTS.md Rule 3.3.
#[repr(C, align(64))]
pub struct CachePaddedAtomicUsize {
    value: AtomicUsize,
    _pad: [u8; 64 - core::mem::size_of::<AtomicUsize>()],
}

static_assertions::const_assert_eq!(core::mem::size_of::<CachePaddedAtomicUsize>(), 64);
static_assertions::const_assert_eq!(core::mem::align_of::<CachePaddedAtomicUsize>(), 64);

impl CachePaddedAtomicUsize {
    /// Initialize with a starting value and zeroed padding.
    #[inline]
    pub const fn new(val: usize) -> Self {
        Self {
            value: AtomicUsize::new(val),
            _pad: [0u8; 64 - core::mem::size_of::<AtomicUsize>()],
        }
    }

    /// Load the atomic value with specified memory ordering.
    #[inline]
    pub fn load(&self, order: Ordering) -> usize {
        self.value.load(order)
    }

    /// Store the atomic value with specified memory ordering.
    #[inline]
    pub fn store(&self, val: usize, order: Ordering) {
        self.value.store(val, order);
    }
}

/// Cache-line-aligned wait-free Single-Producer Single-Consumer (SPSC) ring buffer.
///
/// # Invariants
/// - Single writer calls `try_push`.
/// - Single reader calls `try_pop`.
/// - Memory ordering: `Acquire` / `Release` pairs on head and tail.
/// - Strictly 0 heap allocations on hot path (AGENTS.md Rule 3.1).
pub struct SpscRingBuffer<T: Copy, const N: usize> {
    head: CachePaddedAtomicUsize,
    tail: CachePaddedAtomicUsize,
    slots: [UnsafeCell<MaybeUninit<T>>; N],
}

unsafe impl<T: Copy + Send, const N: usize> Send for SpscRingBuffer<T, N> {}
unsafe impl<T: Copy + Send, const N: usize> Sync for SpscRingBuffer<T, N> {}

impl<T: Copy, const N: usize> SpscRingBuffer<T, N> {
    /// Create a new empty SPSC ring buffer.
    pub fn new() -> Self {
        // Construct array of uninitialized UnsafeCells
        // Safety: An uninitialized MaybeUninit is valid representation for MaybeUninit<T>
        let slots = unsafe {
            let mut arr: MaybeUninit<[UnsafeCell<MaybeUninit<T>>; N]> = MaybeUninit::uninit();
            let ptr = arr.as_mut_ptr() as *mut UnsafeCell<MaybeUninit<T>>;
            for i in 0..N {
                core::ptr::write(ptr.add(i), UnsafeCell::new(MaybeUninit::uninit()));
            }
            arr.assume_init()
        };

        Self {
            head: CachePaddedAtomicUsize::new(0),
            tail: CachePaddedAtomicUsize::new(0),
            slots,
        }
    }

    /// Capacity of the ring buffer.
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Current number of occupied slots in the ring buffer.
    #[inline]
    pub fn len(&self) -> usize {
        let h = self.head.load(Ordering::Acquire);
        let t = self.tail.load(Ordering::Acquire);
        h.wrapping_sub(t)
    }

    /// Returns `true` if the ring buffer contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if the ring buffer has reached its capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() >= N
    }

    /// Enqueue an element into the ring buffer (wait-free, producer-only).
    ///
    /// Returns `Err(AetherError::ScratchpadExhausted)` if full.
    #[inline]
    pub fn try_push(&self, value: T) -> Result<(), AetherError> {
        if N == 0 {
            return Err(AetherError::ScratchpadExhausted {
                requested: 1,
                available: 0,
            });
        }

        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        if head.wrapping_sub(tail) >= N {
            return Err(AetherError::ScratchpadExhausted {
                requested: 1,
                available: 0,
            });
        }

        let slot_idx = head % N;
        let slot_ref = match self.slots.get(slot_idx) {
            Some(s) => s,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: slot_idx,
                    limit: N,
                });
            }
        };

        unsafe {
            let slot_ptr = slot_ref.get();
            (*slot_ptr).write(value);
        }

        self.head.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Dequeue an element from the ring buffer (wait-free, consumer-only).
    ///
    /// Returns `None` if the queue is empty.
    #[inline]
    pub fn try_pop(&self) -> Option<T> {
        if N == 0 {
            return None;
        }

        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail == head {
            return None;
        }

        let slot_idx = tail % N;
        let slot_ref = self.slots.get(slot_idx)?;

        let val = unsafe {
            let slot_ptr = slot_ref.get();
            (*slot_ptr).assume_init()
        };

        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Some(val)
    }
}

impl<T: Copy, const N: usize> Default for SpscRingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

