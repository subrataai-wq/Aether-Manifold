# Project Aether-Manifold: Rust Crate Ecosystem & Implementation Guide
## Hardware-Sympathetic Dependency Architecture & Agent Governance

---

## 1. Dependency Philosophy: The "Zero-Bloat, Zero-Clone" Filter

To strictly enforce **`AGENTS.md`**, Project Aether-Manifold rejects standard high-level Rust ecosystem defaults that introduce hidden heap allocations, work-stealing thread migrations, virtual dispatch, or serialization overhead.

### 1.1 Strictly Banned Crates (Immediate PR Rejection by `Agent-Feynman`)
| Banned Crate / Category | Reason for Prohibition (`AGENTS.md` Violation) | Approved Aether-Manifold Alternative |
| :--- | :--- | :--- |
| **`tokio` (multi-thread)**, **`async-std`**, **`rayon`** | Work-stealing schedulers migrate tasks across CPU cores, destroying L1/L2 cache locality and violating **Rule 3.2 (Shared-Nothing Thread-Per-Core)**. | Raw `io-uring` + pinned single-threaded cooperative event loops (`core_affinity`). |
| **`rocksdb`**, **`sled`**, **`redb`**, **`lmdb`** | Violates **Rule 1 (Zero-Plagiarism)** and lacks native Riemannian Voronoi compaction or 4KB Tensor-Graph page co-location. | Custom **KLSM-Tree** engine built directly over `io-uring` `O_DIRECT`. |
| **`hnsw_rs`**, **`instant-distance`**, **`hora`**, **`petgraph`** | Violates **Rule 1 (Zero-Plagiarism)** and **Rule 3.1 (Zero Hot-Path Allocations)** due to pointer-chasing `Vec` allocations. | Custom 4KB Tensor-Graph Pages + Cellular Sheaf CSR-Delta buffers. |
| **`serde_json`**, **`bincode`**, **`prost`**, **`flatbuffers`** | Violates **Rule 3.3 (Zero-Deserialization Mandate)** by requiring CPU parsing between NVMe pages and SIMD registers. | `zerocopy` and `bytemuck` with `#[repr(C, align(4096))]` direct pointer casts. |
| **`ndarray`**, **`nalgebra` (dynamic `DMatrix`)** | Dynamic matrix operations allocate temporary buffers on the heap during linear algebra evaluation, violating **Rule 3.1**. | `faer-core` with pre-allocated `dyn-stack::PodStack` scratchpads + `pulp` SIMD. |

---

## 2. `Agent-Riemann` & `Agent-Hopper`: SIMD, Manifolds & Sheaf Linear Algebra

These crates power the **Riemannian Product Manifold** ($\mathcal{M} = \bigotimes \mathbb{H}_{\kappa_i}^{d_{H,i}} \times \bigotimes \mathbb{S}_{\tau_j}^{d_{S,j}} \times \mathbb{R}^{d_E}$), **Cellular Sheaf Laplacian** ($L_{\mathcal{F}}$), **Sinkhorn-Knopp Optimal Transport**, and **Anisotropic Tangent-Space Residual Quantization (AT-RQ)**.

### 2.1 `pulp` + `core::arch` (Runtime-Dispatched Zero-Cost SIMD)
* **Role in Architecture**: Executes vectorized fused-multiply-add (FMA) kernels for Lorentz inner products $\langle x, y \rangle_{\mathcal{L}} = -x_0 y_0 + \sum_{l=1}^{d} x_l y_l$, AT-RQ asymmetric look-up tables (LUTs), and Sinkhorn matrix scaling across AVX-512, AVX2, and ARM NEON.
* **How It Works**: `pulp` provides safe, zero-cost type-level SIMD dispatch without dynamic heap allocation or trait-object `vtable` overhead, falling back cleanly to raw `core::arch::x86_64` intrinsics (such as `_mm512_fmadd_ps` and `_mm512_dpbusd_epi32` for VNNI quantized dot products).
* **Mandatory Agent Guidance**:
  * Every SIMD loop **must** unroll 4x register accumulators to hide FMA pipeline latency ($4\text{ cycles}$ on modern x86).
  * In accordance with **Rule 4.1**, the Lorentz clamp $z = \max\left(1 + \epsilon_{\text{mach}}, \, \kappa \langle x, y \rangle_{\mathcal{L}}\right)$ must be executed inside SIMD registers using `_mm512_max_ps` before calling vectorized $\operatorname{arcosh}(z)$.

### 2.2 `faer-core` + `dyn-stack` (Zero-Allocation Matrix & Sheaf Operations)
* **Role in Architecture**: Computes orthogonal restriction maps $\mathcal{F}_{v \trianglelefteq e} \in O(d)$, block-sparse Sheaf Laplacian evaluations $\left(L_{\mathcal{F}}\right)_{uv}$, and SVD/QR decompositions required during coarse Riemannian centroid training and Ricci Flow curvature updates.
* **How It Works**: Unlike `nalgebra` or `ndarray`, `faer-core` separates matrix algorithms from memory allocation via `dyn-stack::PodStack`. Agents pass a pre-allocated slice of thread-local arena memory (`&mut [u8]`) as the scratchpad workspace.
* **Mandatory Agent Guidance**:
  * **Never** call high-level `faer` convenience functions that allocate internally. Always invoke `faer_core` routines passing a `PodStack::new(thread_local_scratchpad)` to strictly satisfy **Rule 3.1 (Zero Hot-Path Allocations)**.
  * Use `faer`'s Householder QR to enforce **Rule 4.2**: $\left\| \mathcal{F}_{v \trianglelefteq e} \right\|_2 \le 1$.

### 2.3 `half` (Hardware-Accelerated FP16 & BF16 Quantization Spillover)
* **Role in Architecture**: Implements the high-precision residual spillover slots required by **Rule 4.3 (Anisotropic Quantization Distortion Certificate)** when a high-curvature vector violates $\left| d_{\mathcal{M}}(x, y) - d_{\mathcal{M}}(\hat{x}, \hat{y}) \right| \le \delta_{\max}$.
* **How It Works**: Provides `f16` and `bf16` types with native hardware conversion intrinsics (`F16C` and `AVX512-FP16`).
* **Mandatory Agent Guidance**: Store edge restriction morphisms $\mathcal{F}_{v \trianglelefteq e}$ inside the $1216\text{-byte}$ trailer of the 4KB Tensor Page as `bf16` blocks, expanding them to `f32` directly in SIMD registers during Sheaf Coboundary evaluation $\delta x$.

### 2.4 `libm` + Custom Vectorized Polynomials (Transcendental Geodesic Math)
* **Role in Architecture**: Computes deterministic, platform-independent $\operatorname{arcosh}$, $\operatorname{arsinh}$, $\cosh$, $\sinh$, and $\arccos$ for exact exponential $\exp_x(v)$ and logarithmic $\log_x(y)$ maps across nodes.
* **Mandatory Agent Guidance**: Standard library `f32::acosh` relies on system `libc`, which can vary by $1\text{ ULP}$ across Linux distributions and break **Stage 3 Deterministic Simulation**. Use minimax Chebyshev/Padé rational approximations in SIMD for bulk geodesic distances, and `libm` for scalar boundary verification.

---

## 3. `Agent-Stonebraker`: KLSM-Tree, `io_uring` DMA & Zero-Copy Storage

These crates implement the **Kinetic Log-Structured Manifold Tree (KLSM-Tree)**, the **4KB Co-Located Tensor-Graph Pages**, and **Boundary-Only Voronoi Compaction**.

### 3.1 `io-uring` (Direct Kernel Submission/Completion Ring DMA)
* **Role in Architecture**: Drives all NVMe Gen5 reads and compaction writes across $L_1 \dots L_k$ without OS context switches or blocking syscalls (**Rule 3.2**).
* **How It Works**: Exposes the raw Linux `io_uring` Submission Queue (SQ) and Completion Queue (CQ) shared memory rings.
* **Mandatory Agent Guidance**:
  * Initialize one isolated `IoUring` instance per physical CPU core using `IORING_SETUP_SQPOLL` and `IORING_SETUP_SINGLE_ISSUER`.
  * Register the thread's pre-allocated 4KB Tensor Page arena at boot using `Submitter::register_buffers` (`IORING_REGISTER_BUFFERS`) and execute all NVMe reads via `opcode::ReadFixed` to achieve true zero-copy DMA from NVMe controller to CPU L3 cache.
  * Couple SIMD traversal at hop $t$ with speculative `opcode::ReadFixed` submissions for hop $t+1$ candidate pages.

### 3.2 `zerocopy` & `bytemuck` (Compile-Time Verified Zero-Deserialization)
* **Role in Architecture**: Enforces **Rule 3.3 (Alignment & Zero-Deserialization Mandate)** for the 4KB Tensor-Graph Page:

$$\text{Page}_{4\text{KB}}(v) = \text{Header}_{64\text{B}} \parallel \hat{x}_v \parallel \{\text{Neighbor IDs } u_i\} \parallel \{\text{Quantized } \hat{x}_{u_i}\} \parallel \{\text{Sheaf Maps } \mathcal{F}_{v \trianglelefteq e_i}\}$$

* **How It Works**: `bytemuck::{Pod, Zeroable}` and `zerocopy::{FromBytes, IntoBytes, KnownLayout, Immutable}` prove at compile time that a Rust `struct` has no padding bytes, no uninitialized memory, and valid bit patterns for any byte sequence.
* **Mandatory Agent Guidance**:
  * Every on-disk and in-memory page struct **must** derive `#[derive(FromBytes, IntoBytes, KnownLayout, Immutable)]` and be annotated with `#[repr(C, align(4096))]`.
  * Acquiring a page from an `io-uring` fixed buffer must compile down to a single pointer offset calculation (`zerocopy::Ref::from_bytes`), taking **0 CPU cycles** of deserialization.

### 3.3 `rustix` + `libc` (Low-Level NUMA, HugePages & `O_DIRECT` Control)
* **Role in Architecture**: Allocates $2\text{ MB}$ and $1\text{ GB}$ HugePages (`MAP_HUGETLB`) for the $L_0$ Morphism Log and $L_1$ Voronoi shards to eliminate TLB (Translation Lookaside Buffer) misses across billion-scale graphs.
* **How It Works**: Provides safe, zero-overhead syscalls directly to the Linux kernel without `std::fs` buffering.
* **Mandatory Agent Guidance**:
  * All NVMe Manifold Run files must be opened with `O_DIRECT | O_NOATIME | O_RDONLY` (or `O_WRONLY` for immutable run flushes) to completely bypass the Linux OS Page Cache. Aether-Manifold manages its own 4KB Tensor Page cache per core.

### 3.4 `crc32fast` (SIMD-Accelerated Page Integrity)
* **Role in Architecture**: Validates the $64\text{-byte}$ `Header` of every 4KB Tensor-Graph Page against silent NVMe bit-rot or torn writes (**Stage 3 Chaos Verification**).
* **How It Works**: Uses hardware `PCLMULQDQ` and AVX-512 carryless multiplication instructions to compute checksums at $> 30\text{ GB/s}$ per core.

---

## 4. `Agent-Lamport`: Thread-Per-Core Concurrency, Arenas & Causal MVCC

These crates enforce the **Shared-Nothing Thread-Per-Core** runtime, **Hybrid Logical Clocks (HLC)**, and **Zero-Allocation Hot Paths**.

### 4.1 `core_affinity` + `hwloc2` (1:1 Physical Core & NUMA Pinning)
* **Role in Architecture**: Enforces **Rule 3.2** by binding each worker thread to a dedicated physical CPU core and mapping its memory arena to the local NUMA node and PCIe root complex of the target NVMe drive.
* **Mandatory Agent Guidance**: Inspect CPU topology at startup to avoid pinning two heavy SIMD worker threads to sibling hyperthreads (SMT) sharing the same physical AVX-512 execution port.

### 4.2 `crossbeam-utils` & `portable-atomic` (Cache-Line Isolation & 128-bit Atomics)
* **Role in Architecture**: Powers the lock-free Single-Producer Single-Consumer (SPSC) cross-core ring buffers and the $64\text{-bit}$ / $128\text{-bit}$ atomic Hybrid Logical Clock (HLC) updates.
* **How It Works**:
  * `crossbeam_utils::CachePadded<T>` aligns head and tail atomic indices to $64\text{-byte}$ (or $128\text{-byte}$ on spatial prefetchers) boundaries, eliminating L1 cache-line ping-ponging (false sharing) between producer and consumer cores.
  * `portable_atomic::AtomicU128` leverages hardware `CMPXCHG16B` instructions for lock-free atomic updates of $(t_{\text{physical}}, c_{\text{logical}}, \text{PageOffset})$ tuples in $L_0$.
* **Mandatory Agent Guidance**: Never use `crossbeam-channel` MP-MC queues on the hot path; build strict point-to-point SPSC meshes using `Ordering::Acquire` and `Ordering::Release`.

### 4.3 `bumpalo` (Epoch-Based Pre-Allocated Query Arenas)
* **Role in Architecture**: Provides temporary scratchpad memory for the **Semiring-Sheaf Algebraic Query Compiler** (such as Geodesic $A^*$ priority queues and active vertex frontiers $\psi^{(t)}$) without ever touching the global OS heap (**Rule 3.1**).
* **How It Works**: Each core pre-allocates a fixed $64\text{ MB}$ backing buffer inside HugePages at startup and wraps it in a thread-local bump arena.
* **Mandatory Agent Guidance**:
  * Allocate all per-query frontier structures inside the core's local arena.
  * When the query completes (or yields at the $250\text{ }\mu\text{s}$ cooperative boundary per **Rule 3.4**), reset the arena pointer in $O(1)$ via `.reset()` ($1$ CPU instruction). Configure the arena to panic or return an error if capacity is exceeded—never reallocate from the OS heap.

### 4.4 `static_assertions` (Compile-Time Structural Invariants)
* **Role in Architecture**: Guarantees at compile time that cache-aligned structures and 4KB Tensor Pages never accidentally drift in size due to code changes.
* **Mandatory Agent Guidance**: Every core structure must include:

```rust
static_assertions::const_assert_eq!(core::mem::size_of::<TensorGraphPage>(), 4096);
static_assertions::const_assert_eq!(core::mem::align_of::<TensorGraphPage>(), 4096);
static_assertions::const_assert_eq!(core::mem::size_of::<PageHeader>(), 64);
```

---

## 5. `Agent-Feynman`: Formal Verification, Fuzzing & Hardware Auditing

These crates implement the mandatory **4-Stage Gate Protocol** defined in Section 5 of `AGENTS.md`.

### 5.1 Stage 1 — `loom` & `kani` (Concurrency & Bit-Precise Formal Verification)
* **`loom`**: Intercepts all `atomic` operations in the SPSC ring buffers, $L_0$ Morphism Log, and HLC state transitions, exhaustively exploring C11 memory model thread interleavings to mathematically prove the absence of data races or lost updates.
* **`kani` (Rust Model Checker)**: Uses SAT/SMT solvers to formally verify that pointer arithmetic inside 4KB Tensor Pages can never trigger out-of-bounds memory access and that the Lorentz hyperbolic clamp never passes $z < 1$ into $\operatorname{arcosh}(z)$.

### 5.2 Stage 2 — `proptest` (Property-Based Riemannian & Sheaf Fuzzing)
* **Role in Architecture**: Executes the $10^8$ randomized manifold property tests required by **Stage 2**.
* **Mandatory Agent Guidance**: Every commit to `crates/aether-manifold` must pass `proptest` suites asserting:
  1. **Riemannian Triangle Inequality**: $d_{\mathcal{M}}(u, w) \le d_{\mathcal{M}}(u, v) + d_{\mathcal{M}}(v, w) + \epsilon$ across extreme negative curvatures $\kappa \in [-100.0, -10^{-5}]$.
  2. **Geodesic Bijection Identity**: $\left\| \log_x(\exp_x(v)) - v \right\|_2 \le 10^{-5} \|v\|_2$.
  3. **Sheaf Positive Semi-Definiteness**: $x^T L_{\mathcal{F}} x \ge 0$ for arbitrary randomized vertex stalks $x \in C^0(G; \mathcal{F})$.

### 5.3 Stage 3 — Custom Deterministic Simulator + `oorandom` (Seeded I/O Chaos)
* **Role in Architecture**: Replaces `io-uring` and the physical clock via a zero-cost generic trait (`StorageBackend` / `ClockSource`) during test builds, driving the entire KLSM-Tree with a deterministic PRNG (`oorandom::Rand64`).
* **Mandatory Agent Guidance**: Inject simulated torn 4KB writes, bit flips, and power loss mid-Voronoi compaction, asserting that HLC recovery reconstructs the exact pre-crash Sheaf topology.

### 5.4 Stage 4 — `perf-event` + Custom `ZeroAllocGuard` (Hardware Counter Gate)
* **`perf-event`**: Reads Linux hardware PMU (Performance Monitoring Unit) counters directly inside Rust integration tests to enforce **Stage 4** gates:
  * `Counter::Instructions` / `Counter::CpuCycles` $\ge 2.8\text{ IPC}$.
  * `CacheId::L1D` miss rate $< 3\%$.
* **Custom `#[global_allocator]` Gate**: Wrap `std::alloc::System` in a custom allocator that checks a thread-local `HOT_PATH_ACTIVE: Cell<bool>` flag (`HotPathAllocGuard`). If any code between query start and query completion triggers `alloc()`, the test immediately aborts with a **Rule 3.1 Violation**.

---

## 6. Reference Workspace Layout & Compiler Configuration

To enforce jurisdictional boundaries between agents, the repository is structured as a strict Cargo Workspace with `#[deny(unsafe_op_in_unsafe_fn)]` and `#![no_std]` (with `alloc` permitted only in initialization and harness crates):

```toml
# Cargo.toml (Workspace Root)
[workspace]
resolver = "2"
members = [
    "crates/aether-core",       # Agent-Hopper & Agent-Lamport (4KB Page layout, HLC Monotonic Clock, Alloc Guard)
    "crates/aether-manifold",   # Agent-Riemann & Agent-Hopper (#![no_std], SIMD, Manifolds, Cellular Sheaves, AT-RQ)
    "crates/aether-storage",    # Agent-Stonebraker (KLSM-Tree, Voronoi Compaction, io_uring WAL)
    "crates/aether-query",      # Agent-Hopper (Semiring-Sheaf Algebraic Query Compiler, Sinkhorn OT)
    "crates/aether-consensus",  # Agent-Lamport (Shared-Nothing Raft, SPSC Rings, Network Simulation)
    "crates/aether-verify",     # Agent-Feynman (Deterministic Chaos, Proptest, PMU Perf Gates)
]

[workspace.package]
version = "0.1.0"
edition = "2021"
authors = ["Aether-Manifold Engineering Team"]
license = "Apache-2.0"

[workspace.dependencies]
# Zero-Copy & Alignment (Rule 3.3)
zerocopy = { version = "0.8", features = ["derive", "simd"] }
bytemuck = { version = "1.21", features = ["derive", "min_const_generics"] }
static_assertions = "1.1"

# SIMD, Math & Quantization (Rule 2.1, 2.4, 4.1-4.3)
pulp = { version = "0.21", default-features = false }
faer-core = { package = "faer", version = "0.24", default-features = false }
dyn-stack = { version = "0.10", default-features = false }
half = { version = "2.4", default-features = false, features = ["bytemuck", "zerocopy"] }
libm = "0.2"

# Storage, Checksums & Hashes (Rule 3.2, 3.3)
io-uring = "0.7"
crc32fast = { version = "1.4", default-features = false }
xxhash-rust = { version = "0.8", default-features = false, features = ["xxh3"] }

# Concurrency, Pinning & Pre-Allocated Arenas (Rule 3.1, 3.2)
crossbeam-utils = { version = "0.8", default-features = false }
portable-atomic = { version = "1.10", default-features = false }
bumpalo = { version = "3.16", features = ["collections"] }

# Error Primitives
thiserror = { version = "2.0", default-features = false }

# Verification, Fuzzing & Hardware Counters (Rule 5)
proptest = { version = "1.6", default-features = false, features = ["std"] }
loom = "0.7"
oorandom = "11.1"

# Internal Workspace Crates
aether-core = { path = "crates/aether-core" }
aether-manifold = { path = "crates/aether-manifold" }
aether-storage = { path = "crates/aether-storage" }
aether-query = { path = "crates/aether-query" }
aether-consensus = { path = "crates/aether-consensus" }
aether-verify = { path = "crates/aether-verify" }

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = false # Retain symbols for Agent-Feynman `perf stat` and flamegraph auditing
```