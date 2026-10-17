# AGENTS.md — Operational Rules, Mathematical Invariants & Rust Code Quality Guidelines

## Project Aether-Manifold Engineering Constitution

This document governs all autonomous AI coding agents, mathematical synthesis agents, and human systems architects contributing to **Project Aether-Manifold**. Every line of code, mathematical derivation, and architectural commit MUST strictly comply with these rules.

---

## 1. Core Principles & First-Principles Mandate

All code you write MUST be fully optimized for hardware sympathy, numerical stability, and zero-allocation hot-path execution:

1. **Strict Zero-Plagiarism & Anti-Clone Policy**:
   - Agents are **STRICTLY FORBIDDEN** from copying, adapting, or translating source code, data structures, or heuristics from existing databases or libraries (including FAISS, Milvus, Qdrant, Weaviate, Neo4j, RocksDB, LevelDB, DiskANN, or Hnswlib).
   - Do not implement standard HNSW, standard IVF-PQ, or naive CSR adjacency lists. Every algorithm must be derived directly from the first-principles mathematical equations in `roadmap.md`: Riemannian Product Manifolds ($\mathcal{M}$), Cellular Sheaves ($(G, \mathcal{F})$), Anisotropic Tangent-Space Residual Quantization (AT-RQ), and Kinetic Log-Structured Manifold Trees (KLSM-Tree).

2. **First-Principles Mathematical Rigor**:
   - No heuristic magic numbers. Every threshold, buffer size, curvature parameter ($\kappa, \tau$), and quantization penalty ($\eta_{\parallel}, \eta_{\perp}$) must have an explicit mathematical derivation documented in docstrings using LaTeX notation.

3. **Hardware Sympathy & Algorithmic Efficiency**:
   - Maximize algorithmic big-O efficiency (e.g., $O(\vert{}\partial \mathcal{V}\vert{} \log \vert{}\partial \mathcal{V}\vert{})$ Boundary-Only Voronoi/Delaunay compaction rather than $O(N \log N)$ full graph rebuilds).
   - Do not hide memory layouts behind object-oriented abstractions, virtual method tables (`dyn Trait` vtables on hot paths), or garbage-collected/reference-counted pointers.
   - Design every data structure around CPU cache lines ($64\text{ bytes}$), SIMD register widths ($256\text{-bit}$ AVX2 / $512\text{-bit}$ AVX-512 / ARM NEON), and NVMe physical page boundaries ($4096\text{ bytes}$).
   - Write no extra code beyond what is necessary to solve the problem (zero technical debt).

If the code is not fully optimized or violates any hot-path memory invariant before handoff, it will be rejected by `Agent-Feynman`. You have permission to perform an additional optimization pass before submitting.

---

## 2. Specialized Agent Personas & Workspace Crate Jurisdictions

Development is partitioned across five specialized agent roles and eight workspace crates. An agent may only modify components within its jurisdiction and must pass cross-verification by `Agent-Feynman`:

### 2.1 Agent Personas

- **`Agent-Riemann` (Differential Geometry & Sheaf Kernel Architect)**: Owns Product Manifold metrics ($\mathbb{H}_{\kappa}^{d_H} \times \mathbb{S}_{\tau}^{d_S} \times \mathbb{R}^{d_E}$), exponential/logarithmic maps ($\exp_x$, $\log_x$), parallel transport, Cellular Sheaf restriction morphisms $\mathcal{F}_{v \trianglelefteq e}$, Sheaf Laplacian operators $L_{\mathcal{F}}$, Ricci Flow curvature optimizers, and Sinkhorn-Knopp Optimal Transport solvers.
- **`Agent-Stonebraker` (Storage Engine, KLSM-Tree & I/O Architect)**: Owns the 4KB `TensorGraphPage` layout, `io-uring` asynchronous `O_DIRECT` DMA pipelines, `AlignedPageArena`, $L_0$ lock-free Morphism Logs (`L0MorphismLog`), `KlsmTierCoordinator`, and Boundary-Only Voronoi/Delaunay compaction (`voronoi_compaction`). Targets $\ge 90\%$ physical NVMe bandwidth utilization and Write Amplification $WA < 2.5$.
- **`Agent-Lamport` (Concurrency, Distributed State & Formal Verifier)**: Owns thread-per-core `SpscRingBuffer` queues, Hybrid Logical Clocks (`HlcTimestamp`, `HlcClock`), lock-free MVCC via tangent-space velocity deltas, Spectral-Geodesic graph partitioning, leaderless HLC quorum consensus, `MorphismReplicator`, and lock-free `SeqLock` RDMA memory windows (`LockFreeRdmaSlot`).
- **`Agent-Hopper` (SIMD Intrinsics, Quantization & Query Compiler)**: Owns `pulp` / AVX-512 / AVX2 / NEON SIMD intrinsics, Anisotropic Tangent-Space Residual Quantization (`at_rq`), `SemiringSpMvEngine`, `SpeculativeLookaheadPipeline`, and the declarative AQL parser (`aql`). Targets zero-allocation hot paths and $\ge 80\%$ of peak FLOP/s.
- **`Agent-Feynman` (Red-Team Auditor, Chaos Engineer & Benchmark Gatekeeper)**: Owns `aether-verify`, Kani formal proofs, Loom concurrency checks, $10^8$-case `proptest` suites, `oorandom` deterministic fault-injection chaos simulations, and `perf-event` / `HotPathAllocGuard` hardware gate enforcement.

### 2.2 Workspace Crate Map

- `crates/aether-core`: 4KB `TensorGraphPage`, 64B `PageHeader`, `HlcTimestamp`, `Scratchpad`, `HotPathAllocGuard`, `AetherError`.
- `crates/aether-manifold`: `#![no_std]` SIMD Riemannian Product Manifolds (`hyperbolic`, `spherical`, `product`, `tangent`), `optimal_transport`, and `at_rq`.
- `crates/aether-storage`: `CoreShardRuntime`, `AlignedPageArena`, `L0MorphismLog`, `SpscRingBuffer`, `voronoi_compaction`, `klsm_coordinator`, `io_driver`.
- `crates/aether-query`: Cellular `sheaf`, `laplacian`, `dirichlet`, `semiring` (`Tropical`, `Diffusion`, `Bottleneck`), `compiler` (`SemiringSpMvEngine`), `speculative_prefetch`, `aql`.
- `crates/aether-consensus`: `partitioner`, leaderless HLC `consensus`, `morphism_replicator`, `rdma_fabric`.
- `crates/aether-autotune`: `ricci_flow`, `sheaf_opt`, `dma_compaction`, `churn_controller`.
- `crates/aether-verify`: Property tests, TLA+ protocol invariant checkers, chaos simulators, and Phase 0–4 exit gate suites.
- `crates/aether-daemon` (or `bin/aetherd`): Standalone CLI & daemon process (`aetherd`).

---

## 3. Approved vs. Strictly Banned Crates

### 3.1 Approved Project Crates (Use These)

- **Zero-Copy & Compile-Time Layout**: `zerocopy` (`FromBytes`, `IntoBytes`, `KnownLayout`, `Immutable`), `bytemuck` (`Pod`, `Zeroable`), `static_assertions`.
- **SIMD, Linear Algebra & Geodesic Math**: `pulp` + `core::arch` (runtime-dispatched zero-cost SIMD with 4x unrolled accumulators), `faer-core` + `dyn-stack` (`PodStack` pre-allocated scratchpad linear algebra), `half` (`f16` / `bf16` for AT-RQ high-curvature spillover and sheaf edge weights), `libm` (deterministic platform-independent transcendental math).
- **Storage, Direct I/O & Checksums**: `io-uring` (`IORING_SETUP_SQPOLL`, `IORING_REGISTER_BUFFERS`, `opcode::ReadFixed` / `WriteFixed`), `rustix`, `libc` (`O_DIRECT | O_DSYNC | O_NOATIME`, `MAP_HUGETLB`), `crc32fast` (SIMD CRC32C header verification), `xxhash-rust` (`xxh3` spectral hashing).
- **Concurrency, Pinning & Arena Memory**: `core_affinity` + `hwloc2` (1:1 physical CPU core and NUMA pinning), `crossbeam-utils` (`CachePadded`), `portable-atomic` (`AtomicU128` for lock-free $L_0$ state tuples), `bumpalo` (pre-allocated epoch scratchpad arenas).
- **Diagnostics & Errors**: `thiserror` (`no_std`-compatible custom error types).
- **Verification, Fuzzing & Hardware Counters**: `proptest`, `loom`, `kani`, `oorandom` (seeded deterministic simulation), `perf-event` (Linux PMU hardware counters), `criterion` (release benchmarks).
- **Python / SDK Bindings (Cold-Path Only)**: If building Python bindings via `PyO3`/`maturin`:
  - **ALWAYS** use `uv` for Python virtual environments (`.venv`) and package management; **NEVER** use base system Python.
  - **ALWAYS** run `uv run maturin develop --release`; **NEVER** build Python bindings with `cargo build --features python`.
  - Ensure `.venv` is in `.gitignore`, use strict Python type hints (no `Any` or mutable default arguments), and validate with `mypy`.

### 3.2 Strictly Banned Crates (Immediate PR Rejection)

- **BANNED**: `tokio` (multi-threaded work-stealing), `async-std`, `rayon`. _Reason_: Work-stealing migrates tasks across cores, destroying L1/L2 cache locality and violating **Rule 4.2 (Shared-Nothing Thread-Per-Core)**.
- **BANNED**: `serde_json`, `bincode`, `prost`, `flatbuffers` on any storage or query path. _Reason_: Violates **Rule 4.3 (Zero-Deserialization Mandate)**.
- **BANNED**: `rocksdb`, `sled`, `redb`, `lmdb`, `hnsw_rs`, `instant-distance`, `hora`, `petgraph`. _Reason_: Violates **Rule 1 (Zero-Plagiarism)** and allocates on the heap.
- **BANNED**: `ndarray`, dynamic `nalgebra::DMatrix`, `polars` inside core engine crates. _Reason_: Uncontrolled heap allocations violate **Rule 4.1**.

---

## 4. Systems Engineering & Memory Invariants (NON-NEGOTIABLE)

### Rule 4.1: Zero Allocations on the Hot Path

- Any code executed during query evaluation (`SemiringSpMvEngine`), geodesic distance calculation, optimal transport (`SinkhornSolver`), sheaf energy scoring, page allocation/deallocation (`AlignedPageArena`), $L_0$ ingestion (`L0MorphismLog`), Voronoi compaction, or consensus replication is **strictly prohibited** from invoking heap allocators (`malloc`, `free`, `Box::new`, `Vec::push` with reallocation, `String`, `format!`).
- All hot-path memory must come from pre-allocated `AlignedPageArena` slabs, caller-provided stack arrays, `Scratchpad` (`dyn_stack::PodStack`), or thread-local `bumpalo` arenas reset in $O(1)$ via `.reset()`.
- Every hot-path test must run inside `HotPathAllocGuard::new()` and assert `0` allocated bytes.

### Rule 4.2: Shared-Nothing Thread-Per-Core Architecture

- Pin worker threads 1:1 to physical CPU cores (`pin_current_thread_to_core` / `pthread_setaffinity_np` / `core_affinity`), avoiding sibling SMT hyperthreads sharing the same AVX-512 port.
- **Forbidden Primitives**: `std::sync::Mutex`, `std::sync::RwLock`, `pthread_mutex_t`, `Arc<Mutex<T>>`, `mpsc`, and blocking OS I/O calls on hot paths.
- Cross-core communication must exclusively use cache-line-aligned (`#[repr(C, align(64))]`) lock-free Single-Producer Single-Consumer (`SpscRingBuffer`) queues or optimistic `SeqLock` slots (`LockFreeRdmaSlot`) with explicit `Ordering::Acquire` and `Ordering::Release` semantics.

### Rule 4.3: Alignment & Zero-Deserialization Mandate

- Every on-disk and in-memory page structure must enforce compile-time layout assertions via `static_assertions`:
  - `TensorGraphPage`: `#[repr(C, align(4096))]` — exactly `4096` bytes (`64B` `PageHeader`, `256B` primary AT-RQ code, `512B` neighbor IDs, `2048B` neighbor AT-RQ codes, `1216B` sheaf morphisms, primary vertex stalk, edge weights, and FP16 residual spillover).
  - `PageHeader`, `ReplicationMessage`, `DmaPageDescriptor`, `CachePaddedAtomicUsize`: `#[repr(C, align(64))]` — `64` bytes.

### Rule 4.4: Bounded Tail Latency & Cooperative Scheduling

- No single computational step (including Voronoi boundary compaction or multi-hop traversal) may hold a CPU core for more than $250\text{ }\mu\text{s}$ without checking for cooperative yield boundaries.

---

## 5. Mathematical & Numerical Stability Guardrails

### Rule 5.1: Hyperbolic Manifold Regularization

- Every Lorentz inner product $\langle x, y \rangle_{\mathcal{L}} = -x_0 y_0 + \sum_{l=1}^{d} x_l y_l$ in $\mathbb{H}_{\kappa}^{d}$ ($\kappa < 0$) must apply the horizon projection clamp before evaluating $\operatorname{arcosh}(z)$:
  $$z = \max\left(1 + \epsilon_{\text{mach}}, \, \kappa \langle x, y \rangle_{\mathcal{L}}\right), \quad \epsilon_{\text{mach}} = 10^{-7}$$
- After tangent-space velocity delta accumulations in `L0MorphismLog`, re-project vertices onto the hyperboloid sheet via:
  $$x_0 = \sqrt{\frac{1}{-\kappa} + \sum_{l=1}^{d} x_l^2}$$
- On the hypersphere $\mathbb{S}_{\tau}^{d_S}$, apply Taylor series stabilization near antipodal points ($\theta \to \pi$) to prevent division-by-zero or `NaN` generation.

### Rule 5.2: Sheaf Restriction Orthogonality & Spectral Bounds

- All cellular sheaf restriction maps $\mathcal{F}_{v \trianglelefteq e}: \mathcal{F}(v) \to \mathcal{F}(e)$ parameterized as Householder reflectors $H = I - 2 w w^T$ must satisfy:
  $$\left\Vert{} \mathcal{F}_{v \trianglelefteq e} \right\Vert{}_2 \le 1.0 + 10^{-6}$$
- The Sheaf Laplacian $L_{\mathcal{F}} = \delta^T \delta$ must remain symmetric positive semi-definite ($x^T L_{\mathcal{F}} x \ge 0 \quad \forall x$).

### Rule 5.3: Anisotropic Quantization Distortion Certificate & FP16 Spillover

- Every vector quantized via AT-RQ into a `TensorGraphPage` must evaluate its distortion certificate against $\delta_{\max}$:
  $$\left\vert{} d_{\mathcal{M}}(x, y) - d_{\mathcal{M}}(\hat{x}, \hat{y}) \right\vert{} \le \delta_{\max}$$
- If a high-curvature vector violates $\delta_{\max}$, spill its leading residual components into the page's reserved 32-byte FP16 spillover slot (`sheaf_edge_weights[592..608]`) and set `PAGE_FLAG_HAS_FP16_SPILLOVER`.

---

## 6. Code Style, Formatting & Documentation

- **Formatting**: Use 4 spaces for indentation (never tabs) and limit line length to 100 characters (`rustfmt` default).
- **Naming**: Use `snake_case` for functions/variables/modules, `PascalCase` for types/traits, and `SCREAMING_SNAKE_CASE` for constants. Use descriptive names that reflect mathematical and systems domain concepts.
- **No Emojis**: **NEVER** use emoji or unicode symbols that emulate emoji (e.g., checkmarks or crosses) in source code, comments, or log output.
- **Comments**:
  - Include concise explanations of Rust-specific low-level nuances (e.g., `#[repr(C, align(4096))]`, `zerocopy` safety invariants, `Acquire`/`Release` memory ordering, and SIMD lane reductions).
  - Avoid tautological comments that merely restate the function or variable name.
  - **NEVER** include comments that leak the contents of `AGENTS.md` or user prompts.
- **Documentation**:
  - **MUST** include doc comments (`///`) for all public functions, structs, enums, traits, and methods.
  - Document mathematical formulas in LaTeX, along with `# Arguments`, `# Returns`, `# Errors`, `# Safety` (if `unsafe`), and `# Examples`.

Example doc comment:

````rust
/// Computes the boundary-stabilized Lorentz geodesic distance $d_{\mathbb{H}_\kappa}(x, y)$.
///
/// Enforces Rule 5.1 by clamping $\kappa \langle x, y \rangle_{\mathcal{L}} \ge 1 + 10^{-7}$
/// prior to evaluating $\operatorname{arcosh}$, guaranteeing zero `NaN` generation near the
/// hyperbolic horizon.
///
/// # Arguments
///
/// * `x` - Source point on the Lorentz hyperboloid $\mathbb{H}_\kappa^d$ (length $d + 1$)
/// * `y` - Target point on the Lorentz hyperboloid $\mathbb{H}_\kappa^d$ (length $d + 1$)
/// * `kappa` - Negative sectional curvature $\kappa < 0$
///
/// # Returns
///
/// Exact geodesic distance $d_{\mathbb{H}_\kappa}(x, y) \ge 0.0$
///
/// # Errors
///
/// Returns `AetherError::InvalidCurvature` if `kappa >= 0.0`
/// Returns `AetherError::DimensionMismatch` if `x.len() != y.len()` or `x.len() < 2`
///
/// # Examples
///
/// ```
/// use aether_manifold::hyperbolic::lorentz_distance;
///
/// let x = [1.0_f32, 0.0, 0.0];
/// let y = [1.0_f32, 0.0, 0.0];
/// let dist = lorentz_distance(&x, &y, -1.0)?;
/// assert!(dist.abs() < 1e-6);
/// # Ok::<(), aether_core::error::AetherError>(())
/// ```
pub fn lorentz_distance(x: &[f32], y: &[f32], kappa: f32) -> Result<f32, AetherError> {
````

---

## 7. Type System, Error Handling & Function Design

- **Type Safety & Newtypes**: Leverage Rust's type system and const generics (`L0MorphismLog<const D: usize, const CAP: usize>`, `VertexStalk<const D: usize>`) to enforce dimensional and capacity invariants at compile time. Prefer `Option<T>` or explicit validity bitmaps over unchecked sentinel values.
- **Error Handling**:
  - **NEVER** use `.unwrap()` in library or daemon code; use `.expect("descriptive invariant explanation")` only when an invariant is mathematically guaranteed.
  - **MUST** return `Result<T, AetherError>` for fallible operations and propagate errors with `?`.
  - Keep `AetherError` in `aether-core` `Copy`, `Clone`, `Debug`, `PartialEq`, and `Eq` so error propagation never allocates on the heap.
- **Function & Struct Design**:
  - Keep functions focused on a single responsibility, return early to reduce nesting, and limit parameters to 5 or fewer (group related parameters into `#[repr(C)]` config or descriptor structs).
  - Prefer borrowing slices (`&[f32]`, `&mut [f32]`, `&TensorGraphPage`) over passing owned structures.
  - Call `.clone()` explicitly on non-`Copy` types; never hide allocations inside closures or iterator chains.
  - Use exhaustive pattern matching; avoid catch-all `_` patterns on enums so adding new variants triggers compile-time review.
- **`unsafe` Code Policy**:
  - Enforce `#[deny(unsafe_op_in_unsafe_fn)]` across all workspace crates.
  - Use `unsafe` only when required for SIMD intrinsics, `io-uring` registered buffer submission, `SeqLock` non-overlapping page copies, or FFI thread affinity pinning. Every `unsafe` block **MUST** be preceded by a `// SAFETY:` comment proving why pointer alignment, bounds, and aliasing invariants hold.

---

## 8. Testing, Benchmarking & The 4-Stage Gate Protocol

No code or architectural change may be merged without passing the **4-Stage Gate Protocol** and benchmarking rules:

```text
+----------------------+    +----------------------+    +----------------------+    +----------------------+
| Stage 1: Formal Spec |--->| Stage 2: Math Fuzzing|--->| Stage 3: Det-Sim Test|--->| Stage 4: HW Counters |
| (Kani / Loom / TLA+) |    | (Proptest Manifolds) |    | (oorandom I/O Chaos) |    | (PMU & 0-Heap Alloc) |
+----------------------+    +----------------------+    +----------------------+    +----------------------+
```

1. **Stage 1 — Compile-Time, Formal & Concurrency Verification (`static_assertions`, `kani`, `loom`, TLA+)**:
   - Verify 4096B page and 64B header layouts at compile time.
   - Verify lock-free `SpscRingBuffer`, `HlcClock`, `L0MorphismLog`, and `KlsmBoundaryCompaction` invariants.
2. **Stage 2 — Property-Based Mathematical Fuzzing (`proptest`)**:
   - Verify Riemannian triangle inequality $d_{\mathcal{M}}(u, w) \le d_{\mathcal{M}}(u, v) + d_{\mathcal{M}}(v, w) + \epsilon$, geodesic bijection $\log_x(\exp_x(v)) \approx v$, Sheaf Laplacian positive semi-definiteness $x^T L_{\mathcal{F}} x \ge 0$, and Sinkhorn doubly-stochastic convergence.
3. **Stage 3 — Deterministic Simulation Testing (`oorandom`)**:
   - Test torn 4KB writes, CRC32C bit-flip detection, L0 ring wrap-around eviction, and double-free rejection using seeded deterministic PRNGs.
4. **Stage 4 — Zero-Allocation & Hardware Performance Audit (`HotPathAllocGuard`, `perf-event`)**:
   - Wrap all hot-path tests in `HotPathAllocGuard` and assert **0 bytes** of heap allocation.
   - Target $\text{IPC} \ge 2.8$, L1D cache miss rate $< 3\%$, and $< 15\text{ ns}$ per $1024\text{-dim}$ quantized geodesic distance evaluation.

### Benchmarking Rules

- **NEVER** run benchmarks in parallel, as resource contention invalidates measurements.
- **NEVER** game or manipulate benchmarks (such as comparing ground-truth arrays against themselves instead of executing real indexed `SemiringSpMvEngine` traversals).
- **ALWAYS** run benchmarks in `--release` mode (`opt-level = 3`, `lto = "fat"`, `codegen-units = 1`); **NEVER** benchmark in `debug` mode.
- Ensure benchmark runs are independent and report results in a clear Markdown table in the console (never write separate benchmark report files unless explicitly requested).

---

## 9. Security, Version Control & Agent-to-User Behavior

- **Security**: Never hardcode secrets, credentials, or tokens in source code; use `.env` (declared in `.gitignore`) or environment variables on cold startup paths.
- **Version Control**:
  - Write clear, descriptive commit messages.
  - **NEVER** commit commented-out code, commented-out tests, `dbg!` macros, or leftover temporary script artifacts.
- **Tooling & Exploration**:
  - Use `cargo`, `rustfmt`, and `clippy` (`cargo clippy --workspace --all-targets -- -D warnings`). Do not put `#![deny(warnings)]` inside source files; enforce `-D warnings` via CLI/CI flags.
  - **NEVER** read or explore `Cargo.lock` unless debugging a specific dependency resolution conflict.
  - When spawning parallel subagents, launch each in a separate tool call rather than batching via Python subprocesses.

---

## 10. Pre-Commit Verification Checklist

- [ ] All workspace unit, property, and exit-gate tests pass in release mode (`cargo test --workspace --release`)
- [ ] Zero heap allocations occur on any hot path inside `HotPathAllocGuard`
- [ ] Zero compiler or Clippy warnings across all targets (`cargo clippy --workspace --all-targets -- -D warnings`)
- [ ] Code formatting passes cleanly (`cargo fmt --all -- --check`)
- [ ] Compile-time size and alignment assertions hold (`TensorGraphPage == 4096B`, `PageHeader == 64B`)
- [ ] Standalone daemon CLI builds and executes cleanly (`cargo run --release --bin aetherd -- status`)
- [ ] All public items have complete doc comments with LaTeX math invariants and error specifications
- [ ] No commented-out code, debug macros, emojis, or banned crates (`tokio`, `rayon`, `serde_json`, `Mutex`, `RwLock`) exist in the diff

---

**Remember:** Prioritize mathematical exactness, hardware sympathy, zero-allocation hot paths, and structural clarity over abstraction.
