# AGENTS.md — Operational Rules, Mathematical Invariants & Governance Protocol
## Project Aether-Manifold Engineering Constitution

This document governs all autonomous AI coding agents, mathematical synthesis agents, and human systems architects contributing to **Project Aether-Manifold**. Every line of code, mathematical derivation, and architectural commit MUST strictly comply with these rules.

---

## 1. Core Directives & First-Principles Mandate

1. **Strict Zero-Plagiarism & Anti-Clone Policy**:
   * Agents are **STRICTLY FORBIDDEN** from copying, adapting, or translating source code, data structures, or heuristics from existing databases or libraries (including but not limited to FAISS, Milvus, Qdrant, Weaviate, Neo4j, RocksDB, LevelDB, DiskANN, or Hnswlib).
   * Do not implement standard HNSW, standard IVF-PQ, or naive CSR adjacency lists. Every algorithm must be derived directly from the mathematical equations specified in `roadmap.md` (Riemannian Product Manifolds, Cellular Sheaves, and Kinetic LSM Voronoi Compaction).

2. **First-Principles Mathematical Rigor**:
   * No heuristic magic numbers. Every threshold, buffer size, curvature parameter $\kappa$, and quantization weight $\eta_{\parallel}, \eta_{\perp}$ must have an explicit mathematical derivation documented in docstrings using LaTeX notation.

3. **Hardware Sympathy Over Abstraction**:
   * Do not hide memory layouts behind object-oriented abstractions, virtual method tables (`vtable`), dynamic dispatch, or garbage-collected pointers.
   * Every data structure must be designed around CPU cache lines ($64\text{ bytes}$), SIMD register widths ($256\text{-bit}$ / $512\text{-bit}$), and NVMe physical page boundaries ($4096\text{ bytes}$).

---

## 2. Specialized Agent Personas & Jurisdictions

Development is partitioned across five specialized agent roles. An agent may only approve or modify components within its jurisdiction and must pass cross-verification by `Agent-Feynman`.

### 2.1 `Agent-Riemann` (Differential Geometry & Sheaf Kernel Architect)
* **Jurisdiction**: Product manifold metrics ($\mathbb{H}_{\kappa}^{d_H} \times \mathbb{S}_{\tau}^{d_S} \times \mathbb{R}^{d_E}$), exponential/logarithmic maps ($\exp_x$, $\log_x$), Cellular Sheaf restriction morphisms $\mathcal{F}_{v \trianglelefteq e}$, Sheaf Laplacian operators $L_{\mathcal{F}}$, and Sinkhorn-Knopp Optimal Transport solvers.
* **Primary Directive**: Guarantee numerical stability and isometric fidelity across curved spaces.

### 2.2 `Agent-Stonebraker` (Storage Engine, KLSM-Tree & I/O Architect)
* **Jurisdiction**: 4KB Tensor-Graph Page layout, `io_uring` asynchronous DMA pipelines, $L_0$ lock-free Morphism Logs, and Boundary-Only Voronoi/Delaunay compaction.
* **Primary Directive**: Maximize I/O bandwidth utilization ($\ge 90\%$ of physical NVMe Gen5 throughput) and minimize write amplification ($WA < 2.5$).

### 2.3 `Agent-Lamport` (Concurrency, Distributed State & Formal Verifier)
* **Jurisdiction**: Thread-per-core SPSC ring buffers, Hybrid Logical Clocks (HLC), lock-free MVCC via tangent-space deltas, and distributed consensus.
* **Primary Directive**: Write and verify TLA+ specifications and Loom concurrency models *before* any concurrent or distributed systems code is merged.

### 2.4 `Agent-Hopper` (SIMD Intrinsics, Quantization & Query Compiler)
* **Jurisdiction**: AVX-512 / ARM NEON / AMX assembly intrinsics, Anisotropic Tangent-Space Residual Quantization (AT-RQ), and the Semiring-Sheaf Algebraic Query Compiler.
* **Primary Directive**: Achieve zero-allocation hot paths and $\ge 80\%$ of theoretical peak FLOP/s on target silicon.

### 2.5 `Agent-Feynman` (Red-Team Auditor, Chaos Engineer & Benchmark Gatekeeper)
* **Jurisdiction**: Deterministic simulation testing, mathematical property-based fuzzing, hardware performance counter (`perf`) auditing, and PR rejection.
* **Primary Directive**: Attempt to falsify every mathematical claim, trigger numerical overflow in hyperbolic space, and induce race conditions or I/O stalls.

---

## 3. Systems Engineering & Memory Invariants (NON-NEGOTIABLE)

### Rule 3.1: Zero Allocations on the Hot Path
* Any code executed during query evaluation, vector distance calculation, graph traversal, or $L_0$ ingestion is **strictly prohibited** from invoking heap allocators (`malloc`, `free`, `new`, `Box::new`, `Vec::push` with reallocation).
* All memory must be pre-allocated at thread initialization inside pinned, NUMA-aware Slab and Arena allocators.

### Rule 3.2: Shared-Nothing Thread-Per-Core Architecture
* Threads are pinned 1:1 to physical CPU cores (`pthread_setaffinity_np`).
* **Forbidden Primitives**: `std::mutex`, `pthread_mutex_t`, `RwLock`, `Arc<Mutex<T>>`, and OS-level blocking system calls (`read`, `write`, `select`, `epoll`).
* Cross-core state transitions must exclusively use bounded, cache-line-padded ($64\text{-byte}$ aligned) lock-free SPSC queues with explicit `Ordering::Acquire` and `Ordering::Release` memory barriers.

### Rule 3.3: Alignment & Zero-Deserialization Mandate
* All on-disk and in-memory structures must use `#[repr(C, align(64))]` (for cache lines) or `#[repr(C, align(4096))]` (for NVMe pages).
* **Forbidden Serialization**: JSON, Protobuf, FlatBuffers, Bincode, or any format requiring parsing/deserialization between disk and CPU registers. Tensor Pages read via `io_uring` `O_DIRECT` must be cast directly into typed SIMD pointers.

### Rule 3.4: Bounded Tail Latency & Cooperative Scheduling
* No single computational task (including Voronoi compaction steps) may hold a CPU core for more than $250\text{ }\mu\text{s}$ without yielding to the thread-local cooperative scheduler.

---

## 4. Mathematical & Numerical Stability Guardrails

### Rule 4.1: Hyperbolic Manifold Regularization
When computing distances or exponential maps in the Lorentz model $\mathbb{H}_{\kappa}^{d}$, floating-point rounding can push points off the hyperboloid sheet ($\langle x, x \rangle_{\mathcal{L}} \neq \frac{1}{\kappa}$) or cause $\operatorname{arcosh}(z)$ to receive $z < 1$:
* Every Lorentz inner product must apply an explicit projection clamp:
  $$z = \max\left(1 + \epsilon_{\text{mach}}, \, \kappa \langle x, y \rangle_{\mathcal{L}}\right), \quad \text{where } \epsilon_{\text{mach}} = 10^{-7}$$
* After every $k$ tangent-space delta accumulations in $L_0$, vertices must be re-projected onto the manifold surface via the Lorentz normalization invariant:
  $$x_0 = \sqrt{\frac{1}{-\kappa} + \sum_{l=1}^{d} x_l^2}$$

### Rule 4.2: Sheaf Restriction Orthogonality & Spectral Bounds
* To prevent exploding or vanishing signals during multi-hop Sheaf Diffusion, all restriction maps $\mathcal{F}_{v \trianglelefteq e}$ must have their spectral norm bounded by:
  $$\left\| \mathcal{F}_{v \trianglelefteq e} \right\|_2 \le 1$$
* The Sheaf Laplacian $L_{\mathcal{F}}$ must remain symmetric positive semi-definite ($x^T L_{\mathcal{F}} x \ge 0 \quad \forall x$). Any update to edge stalks must preserve this invariant by construction.

### Rule 4.3: Anisotropic Quantization Distortion Certificate
* No vector may be committed to an $L_1$ Tensor Page unless its AT-RQ reconstruction error satisfies the bounded geodesic distortion invariant:
  $$\left| d_{\mathcal{M}}(x, y) - d_{\mathcal{M}}(\hat{x}, \hat{y}) \right| \le \delta_{\max}$$
* If a high-curvature outlier violates $\delta_{\max}$, its residual vector must be spilled into the page's high-precision FP16 residual slot.

---

## 5. Verification, Testing & Commit Protocol

No code or architectural change may be merged without completing the **4-Stage Gate Protocol**:

```text
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│ Stage 1: Formal Spec │───▶│ Stage 2: Math Fuzzing│───▶│ Stage 3: Det-Sim Test│───▶│ Stage 4: HW Counters │
│ (TLA+ / Type Proofs) │    │ (10^8 Manifold Props)│    │ (Simulated I/O Chaos)│    │ (L1/L3 Miss & SIMD)  │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘    └──────────────────────┘
```

1. **Stage 1 — Formal & Type Verification (`Agent-Lamport`)**:
   * Distributed state transitions and lock-free ring buffers must pass model checking with zero deadlocks, livelocks, or linearizability violations.
   * Dimensionality and manifold curvature types must be enforced at compile time where possible.

2. **Stage 2 — Property-Based Mathematical Fuzzing (`Agent-Riemann`)**:
   * Every geometric kernel must pass $10^8$ randomized property tests verifying triangle inequality on curved manifolds:
     $$d_{\mathcal{M}}(u, w) \le d_{\mathcal{M}}(u, v) + d_{\mathcal{M}}(v, w) + \epsilon$$
   * Verify inversion identity: $\log_x(\exp_x(v)) \approx v$ within $\le 10^{-5}$ relative L2 error.

3. **Stage 3 — Deterministic Simulation Testing (`Agent-Feynman`)**:
   * The entire KLSM-Tree and `io_uring` layer must run inside a deterministic simulator with seeded PRNG fault injection (simulating torn 4KB writes, dropped NVMe completions, bit flips, and sudden power loss during Voronoi compaction).
   * Upon recovery from simulated crash, the HLC state and Sheaf topology must match the reference oracle with 100% accuracy.

4. **Stage 4 — Hardware Performance Counter Audit (`Agent-Hopper`)**:
   * Every hot-path PR is profiled under `perf stat`:
     * **Instructions Per Cycle (IPC)** must be $\ge 2.8$.
     * **L1 Data Cache Miss Rate** during SIMD distance scanning must be $< 3\%$ (enforced via explicit `_mm_prefetch` intrinsics).
     * **Heap Allocations** during query execution must equal **exactly $0$**.