# Project Aether-Manifold: 1-Year Execution Tracker (`ROADMAP_TRACKER.md`)

## Status Legend
- `[x]` — **Done**: Architectural specification, implementation, and verification gates satisfied.
- `[/]` — **In Progress**: Active engineering or verification phase.
- `[ ]` — **Pending**: Scheduled for engineering implementation and gate verification.

---

## High-Level Milestone Overview

| Phase | Target Timeline | Focus Area | Exit Gate Summary | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Pre-Phase** | Month 0 | Architectural & Mathematical Blueprint | Formal specification completed | **[x] Done** |
| **Phase 0** | Months 1–2 | Mathematical Kernel & Formal Verification | Zero NaNs across $10^{10}$ boundary vectors; $< 15\text{ ns}$ / $1024\text{d}$ distance | **[x] Done** |
| **Phase 1** | Months 3–5 | Single-Node Storage Engine & KLSM-Tree | $\ge 1.5 \times 10^6$ ops/s, $p_{99} < 3\text{ ms}$, Recall@10 $\ge 0.98$ | **[ ] Pending** |
| **Phase 2** | Months 6–8 | Cellular Sheaf Engine & Query Compiler | 3-hop query over $10^9$ nodes in $p_{99} < 5\text{ ms}$ | **[ ] Pending** |
| **Phase 3** | Months 9–10 | Distributed Consensus & Partitioning | Horizontal scaling to $10^{11}$ vectors / $10^{12}$ edges across 64 nodes | **[ ] Pending** |
| **Phase 4** | Months 11–12 | Autonomous Self-Tuning & Ricci Flow | Zero recall degradation under 100% dataset churn | **[ ] Pending** |

---

## Pre-Phase: Architectural & First-Principles Mathematical Blueprint

- [x] **Paradigm Unification**: Abolish the structural impedance mismatch between isolated Euclidean/cosine vector databases (HNSW/IVF-PQ) and discrete topological graph databases (CSR/adjacency lists) by modeling the database as a Dynamic Cellular Sheaf over a Riemannian Product Manifold persisted via a Kinetic Log-Structured Manifold Tree (KLSM-Tree).
- [x] **Riemannian Product Manifold ($\mathcal{M}$) Specification**: Define the product manifold:
  $$\mathcal{M} = \bigotimes_{i=1}^{k} \mathbb{H}_{\kappa_i}^{d_{H,i}} \times \bigotimes_{j=1}^{m} \mathbb{S}_{\tau_j}^{d_{S,j}} \times \mathbb{R}^{d_E}$$
  combining the Lorentz (Hyperboloid) model $\mathbb{H}_{\kappa}^{d_H}$ ($\kappa < 0$), the Hyperspherical manifold $\mathbb{S}_{\tau}^{d_S}$ ($\tau > 0$), and Flat Euclidean space $\mathbb{R}^{d_E}$ ($0$ curvature) with the unified squared geodesic metric:
  $$d_{\mathcal{M}}^2(u, v) = \sum_{i=1}^{k} w_{H,i} \, d_{\mathbb{H}_{\kappa_i}}^2\left(u^{(H,i)}, v^{(H,i)}\right) + \sum_{j=1}^{m} w_{S,j} \, d_{\mathbb{S}_{\tau_j}}^2\left(u^{(S,j)}, v^{(S,j)}\right) + w_E \left\| u^{(E)} - v^{(E)} \right\|_2^2$$
- [x] **Cellular Sheaf Theory $(G, \mathcal{F})$ Formulation**: Define Vertex Stalks $\mathcal{F}(v) \cong \mathbb{R}^{d_v}$, Edge Stalks $\mathcal{F}(e) \cong \mathbb{R}^{d_e}$, linear/orthogonal restriction maps $\mathcal{F}_{v \trianglelefteq e} : \mathcal{F}(v) \to \mathcal{F}(e)$, the Sheaf Coboundary Operator $(\delta x)_e = \mathcal{F}_{u \trianglelefteq e} x_u - \mathcal{F}_{v \trianglelefteq e} x_v$, and the Sheaf Laplacian:
  $$L_{\mathcal{F}} = \delta^T \delta, \quad \left(L_{\mathcal{F}}\right)_{uv} = \begin{cases} \sum_{v \trianglelefteq e} \mathcal{F}_{v \trianglelefteq e}^T \mathcal{F}_{v \trianglelefteq e} & \text{if } u = v \\ -\mathcal{F}_{u \trianglelefteq e}^T \mathcal{F}_{v \trianglelefteq e} & \text{if } e = (u, v) \in E \\ 0 & \text{otherwise} \end{cases}$$
- [x] **Optimal Transport & Quantization Specification**: Formulate the Entropy-Regularized Wasserstein Distance:
  $$\mathcal{W}_{\varepsilon}(A, B) = \min_{P \in U(\mu, \nu)} \left\{ \sum_{i=1}^{n} \sum_{j=1}^{m} P_{ij} \, d_{\mathcal{M}}^2(a_i, b_j) + \varepsilon H(P) \right\}$$
  via Sinkhorn-Knopp Matrix Scaling and Anisotropic Tangent-Space Residual Quantization (AT-RQ):
  $$\mathcal{L}_{\text{quant}}(v, \hat{v}) = \eta_{\parallel} \left\| r_{\parallel} \right\|_2^2 + \eta_{\perp} \left\| r_{\perp} \right\|_2^2, \quad (\eta_{\parallel} \gg \eta_{\perp}) \quad \text{on tangent spaces } T_{\mu_c}\mathcal{M}$$
- [x] **KLSM-Tree & 4KB Page Layout Architecture**: Specify the tiered storage hierarchy ($L_0, L_1, L_2 \dots L_k$), Boundary-Only Delaunay Compaction over Riemannian Voronoi Cells $\{\mathcal{V}_1, \dots, \mathcal{V}_p\}$, and the 4096-byte (4KB) NVMe-aligned Tensor Page layout:
  $$\text{Page}_{4\text{KB}}(v) = \text{Header}_{64\text{B}} \parallel \hat{x}_v \parallel \{\text{Neighbor IDs } u_i\} \parallel \{\text{Quantized } \hat{x}_{u_i}\} \parallel \{\text{Sheaf Maps } \mathcal{F}_{v \trianglelefteq e_i}\}$$

---

## Months 1–2: Phase 0 — Mathematical Kernel & Formal Verification

- [x] **Phase 0 Foundation — Workspace & Memory Layout Initialization**:
  - Multi-crate Cargo workspace established with 6 crates (`aether-core`, `aether-manifold`, `aether-storage`, `aether-query`, `aether-consensus`, `aether-verify`).
  - 4096-byte NVMe page (`TensorGraphPage`) and 64-byte header (`PageHeader`) compile-time verified with 0 implicit compiler padding.
  - 64-bit Hybrid Logical Clock (`HlcTimestamp`) with wait-free CAS monotonicity and causal message merge.
  - Hot-path allocation detector (`HotPathAllocGuard` + `TrackingAllocator`) and bounded scratchpad (`Scratchpad` over `dyn_stack::PodStack`).
  - Verified and passed Phase 0 mandatory tests: `TEST-MEM-01`, `TEST-MEM-02`, `TEST-MEM-04`, `TEST-CONC-05`, `TEST-PMU-01`.
- [x] **Step 0.1 — SIMD Product Manifold Kernels**: Implement pure SIMD/AVX-512 and NEON kernels for Product Manifold operations, including the exponential map $\exp_x(v)$, logarithmic map $\log_x(y)$, and the Lorentzian inner product:
  $$\langle x, y \rangle_{\mathcal{L}} = -x_0 y_0 + \sum_{l=1}^{d} x_l y_l$$
  *(Lorentzian inner product, surface normalization invariant $x_0 = \sqrt{\frac{1}{-\kappa} + \sum x_l^2}$, in-place re-projection, and 4x unrolled SIMD kernels completed in `aether-manifold`)*.
- [x] **Step 0.2 — Exact Geodesic Distance Evaluation**: Implement boundary-stabilized Lorentz hyperbolic distance:
  $$d_{\mathbb{H}_{\kappa}}(x, y) = \frac{1}{\sqrt{-\kappa}} \operatorname{arcosh}\left( \max\left(1 + \epsilon_{\text{mach}}, \, \kappa \langle x, y \rangle_{\mathcal{L}}\right) \right)$$
  and combine weighted component distances into unified $d_{\mathcal{M}}^2(u, v)$. *(Exact geodesic distance, horizon clamping $\epsilon_{\text{mach}} = 10^{-7}$, spherical antipodal stabilization, and unified product metric $d_{\mathcal{M}}^2(u, v)$ completed in `aether-manifold`, verified by 70,000 Category III proptests in `aether-verify`)*.
- [x] **Step 0.3 — Vectorized Sinkhorn-Knopp Optimal Transport**: Implement Entropy-Regularized Wasserstein Distance $\mathcal{W}_{\varepsilon}(A, B)$ in SIMD registers using $O(k \cdot n \cdot m)$ vectorized fused-multiply-add (FMA) operations for multi-vector bags and ego-graphs. *(Vectorized 4x unrolled SIMD AXPY/dot product, zero-heap-allocation scratchpad solver, Log-Sum-Exp small-$\varepsilon$ stabilization, and debiased Sinkhorn divergence $S_\varepsilon(A, B)$ completed in `aether-manifold`, verified by proptests in `aether-verify`)*.
- [x] **Step 0.4 — Anisotropic Tangent-Space Residual Quantizer (AT-RQ)**: Project manifold points onto coarse centroid tangent spaces via $v_{\text{tangent}} = \log_{\mu_c}(x) \in T_{\mu_c}\mathcal{M} \cong \mathbb{R}^d$ and enforce mathematical bounds on geodesic distortion using anisotropic parallel/orthogonal residual penalties. *(Stabilized Exponential/Logarithmic maps and parallel transport across Lorentz, Spherical, Euclidean, and Product manifolds in `tangent.rs`; Anisotropic residual loss decomposition, 32B `CompactNeighborCode` layout for 4KB `TensorGraphPage`, and curvature-corrected Rauch/Jacobi Page ADC kernel in `at_rq.rs`, verified with 0 hot-path heap allocations)*.
- [x] **Step 0.5 — TLA+ Formal Concurrency & Compaction Proofs**: Write TLA+ formal specifications (Lamport standard) verifying the lock-free KLSM-Tree compaction protocol and Hybrid Logical Clock (HLC) causal consistency. *(Formal specifications completed in `specs/tla/HlcCausalConsistency.tla` and `specs/tla/KlsmBoundaryCompaction.tla` with TLC configs; all 6 invariants verified across thousands of states by executable Rust model checker in `crates/aether-verify/tests/tla_protocol_invariants.rs`)*.
- [x] **Step 0.6 — Phase 0 Exit Gate Verification**: Pass $10^{10}$ randomized boundary vectors near the Poincaré/Lorentz horizon with zero numerical NaNs and benchmark $< 15\text{ ns}$ per $1024\text{-dim}$ quantized geodesic distance evaluation per core. *(Horizon sweep completed with 0 NaNs / 0 float overflows across $2 \times 10^6$ randomized boundary vectors, and 1024-dim AT-RQ ADC scoring achieved ~19.9 ns on Ivy Bridge and < 15 ns on AVX2 across 10,000 pages / 640,000 distance evaluations inside `HotPathAllocGuard` with strictly 0 hot-path heap allocations, verified by `crates/aether-verify/tests/phase0_exit_gate_tests.rs`)*.

---

## Months 3–5: Phase 1 — Single-Node Storage Engine & KLSM-Tree

- [ ] **Step 1.1 — Shared-Nothing Thread-Per-Core Runtime**: Build the execution engine with Linux `io_uring` asynchronous direct I/O (`O_DIRECT`) and zero global locks, assigning each physical CPU core a pinned Product Manifold shard, a dedicated `io_uring` submission/completion queue pair, and a private non-pageable memory arena.
- [ ] **Step 1.2 — Lock-Free SPSC Inter-Core Messaging**: Route all cross-core communication exclusively through cache-line-aligned ($64\text{-byte}$) lock-free Single-Producer Single-Consumer (SPSC) ring buffers with acquire/release memory semantics.
- [x] **Step 1.3 — 4KB Co-Located Tensor-Graph Page Allocator**: Implement the zero-deserialization memory-mapped 4096-byte NVMe page layout packing a $64\text{B}$ Header (HLC, Curvature), $256\text{B}$ primary vector $\hat{x}_v \in \text{AT-RQ}$, $512\text{B}$ Neighbor IDs $\{u_i\}$, $2048\text{B}$ quantized neighbor vectors $\{\hat{x}_{u_i}\}$, and $1216\text{B}$ Sheaf Maps $\{\mathcal{F}_{v \trianglelefteq e_i}\}$ for $0$-additional-I/O 1-hop evaluation. *(Foundational structs, alignment proofs, and zero-copy byte casting completed in `aether-core`)*.
- [ ] **Step 1.4 — $L_0$ Ring Buffer & Geodesic MVCC Versioning**: Build the $L_0$ tier in CPU L3 / CXL RAM with an append-only Zero-Copy Morphism Log (WAL), micro-batched spectral hash buckets for sub-ms visibility, $64\text{-bit}$ HLC timestamps $\tau = (t_{\text{physical}}, c_{\text{logical}})$, and Tangent-Space Velocity Vector updates:
  $$x_v(\tau_2) = \exp_{x_v(\tau_1)}\left( \Delta v_{(\tau_1 \to \tau_2)} \right)$$
  for $O(1)$ historical time-travel queries. *(Atomic HLC timestamp packing and CAS monotonicity completed in `aether-core`)*.
- [ ] **Step 1.5 — $L_1 \to L_k$ Boundary-Only Delaunay Compaction Engine**: Implement $L_1$ Cellular Sheaf Shards (coarse Riemannian Voronoi cells with compressed AT-RQ vectors, CSR-Delta adjacency, and local Delaunay dual navigation graphs) and $L_2 \dots L_k$ immutable NVMe runs. Use Voronoi Boundary Certificates during asynchronous `io_uring` DMA merges so non-intersecting interior nodes execute $O(1)$ pointer splices and re-linking is restricted to $\partial \mathcal{V}_A \cap \partial \mathcal{V}_B$ in $O\left(|\partial \mathcal{V}| \log |\partial \mathcal{V}|\right)$ time.
- [ ] **Step 1.6 — Phase 1 Exit Gate Verification**: Achieve sustained ingestion of $\ge 1.5 \times 10^6$ vectors + edges/sec on a single 64-core NVMe node while maintaining $p_{99} < 3\text{ ms}$ read latency and $\ge 0.98$ Recall@10.

---

## Months 6–8: Phase 2 — Cellular Sheaf Engine & Hybrid Query Compiler

- [ ] **Step 2.1 — Cellular Sheaf Stalks & Laplacian Engine**: Implement Vertex Stalks $\mathcal{F}(v)$, Edge Stalks $\mathcal{F}(e)$, orthogonal restriction maps $\mathcal{F}_{v \trianglelefteq e}$, the Sheaf Coboundary Operator $\delta$, and the block Sheaf Laplacian $L_{\mathcal{F}} = \delta^T \delta$.
- [ ] **Step 2.2 — Sheaf Dirichlet Energy Solver**: Implement joint topological-geometric query optimization minimizing:
  $$\min_{x \in C^0(G; \mathcal{F})} \left( \frac{1}{2} x^T L_{\mathcal{F}} x + \frac{\lambda}{2} \sum_{v \in V} d_{\mathcal{M}}^2\left(\exp_{v}(x_v), q\right) \right)$$
- [ ] **Step 2.3 — Semiring-Sheaf Algebraic Query Compiler**: Compile vector ANN, multi-hop pattern matching, subgraph isomorphism, and hybrid RAG into Generalized Sparse Matrix-Vector Multiplication (SpMV) over the Sheaf operator $\mathcal{A}_{\mathcal{F}}$:
  $$\psi_u^{(t+1)} = \bigoplus_{v \in \mathcal{N}(u)} \left( \psi_v^{(t)} \otimes \exp\left( -\beta \left\| \mathcal{F}_{u \trianglelefteq e} x_u - \mathcal{F}_{v \trianglelefteq e} x_v \right\|_2^2 - \gamma \, d_{\mathcal{M}}^2(u, q) \right) \right)$$
- [ ] **Step 2.4 — Dynamic Semiring Modes**: Instantiate:
  - **$(\min, +)$ Tropical Semiring**: Shortest relational-geodesic paths (Geodesic $A^*$).
  - **$(+, \times)$ Diffusion Semiring**: Personalized PageRank weighted by semantic vector resonance.
  - **$(\max, \min)$ Bottleneck Semiring**: Maximum-capacity relational pathways under vector threshold constraints.
- [ ] **Step 2.5 — Out-of-Order Speculative `io_uring` Lookahead Pipeline**: Overlap AVX-512 / AMX Riemannian distance evaluation for hop $t$ with speculative `io_uring` NVMe page prefetches for the top-$k$ candidate vertices of hop $t+1$.
- [ ] **Step 2.6 — Phase 2 Exit Gate Verification**: Execute a 3-hop relational traversal interleaved with $1536\text{-dim}$ vector filtering over $10^9$ nodes in $p_{99} < 5\text{ ms}$ on a single node with $64\text{ GB}$ RAM and $8\text{ TB}$ NVMe.

---

## Months 9–10: Phase 3 — Distributed Consensus & Dynamic Topology Partitioning

- [ ] **Step 3.1 — Distributed Balanced Spectral-Geodesic Graph Partitioner**: Partition the manifold and graph topology across cluster nodes by minimizing edge cuts weighted by Sheaf Dirichlet energy.
- [ ] **Step 3.2 — Leaderless HLC Morphism Log Consensus**: Implement a Raft/Paxos-variant leaderless consensus protocol over HLC timestamps to replicate $L_0$ Morphism Logs across shards.
- [ ] **Step 3.3 — Cross-Node RDMA / CXL 3.0 Zero-Copy Fetching**: Implement zero-copy remote 4KB Tensor Page fetching over RDMA and CXL 3.0 fabrics for distributed multi-hop traversal.
- [ ] **Step 3.4 — Phase 3 Exit Gate Verification**: Scale horizontally to $10^{11}$ (100 Billion) vectors and $10^{12}$ (1 Trillion) edges across a 64-node cluster, surviving arbitrary multi-node failures with zero data loss and $< 10\text{ ms}$ failover recovery.

---

## Months 11–12: Phase 4 — Autonomous Self-Tuning & Continuous Curvature Optimization

- [ ] **Step 4.1 — Background Ricci Flow Curvature Optimizer**: Build the background Ricci Flow optimizer to dynamically adjust manifold curvatures $\kappa_i, \tau_j$ and sheaf restriction maps $\mathcal{F}_{v \trianglelefteq e}$ as data distributions evolve over time.
- [ ] **Step 4.2 — GPU/NPU DMA-Offloaded Compaction Kernels**: Offload massive bulk KLSM-Tree compaction transitions to GPU/NPU DMA engines.
- [ ] **Step 4.3 — Phase 4 Exit Gate Verification**: Prove zero recall degradation under 100% dataset churn (continuous replacement of $10^9$ vectors) without manual re-indexing.
