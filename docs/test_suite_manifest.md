# Project Aether-Manifold: Master Pre-Publish Test Suite Manifest
## Exhaustive Verification, Fuzzing, Chaos & Hardware Gate Specification

---

## 1. Release Verification Philosophy

In accordance with [`agents.md`](./agents.md) and [`roadmap.md`](./roadmap.md), Project Aether-Manifold **cannot be published or tagged for release** until 100% of the tests in this manifest pass across all four verification stages:
1. **Stage 1: Compile-Time, Bit-Precise Formal Proofs & Concurrency Model Checking** (`static_assertions`, `kani`, `loom`, TLA+)
2. **Stage 2: Property-Based Mathematical & Geometric Fuzzing** (`proptest` at $10^8$ iterations per kernel)
3. **Stage 3: Deterministic Simulation Testing (DST) & Fault Injection Chaos** (`oorandom` seeded simulation)
4. **Stage 4: Hardware PMU Counter, Zero-Allocation & Scale Benchmarks** (`perf-event`, `HotPathAllocGuard`)

Every test listed below is mandatory. Skipping, stubbing, or weakening any tolerance bound results in immediate release rejection by **`Agent-Feynman`**.

---

## 2. Category I: Compile-Time Layout & Bit-Precise Formal Verification (`kani`)
**Jurisdiction:** `Agent-Hopper` & `Agent-Riemann` | **Crate:** `aether-core`, `aether-manifold`

These tests mathematically prove memory safety, alignment invariants, and floating-point domain bounds using SAT/SMT model checking (`cargo kani`) and compile-time assertions.

| # | Exact Test Name | Tool | Description & Verified Invariant |
| :--- | :--- | :--- | :--- |
| **1.1** | `static_assert_tensor_graph_page_4kb_alignment` | `static_assertions` | Verifies at compile time that `size_of::<TensorGraphPage>() == 4096` and `align_of::<TensorGraphPage>() == 4096` (**Rule 3.3**). |
| **1.2** | `static_assert_page_header_64b_cacheline` | `static_assertions` | Verifies that `size_of::<PageHeader>() == 64` and `align_of::<PageHeader>() == 64` so headers never straddle CPU L1 cache lines. |
| **1.3** | `static_assert_spsc_slot_cache_padded` | `static_assertions` | Verifies that cross-core SPSC ring buffer slots and atomic cursors are isolated on $64\text{-byte}$ boundaries (`align_of::<RingCursor>() >= 64`). |
| **1.4** | `static_assert_zerocopy_pod_invariants` | `zerocopy` / `bytemuck` | Proves at compile time that `TensorGraphPage`, `PageHeader`, `AtRqCodeword`, and `SheafEdgeMorphism` contain zero padding bytes and are valid for arbitrary bit patterns (`FromBytes`, `IntoBytes`, `Immutable`). |
| **1.5** | `proof_lorentz_inner_product_horizon_clamp` | `kani` | Formally proves for all symbolic IEEE-754 `f32` inputs that the Lorentz projection clamp $z = \max\left(1 + \epsilon_{\text{mach}}, \, \kappa \langle x, y \rangle_{\mathcal{L}}\right)$ never passes $z < 1.0$ into $\operatorname{arcosh}(z)$, guaranteeing zero `NaN` generation (**Rule 4.1**). |
| **1.6** | `proof_tensor_page_slice_bounds_safety` | `kani` | Formally proves that dynamic neighbor indexing `u_i` and sheaf restriction map offsets inside a 4KB `TensorGraphPage` can never read or write past byte offset $4095$ for any `neighbor_count` $\in [0, 255]$. |
| **1.7** | `proof_at_rq_codebook_lookup_inbounds` | `kani` | Formally proves that packed $4\text{-bit}$ and $8\text{-bit}$ Anisotropic Residual Quantization sub-vector indices never index out-of-bounds during AVX-512 `pshufb` / LUT reconstruction. |
| **1.8** | `proof_hlc_timestamp_pack_unpack_bijection` | `kani` | Formally proves that packing a $48\text{-bit}$ physical clock $t_{\text{physical}}$ and $16\text{-bit}$ logical counter $c_{\text{logical}}$ into a $64\text{-bit}$ `u64` HLC stamp is strictly order-preserving and bijective. |

---

## 3. Category II: Lock-Free Concurrency & Memory Model Exhaustion (`loom`)
**Jurisdiction:** `Agent-Lamport` | **Crate:** `aether-core`, `aether-consensus`

These tests exhaustively permute C11/Rust atomic memory orderings (`Acquire`, `Release`, `AcqRel`, `SeqCst`) under `cargo test --cfg loom` to prove zero data races, deadlocks, livelocks, or ABA bugs (**Rule 3.2**).

| # | Exact Test Name | Tool | Description & Verified Invariant |
| :--- | :--- | :--- | :--- |
| **2.1** | `loom_spsc_ring_buffer_acquire_release_linearizability` | `loom` | Verifies that a single producer core and single consumer core exchanging page pointers across a bounded SPSC queue never observe torn writes, out-of-order delivery, or stale ring wrap-around states. |
| **2.2** | `loom_l0_morphism_log_atomic_append_visibility` | `loom` | Verifies that concurrent readers scanning $L_0$ tangent-space deltas observe a strictly monotonic, wait-free view of appended mutations via `AtomicU128` (`CMPXCHG16B`) state pointers. |
| **2.3** | `loom_hlc_monotonic_causal_ordering` | `loom` | Proves that across all thread interleavings and simulated physical clock regressions (NTP backward jumps), `HlcClock::now()` and `HlcClock::update_with_remote()` produce strictly increasing timestamps $\tau_1 < \tau_2$. |
| **2.4** | `loom_epoch_arena_reclamation_no_use_after_reset` | `loom` | Proves that `bumpalo` query scratchpad arenas and $L_0$ epoch buffers are never reset or reclaimed while an active reader holds a pinned epoch guard. |
| **2.5** | `loom_voronoi_cell_pointer_splice_atomicity` | `loom` | Proves that swapping a compacted $L_1$ Voronoi shard pointer into the active routing directory is atomic and wait-free for concurrent hop traversals. |
| **2.6** | `loom_cooperative_scheduler_yield_fairness` | `loom` | Proves that the thread-local cooperative task queue never starves high-priority read queries when background Voronoi compaction tasks yield at the $250\text{ }\mu\text{s}$ boundary (**Rule 3.4**). |

---

## 4. Category III: Riemannian Product Manifold & Geodesic Math (`proptest`)
**Jurisdiction:** `Agent-Riemann` & `Agent-Hopper` | **Crate:** `aether-manifold`

Executed across $10^8$ randomized inputs per test (`PROPTEST_CASES=100000000`), covering boundary horizons, extreme curvatures $\kappa \in [-100.0, -10^{-6}]$, and mixed dimensions.

| # | Exact Test Name | Tool | Description & Verified Invariant |
| :--- | :--- | :--- | :--- |
| **3.1** | `prop_lorentz_manifold_surface_invariant` | `proptest` | Verifies that for any generated point $x \in \mathbb{H}_{\kappa}^{d_H}$, the Lorentzian inner product satisfies $\left\vert \langle x, x \rangle_{\mathcal{L}} - \frac{1}{\kappa} \right\vert \le 10^{-6}$. |
| **3.2** | `prop_lorentz_reprojection_drift_correction` | `proptest` | Applies $10^4$ sequential tangent-space delta accumulations in $L_0$ and verifies that the Lorentz normalization rule $x_0 = \sqrt{\frac{1}{-\kappa} + \sum_{l=1}^{d} x_l^2}$ restores the point to $\mathbb{H}_{\kappa}^{d_H}$ with $< 10^{-7}$ error (**Rule 4.1**). |
| **3.3** | `prop_product_manifold_metric_non_negativity_and_identity` | `proptest` | Verifies $d_{\mathcal{M}}^2(u, v) \ge 0$ for all $u, v \in \mathcal{M}$ and $d_{\mathcal{M}}^2(u, u) = 0$ within $\le 10^{-7}$ across $\mathbb{H}_{\kappa}^{d_H} \times \mathbb{S}_{\tau}^{d_S} \times \mathbb{R}^{d_E}$. |
| **3.4** | `prop_product_manifold_metric_symmetry` | `proptest` | Verifies exact symmetry $\left\vert d_{\mathcal{M}}(u, v) - d_{\mathcal{M}}(v, u) \right\vert \le 10^{-7}$ across all SIMD instruction sets. |
| **3.5** | `prop_product_manifold_riemannian_triangle_inequality` | `proptest` | Verifies $d_{\mathcal{M}}(u, w) \le d_{\mathcal{M}}(u, v) + d_{\mathcal{M}}(v, w) + 10^{-5}$ for $10^8$ random triplets $(u, v, w) \in \mathcal{M}^3$, including points near the hyperbolic boundary ($x_0 > 10^4$). |
| **3.6** | `prop_exp_log_map_bijection_identity` | `proptest` | Verifies that the exponential map $\exp_x: T_x\mathcal{M} \to \mathcal{M}$ and logarithmic map $\log_x: \mathcal{M} \to T_x\mathcal{M}$ are exact inverses: $\left\Vert \log_x(\exp_x(v)) - v \right\Vert_2 \le 10^{-5} \Vert v \Vert_2$ (**Stage 2 Gate**). |
| **3.7** | `prop_geodesic_distance_equals_tangent_norm` | `proptest` | Verifies that the Riemannian metric on the tangent space $T_x\mathcal{M}$ preserves geodesic length: $\left\vert d_{\mathcal{M}}(x, y) - \left\Vert \log_x(y) \right\Vert_{g_x} \right\vert \le 10^{-5}$. |
| **3.8** | `prop_parallel_transport_metric_isometry` | `proptest` | Verifies that parallel transport $P_{x \to y}: T_x\mathcal{M} \to T_y\mathcal{M}$ along the geodesic connecting $x$ and $y$ preserves inner products: $\left\vert \langle u, v \rangle_{g_x} - \langle P_{x \to y}(u), P_{x \to y}(v) \rangle_{g_y} \right\vert \le 10^{-5}$. |
| **3.9** | `prop_hypersphere_antipodal_numerical_stability` | `proptest` | Tests $\exp_x(v)$, $\log_x(y)$, and $d_{\mathbb{S}_{\tau}}(x, y)$ when $y \approx -x$ ($\theta \to \pi$) on $\mathbb{S}_{\tau}^{d_S}$ to guarantee zero division-by-zero or `NaN` singularities. |
| **3.10** | `test_simd_vs_scalar_manifold_bit_equivalence` | `pulp` / `proptest` | Compares AVX-512, AVX2, ARM NEON, and scalar `libm` implementations of $d_{\mathcal{M}}^2(u, v)$, $\exp_x(v)$, and $\log_x(y)$, asserting deviation $\le 2\text{ ULP}$. |

---

## 5. Category IV: Anisotropic Tangent-Space Residual Quantization (AT-RQ)
**Jurisdiction:** `Agent-Hopper` | **Crate:** `aether-manifold`

Tests the compression of manifold points via logarithmic projection $v = \log_{\mu_c}(x) \in T_{\mu_c}\mathcal{M}$ and anisotropic quantization ($\eta_{\parallel} \gg \eta_{\perp}$).

| # | Exact Test Name | Tool | Description & Verified Invariant |
| :--- | :--- | :--- | :--- |
| **4.1** | `prop_at_rq_parallel_error_prioritization` | `proptest` | Verifies that for any vector $v \in T_{\mu_c}\mathcal{M}$ and its quantized reconstruction $\hat{v}$, the parallel residual error satisfies $\frac{\Vert r_{\parallel} \Vert_2^2}{\Vert v_{\parallel} \Vert_2^2} < \frac{\Vert r_{\perp} \Vert_2^2}{\Vert v_{\perp} \Vert_2^2}$ when $\eta_{\parallel} = 10 \eta_{\perp}$. |
| **4.2** | `prop_at_rq_distortion_certificate_enforcement` | `proptest` | Verifies **Rule 4.3**: Every vector committed to a 4KB Tensor Page satisfies $\left\vert d_{\mathcal{M}}(x, y) - d_{\mathcal{M}}(\hat{x}, \hat{y}) \right\vert \le \delta_{\max}$. |
| **4.3** | `test_at_rq_high_curvature_fp16_spillover` | `half` / Unit | Injects synthetic extreme-curvature outliers violating $\delta_{\max}$ and verifies automatic spillover into the page's `f16`/`bf16` residual slot, restoring $\left\vert d_{\mathcal{M}}(x, y) - d_{\mathcal{M}}(\hat{x}, \hat{y}) \right\vert \le \frac{1}{10}\delta_{\max}$. |
| **4.4** | `test_at_rq_vnni_lut_asymmetric_distance_accuracy` | `pulp` | Verifies that SIMD asymmetric LUT distance computation (unquantized query $q$ against AT-RQ packed codes $\hat{x}$) matches full `f32` reconstructed distance within $10^{-4}$ relative error. |
| **4.5** | `test_riemannian_karcher_mean_centroid_convergence` | `faer-core` | Verifies that Riemannian Fréchet/Karcher mean clustering on $\mathcal{M}$ converges monotonically in $< 25$ iterations using zero-allocation `PodStack` scratchpads. |

---

## 6. Category V: Cellular Sheaf Topology, Laplacian & Optimal Transport
**Jurisdiction:** `Agent-Riemann` | **Crate:** `aether-manifold`

Tests the unification of graph topology and vector spaces via Vertex Stalks $\mathcal{F}(v)$, Edge Stalks $\mathcal{F}(e)$, Restriction Maps $\mathcal{F}_{v \trianglelefteq e}$, the Sheaf Laplacian $L_{\mathcal{F}} = \delta^T \delta$, and Wasserstein-Sinkhorn Optimal Transport.

| # | Exact Test Name | Tool | Description & Verified Invariant |
| :--- | :--- | :--- | :--- |
| **5.1** | `prop_sheaf_restriction_map_spectral_norm_bound` | `proptest` / `faer` | Verifies **Rule 4.2**: Every restriction morphism $\mathcal{F}_{v \trianglelefteq e}: \mathcal{F}(v) \to \mathcal{F}(e)$ satisfies $\left\Vert \mathcal{F}_{v \trianglelefteq e} \right\Vert_2 \le 1.0 + 10^{-6}$ via Householder QR / SVD projection. |
| **5.2** | `prop_sheaf_coboundary_operator_linearity` | `proptest` | Verifies that $(\delta(a x + b y))_e = a(\delta x)_e + b(\delta y)_e$ for all 0-cochains $x, y \in C^0(G; \mathcal{F})$ and scalars $a, b \in \mathbb{R}$. |
| **5.3** | `prop_sheaf_laplacian_symmetry_and_psd` | `proptest` | Verifies **Rule 4.2**: For any generated graph $G$ and cellular sheaf $\mathcal{F}$, $L_{\mathcal{F}}^T = L_{\mathcal{F}}$ and the quadratic form (Sheaf Dirichlet Energy) satisfies $x^T L_{\mathcal{F}} x = \Vert \delta x \Vert_2^2 \ge 0$ for all $x \in C^0(G; \mathcal{F})$. |
| **5.4** | `test_sheaf_harmonic_extension_global_section_zero_energy` | Unit | Constructs a consistent global section $x \in H^0(G; \mathcal{F})$ where $\mathcal{F}_{u \trianglelefteq e} x_u = \mathcal{F}_{v \trianglelefteq e} x_v$ across all edges and verifies $x^T L_{\mathcal{F}} x = 0$. |
| **5.5** | `test_sheaf_diffusion_convergence_stability` | Unit | Runs 100 steps of discrete sheaf diffusion $x^{(t+1)} = (I - \alpha L_{\mathcal{F}}) x^{(t)}$ and proves zero signal explosion ($\Vert x^{(t+1)} \Vert_2 \le \Vert x^{(t)} \Vert_2$). |
| **5.6** | `prop_sinkhorn_knopp_marginal_convergence` | `proptest` | Verifies that the SIMD Sinkhorn-Knopp solver for $\mathcal{W}_{\varepsilon}(A, B)$ converges to doubly-stochastic transport plan marginals $\left\Vert P \mathbf{1}_m - \mu \right\Vert_1 \le 10^{-5}$ and $\left\Vert P^T \mathbf{1}_n - \nu \right\Vert_1 \le 10^{-5}$ within $\le 50$ iterations. |
| **5.7** | `prop_wasserstein_sinkhorn_metric_properties` | `proptest` | Verifies that Debiased Sinkhorn Divergence $S_{\varepsilon}(A, B)$ satisfies non-negativity $S_{\varepsilon}(A, B) \ge 0$, identity $S_{\varepsilon}(A, A) = 0$, and symmetry $S_{\varepsilon}(A, B) = S_{\varepsilon}(B, A)$ over multi-vector bags on $\mathcal{M}$. |
| **5.8** | `test_sinkhorn_log_domain_stabilization_small_epsilon` | Unit | Tests entropic regularization down to $\varepsilon = 10^{-4}$ to verify zero SIMD floating-point underflow (`0.0`) or overflow (`Inf`) during scaling. |

---

## 7. Category VI: KLSM-Tree, 4KB Tensor Pages & Boundary-Only Voronoi Compaction
**Jurisdiction:** `Agent-Stonebraker` & `Agent-Hopper` | **Crate:** `aether-core`, `aether-storage`

Tests the physical storage layout, `io_uring` DMA pipeline, and the $O\left(\vert \partial \mathcal{V} \vert \log \vert \partial \mathcal{V} \vert\right)$ Boundary-Only Delaunay Compaction engine.

| # | Exact Test Name | Tool | Description & Verified Invariant |
| :--- | :--- | :--- | :--- |
| **6.1** | `test_tensor_page_zero_cycle_cast_and_crc32_integrity` | `zerocopy` / `crc32fast` | Constructs a 4KB `TensorGraphPage`, flushes via `O_DIRECT`, reads back into a registered fixed buffer, casts in $0$ deserialization cycles, and validates the `Header` CRC32C checksum. |
| **6.2** | `test_single_hop_zero_additional_io_invariant` | `io-uring` | Proves the Section 3.2 architectural guarantee: fetching vertex $v$'s 4KB page provides all quantized neighbor vectors $\{\hat{x}_{u_i}\}$ and sheaf maps $\{\mathcal{F}_{v \trianglelefteq e_i}\}$ required to score hop $t+1$ with **0 additional NVMe reads**. |
| **6.3** | `test_l0_tangent_delta_mvcc_time_travel_reconstruction` | Unit | Applies 50 sequential geodesic updates $x_v(\tau_{k+1}) = \exp_{x_v(\tau_k)}(\Delta v_k)$ stamped with HLC timestamps and verifies that querying at any historical $\tau_q \in [\tau_0, \tau_{50}]$ reconstructs $x_v(\tau_q)$ in $O(1)$ time with $< 10^{-5}$ error. |
| **6.4** | `prop_voronoi_boundary_certificate_pruning_exactness` | `proptest` | Proves that during $L_i \to L_{i+1}$ compaction, any interior vertex whose Voronoi hyperplane certificate does not intersect the modified run's bounding locus has **identical** true top-$k$ Delaunay neighbors before and after merge (0% false pruning). |
| **6.5** | `test_boundary_only_delaunay_compaction_complexity_scaling` | Benchmark / Unit | Merges two runs of $N = 10^6$ vertices where $5\%$ of Voronoi cells overlap and verifies that distance evaluations are strictly bounded by $\le 2 \times \vert \partial \mathcal{V}_A \cap \partial \mathcal{V}_B \vert \log(\vert \partial \mathcal{V} \vert)$, bypassing $> 90\%$ of interior nodes. |
| **6.6** | `test_delaunay_graph_connectivity_no_isolated_islands` | Unit | Performs 100 consecutive cycles of high-velocity insertions and tombstone deletions ($50\%$ churn) followed by Voronoi compaction, asserting that the resulting navigation graph remains $100\%$ strongly connected with $0$ unreachable islands. |
| **6.7** | `test_klsm_write_amplification_bound` | Integration | Streams $10^7$ vector + edge mutations through $L_0 \to L_3$ and verifies that physical NVMe bytes written divided by logical bytes ingested satisfies Write Amplification $WA < 2.5$ (**`Agent-Stonebraker` Primary Directive**). |
| **6.8** | `test_io_uring_sqpoll_registered_buffer_dma` | `io-uring` | Verifies that `opcode::ReadFixed` and `opcode::WriteFixed` execute via kernel `IORING_SETUP_SQPOLL` with zero `io_uring_enter` blocking syscalls when the SQ ring is active. |

---

## 8. Category VII: Semiring-Sheaf Query Compiler & Speculative Execution
**Jurisdiction:** `Agent-Hopper` | **Crate:** `aether-query`

Tests the compilation of hybrid vector + multi-hop graph queries into Generalized SpMV over Tropical, Diffusion, and Bottleneck semirings:

$$\psi_u^{(t+1)} = \bigoplus_{v \in \mathcal{N}(u)} \left( \psi_v^{(t)} \otimes \exp\left( -\beta \left\Vert \mathcal{F}_{u \trianglelefteq e} x_u - \mathcal{F}_{v \trianglelefteq e} x_v \right\Vert_2^2 - \gamma \, d_{\mathcal{M}}^2(u, q) \right) \right)$$

| # | Exact Test Name | Tool | Description & Verified Invariant |
| :--- | :--- | :--- | :--- |
| **7.1** | `test_tropical_semiring_geodesic_astar_optimality` | Unit / Oracle | Instantiates $(\oplus, \otimes) = (\min, +)$ and verifies that Geodesic $A^*$ finds the exact minimum-energy relational-geodesic path matching an exhaustive Bellman-Ford oracle across $10^5$ nodes. |
| **7.2** | `test_diffusion_semiring_sheaf_pagerank_convergence` | Unit | Instantiates $(\oplus, \otimes) = (+, \times)$ and verifies that Personalized Sheaf PageRank weighted by vector proximity converges to the stationary distribution $\psi^*$ within $\left\Vert \psi^{(t+1)} - \psi^{(t)} \right\Vert_1 \le 10^{-6}$. |
| **7.3** | `test_bottleneck_semiring_max_min_pathway` | Unit | Instantiates $(\oplus, \otimes) = (\max, \min)$ and verifies exact identification of the widest-capacity relational bottleneck path under a hard geodesic radius threshold $d_{\mathcal{M}}(u, q) \le r_{\max}$. |
| **7.4** | `test_hybrid_sheaf_dirichlet_query_vs_bruteforce_recall` | Integration | Evaluates 3-hop typed relational queries combined with $1536\text{-dim}$ Product Manifold similarity against an exact brute-force ground-truth solver, asserting $\text{Recall@10} \ge 0.98$. |
| **7.5** | `test_speculative_io_uring_prefetch_pipeline_overlap` | Integration | Verifies that while AVX-512 computes geodesic distances for hop $t$, `io_uring` asynchronously prefetches the top-$k$ candidate 4KB pages for hop $t+1$, achieving $\ge 85\%$ I/O-compute overlap. |
| **7.6** | `test_query_cooperative_yield_at_250us_boundary` | Integration | Injects a massive 10-hop broad-fanout query and verifies via hardware TSC (`rdtsc`) that the query execution loop yields control to the thread-local scheduler within $\le 250\text{ }\mu\text{s}$ (**Rule 3.4**). |

---

## 9. Category VIII: Deterministic Chaos & NVMe Fault Injection (`aether-verify`)
**Jurisdiction:** `Agent-Feynman` | **Crate:** `aether-verify`

Implements **Stage 3 of the Gate Protocol**. All I/O, timers, and thread scheduling are driven by a seeded deterministic PRNG (`oorandom::Rand64`) so any failure is $100\%$ reproducible from its $64\text{-bit}$ seed.

| # | Exact Test Name | Tool | Description & Verified Invariant |
| :--- | :--- | :--- | :--- |
| **8.1** | `sim_torn_4kb_tensor_page_write_recovery` | `oorandom` | Simulates a power loss after writing only $1024$ bytes of a $4096\text{-byte}$ `TensorGraphPage` during compaction. Verifies that recovery detects the CRC32C mismatch, discards the uncommitted run, and replays the $L_0$ Morphism WAL with $0$ data loss. |
| **8.2** | `sim_nvme_silent_bitflip_corruption_detection` | `oorandom` | Randomly flips $1$ to $4$ bits inside on-disk quantized vectors and sheaf maps across $10^5$ page reads. Verifies $100\%$ detection via page header checksums before corrupt data can enter SIMD registers. |
| **8.3** | `sim_power_loss_mid_voronoi_boundary_compaction` | `oorandom` | Terminates the storage engine at 1,000 randomized instruction points during $L_1 \to L_2$ Boundary-Only Delaunay Compaction. Verifies that post-crash recovery restores the exact pre-crash HLC snapshot state and graph topology. |
| **8.4** | `sim_io_uring_dropped_and_reordered_completions` | `oorandom` | Reorders NVMe CQE (Completion Queue Entry) arrivals and injects transient `EIO`/`EAGAIN` kernel errors. Verifies that the speculative traversal state machine retries cleanly without hanging or leaking fixed buffer slots. |
| **8.5** | `sim_hlc_ntp_clock_skew_and_leap_second_chaos` | `oorandom` | Injects forward/backward physical clock jumps up to $\pm 5000\text{ ms}$ across simulated cores and verifies that causal MVCC snapshot isolation never exposes future uncommitted mutations or drops committed edges. |
| **8.6** | `sim_arena_exhaustion_graceful_backpressure` | `oorandom` | Floods a single core with concurrent complex queries until the $64\text{ MB}$ `bumpalo` arena reaches $100\%$ capacity. Verifies that the engine returns a deterministic backpressure status without invoking OS heap allocation (`malloc`) or panicking. |

---

## 10. Category IX: Distributed Consensus, Partitioning & Autonomous Ricci Flow
**Jurisdiction:** `Agent-Lamport` & `Agent-Riemann` | **Crate:** `aether-consensus`, `aether-manifold`

Covers Phase 3 (Distributed Scaling) and Phase 4 (Autonomous Curvature Optimization) of [`roadmap.md`](./roadmap.md).

| # | Exact Test Name | Tool | Description & Verified Invariant |
| :--- | :--- | :--- | :--- |
| **9.1** | `test_spectral_geodesic_graph_partitioner_cut_minimization` | Integration | Partitions a $10^6\text{-node}$ sheaf graph across $64$ shards and verifies that the Spectral-Geodesic partitioner reduces cross-shard Sheaf Dirichlet cut energy by $\ge 40\%$ compared to random or hash partitioning while maintaining shard load balance within $\pm 5\%$. |
| **9.2** | `sim_distributed_leaderless_hlc_consensus_network_partition` | `aether-verify` | Simulates a symmetric and asymmetric network split (Jepsen-style nemesis) across a 7-node cluster replicating $L_0$ Morphism Logs. Proves zero split-brain commits and $< 10\text{ ms}$ failover recovery upon partition healing. |
| **9.3** | `test_rdma_cxl_zero_copy_remote_tensor_page_fetch` | Integration | Verifies that a remote hop across shard boundaries fetches the target 4KB `TensorGraphPage` directly into the requesting core's registered buffer without intermediate serialization. |
| **9.4** | `test_autonomous_ricci_flow_curvature_adaptation` | Integration | Gradually shifts an ingested dataset from a flat Euclidean distribution ($\kappa \to 0$) to a deep hierarchical tree ($\kappa = -2.5$). Verifies that the background Ricci Flow optimizer adapts manifold curvatures $\kappa_i, \tau_j$ and lowers average geodesic distortion by $\ge 30\%$. |
| **9.5** | `test_100_percent_dataset_churn_zero_recall_degradation` | Endurance | Continuously inserts and deletes $10^7$ vectors and edges until $100\%$ of the initial dataset is replaced ($100\%$ churn) without manual re-indexing. Asserts that $\text{Recall@10}$ remains $\ge 0.98$ throughout the entire transition (**Phase 4 Exit Gate**). |

---

## 11. Category X: Stage 4 Hardware PMU Counters & Zero-Allocation Gates
**Jurisdiction:** `Agent-Feynman` & `Agent-Hopper` | **Crate:** `aether-verify`, `aether-core`

These tests wrap execution in the custom `HotPathAllocGuard` global allocator and read Linux hardware performance counters (`perf-event`) to enforce the non-negotiable physical hardware gates in [`agents.md`](./agents.md) and [`roadmap.md`](./roadmap.md).

| # | Exact Test Name | Tool | Required Hardware Gate to Pass |
| :--- | :--- | :--- | :--- |
| **10.1** | `gate_zero_heap_allocations_on_hot_path` | `HotPathAllocGuard` | Executes $10^6$ hybrid queries, $10^6$ $L_0$ ingestions, and $100$ Voronoi compactions with `HOT_PATH_ACTIVE = true`. **Pass Criterion:** Exactly $0$ calls to `alloc()` or `dealloc()` (**Rule 3.1**). |
| **10.2** | `gate_perf_ipc_minimum_2_8_on_simd_traversal` | `perf-event` | Measures `Counter::Instructions` / `Counter::CpuCycles` across $10^7$ Product Manifold geodesic + sheaf evaluations. **Pass Criterion:** $\text{IPC} \ge 2.8$ (**Stage 4 Gate**). |
| **10.3** | `gate_perf_l1d_cache_miss_rate_under_3_percent` | `perf-event` | Measures `CacheId::L1D` read misses during SIMD distance scanning with `_mm_prefetch` intrinsics. **Pass Criterion:** L1D miss rate $< 3.0\%$ (**Stage 4 Gate**). |
| **10.4** | `gate_1024d_quantized_geodesic_distance_under_15ns` | Hardware TSC | Benchmarks a single core evaluating $1024\text{-dim}$ AT-RQ Product Manifold distances. **Pass Criterion:** $< 15\text{ ns}$ per evaluation (**Phase 0 Exit Gate**). |
| **10.5** | `gate_single_node_ingestion_1_5m_mutations_per_sec` | Benchmark |Sustained concurrent ingestion of vectors + sheaf edges on a 64-core NVMe node. **Pass Criterion:** $\ge 1.5 \times 10^6\text{ mutations/sec}$ while maintaining read $p_{99} < 3.0\text{ ms}$ (**Phase 1 Exit Gate**). |
| **10.6** | `gate_3hop_hybrid_query_billion_scale_under_5ms_p99` | Benchmark | Executes 3-hop relational sheaf traversal interleaved with $1536\text{-dim}$ vector filtering over $10^9$ vertices ($64\text{ GB}$ RAM, $8\text{ TB}$ NVMe). **Pass Criterion:** $p_{99} < 5.0\text{ ms}$ and $\text{Recall@10} \ge 0.98$ (**Phase 2 Exit Gate**). |

---

## 12. Master Pre-Publish Verification Execution Script

Before any release tag can be created, `Agent-Feynman` executes the following mandatory verification pipeline:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== [STAGE 1A] Compile-Time Layout & Bit-Precise Kani Proofs ==="
cargo test --workspace --lib static_assert_
cargo kani --workspace --tests

echo "=== [STAGE 1B] Exhaustive C11 Concurrency Model Checking (Loom) ==="
RUSTFLAGS="--cfg loom" cargo test --package aether-core --package aether-consensus --release loom_

echo "=== [STAGE 2] 10^8 Property-Based Riemannian & Sheaf Fuzzing ==="
PROPTEST_CASES=100000000 cargo test --package aether-manifold --package aether-storage --release prop_

echo "=== [STAGE 3] Deterministic Chaos & NVMe Fault Injection Simulation ==="
cargo test --package aether-verify --release sim_

echo "=== [STAGE 4] Hardware PMU Counter & Zero-Allocation Gates ==="
cargo test --package aether-verify --release gate_ -- --nocapture
```