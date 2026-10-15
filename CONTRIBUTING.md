# Contributing to Project Aether-Manifold

Welcome to **Project Aether-Manifold**. We are building a next-generation unified Vector and Graph Database from first principles—modeling data as a **Dynamic Cellular Sheaf over a Riemannian Product Manifold** ($\mathcal{M} = \bigotimes_{i=1}^{k} \mathbb{H}_{\kappa_i}^{d_{H,i}} \times \bigotimes_{j=1}^{m} \mathbb{S}_{\tau_j}^{d_{S,j}} \times \mathbb{R}^{d_E}$) persisted via a hardware-sympathetic **Kinetic Log-Structured Manifold Tree (KLSM-Tree)**.

Before opening an issue or submitting a Pull Request (PR), every contributor—human or autonomous AI agent—must read and internalize our core specification suite:

1. **[`roadmap.md`](./docs/roadmap.md)** — Mathematical & Systems Architecture Blueprint
2. **[`agents.md`](./docs/agents.md)** — Operational Rules, Mathematical Invariants & Governance Protocol
3. **[`rust_crates_guide.md`](./docs/rust_crates_guide.md)** — Approved Crates & Banned Dependencies
4. **[`test_suite_manifest.md`](./docs/test_suite_manifest.md)** — Pre-Publish Master Test Suite Manifest
5. **[`ROADMAP_TRACKER.md`](./docs/ROADMAP_TRACKER.md)** — Implementation Status & Verification Tracker

---

## 1. The First-Principles Ethos

Most database projects accept incremental patches that wrap existing libraries or trade hardware sympathy for developer convenience. **Aether-Manifold operates like an aerospace or high-frequency trading kernel:**

- **Zero Plagiarism / Zero Clones**: You may never copy, port, or adapt code from FAISS, Hnswlib, DiskANN, Milvus, Qdrant, Neo4j, or RocksDB. Every algorithm in this repository is derived directly from differential geometry, algebraic topology (cellular sheaves), optimal transport, and kinetic Voronoi/Delaunay compaction.
- **Math First, Code Second**: No algorithm or constant enters the codebase without an explicit mathematical derivation in its module docstring using LaTeX notation.
- **Mechanical Sympathy**: We design around $64\text{-byte}$ CPU cache lines, $512\text{-bit}$ SIMD registers, and $4096\text{-byte}$ NVMe `O_DIRECT` pages—never around object-oriented abstractions or garbage-collected/heap-allocated pointers.

---

## 2. Choosing Your Jurisdiction & Crate

All contributions map to one of our six workspace crates and five architectural jurisdictions defined in [`agents.md`](./docs/agents.md):

| Workspace Crate            | Primary Jurisdiction             | Core Focus Area                                                                                                                                                             |
| :------------------------- | :------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `crates/aether-core`       | `Agent-Hopper` & `Agent-Lamport` | $4096\text{-byte}$ `TensorGraphPage`, $64\text{-byte}$ `PageHeader`, `HlcTimestamp` monotonic clock, zero-alloc tracking guard, and foundational error primitives.         |
| `crates/aether-manifold`   | `Agent-Riemann` & `Agent-Hopper` | Lorentz $\mathbb{H}_{\kappa}^{d_H}$, Spherical $\mathbb{S}_{\tau}^{d_S}$, Euclidean $\mathbb{R}^{d_E}$ SIMD kernels, Cellular Sheaf Laplacian $L_{\mathcal{F}}$, and AT-RQ. |
| `crates/aether-storage`    | `Agent-Stonebraker`              | KLSM-Tree hierarchy, `io_uring` thread-per-core reactor (`O_DIRECT` DMA), Boundary-Only Voronoi compaction, and NVMe WAL.                                                 |
| `crates/aether-query`      | `Agent-Hopper`                   | Semiring-Sheaf Algebraic Query Compiler (Tropical, Diffusion, Bottleneck semirings), Sinkhorn-Knopp Optimal Transport solver, and speculative prefetching.                  |
| `crates/aether-consensus`  | `Agent-Lamport`                  | Shared-nothing thread-per-core actor reactor, SPSC lock-free ring buffers, Raft state machine, and deterministic network harness.                                           |
| `crates/aether-verify`     | `Agent-Feynman`                  | Deterministic simulation testing (DST), Kani SAT/SMT bit-precise verification, Loom atomic model tests, property fuzzing, and PMU hardware performance gates.              |

---

## 3. Development Environment Setup

### 3.1 System & Kernel Prerequisites

To run the full storage engine and Stage 4 hardware performance gates locally, your environment should provide:

- **Linux Kernel $\ge 6.1$**: Required for `io_uring` features including `IORING_SETUP_SQPOLL`, `IORING_SETUP_SINGLE_ISSUER`, and registered fixed buffers (`IORING_REGISTER_BUFFERS`). On non-Linux platforms (Windows/macOS), cross-compilation and non-`io-uring` crates build and test cleanly.
- **CPU ISA**: `x86_64` with AVX-512 (`AVX-512F`, `AVX-512DQ`, `AVX-512BW`, `AVX-512VL`) or AVX2/FMA with dynamic SIMD dispatch via `pulp`. ARM64 NEON is supported via scalar/NEON fallback kernels.
- **Rust Toolchain**: Pinned via `rust-toolchain.toml` with `clippy`, `rustfmt`, and `miri` components installed (`rustup component add clippy rustfmt miri`).
- **PMU Access**: Linux `perf` subsystem access (`/proc/sys/kernel/perf_event_paranoid` $\le 1$) for hardware instruction retired, cache miss, and branch misprediction measurements in `aether-verify`.

### 3.2 Required Build & Verification Tools

Install the formal verification and verification tooling:

```bash
cargo install --locked kani-verifier
cargo kani setup
cargo install --locked cargo-fuzz
```

---

## 4. Non-Negotiable Coding Standards

Before writing any code, ensure you adhere to the hard constraints enforced by CI:

1. **Zero Banned Crates**: Never introduce multi-threaded `tokio`, `rayon`, `serde`, `bincode`, `prost`, `rocksdb`, `petgraph`, `ndarray`, or `nalgebra`. Consult [`rust_crates_guide.md`](./docs/rust_crates_guide.md) for the complete whitelist.
2. **Zero Panic Primitives**: `.unwrap()`, `.expect()`, `panic!()`, `todo!()`, and `unimplemented!()` are strictly forbidden in non-test library code and will trigger immediate CI rejection (`clippy::panic`, `clippy::unwrap_used`, `clippy::expect_used`).
3. **No Unaligned Page Structures**: Disk and memory pages must be strictly $4096\text{-byte}$ aligned (`#[repr(C, align(4096))]`) with zero implicit compiler padding. Verify with `TEST-MEM-01`.
4. **Zero Hot-Path Heap Allocations**: All query, traversal, compaction, and consensus loops must run with zero heap allocations, using `dyn-stack::PodStack` or thread-local `bumpalo::Bump` arenas, protected by `HotPathAllocGuard`.
5. **Strict Monotonicity & Ordering**: All timestamping must utilize wait-free `HlcTimestamp` primitives.

---

## 5. Pull Request Verification Protocol

Every PR must pass the **4-Stage Verification Gate** locally before requesting review:

```bash
# 1. Format and Lints
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings

# 2. Workspace Unit & Integration Tests
cargo test --workspace

# 3. Memory & Layout Proofs
cargo test --package aether-core static_assert_

# 4. Atomic Concurrency Verification (Loom)
RUSTFLAGS="--cfg loom" cargo test --package aether-core --package aether-consensus loom_

# 5. Property-Based Numerical Fuzzing (10^7 iterations)
PROPTEST_CASES=10000000 cargo test --package aether-manifold --package aether-verify prop_
```

---

## 6. PR Review & Merge Checklist

Your PR description must include:
- [ ] Identification of the relevant **Agent Jurisdiction** (`Agent-Hopper`, `Agent-Riemann`, `Agent-Stonebraker`, `Agent-Lamport`, or `Agent-Feynman`).
- [ ] Reference to the corresponding **Roadmap Milestone** (from [`ROADMAP_TRACKER.md`](./docs/ROADMAP_TRACKER.md)).
- [ ] Identification of covered **Test IDs** (from [`test_suite_manifest.md`](./docs/test_suite_manifest.md)).
- [ ] Proof of zero clippy warnings (`cargo clippy --workspace --all-targets -- -D warnings`).
- [ ] Output log demonstrating passing tests and zero hot-path allocations (`HotPathAllocGuard`).
