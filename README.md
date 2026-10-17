# Lets check my stupid Database Engine Design (-\_-)

# Aether-Manifold

[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Zero Hot-Path Allocations](https://img.shields.io/badge/Hot--Path_Allocations-0_Bytes-brightgreen.svg)](#)
[![Verification](https://img.shields.io/badge/Verified-Kani_%7C_Loom_%7C_Proptest-purple.svg)](#)

**Aether-Manifold** is a next-generation unified Vector and Graph Database engine written from first principles in bare-metal Rust. It unifies continuous metric spaces and discrete graph topologies by modeling data as a **Dynamic Cellular Sheaf over a Riemannian Product Manifold**, persisted via a hardware-sympathetic **Kinetic Log-Structured Manifold Tree (KLSM-Tree)**.

---

## Key Architectural Principles

1. **Riemannian Product Manifold ($\mathcal{M}$)**: Combines Hyperbolic Lorentz spaces ($\mathbb{H}_{\kappa}^{d_H}$, $\kappa < 0$) for exponential hierarchies, Hyperspherical spaces ($\mathbb{S}_{\tau}^{d_S}$, $\tau > 0$) for directional semantics, and Euclidean space ($\mathbb{R}^{d_E}$) with fused SIMD metric evaluation.
2. **Cellular Sheaf Theory**: Unifies graph queries and vector similarity into Generalized Sparse Matrix-Vector Multiplication (SpMV) over Tropical, Diffusion, and Bottleneck semirings via the Sheaf Laplacian ($L_{\mathcal{F}} = \delta^T \delta$).
3. **Co-Located 4096-Byte Tensor-Graph Pages**: Co-locates node vectors, 1-hop neighbor IDs, quantized neighbor vectors, and Sheaf restriction maps into a single `#[repr(C, align(4096))]` NVMe page. Hop $t+1$ candidate scoring requires **0 additional disk reads**.
4. **Boundary-Only Voronoi Compaction**: Compaction between immutable KLSM runs relinks Delaunay edges strictly along intersecting cell boundaries ($\partial \mathcal{V}_A \cap \partial \mathcal{V}_B$), eliminating full $O(N \log N)$ index rebuilds.
5. **Zero Hot-Path Allocations**: Hot-path query, traversal, and metric routines allocate **0 heap bytes** (`HotPathAllocGuard` enforced), operating purely via pre-allocated scratchpads and arena buffers.
6. **Shared-Nothing Thread-Per-Core**: Physical 1:1 core pinning, lock-free SPSC cross-core ring buffers, and asynchronous Linux `io_uring` direct I/O (`O_DIRECT`).

---

## Physical Page Layout (4KB NVMe-Aligned)

```text
+-----------------------------------------------------------------------------+
|                4096-BYTE (4KB) NVMe O_DIRECT TENSOR-GRAPH PAGE              |
+------------------+------------------+------------------+--------------------+
|   Header (64B)   |  Vertex AT-RQ    |  1-Hop Neighbor  |   Quantized 1-Hop  |
|  CRC32C, HLC,    |  Manifold Vector |  Vertex IDs      |   Neighbor Vectors |
|  CurvatureBits   |     (256B)       |     (512B)       |      (2048B)       |
+------------------+------------------+------------------+--------------------+
|             Orthogonal Sheaf Edge Morphisms F_{v <= e} (1216B)              |
|           (Packed BF16 / Householder Reflectors + FP16 Spillover)           |
+-----------------------------------------------------------------------------+
```

---

## Workspace Architecture

The project is structured as a Cargo workspace with strict jurisdictional boundaries:

| Crate                                                  | Responsibility                                                                                           | Key Technologies                         |
| :----------------------------------------------------- | :------------------------------------------------------------------------------------------------------- | :--------------------------------------- |
| [`crates/aether-core`](./crates/aether-core)           | 4KB `TensorGraphPage`, 64B `PageHeader`, `HlcTimestamp` clock, `HotPathAllocGuard`                       | `zerocopy`, `bytemuck`, `crc32fast`      |
| [`crates/aether-manifold`](./crates/aether-manifold)   | Product Manifold SIMD metrics ($\mathbb{H} \times \mathbb{S} \times \mathbb{R}$), Sheaf Laplacian, AT-RQ | `pulp` SIMD, `libm`, `faer-core`, `half` |
| [`crates/aether-storage`](./crates/aether-storage)     | KLSM-Tree, $L_0$ Morphism Log, Boundary-Only Voronoi compaction, `io_uring` WAL                          | `io-uring`, `portable-atomic`, `bumpalo` |
| [`crates/aether-query`](./crates/aether-query)         | Semiring-Sheaf Algebraic Compiler, Tropical/Diffusion SpMV, speculative lookahead                        | `bumpalo`, `pulp`, `dyn-stack`           |
| [`crates/aether-consensus`](./crates/aether-consensus) | Thread-per-core actor reactor, lock-free SPSC rings, leaderless HLC consensus                            | `crossbeam-utils`, `portable-atomic`     |
| [`crates/aether-verify`](./crates/aether-verify)       | 4-Stage verification gate, deterministic I/O chaos, property fuzzing, PMU counters                       | `proptest`, `loom`, `oorandom`           |

---

## Quickstart

### Prerequisites

- **Rust**: Toolchain pinned via `rust-toolchain.toml` (1.80+ stable).
- **Target Platforms**:
  - Cross-platform (Linux, Windows, macOS) for core math, memory layouts, and verification suites.
  - Bare-metal Linux ($\ge 6.1$) recommended for production `io_uring` direct I/O and hardware PMU counters.

### Building & Testing

```bash
# Build workspace in release mode
cargo build --release

# Run all workspace unit and integration tests
cargo test --workspace

# Run Clippy with zero warnings enforced
cargo clippy --workspace --all-targets -- -D warnings
```

---

## Verification Pipeline

Releases require passing the 4-Stage Gate Protocol defined in [`docs/test_suite_manifest.md`](./docs/test_suite_manifest.md):

```bash
# Stage 1: Memory Layout & Concurrency Model Checking
cargo test --workspace --lib static_assert_
RUSTFLAGS="--cfg loom" cargo test --package aether-core --package aether-consensus loom_

# Stage 2: Property-Based Mathematical Fuzzing
cargo test --package aether-verify prop_

# Stage 3 & 4: Deterministic Chaos & Hardware PMU Gates
cargo test --package aether-verify --release
```

---

## Documentation

- [Architectural Blueprint (`docs/roadmap.md`)](./docs/roadmap.md) — Mathematical derivations and 1-year phased roadmap.
- [Engineering Constitution (`docs/agents.md`)](./docs/agents.md) — Non-negotiable hardware sympathy rules and agent jurisdictions.
- [Rust Crate Blueprint (`docs/rust_crates_guide.md`)](./docs/rust_crates_guide.md) — Approved bare-metal dependencies and banned ecosystem libraries.
- [Test Suite Manifest (`docs/test_suite_manifest.md`)](./docs/test_suite_manifest.md) — Master 60-gate pre-publish verification catalog.
- [Roadmap Tracker (`docs/ROADMAP_TRACKER.md`)](./docs/ROADMAP_TRACKER.md) — Active milestone and implementation tracker.
- [Contributing Guide (`CONTRIBUTING.md`)](./CONTRIBUTING.md) — Code standards, PR checklist, and verification protocols.

---

## License

Licensed under the [Apache License, Version 2.0](./LICENSE).
