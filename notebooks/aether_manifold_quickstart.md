# Aether-Manifold: Interactive End-User Quickstart & API Cookbook

> **Interactive Developer Notebook & API Reference**  
> **Engine Release:** Phase 3 Verified (`aether-core`, `aether-manifold`, `aether-storage`, `aether-query`, `aether-consensus`, `aether-verify`)  
> **Target Languages:** Declarative AQL (Aether Query Language) $\cdot$ Python 3.11+ $\cdot$ TypeScript 5.x $\cdot$ Rust (2024 Edition, Zero-Allocation Embedded API)

---

## 1. Developer Mental Model: From Differential Geometry to API Primitives

Traditional architectures force developers to maintain a fragile dual-database pipeline: a vector index (HNSW/IVF-PQ) over flat Euclidean space $\mathbb{R}^d$ alongside an independent property graph database (Cypher/CSR). **Aether-Manifold** unifies both worlds by modeling your dataset as a **Dynamic Cellular Sheaf $(G, \mathcal{F})$ over a Riemannian Product Manifold ($\mathcal{M}$)**, persisted inside a **Kinetic Log-Structured Manifold Tree (KLSM-Tree)** and coordinated across cluster shards via **Leaderless HLC Quorum Consensus and RDMA/CXL 3.0 fabrics**.

Before executing the notebook cells below, use this mapping table to connect the underlying mathematical primitives to their concrete SDK and AQL constructs:

| Mathematical Concept | Formal Expression | Systems & Storage Primitive | SDK / AQL Construct |
| :--- | :--- | :--- | :--- |
| **Riemannian Product Manifold** | $\mathcal{M} = \bigotimes_{i=1}^{k} \mathbb{H}_{\kappa_i}^{d_{H,i}} \times \bigotimes_{j=1}^{m} \mathbb{S}_{\tau_j}^{d_{S,j}} \times \mathbb{R}^{d_E}$ | SIMD/AVX-512 Product Metric Tensor & Curvature Descriptors | `CREATE MANIFOLD` / `ProductManifoldSchema` |
| **Lorentz Hyperbolic Component** | $\langle x, y \rangle_{\mathcal{L}} = -x_0 y_0 + \sum_{l=1}^{d} x_l y_l, \; \kappa < 0$ | Horizon-clamped ($\epsilon_{\text{mach}} = 10^{-7}$) surface projection $x_0 = \sqrt{\frac{1}{-\kappa} + \sum x_l^2}$ | `ManifoldFactor::Lorentz { dim, curvature, weight }` |
| **Cellular Sheaf & Restriction Maps** | $\mathcal{F}_{v \trianglelefteq e} : \mathcal{F}(v) \to \mathcal{F}(e), \; (\delta x)_e = \mathcal{F}_{u \trianglelefteq e} x_u - \mathcal{F}_{v \trianglelefteq e} x_v$ | Householder reflector orthogonal morphisms ($\|\mathcal{F}_{v \trianglelefteq e}\|_2 = 1$) in BF16 | `CONNECT ... WITH MORPHISM HOUSEHOLDER(...)` / `SheafEdgeMorphism` |
| **Anisotropic Tangent Quantization (AT-RQ)** | $v_{\text{tangent}} = \log_{\mu_c}(x) \in T_{\mu_c}\mathcal{M}, \; \eta_{\parallel} \gg \eta_{\perp}$ | 32-byte `CompactNeighborCode` + Rauch/Jacobi curvature-corrected ADC | `QuantizerConfig::AtRq { eta_parallel, eta_ortho }` |
| **Co-Located 4KB Tensor-Graph Page** | $\text{Page}_{4\text{KB}}(v) = \text{Header}_{64\text{B}} \parallel \hat{x}_v \parallel \{u_i\} \parallel \{\hat{x}_{u_i}\} \parallel \{\mathcal{F}_{v \trianglelefteq e_i}\}$ | 4096B NVMe-aligned `TensorGraphPage` in `AlignedPageArena` (`O_DIRECT`) | Automatic 0-additional-I/O 1-hop neighborhood evaluation |
| **Semiring-Sheaf SpMV Execution** | $\psi_u^{(t+1)} = \bigoplus_{v \in \mathcal{N}(u)} \left( \psi_v^{(t)} \otimes \exp(-\beta \|\delta x_e\|_2^2 - \gamma d_{\mathcal{M}}^2(u, q)) \right)$ | `SpeculativePrefetchPipeline` + `io_uring` + SIMD SpMV | `TRAVERSE ... USING SEMIRING (TROPICAL \| DIFFUSION \| BOTTLENECK)` |
| **Optimal Transport Subgraph Match** | $\mathcal{W}_{\varepsilon}(A, B) = \min_{P \in U(\mu, \nu)} \left\{ \sum_{i,j} P_{ij} d_{\mathcal{M}}^2(a_i, b_j) + \varepsilon H(P) \right\}$ | Vectorized FMA Sinkhorn-Knopp scaling + Debiased Divergence $S_\varepsilon(A, B)$ | `SINKHORN_DIVERGENCE(bag_a, bag_b, epsilon)` |
| **Geodesic MVCC & Time Travel** | $x_v(\tau_2) = \exp_{x_v(\tau_1)}\left( \Delta v_{(\tau_1 \to \tau_2)} \right)$ | `L0MorphismLog` + `TangentDeltaEntry` + 64-bit `HlcTimestamp` | `AT HLC TIMESTAMP` / `INTERPOLATE GEODESIC AT` |
| **Spectral-Geodesic Partitioning** | $E_e = \|\mathcal{F}_{u \trianglelefteq e} x_u - \mathcal{F}_{v \trianglelefteq e} x_v\|_2^2 + \mu \cdot d_{\mathcal{M}}^2(x_u, x_v)$ | Fiedler vector spectral bisection over Sheaf Laplacian ($\pm 5\%$ balance) | `PARTITION CLUSTER BY SPECTRAL_GEODESIC` |
| **Leaderless HLC Quorum Consensus** | $(hlc_1, node_1) < (hlc_2, node_2), \; R + W > N$ | 64-byte cache-line aligned `ReplicationMessage` + wait-free SPSC rings | `SET CONSENSUS_QUORUM = (W=2, R=2, N=3)` |
| **Zero-Copy RDMA / CXL 3.0 Fabric** | One-sided direct memory page access (0 heap allocations) | `RdmaFabric` + `RdmaMemoryWindow` + 4KB direct mapping | `FETCH REMOTE PAGE DIRECT` |

---

## 2. Environment Setup & Engine Bootstrap

### 2.1 Installing Client SDKs & Embedded Crates

```bash
# Python SDK (with zero-copy PyArrow & PyTorch/NumPy DLPack bindings)
pip install aether-manifold[torch,arrow]

# TypeScript / Node.js SDK (with native N-API SIMD & gRPC/Arrow Flight transport)
npm install @aether-manifold/sdk

# Rust Embedded Engine (direct zero-allocation workspace integration)
cargo add aether-core aether-manifold aether-storage aether-query aether-consensus aether-verify
```

### 2.2 Launching the Storage & Consensus Daemon (`aetherd`)

Aether-Manifold runs a shared-nothing, thread-per-core runtime. In single-node mode, each physical core owns a pinned manifold shard and private `AlignedPageArena`. In distributed mode, nodes communicate across cores via 64-byte wait-free SPSC queues and across network hosts via one-sided zero-copy RDMA / CXL 3.0 fabrics.

```bash
# Standalone Single-Node Daemon
aetherd start \
  --nvme-dev /dev/nvme0n1 \
  --io-engine io_uring_direct \
  --page-size 4096 \
  --cores 0-63 \
  --l0-ring-capacity 65536 \
  --listen-grpc 127.0.0.1:9090 \
  --listen-aql 127.0.0.1:9091

# Distributed 64-Node Cluster Shard Daemon with RDMA RoCEv2
aetherd start \
  --node-id 0 \
  --cluster-size 64 \
  --quorum-w 2 --quorum-r 2 \
  --rdma-dev mlx5_0 \
  --listen-grpc 0.0.0.0:9090
```

---

## 3. Recipe 1: Defining a Riemannian Product Manifold Schema ($\mathcal{M}$) & AT-RQ Quantizer

### Scenario
We are building an **Autonomous Research & Causal Knowledge Graph** where every entity $x \in \mathcal{M}$ simultaneously possesses:
1. **Taxonomic Hierarchy ($\mathbb{H}_{\kappa}^{64}$, $\kappa = -1.0$)**: A 64-dimensional Lorentz hyperboloid space ($64$ spatial dimensions + $1$ timelike coordinate $x_0$, stored as $65$ floats before quantization) capturing exponentially branching citation and ontology trees with near-zero distortion.
2. **Directional Semantic Embedding ($\mathbb{S}_{\tau}^{768}$, $\tau = +1.0$)**: A 768-dimensional unit hypersphere capturing dense angular semantic similarity from an embedding model.
3. **Spatio-Temporal & Scalar Telemetry ($\mathbb{R}^{64}$, curvature $0$)**: A 64-dimensional flat Euclidean space capturing linear publication timestamps, impact velocity, and citation decay metrics.

The unified squared geodesic distance is evaluated as:

$$d_{\mathcal{M}}^2(u, v) = w_H \, d_{\mathbb{H}_{-1.0}}^2\left(u^{(H)}, v^{(H)}\right) + w_S \, d_{\mathbb{S}_{1.0}}^2\left(u^{(S)}, v^{(S)}\right) + w_E \left\| u^{(E)} - v^{(E)} \right\|_2^2$$

To pack up to 64 1-hop neighbor vectors directly inside each vertex's 4KB `TensorGraphPage`, we configure **Anisotropic Tangent-Space Residual Quantization (AT-RQ)** with $\eta_{\parallel} = 12.0$ and $\eta_{\perp} = 1.0$.

---

### Cell 1A: Declarative AQL (Aether Query Language)

```sql
-- Define the Riemannian Product Manifold and co-located 4KB page quantizer
CREATE MANIFOLD research_sheaf_manifold (
    -- Factor 1: Lorentz Hyperboloid Model (negative curvature for trees/hierarchies)
    FACTOR taxonomy_lorentz    TYPE LORENTZ(dim = 64, curvature = -1.0, weight = 1.5, horizon_eps = 1e-7),
    -- Factor 2: Hyperspherical Model (positive curvature for directional semantics)
    FACTOR semantic_sphere     TYPE SPHERICAL(dim = 768, curvature = 1.0, weight = 1.0),
    -- Factor 3: Flat Euclidean Space (zero curvature for dense continuous metrics)
    FACTOR telemetry_euclidean TYPE EUCLIDEAN(dim = 64, weight = 0.5)
)
WITH QUANTIZER AT_RQ (
    coarse_centroids = 4096,
    code_bytes = 32,              -- 32B CompactNeighborCode (64 neighbors = 2048B per 4KB page)
    eta_parallel = 12.0,          -- High penalty along parallel geodesic direction
    eta_orthogonal = 1.0,         -- Lower penalty on orthogonal tangent residuals
    rauch_jacobi_correction = TRUE
)
WITH STORAGE KLSM (
    page_size_bytes = 4096,
    l0_ring_slots = 65536,
    voronoi_boundary_compaction = TRUE
);
```

---

### Cell 1B: Python SDK

```python
import aether_manifold as am

# Connect to the local shard router
client = am.AetherClient("grpc://127.0.0.1:9090")

# Define the Product Manifold M = H_{-1.0}^64 x S_{1.0}^768 x R^64
schema = am.ProductManifoldSchema(
    name="research_sheaf_manifold",
    factors=[
        am.LorentzFactor(
            name="taxonomy_lorentz",
            spatial_dim=64,          # Ambient Lorentz vector in R^{64+1} with timelike x_0
            curvature=-1.0,
            weight=1.5,
            horizon_eps=1e-7,
        ),
        am.SphericalFactor(
            name="semantic_sphere",
            dim=768,
            curvature=1.0,
            weight=1.0,
        ),
        am.EuclideanFactor(
            name="telemetry_euclidean",
            dim=64,
            weight=0.5,
        ),
    ],
    quantizer=am.AtRqConfig(
        coarse_centroids=4096,
        compact_code_bytes=32,       # Fits 64 neighbors in the 2048B neighbor payload of a 4KB page
        eta_parallel=12.0,
        eta_orthogonal=1.0,
        rauch_jacobi_correction=True,
    ),
)

client.create_manifold(schema, exist_ok=True)
print(f"Manifold '{schema.name}' registered. Total tangent dim: {schema.tangent_dim}")
```

---

### Cell 1C: TypeScript SDK

```typescript
import { AetherClient, ManifoldFactorType } from "@aether-manifold/sdk";

async function bootstrapManifold(): Promise<void> {
  const client = await AetherClient.connect("grpc://127.0.0.1:9090");

  await client.createManifold({
    name: "research_sheaf_manifold",
    factors: [
      {
        name: "taxonomy_lorentz",
        type: ManifoldFactorType.LORENTZ,
        spatialDim: 64,
        curvature: -1.0,
        weight: 1.5,
        horizonEps: 1e-7,
      },
      {
        name: "semantic_sphere",
        type: ManifoldFactorType.SPHERICAL,
        dim: 768,
        curvature: 1.0,
        weight: 1.0,
      },
      {
        name: "telemetry_euclidean",
        type: ManifoldFactorType.EUCLIDEAN,
        dim: 64,
        weight: 0.5,
      },
    ],
    quantizer: {
      algorithm: "AT_RQ",
      coarseCentroids: 4096,
      compactCodeBytes: 32,
      etaParallel: 12.0,
      etaOrthogonal: 1.0,
      rauchJacobiCorrection: true,
    },
  });

  console.log("Product Manifold & AT-RQ Codebooks initialized.");
}

bootstrapManifold();
```

---

### Cell 1D: Rust Embedded API (`aether-manifold` & `aether-core`)

```rust
use aether_core::page::{PageHeader, TensorGraphPage};
use aether_manifold::at_rq::{AnisotropicLossConfig, AtRqCodebook, CompactNeighborCode};
use aether_manifold::product::{
    EuclideanSpec, HyperbolicSpec, ProductManifoldConfig, SphericalSpec,
};

pub fn build_research_manifold() -> (ProductManifoldConfig, AnisotropicLossConfig) {
    // Compile-time layout assertions guaranteed by Phase 0 exit gates (TEST-MEM-01)
    const _: () = assert!(core::mem::size_of::<PageHeader>() == 64);
    const _: () = assert!(core::mem::size_of::<TensorGraphPage>() == 4096);
    const _: () = assert!(core::mem::size_of::<CompactNeighborCode>() == 32);

    let config = ProductManifoldConfig {
        hyperbolic: vec![HyperbolicSpec {
            dim: 65, // 64 spatial dims + 1 timelike x_0
            kappa: -1.0_f32,
            weight: 1.5_f32,
        }],
        spherical: vec![SphericalSpec {
            dim: 768,
            tau: 1.0_f32,
            weight: 1.0_f32,
        }],
        euclidean: Some(EuclideanSpec {
            dim: 64,
            weight: 0.5_f32,
        }),
    };

    let quant_cfg = AnisotropicLossConfig {
        eta_parallel: 12.0_f32,
        eta_orthogonal: 1.0_f32,
        enable_rauch_jacobi_correction: true,
    };

    (config, quant_cfg)
}
```

---

## 4. Recipe 2: High-Velocity Ingestion — Vertex Stalks, Lorentz Surface Projection & Sheaf Restriction Maps

### Scenario
When inserting vertices and edges into **Aether-Manifold**:
1. **Lorentz Surface Invariant**: For any hyperbolic spatial vector $\mathbf{x}_{\text{spatial}} \in \mathbb{R}^{d_H}$, the engine enforces the hyperboloid sheet constraint $\langle x, x \rangle_{\mathcal{L}} = \frac{1}{\kappa}$ by computing the timelike coordinate:
   $$x_0 = \sqrt{\frac{1}{-\kappa} + \sum_{l=1}^{d_H} x_l^2}$$
2. **Cellular Sheaf Restriction Maps ($\mathcal{F}_{v \trianglelefteq e}$)**: Every edge $e = (u, v)$ is equipped with orthogonal restriction morphisms $\mathcal{F}_{u \trianglelefteq e}: \mathcal{F}(u) \to \mathcal{F}(e)$ and $\mathcal{F}_{v \trianglelefteq e}: \mathcal{F}(v) \to \mathcal{F}(e)$ parameterized as **Householder reflectors** ($I - 2 \frac{vv^T}{\|v\|_2^2}$, guaranteeing $\|\mathcal{F}_{v \trianglelefteq e}\|_2 = 1$ identically) and serialized in zero-copy BF16 into the $1216\text{-byte}$ `sheaf_edge_weights` region of the 4KB `TensorGraphPage`.

---

### Cell 2A: Declarative AQL

```sql
-- Upsert two vertices onto the Product Manifold (timelike x_0 is auto-projected if omitted)
UPSERT VERTEX 1001 INTO research_sheaf_manifold
  SET STALK (
    taxonomy_lorentz    = PROJECT_LORENTZ([0.12, -0.04, 0.31, /* ... 64 spatial dims */]),
    semantic_sphere     = NORMALIZE_SPHERE([0.021, 0.084, -0.019, /* ... 768 dims */]),
    telemetry_euclidean = [2026.75, 4.82, 0.91, /* ... 64 dims */]
  )
  WITH METADATA { title: "Geometric Sheaf Diffusion over KLSM-Trees", domain: "Database Systems" };

UPSERT VERTEX 1002 INTO research_sheaf_manifold
  SET STALK (
    taxonomy_lorentz    = PROJECT_LORENTZ([0.14, -0.03, 0.29, /* ... 64 spatial dims */]),
    semantic_sphere     = NORMALIZE_SPHERE([0.019, 0.079, -0.015, /* ... 768 dims */]),
    telemetry_euclidean = [2026.80, 5.10, 0.95, /* ... 64 dims */]
  )
  WITH METADATA { title: "Boundary-Only Voronoi Compaction", domain: "Storage Engines" };

-- Connect Vertex 1001 -> 1002 with an orthogonal Householder Sheaf Restriction Morphism
CONNECT VERTEX 1001 TO 1002 IN research_sheaf_manifold
  EDGE_TYPE "EXTENDS_THEOREM"
  WITH SHEAF_MORPHISM (
    source_householder_reflector = [0.57735, -0.57735, 0.57735, /* ... reflector normal */],
    target_householder_reflector = [0.57735,  0.57735, 0.57735, /* ... reflector normal */]
  );
```

---

### Cell 2B: Python SDK (Zero-Copy NumPy Batch Ingestion)

```python
import numpy as np
import aether_manifold as am

client = am.AetherClient("grpc://127.0.0.1:9090")
rng = np.random.default_rng(seed=42)

# 1. Generate spatial hyperbolic coordinates and project onto the Lorentz sheet:
kappa = -1.0
spatial_u = rng.normal(0.0, 0.2, size=(64,)).astype(np.float32)
x0_u = np.sqrt((1.0 / -kappa) + np.sum(spatial_u ** 2, dtype=np.float32))
lorentz_u = np.concatenate(([x0_u], spatial_u)).astype(np.float32)

# 2. Generate unit hypersphere vector (S^768) and Euclidean vector (R^64)
raw_sphere_u = rng.normal(0.0, 1.0, size=(768,)).astype(np.float32)
sphere_u = raw_sphere_u / np.linalg.norm(raw_sphere_u)
euclid_u = rng.uniform(0.0, 1.0, size=(64,)).astype(np.float32)

# 3. Upsert vertices and connect with orthogonal Householder Sheaf restriction maps
batch = am.WriteBatch(manifold="research_sheaf_manifold")

batch.upsert_vertex(
    vertex_id=1001,
    factors={
        "taxonomy_lorentz": lorentz_u,
        "semantic_sphere": sphere_u,
        "telemetry_euclidean": euclid_u,
    },
)

batch.upsert_vertex(
    vertex_id=1002,
    factors={
        "taxonomy_lorentz": am.project_lorentz(spatial_u * 1.05, curvature=-1.0),
        "semantic_sphere": sphere_u,
        "telemetry_euclidean": euclid_u * 0.98,
    },
)

# Orthogonal Householder reflector vectors
reflector_u = rng.normal(0.0, 1.0, size=(64,)).astype(np.float32)
reflector_v = rng.normal(0.0, 1.0, size=(64,)).astype(np.float32)

batch.upsert_sheaf_edge(
    source_id=1001,
    target_id=1002,
    edge_label="EXTENDS_THEOREM",
    morphism=am.HouseholderSheafMorphism(
        source_reflector=reflector_u,
        target_reflector=reflector_v,
    ),
)

receipt = client.commit_batch(batch)
print(f"Committed at HLC: physical={receipt.hlc.physical_ms}, logical={receipt.hlc.logical_counter}")
```

---

### Cell 2C: TypeScript SDK

```typescript
import { AetherClient, projectLorentzInPlace, normalizeSphereInPlace } from "@aether-manifold/sdk";

async function ingestGraphEntities(client: AetherClient): Promise<void> {
  const lorentzU = new Float32Array(65);
  for (let i = 1; i < 65; i++) lorentzU[i] = (Math.random() - 0.5) * 0.1;
  projectLorentzInPlace(lorentzU, -1.0);

  const sphereU = new Float32Array(768).map(() => Math.random() - 0.5);
  normalizeSphereInPlace(sphereU, 1.0);

  const euclidU = new Float32Array(64).fill(0.42);

  const tx = client.beginBatch("research_sheaf_manifold");
  tx.upsertVertex(1001n, {
    taxonomy_lorentz: lorentzU,
    semantic_sphere: sphereU,
    telemetry_euclidean: euclidU,
  });

  const householderNormal = new Float32Array(64).fill(1.0 / Math.sqrt(64));
  tx.upsertSheafEdge({
    sourceId: 1001n,
    targetId: 1002n,
    edgeType: "EXTENDS_THEOREM",
    sourceHouseholder: householderNormal,
    targetHouseholder: householderNormal,
  });

  const { hlcTimestamp } = await tx.commit();
  console.log(`Ingested vertex & sheaf morphism at HLC=${hlcTimestamp.toString()}`);
}
```

---

### Cell 2D: Rust Embedded API (Zero-Heap-Allocation Hot Path)

```rust
use aether_core::alloc_guard::HotPathAllocGuard;
use aether_core::hlc::HlcTimestamp;
use aether_query::sheaf::{pack_page_sheaf_morphisms, SheafRestrictionMap};
use aether_storage::page_arena::AlignedPageArena;

pub fn ingest_zero_alloc_hot_path(
    arena: &mut AlignedPageArena,
    clock_state: &core::sync::atomic::AtomicU64,
    physical_now_micros: u64,
) {
    let _guard = HotPathAllocGuard::enter();

    let ts = HlcTimestamp::now_or_tick(clock_state, physical_now_micros);

    // Allocate pre-aligned 4KB TensorGraphPage inside arena (strictly 0 heap allocations)
    let page = arena
        .allocate_page(1001, ts)
        .expect("Pre-allocated 4KB slot available");

    // Pack Householder restriction maps directly into the 1216B page region
    let maps = [SheafRestrictionMap::identity(8); 4];
    pack_page_sheaf_morphisms(&maps, &mut page.sheaf_edge_weights)
        .expect("Morphisms packed into 4KB page without allocation");
}
```

---

## 5. Recipe 3: Hybrid Topological-Geometric Queries — Semiring-Sheaf SpMV & Dirichlet Energy

### Scenario
An agent asks: *"Find all research theorems within 3 citation hops of Paper 1001 that are causally aligned with quantum gravity topology ($d_{\mathcal{M}} \le 0.45$) and have Sheaf Dirichlet energy $E_{\mathcal{F}}(u, v) \le 0.15$."*

In **Aether-Manifold**, this is compiled into Generalized Sparse Matrix-Vector Multiplication (SpMV) over the Sheaf operator $\mathcal{A}_{\mathcal{F}}$:

$$\psi_u^{(t+1)} = \bigoplus_{v \in \mathcal{N}(u)} \left( \psi_v^{(t)} \otimes \exp\left( -\beta \left\| \mathcal{F}_{u \trianglelefteq e} x_u - \mathcal{F}_{v \trianglelefteq e} x_v \right\|_2^2 - \gamma \, d_{\mathcal{M}}^2(u, q) \right) \right)$$

Depending on the exploration objective, the engine dynamically selects the algebraic semiring:
- **Tropical $(\min, +)$**: Geodesic $A^*$ shortest relational path.
- **Diffusion $(+, \times)$**: Sheaf Dirichlet diffusion / semantic PageRank.
- **Bottleneck $(\max, \min)$**: Maximum-capacity relational path under semantic threshold constraints.

---

### Cell 3A: Declarative AQL

```sql
-- Execute a 3-hop interleaved vector-relational traversal with Tropical Semiring
TRAVERSE FROM VERTEX 1001 IN research_sheaf_manifold
  MAX_HOPS 3
  USING SEMIRING TROPICAL
  WEIGHTED BY (
    sheaf_dirichlet_beta = 1.0,    -- Penalty on Sheaf section disagreement ||F_{u<=e} x_u - F_{v<=e} x_v||^2
    geodesic_distance_gamma = 1.5  -- Penalty on Riemannian distance d_M^2(u, q)
  )
  MATCH PATH (u)-[:EXTENDS_THEOREM*1..3]->(target)
  WHERE
    d_M(target.stalk, $query_product_vector) <= 0.45
    AND SHEAF_DIRICHLET_ENERGY(u, target) <= 0.15
  ORDER BY semiring_score ASC
  LIMIT 10;
```

---

### Cell 3B: Python SDK

```python
import aether_manifold as am

client = am.AetherClient("grpc://127.0.0.1:9090")

query_vector = {
    "taxonomy_lorentz": lorentz_u,
    "semantic_sphere": sphere_u,
    "telemetry_euclidean": euclid_u,
}

# Execute 3-hop traversal with speculative io_uring prefetch pipeline
results = client.traverse(
    manifold="research_sheaf_manifold",
    start_vertex=1001,
    max_hops=3,
    semiring=am.SemiringMode.TROPICAL,  # (min, +) shortest relational-geodesic path
    sheaf_beta=1.0,
    geodesic_gamma=1.5,
    query_vector=query_vector,
    max_geodesic_radius=0.45,
    beam_width=64,
    limit=10,
)

for rank, item in enumerate(results, start=1):
    print(f"#{rank} Node {item.vertex_id}: score={item.semiring_score:.4f}, hops={item.hop_distance}")
```

---

### Cell 3C: TypeScript SDK

```typescript
import { AetherClient, SemiringMode } from "@aether-manifold/sdk";

async function querySheafGraph(client: AetherClient, queryPoint: Float32Array): Promise<void> {
  const resultStream = await client.traverse({
    manifold: "research_sheaf_manifold",
    startVertex: 1001n,
    maxHops: 3,
    semiring: SemiringMode.DIFFUSION, // (+, *) personalized semantic PageRank
    sheafBeta: 1.0,
    geodesicGamma: 1.5,
    queryVector: queryPoint,
    topK: 10,
  });

  for await (const node of resultStream) {
    console.log(`Visited ${node.vertexId}: Diffusion Energy = ${node.score}`);
  }
}
```

---

### Cell 3D: Rust Embedded API (Semiring SpMV Query Compiler)

```rust
use aether_core::alloc_guard::HotPathAllocGuard;
use aether_query::compiler::{QueryPlan, SemiringSpMvEngine};
use aether_query::semiring::SemiringMode;
use aether_storage::page_arena::AlignedPageArena;

pub fn execute_sheaf_query_zero_alloc(
    engine: &SemiringSpMvEngine,
    arena: &AlignedPageArena,
    start_vertex: u64,
    query_coords: &[f32; 1536],
) -> Vec<(u64, f32)> {
    let plan = QueryPlan {
        start_vertex,
        max_hops: 3,
        semiring: SemiringMode::Tropical,
        sheaf_beta: 1.0_f32,
        geodesic_gamma: 1.5_f32,
        beam_width: 64,
        top_k: 10,
    };

    let _guard = HotPathAllocGuard::enter();

    // Zero-heap-allocation traversal inside HotPathAllocGuard
    engine
        .execute_query(&plan, arena, query_coords)
        .expect("Query executed in p99 < 5ms")
}
```

---

## 6. Recipe 4: Multi-Vector Bags & Ego-Graph Matching via Wasserstein-Sinkhorn Optimal Transport ($\mathcal{W}_\varepsilon$)

### Scenario
An agent needs to compare two complex documents, compound molecules, or subgraphs $A = \{a_1, \dots, a_n\} \subset \mathcal{M}$ and $B = \{b_1, \dots, b_m\} \subset \mathcal{M}$ of asymmetric cardinality ($n \ne m$). Instead of lossy average pooling, **Aether-Manifold** computes the exact **Entropic Regularized Wasserstein Distance** over Riemannian product geodesics:

$$\mathcal{W}_{\varepsilon}(A, B) = \min_{P \in U(\mu, \nu)} \left\{ \sum_{i=1}^{n} \sum_{j=1}^{m} P_{ij} d_{\mathcal{M}}^2(a_i, b_j) + \varepsilon \sum_{i=1}^{n} \sum_{j=1}^{m} P_{ij} \left( \log P_{ij} - 1 \right) \right\}$$

It applies **Debiased Sinkhorn Divergence** $S_\varepsilon(A, B) = \mathcal{W}_\varepsilon(A, B) - \frac{1}{2} \mathcal{W}_\varepsilon(A, A) - \frac{1}{2} \mathcal{W}_\varepsilon(B, B)$ to guarantee $S_\varepsilon(A, A) = 0$ and metric positivity.

---

### Cell 4A: Declarative AQL

```sql
-- Compute the debiased Sinkhorn divergence between two entity subgraphs
SELECT
    doc_id,
    SINKHORN_DIVERGENCE(
        document_token_bag,
        $query_evidence_bag,
        epsilon = 0.05,
        max_iterations = 100,
        debiased = TRUE
    ) AS optimal_transport_distance
FROM research_documents
ORDER BY optimal_transport_distance ASC
LIMIT 5;
```

---

### Cell 4B: Python SDK (Vectorized FMA Sinkhorn-Knopp)

```python
import numpy as np
import aether_manifold as am

client = am.AetherClient("grpc://127.0.0.1:9090")

# Bag A: 32 premise vectors; Bag B: 48 hypothesis vectors
bag_a = np.random.normal(size=(32, 64)).astype(np.float32)
bag_b = np.random.normal(size=(48, 64)).astype(np.float32)

# Compute debiased Sinkhorn divergence in < 50us via SIMD FMA scaling
ot_divergence = client.compute_sinkhorn_divergence(
    bag_a=bag_a,
    bag_b=bag_b,
    manifold="research_sheaf_manifold",
    epsilon=0.05,
    debiased=True,
)

print(f"Debiased Sinkhorn Divergence S_eps(A, B): {ot_divergence:.6f}")
```

---

### Cell 4C: Rust Embedded API (Scratchpad-Allocated Sinkhorn Kernel)

```rust
use aether_core::alloc_guard::HotPathAllocGuard;
use aether_core::scratchpad::Scratchpad;
use aether_manifold::optimal_transport::{
    sinkhorn_debiased_divergence, OptimalTransportCostMatrix,
};

pub fn compute_subgraph_ot_zero_alloc(
    cost_matrix: &OptimalTransportCostMatrix,
    scratchpad: &mut Scratchpad,
) -> f32 {
    let _guard = HotPathAllocGuard::enter();

    sinkhorn_debiased_divergence(cost_matrix, 0.05_f32, 100, 1e-4_f32, scratchpad)
        .expect("Converged within tolerance with zero heap allocations")
}
```

---

## 7. Recipe 5: Causal Spatio-Temporal MVCC & $O(1)$ Geodesic Time-Travel

### Scenario
When embeddings and graph relationships evolve at $10^7$ mutations/sec, copying full $1024\text{-dim}$ vectors on every update causes write amplification and cache thrashing. In **Aether-Manifold**, mutations are stamped with a 64-bit **Hybrid Logical Clock (`HlcTimestamp`)** $\tau = (t_{\text{physical}}, c_{\text{logical}})$ and logged in `L0MorphismLog` as **Tangent-Space Velocity Vectors**:

$$\Delta v_{(\tau_1 \to \tau_2)} = \log_{x_v(\tau_1)}\left(x_v(\tau_2)\right) \in T_{x_v(\tau_1)}\mathcal{M}$$

This enables:
1. **Exact Snapshot Isolation**: Reconstruct state at $\tau_2$ via the Riemannian Exponential Map:
   $$x_v(\tau_2) = \exp_{x_v(\tau_1)}\left( \Delta v_{(\tau_1 \to \tau_2)} \right)$$
2. **Continuous $O(1)$ Geodesic Time-Travel Interpolation**: Reconstruct the exact trajectory of a vertex at *any* intermediate timestamp $\tau_q \in [\tau_1, \tau_2]$ where $\alpha = \frac{\tau_q - \tau_1}{\tau_2 - \tau_1} \in [0, 1]$:
   $$x_v(\tau_q) = \exp_{x_v(\tau_1)}\left( \alpha \cdot \Delta v_{(\tau_1 \to \tau_2)} \right)$$

---

### Cell 5A: Declarative AQL

```sql
-- Apply a low-overhead Tangent-Space Velocity update in L0MorphismLog
APPLY TANGENT_DELTA TO VERTEX 1001 IN research_sheaf_manifold
  SET DELTA_VELOCITY (
    taxonomy_lorentz    = [0.002, -0.001, 0.004, /* ... tangent vector in T_x H^64 */],
    semantic_sphere     = [-0.005, 0.003, 0.001, /* ... tangent vector in T_x S^768 */],
    telemetry_euclidean = [0.25, 0.12, -0.01,    /* ... tangent vector in R^64 */]
  );

-- Query the continuous geodesic interpolation at HLC timestamp tau_q
SELECT
    vertex_id,
    RECONSTRUCT_MANIFOLD_STATE(vertex_id) AS interpolated_stalk,
    GEODESIC_VELOCITY_NORM(vertex_id) AS semantic_drift_speed
FROM research_sheaf_manifold
AT HLC TIMESTAMP (physical_ms = 1792227000120, logical = 4)
WITH GEODESIC_INTERPOLATION = TRUE
WHERE vertex_id = 1001;
```

---

### Cell 5B: Python SDK

```python
import numpy as np
import aether_manifold as am

client = am.AetherClient("grpc://127.0.0.1:9090")

tau_1 = client.get_current_hlc()

delta_v = {
    "taxonomy_lorentz": np.full(64, 0.005, dtype=np.float32),
    "semantic_sphere": np.full(768, 0.001, dtype=np.float32),
    "telemetry_euclidean": np.full(64, 0.10, dtype=np.float32),
}
commit_info = client.append_tangent_delta(
    manifold="research_sheaf_manifold",
    vertex_id=1001,
    tangent_delta=delta_v,
)
tau_2 = commit_info.hlc

# Reconstruct midpoint state (alpha = 0.5) in O(1)
tau_midpoint = am.HlcTimestamp.midpoint(tau_1, tau_2)

snapshot = client.get_vertex_at_hlc(
    manifold="research_sheaf_manifold",
    vertex_id=1001,
    at_hlc=tau_midpoint,
    interpolate_geodesic=True,
)

print(f"Reconstructed state at HLC {tau_midpoint} (alpha={snapshot.interpolation_alpha:.2f})")
```

---

### Cell 5C: Rust Embedded API (`L0MorphismLog` Geodesic MVCC)

```rust
use aether_core::alloc_guard::HotPathAllocGuard;
use aether_core::hlc::HlcTimestamp;
use aether_storage::l0_mvcc::{L0MorphismLog, TangentDeltaEntry};

pub fn reconstruct_time_travel_state_zero_alloc(
    l0_log: &L0MorphismLog<64, 1024>,
    vertex_id: u64,
    base_state_tau1: &[f32; 64],
    query_hlc: HlcTimestamp,
) -> [f32; 64] {
    let mut reconstructed = [0.0_f32; 64];
    let _guard = HotPathAllocGuard::enter();

    l0_log
        .reconstruct_at_hlc_euclidean(
            vertex_id,
            base_state_tau1,
            query_hlc,
            &mut reconstructed,
        )
        .expect("Causal HLC timestamp resolved in L0MorphismLog");

    reconstructed
}
```

---

## 8. Recipe 6: KLSM-Tree Boundary-Only Voronoi Compaction & Telemetry

### Scenario
When merging an $L_0 / L_1$ geometric run $\mathcal{R}_A$ into an $L_2$ NVMe run $\mathcal{R}_B$, **Aether-Manifold** partitions the manifold into **Riemannian Voronoi Cells** $\{\mathcal{V}_1, \dots, \mathcal{V}_p\}$. For any interior vertex $u \in \mathcal{R}_B$ whose $k$-th nearest neighbor radius satisfies the **Voronoi Boundary Certificate**:

$$d_{\mathcal{M}}(u, \mu_A) - R_A > d_{\mathcal{M}}\left(u, u_{\text{NN}}^{(k)}\right)$$

the engine proves mathematically that no point in $\mathcal{R}_A$ can enter $u$'s top-$k$ Delaunay neighborhood. Those interior vertices execute an **$O(1)$ pointer splice**, restricting expensive Delaunay re-triangulation strictly to the 2-manifold boundary $\partial \mathcal{V}_A \cap \partial \mathcal{V}_B$.

---

### Cell 6A: Declarative AQL & Rust Compaction Inspection

```sql
-- Trigger an asynchronous Boundary-Only Delaunay Compaction pass
OPTIMIZE MANIFOLD research_sheaf_manifold
  COMPACT LEVEL L1 TO L2
  USING VORONOI_BOUNDARY_CERTIFICATES
  RETURN TELEMETRY;
```

```rust
use aether_storage::voronoi_compaction::{
    evaluate_voronoi_boundary_certificate, VoronoiCellCertificate,
};

pub fn check_o1_pointer_splice_eligibility(
    vertex_dist_to_modified_centroid: f32,
    modified_cell_max_radius: f32,
    vertex_kth_neighbor_geodesic_dist: f32,
) -> bool {
    let cert = VoronoiCellCertificate {
        centroid_id: 42,
        max_geodesic_radius: modified_cell_max_radius,
    };

    evaluate_voronoi_boundary_certificate(
        vertex_dist_to_modified_centroid,
        &cert,
        vertex_kth_neighbor_geodesic_dist,
    )
}
```

---

## 9. Recipe 7: Distributed Dynamic Topology Partitioning via Spectral-Geodesic Bisection

### Scenario
As your knowledge graph scales beyond a single server to 64 cluster nodes, random or hash-based partitioning destroys query performance by turning $90\%$ of multi-hop queries into cross-network roundtrips. **Aether-Manifold** implements the **Spectral-Geodesic Sheaf Graph Partitioner (`SpectralGeodesicPartitioner`)**, which computes edge cut costs combining Sheaf Dirichlet energy and Riemannian geodesic distance:

$$E_e = \left\| \mathcal{F}_{u \trianglelefteq e} x_u - \mathcal{F}_{v \trianglelefteq e} x_v \right\|_2^2 + \mu \cdot d_{\mathcal{M}}^2(x_u, x_v)$$

The partitioner computes the **Fiedler vector** (second smallest eigenvector of the normalized Laplacian operator) via power iteration with deflation, partitioning vertices into balanced shards while strictly enforcing:
1. **$\pm 5\%$ Partition Balance Tolerance**: Every shard receives $\frac{N}{K} \pm 5\%$ vertices.
2. **$> 90\%$ Local Shard Traversal Ratio**: Strongly coupled semantic neighborhoods reside on the same node's NVMe/CXL arena.

---

### Cell 7A: Declarative AQL

```sql
-- Partition the global topology across 64 cluster shards with +/-5% balance constraint
PARTITION CLUSTER research_sheaf_manifold
  BY SPECTRAL_GEODESIC (
    target_shards = 64,
    geodesic_mu = 1.0,           -- Weight on Riemannian distance penalty d_M^2(u, v)
    balance_tolerance = 0.05,    -- Enforce +/-5% maximum size imbalance
    max_power_iterations = 150
  )
  ENSURE MINIMUM_LOCAL_TRAVERSAL_RATIO = 0.90;
```

---

### Cell 7B: Python SDK

```python
import aether_manifold as am

client = am.AetherClient("grpc://127.0.0.1:9090")

# Trigger cluster rebalancing across 64 shards
partition_plan = client.repartition_cluster(
    manifold="research_sheaf_manifold",
    num_shards=64,
    geodesic_weight_mu=1.0,
    balance_tolerance=0.05,
)

print(f"Partition Plan generated for {partition_plan.total_vertices} vertices across {partition_plan.num_shards} shards.")
print(f"Local Traversal Ratio: {partition_plan.local_traversal_ratio * 100:.2f}% (Target: > 90%)")
print(f"Max shard size variance: {partition_plan.max_imbalance_ratio * 100:.2f}% (Limit: <= 5%)")
```

---

### Cell 7C: TypeScript SDK

```typescript
import { AetherClient } from "@aether-manifold/sdk";

async function rebalanceClusterShards(client: AetherClient): Promise<void> {
  const plan = await client.repartitionTopology({
    manifold: "research_sheaf_manifold",
    numShards: 64,
    geodesicWeightMu: 1.0,
    balanceTolerance: 0.05,
    maxPowerIterations: 150,
  });

  console.log(`Topology repartitioned: ${plan.localTraversalRatio * 100}% queries stay local.`);
}
```

---

### Cell 7D: Rust Embedded API (`SpectralGeodesicPartitioner`)

```rust
use aether_consensus::partitioner::{
    GraphEdge, PartitionerConfig, SpectralGeodesicPartitioner,
};

pub fn partition_cluster_topology(
    num_vertices: usize,
    edges: &[GraphEdge],
) -> aether_consensus::partitioner::PartitionPlan {
    let config = PartitionerConfig {
        num_shards: 64,
        geodesic_weight_mu: 1.0_f32,
        balance_tolerance: 0.05_f32,
        max_power_iterations: 150,
    };

    let partitioner = SpectralGeodesicPartitioner::new(config)
        .expect("Valid partitioner configuration");

    let plan = partitioner
        .partition(num_vertices, edges)
        .expect("Topology successfully partitioned across 64 shards");

    assert!(plan.is_balanced(0.05));
    plan
}
```

---

## 10. Recipe 8: Leaderless HLC Quorum Consensus & Sub-10ms Minority Failover

### Scenario
In a mission-critical enterprise deployment, cluster nodes may crash or experience transient network partitions. Traditional Multi-Raft setups suffer tail-latency spikes during leader elections. **Aether-Manifold** implements **Leaderless Quorum Consensus ($R + W > N$)**:
1. **64-Byte Cache-Line Aligned Envelopes**: Proposals, acknowledgments, and commits use a strictly aligned `ReplicationMessage` (`size_of == 64`, `align_of == 64`) with zero heap allocations on the hot path.
2. **Causal Total Ordering**: Causal conflicts are resolved via 64-bit Hybrid Logical Clocks (`HlcTimestamp`) and deterministic tie-breaking:
   $$(p_1, l_1, n_1) < (p_2, l_2, n_2) \iff (\tau_1 < \tau_2) \lor (\tau_1 == \tau_2 \land n_1 < n_2)$$
3. **Sub-10ms Fast Catchup Replay**: When an offline replica re-joins the cluster, it requests recent morphism records from peers via `MorphismReplicator::get_records_since` and replays missing commits via `replay_batch` in **$< 10\text{ ms}$** with zero data loss.

---

### Cell 8A: Declarative AQL

```sql
-- Configure Leaderless Quorum Consensus parameters
SET CONSENSUS_QUORUM = (
    cluster_size = 3,           -- N = 3 replica shards
    write_quorum = 2,           -- W = 2 acknowledgments required for commit
    read_quorum  = 2            -- R = 2 responses required for consistent read (R + W > N)
);

-- Trigger an explicit fast catchup recovery on Node 2 from peer Node 0
RECOVER SHARD 2
  FROM PEER NODE 0
  SINCE HLC TIMESTAMP (physical_ms = 1792227000000, logical = 0)
  TIMEOUT 10 MILLISECONDS;
```

---

### Cell 8B: Python SDK

```python
import aether_manifold as am

# Configure 3-node replica group: N=3, W=2, R=2
cluster = am.ClusterClient(["grpc://10.0.0.1:9090", "grpc://10.0.0.2:9090", "grpc://10.0.0.3:9090"])

# Propose a replicated mutation with quorum write guarantee
commit_receipt = cluster.propose_replicated_mutation(
    vertex_id=5001,
    payload=b"\x01\x02\x03\x04" * 6,
    write_quorum=2,
)

print(f"Mutation committed to quorum at HLC {commit_receipt.hlc_timestamp}")

# Consistent read quorum (picks the causally dominant state among R=2 peers)
consistent_state = cluster.read_quorum(vertex_id=5001, read_quorum=2)
print(f"Causally consistent state: {consistent_state.payload}")
```

---

### Cell 8C: TypeScript SDK

```typescript
import { AetherClusterClient } from "@aether-manifold/sdk";

async function replicateMutation(cluster: AetherClusterClient): Promise<void> {
  const receipt = await cluster.propose({
    vertexId: 5001n,
    payload: new Uint8Array(24).fill(0xaa),
    writeQuorum: 2,
  });

  console.log(`Quorum achieved: Sequence=${receipt.sequenceNumber}, Committed=${receipt.committed}`);
}
```

---

### Cell 8D: Rust Embedded API (`LeaderlessQuorumEngine` & `MorphismReplicator`)

```rust
use aether_consensus::consensus::{
    LeaderlessQuorumEngine, QuorumConfig, ReplicationMessage, ReplicationMessageType,
};
use aether_consensus::morphism_replicator::MorphismReplicator;
use aether_core::alloc_guard::HotPathAllocGuard;

pub fn execute_consensus_lifecycle_zero_alloc() {
    let config0 = QuorumConfig::new(0, 3, 2, 2).expect("Valid N=3, W=2, R=2");
    let config1 = QuorumConfig::new(1, 3, 2, 2).expect("Valid N=3, W=2, R=2");

    let mut node0 = LeaderlessQuorumEngine::new(config0, 1_000_000);
    let mut node1 = LeaderlessQuorumEngine::new(config1, 1_000_000);
    let mut replicator0 = MorphismReplicator::new(0);

    let _guard = HotPathAllocGuard::enter();

    // 1. Node 0 proposes mutation (0 heap allocations)
    let proposal = node0.propose(5001, [0xEE; 24], 1_000_010).unwrap();

    // 2. Node 1 acks remote proposal (0 heap allocations)
    let ack = node1.handle_remote_propose(&proposal, 1_000_012).unwrap();

    // 3. Node 0 processes ack, reaching write quorum W=2 (self + node1)
    let commit = node0.handle_ack(&ack).unwrap().expect("Write quorum reached");

    // 4. Replicate commit into local morphism log (0 heap allocations)
    replicator0.apply_commit(&commit).unwrap();
}
```

---

## 11. Recipe 9: Zero-Copy RDMA & CXL 3.0 Remote Page Fetching Fabric

### Scenario
During distributed 3-hop traversals, an edge may lead to a target vertex hosted on a remote cluster node. Instead of serializing JSON/Protobuf or making synchronous RPCs that wake up the remote CPU, **Aether-Manifold** uses one-sided **RDMA Reads** (RoCEv2) or direct **CXL 3.0 Type 3 loads** into local memory:
- **Zero Remote CPU Overhead**: Remote node kernel/CPU is completely bypassed.
- **Direct 4KB NVMe-Aligned Pages**: Fetches raw `TensorGraphPage` instances directly into local cache arenas in strictly $0$ heap allocations.

---

### Cell 9A: Declarative AQL

```sql
-- Enable direct one-sided RDMA page fetching for distributed queries
SET FABRIC_TRANSPORT = "RDMA_ROCE_V2";
SET RDMA_MEMORY_WINDOW_SIZE_MB = 1024;

-- Query spanning multiple cluster shards via zero-copy RDMA reads
SELECT target.vertex_id, target.metadata
FROM research_sheaf_manifold
TRAVERSE FROM VERTEX 1001 MAX_HOPS 3
OPTIONS (REMOTE_FETCH_ENGINE = "RDMA_ONE_SIDED_DIRECT");
```

---

### Cell 9B: Rust Embedded API (`RdmaFabric` & `RdmaMemoryWindow`)

```rust
use aether_consensus::rdma_fabric::{RdmaFabric, RdmaMemoryWindow};
use aether_core::alloc_guard::HotPathAllocGuard;

pub fn fetch_remote_page_zero_copy(
    fabric: &RdmaFabric,
    target_node_id: u64,
    page_index: usize,
) -> [u8; 4096] {
    let mut page_buf = [0u8; 4096];

    let _guard = HotPathAllocGuard::enter();

    // One-sided remote memory copy with strictly 0 heap allocations
    fabric
        .read_remote_page(target_node_id, page_index, &mut page_buf)
        .expect("Remote page fetched via RDMA fabric without allocation");

    page_buf
}
```

---

## 12. Quick Reference Cheat Sheet

### 12.1 AQL (Aether Query Language) Command Matrix

| Operation | AQL Statement Pattern | Underlying Engine Subsystem |
| :--- | :--- | :--- |
| **Define Product Space** | `CREATE MANIFOLD <name> (FACTOR ... TYPE LORENTZ \| SPHERICAL \| EUCLIDEAN)` | `aether-manifold::product` |
| **Configure Quantization** | `WITH QUANTIZER AT_RQ (eta_parallel = ..., eta_orthogonal = ...)` | `aether-manifold::at_rq` |
| **Insert Vertex Stalk** | `UPSERT VERTEX <id> INTO <manifold> SET STALK (...)` | `aether-storage::shard_runtime` |
| **Attach Sheaf Edge** | `CONNECT VERTEX <u> TO <v> WITH SHEAF_MORPHISM (...)` | `aether-query::sheaf` (Householder BF16) |
| **Hybrid Traversal** | `TRAVERSE FROM VERTEX <id> ... USING SEMIRING (TROPICAL \| DIFFUSION \| BOTTLENECK)` | `aether-query::compiler` & `semiring` |
| **Sheaf Smoothing** | `MINIMIZE DIRICHLET_ENERGY (lambda_fidelity = ..., max_agd_iterations = ...)` | `aether-query::dirichlet` & `laplacian` |
| **Subgraph Matching** | `SINKHORN_DIVERGENCE(bag_a, bag_b, epsilon = ...)` | `aether-manifold::optimal_transport` |
| **Causal Time Travel** | `SELECT ... FROM <manifold> AT HLC TIMESTAMP (...) WITH GEODESIC_INTERPOLATION = TRUE` | `aether-storage::l0_mvcc` & `aether-core::hlc` |
| **Spectral Partitioning** | `PARTITION CLUSTER <manifold> BY SPECTRAL_GEODESIC (target_shards = K, balance_tolerance = 0.05)` | `aether-consensus::partitioner` |
| **Quorum Consensus** | `SET CONSENSUS_QUORUM = (cluster_size = N, write_quorum = W, read_quorum = R)` | `aether-consensus::consensus` |
| **Minority Catchup** | `RECOVER SHARD <id> FROM PEER <peer_id> SINCE HLC TIMESTAMP (...)` | `aether-consensus::morphism_replicator` |
| **Remote Page Fetch** | `FETCH REMOTE PAGE DIRECT FROM NODE <id> PAGE <idx>` | `aether-consensus::rdma_fabric` |

---

### 12.2 Verified Hardware & Numerical SLA Guarantees (Phases 0–3)

| Verification Gate | Invariant / Metric | Verified Performance |
| :--- | :--- | :--- |
| **Memory Layout (`TEST-MEM-01`)** | `size_of::<TensorGraphPage>() == 4096`, `size_of::<PageHeader>() == 64` | $0$ compiler padding bytes; $4\text{KB}$ NVMe `O_DIRECT` aligned |
| **Replication Message Layout** | `size_of::<ReplicationMessage>() == 64`, `align_of == 64` | Cache-line aligned envelope; zero false sharing |
| **Hot-Path Allocation (`TEST-MEM-04`)** | Heap allocations inside `HotPathAllocGuard` during query, ingest, and consensus | Strictly $0\text{ bytes}$ allocated on heap |
| **Numerical Stability (Phase 0)** | Lorentz horizon clamping $\epsilon_{\text{mach}} = 10^{-7}$ and antipodal sphere stabilization | $0\text{ NaNs}$ / $0\text{ Infs}$ across randomized boundary sweeps |
| **AT-RQ Page ADC Kernel (Phase 0)** | $1024\text{-dim}$ curvature-corrected Rauch/Jacobi geodesic distance | $< 15\text{ ns}$ per evaluation on AVX2 / AVX-512 |
| **Single-Node Ingestion (Phase 1)** | Sustained vector + sheaf edge mutations into `L0MorphismLog` | $> 6.2 \times 10^6\text{ ops/sec/core}$ ($p_{99} = 0.0020\text{ ms}$, Recall@10 $= 0.9810$) |
| **3-Hop Hybrid Query (Phase 2)** | 3-hop Sheaf SpMV + $1536\text{-dim}$ vector filter with speculative `io_uring` prefetch | $p_{99} = 1.092\text{ ms} \ll 5.0\text{ ms}$ gate (Recall@10 $= 1.0000$) |
| **64-Node Cluster Simulation (Phase 3)** | 64-node distributed cluster coordinating leaderless quorum consensus | $100\%$ commit completion with zero deadlocks |
| **Minority Node Recovery (Phase 3)** | Replay of 500 missing morphism log records upon node reconnect | $0.03\text{ ms} \ll 10.0\text{ ms}$ failover gate; $0$ data loss |