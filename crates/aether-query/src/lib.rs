//! Project Aether-Manifold: Cellular Sheaf Engine & Hybrid Query Compiler (Phase 2)
//!
//! Provides the core query and topological kernel:
//! - Cellular Sheaf Stalks & Orthogonal Restriction Morphisms (`sheaf`)
//! - Discrete Sheaf Coboundary & Block Sheaf Laplacian (`laplacian`)
//! - Sheaf Dirichlet Energy Solver (`dirichlet`)
//! - Semiring Algebraic Framework (`semiring`)
//! - Semiring-Sheaf Algebraic Query Compiler (`compiler`)
//! - Speculative Lookahead Prefetch Pipeline (`speculative_prefetch`)

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod aql;
pub mod compiler;
pub mod dirichlet;
pub mod laplacian;
pub mod semiring;
pub mod sheaf;
pub mod speculative_prefetch;

// Re-exports for convenient high-level usage
pub use aql::{parse_aql, AqlCompileHelper, AqlStatement, SemiringKind};
pub use compiler::{FrontierNode, QueryPlan, SemiringSpMvEngine, MAX_QUERY_FRONTIER};
pub use dirichlet::{ClusterEdgeDescriptor, DirichletSolverConfig, SheafDirichletSolver};
pub use laplacian::{BlockSheafLaplacian, SheafCoboundary};
pub use semiring::{
    semiring_eval, semiring_one, semiring_zero, BottleneckSemiring, DiffusionSemiring,
    SemiringAlgebra, SemiringMode, TropicalSemiring,
};
pub use sheaf::{
    pack_page_sheaf_morphisms, unpack_page_sheaf_morphism, EdgeStalk, SheafRestrictionMap,
    VertexStalk, DEFAULT_STALK_DIM, MAX_NEIGHBORS_PER_PAGE, PACKED_RESTRICTION_BYTES,
};
pub use speculative_prefetch::{PrefetchReport, SpeculativeLookaheadPipeline};
