//! Project Aether-Manifold: SIMD-Accelerated Riemannian Product Manifold Engine
//!
//! Provides bare-metal SIMD kernels across Lorentz hyperbolic space $\mathbb{H}_{\kappa}^{d_H}$,
//! hyperspherical space $\mathbb{S}_{\tau}^{d_S}$, and flat Euclidean space $\mathbb{R}^{d_E}$,
//! unified into the Riemannian Product Manifold metric:
//!
//! $$\mathcal{M} = \bigotimes_{i=1}^{k} \mathbb{H}_{\kappa_i}^{d_{H,i}} \times \bigotimes_{j=1}^{m} \mathbb{S}_{\tau_j}^{d_{S,j}} \times \mathbb{R}^{d_E}$$
//! $$d_{\mathcal{M}}^2(u, v) = \sum_{i=1}^{k} w_{H,i} d_{\mathbb{H}_{\kappa_i}}^2(u, v) + \sum_{j=1}^{m} w_{S,j} d_{\mathbb{S}_{\tau_j}}^2(u, v) + w_E \|u - v\|_2^2$$
//!
//! Governed by Agent-Riemann & Agent-Hopper in strict compliance with AGENTS.md:
//! - Rule 2.1: 4x register accumulator unrolling to hide FMA latency.
//! - Rule 3.1: Strictly 0 heap allocations on the hot path.
//! - Rule 4.1: Machine epsilon horizon projection clamp $z = \max(1 + \epsilon_{\text{mach}}, \kappa \langle x, y \rangle_{\mathcal{L}})$.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod euclidean;
pub mod hyperbolic;
pub mod product;
pub mod simd;
pub mod spherical;

pub use euclidean::{
    euclidean_distance, euclidean_distance_squared, euclidean_dot_product,
    scalar_euclidean_distance_checked, scalar_euclidean_distance_squared_checked,
    scalar_euclidean_dot_product_checked,
};
pub use hyperbolic::{
    lorentz_compute_x0, lorentz_distance, lorentz_distance_squared, lorentz_inner_product,
    lorentz_reproject, scalar_lorentz_compute_x0, scalar_lorentz_distance,
    scalar_lorentz_inner_product, HORIZON_EPSILON,
};
pub use product::{
    product_manifold_distance, product_manifold_distance_squared,
    scalar_product_manifold_distance, scalar_product_manifold_distance_squared, EuclideanSpec,
    HyperbolicSpec, ProductManifoldConfig, SphericalSpec,
};
pub use simd::{
    scalar_dot_product, scalar_euclidean_distance_squared, scalar_sum_of_squares,
    simd_dot_product, simd_euclidean_distance_squared, simd_sum_of_squares,
};
pub use spherical::{
    scalar_spherical_distance, scalar_spherical_normalize, spherical_distance,
    spherical_distance_squared, spherical_normalize,
};
