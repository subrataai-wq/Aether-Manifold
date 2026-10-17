//! Project Aether-Manifold: Semiring Algebraic Framework
//!
//! Implements abstract semiring algebras $(S, \oplus, \otimes, \mathbb{0}, \mathbb{1})$
//! for unified SpMV query compilation:
//! - Tropical Semiring $(\min, +)$ for Geodesic $A^*$ shortest paths
//! - Diffusion Semiring $(+, \times)$ for Sheaf Dirichlet Diffusion & Personalized PageRank
//! - Bottleneck Semiring $(\max, \min)$ for maximum-capacity threshold paths

/// Core algebraic Semiring trait $(S, \oplus, \otimes, \mathbb{0}, \mathbb{1})$.
pub trait SemiringAlgebra: Copy + Clone + Send + Sync + 'static {
    /// Scalar element type.
    type Elem: Copy + Clone + PartialEq + PartialOrd;

    /// Additive identity $\mathbb{0}$ satisfying $x \oplus \mathbb{0} = x$.
    fn zero() -> Self::Elem;

    /// Multiplicative identity $\mathbb{1}$ satisfying $x \otimes \mathbb{1} = x$.
    fn one() -> Self::Elem;

    /// Additive aggregation operation $\oplus$.
    fn oplus(a: Self::Elem, b: Self::Elem) -> Self::Elem;

    /// Multiplicative combination operation $\otimes$.
    fn otimes(a: Self::Elem, b: Self::Elem) -> Self::Elem;
}

/// Tropical Semiring $(\min, +)$ with $\mathbb{0} = +\infty$ and $\mathbb{1} = 0$.
///
/// Models shortest-path geodesic traversal (Geodesic $A^*$).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TropicalSemiring;

impl SemiringAlgebra for TropicalSemiring {
    type Elem = f32;

    #[inline]
    fn zero() -> Self::Elem {
        f32::INFINITY
    }

    #[inline]
    fn one() -> Self::Elem {
        0.0f32
    }

    #[inline]
    fn oplus(a: Self::Elem, b: Self::Elem) -> Self::Elem {
        a.min(b)
    }

    #[inline]
    fn otimes(a: Self::Elem, b: Self::Elem) -> Self::Elem {
        if a == f32::INFINITY || b == f32::INFINITY {
            f32::INFINITY
        } else {
            a + b
        }
    }
}

/// Diffusion Semiring $(+, \times)$ with $\mathbb{0} = 0$ and $\mathbb{1} = 1$.
///
/// Models continuous probability / heat flow / semantic Personalized PageRank.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct DiffusionSemiring;

impl SemiringAlgebra for DiffusionSemiring {
    type Elem = f32;

    #[inline]
    fn zero() -> Self::Elem {
        0.0f32
    }

    #[inline]
    fn one() -> Self::Elem {
        1.0f32
    }

    #[inline]
    fn oplus(a: Self::Elem, b: Self::Elem) -> Self::Elem {
        a + b
    }

    #[inline]
    fn otimes(a: Self::Elem, b: Self::Elem) -> Self::Elem {
        a * b
    }
}

/// Bottleneck Semiring $(\max, \min)$ with $\mathbb{0} = 0$ and $\mathbb{1} = +\infty$.
///
/// Models maximum capacity / bottleneck routing under threshold constraints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct BottleneckSemiring;

impl SemiringAlgebra for BottleneckSemiring {
    type Elem = f32;

    #[inline]
    fn zero() -> Self::Elem {
        0.0f32
    }

    #[inline]
    fn one() -> Self::Elem {
        f32::INFINITY
    }

    #[inline]
    fn oplus(a: Self::Elem, b: Self::Elem) -> Self::Elem {
        a.max(b)
    }

    #[inline]
    fn otimes(a: Self::Elem, b: Self::Elem) -> Self::Elem {
        a.min(b)
    }
}

/// Enumeration of runtime-selectable semiring operational modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemiringMode {
    /// Tropical $(\min, +)$ for shortest paths
    Tropical,
    /// Diffusion $(+, \times)$ for heat / PageRank
    Diffusion,
    /// Bottleneck $(\max, \min)$ for capacity routing
    Bottleneck,
}

/// Dynamic semiring dispatch helper executing $a \oplus b$ and $a \otimes b$.
#[inline]
pub fn semiring_eval(mode: SemiringMode, a: f32, b: f32, is_mult: bool) -> f32 {
    match mode {
        SemiringMode::Tropical => {
            if is_mult {
                TropicalSemiring::otimes(a, b)
            } else {
                TropicalSemiring::oplus(a, b)
            }
        }
        SemiringMode::Diffusion => {
            if is_mult {
                DiffusionSemiring::otimes(a, b)
            } else {
                DiffusionSemiring::oplus(a, b)
            }
        }
        SemiringMode::Bottleneck => {
            if is_mult {
                BottleneckSemiring::otimes(a, b)
            } else {
                BottleneckSemiring::oplus(a, b)
            }
        }
    }
}

/// Returns the additive identity $\mathbb{0}$ for the chosen mode.
#[inline]
pub fn semiring_zero(mode: SemiringMode) -> f32 {
    match mode {
        SemiringMode::Tropical => TropicalSemiring::zero(),
        SemiringMode::Diffusion => DiffusionSemiring::zero(),
        SemiringMode::Bottleneck => BottleneckSemiring::zero(),
    }
}

/// Returns the multiplicative identity $\mathbb{1}$ for the chosen mode.
#[inline]
pub fn semiring_one(mode: SemiringMode) -> f32 {
    match mode {
        SemiringMode::Tropical => TropicalSemiring::one(),
        SemiringMode::Diffusion => DiffusionSemiring::one(),
        SemiringMode::Bottleneck => BottleneckSemiring::one(),
    }
}
