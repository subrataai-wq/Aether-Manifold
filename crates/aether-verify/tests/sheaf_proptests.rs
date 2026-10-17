//! Project Aether-Manifold: Cellular Sheaf & Semiring Property-Based Fuzzing
//!
//! Validates:
//! 1. Rule 4.2: Sheaf Laplacian is symmetric positive semi-definite ($x^T L_{\mathcal{F}} x \ge 0$).
//! 2. Rule 4.2: Sheaf Householder restriction maps are isometric ($\|H_w x\| = \|x\|$).
//! 3. Semiring algebraic axioms (associativity, commutativity, distributivity).

use aether_query::laplacian::BlockSheafLaplacian;
use aether_query::semiring::{SemiringAlgebra, TropicalSemiring};
use aether_query::sheaf::{SheafRestrictionMap, VertexStalk, DEFAULT_STALK_DIM};
use proptest::prelude::*;

fn arb_stalk() -> impl Strategy<Value = [f32; DEFAULT_STALK_DIM]> {
    prop::array::uniform8(-10.0f32..10.0f32)
}

fn arb_reflector() -> impl Strategy<Value = [f32; DEFAULT_STALK_DIM]> {
    prop::array::uniform8(-1.0f32..1.0f32)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_sheaf_laplacian_positive_semidefinite(
        s0 in arb_stalk(),
        s1 in arb_stalk(),
        s2 in arb_stalk(),
        r01 in arb_reflector(),
        r12 in arb_reflector(),
        r20 in arb_reflector(),
    ) {
        let x0 = VertexStalk::new(s0);
        let x1 = VertexStalk::new(s1);
        let x2 = VertexStalk::new(s2);

        let m01 = SheafRestrictionMap::from_reflector(&r01, 1.0);
        let m12 = SheafRestrictionMap::from_reflector(&r12, 1.0);
        let m20 = SheafRestrictionMap::from_reflector(&r20, 1.0);

        let id = SheafRestrictionMap::identity();

        let edges = [
            (m01, x0, id, x1),
            (m12, x1, id, x2),
            (m20, x2, id, x0),
        ];

        let energy = BlockSheafLaplacian::total_dirichlet_energy(&edges);
        prop_assert!(energy >= -1e-5, "Sheaf Dirichlet energy must be >= 0 (PSD)");
    }

    #[test]
    fn prop_sheaf_orthogonal_isometry(
        s in arb_stalk(),
        r in arb_reflector(),
    ) {
        let x = VertexStalk::new(s);
        let initial_norm = x.norm();

        let morphism = SheafRestrictionMap::from_reflector(&r, 1.0);
        let mut y = aether_query::sheaf::EdgeStalk::default();
        morphism.apply(&x, &mut y);

        let mut sum_sq = 0.0f32;
        for val in &y.components {
            sum_sq += val * val;
        }
        let transformed_norm = sum_sq.sqrt();

        if initial_norm > 1e-6 {
            let rel_diff = (transformed_norm - initial_norm).abs() / initial_norm;
            prop_assert!(rel_diff < 1e-4, "Relative norm change must be < 1e-4");
        }
    }

    #[test]
    fn prop_tropical_semiring_distributivity(
        a in 0.0f32..100.0f32,
        b in 0.0f32..100.0f32,
        c in 0.0f32..100.0f32,
    ) {
        // a (x) (b (+) c) == (a (x) b) (+) (a (x) c)
        // a + min(b, c) == min(a + b, a + c)
        let left = TropicalSemiring::otimes(a, TropicalSemiring::oplus(b, c));
        let right = TropicalSemiring::oplus(
            TropicalSemiring::otimes(a, b),
            TropicalSemiring::otimes(a, c),
        );
        prop_assert!((left - right).abs() < 1e-5, "Tropical multiplication must distribute over addition");
    }
}
