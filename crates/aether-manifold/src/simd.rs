//! Project Aether-Manifold: SIMD Hardware Intrinsics & 4x Unrolled Kernels
//!
//! Provides runtime-dispatched SIMD kernels using `pulp` across AVX-512, AVX2, and NEON,
//! unrolling 4x register accumulators to hide FMA pipeline latency in strict compliance
//! with AGENTS.md Rule 2.1 and Rule 3.1 (Zero Hot-Path Allocations).

use pulp::{Arch, Simd, WithSimd};

/// SIMD dot product kernel with 4x unrolled register accumulators.
struct DotProductKernel<'a> {
    a: &'a [f32],
    b: &'a [f32],
}

impl<'a> WithSimd for DotProductKernel<'a> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (head_a, tail_a) = S::as_simd_f32s(self.a);
        let (head_b, tail_b) = S::as_simd_f32s(self.b);

        if head_a.len() == head_b.len() && !head_a.is_empty() {
            let mut acc0 = simd.splat_f32s(0.0);
            let mut acc1 = simd.splat_f32s(0.0);
            let mut acc2 = simd.splat_f32s(0.0);
            let mut acc3 = simd.splat_f32s(0.0);

            let mut i = 0;
            let len = head_a.len();
            while i + 4 <= len {
                if let (
                    Some(&a0),
                    Some(&b0),
                    Some(&a1),
                    Some(&b1),
                    Some(&a2),
                    Some(&b2),
                    Some(&a3),
                    Some(&b3),
                ) = (
                    head_a.get(i),
                    head_b.get(i),
                    head_a.get(i + 1),
                    head_b.get(i + 1),
                    head_a.get(i + 2),
                    head_b.get(i + 2),
                    head_a.get(i + 3),
                    head_b.get(i + 3),
                ) {
                    acc0 = simd.mul_add_f32s(a0, b0, acc0);
                    acc1 = simd.mul_add_f32s(a1, b1, acc1);
                    acc2 = simd.mul_add_f32s(a2, b2, acc2);
                    acc3 = simd.mul_add_f32s(a3, b3, acc3);
                }
                i += 4;
            }
            while i < len {
                if let (Some(&a_val), Some(&b_val)) = (head_a.get(i), head_b.get(i)) {
                    acc0 = simd.mul_add_f32s(a_val, b_val, acc0);
                }
                i += 1;
            }
            let acc = simd.add_f32s(simd.add_f32s(acc0, acc1), simd.add_f32s(acc2, acc3));
            let mut total = simd.reduce_sum_f32s(acc);
            for (&x, &y) in tail_a.iter().zip(tail_b.iter()) {
                total += x * y;
            }
            total
        } else {
            let (a_chunks, a_rem) = pulp::as_arrays::<4, f32>(self.a);
            let (b_chunks, b_rem) = pulp::as_arrays::<4, f32>(self.b);

            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut acc3 = 0.0f32;

            for (chunk_a, chunk_b) in a_chunks.iter().zip(b_chunks.iter()) {
                acc0 += chunk_a[0] * chunk_b[0];
                acc1 += chunk_a[1] * chunk_b[1];
                acc2 += chunk_a[2] * chunk_b[2];
                acc3 += chunk_a[3] * chunk_b[3];
            }

            let mut total = (acc0 + acc1) + (acc2 + acc3);
            for (&x, &y) in a_rem.iter().zip(b_rem.iter()) {
                total += x * y;
            }
            total
        }
    }
}

/// Compute inner product between two float slices using runtime SIMD dispatch with 4x unrolling.
#[inline]
pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let arch = Arch::new();
    arch.dispatch(DotProductKernel { a, b })
}

/// SIMD squared Euclidean distance kernel: $\sum (a_i - b_i)^2$.
struct EuclideanSqKernel<'a> {
    a: &'a [f32],
    b: &'a [f32],
}

impl<'a> WithSimd for EuclideanSqKernel<'a> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (head_a, tail_a) = S::as_simd_f32s(self.a);
        let (head_b, tail_b) = S::as_simd_f32s(self.b);

        if head_a.len() == head_b.len() && !head_a.is_empty() {
            let mut acc0 = simd.splat_f32s(0.0);
            let mut acc1 = simd.splat_f32s(0.0);
            let mut acc2 = simd.splat_f32s(0.0);
            let mut acc3 = simd.splat_f32s(0.0);

            let mut i = 0;
            let len = head_a.len();
            while i + 4 <= len {
                if let (
                    Some(&a0),
                    Some(&b0),
                    Some(&a1),
                    Some(&b1),
                    Some(&a2),
                    Some(&b2),
                    Some(&a3),
                    Some(&b3),
                ) = (
                    head_a.get(i),
                    head_b.get(i),
                    head_a.get(i + 1),
                    head_b.get(i + 1),
                    head_a.get(i + 2),
                    head_b.get(i + 2),
                    head_a.get(i + 3),
                    head_b.get(i + 3),
                ) {
                    let d0 = simd.sub_f32s(a0, b0);
                    let d1 = simd.sub_f32s(a1, b1);
                    let d2 = simd.sub_f32s(a2, b2);
                    let d3 = simd.sub_f32s(a3, b3);

                    acc0 = simd.mul_add_f32s(d0, d0, acc0);
                    acc1 = simd.mul_add_f32s(d1, d1, acc1);
                    acc2 = simd.mul_add_f32s(d2, d2, acc2);
                    acc3 = simd.mul_add_f32s(d3, d3, acc3);
                }
                i += 4;
            }
            while i < len {
                if let (Some(&a_val), Some(&b_val)) = (head_a.get(i), head_b.get(i)) {
                    let diff = simd.sub_f32s(a_val, b_val);
                    acc0 = simd.mul_add_f32s(diff, diff, acc0);
                }
                i += 1;
            }
            let acc = simd.add_f32s(simd.add_f32s(acc0, acc1), simd.add_f32s(acc2, acc3));
            let mut total = simd.reduce_sum_f32s(acc);
            for (&x, &y) in tail_a.iter().zip(tail_b.iter()) {
                let diff = x - y;
                total += diff * diff;
            }
            total
        } else {
            let (a_chunks, a_rem) = pulp::as_arrays::<4, f32>(self.a);
            let (b_chunks, b_rem) = pulp::as_arrays::<4, f32>(self.b);

            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut acc3 = 0.0f32;

            for (chunk_a, chunk_b) in a_chunks.iter().zip(b_chunks.iter()) {
                let d0 = chunk_a[0] - chunk_b[0];
                let d1 = chunk_a[1] - chunk_b[1];
                let d2 = chunk_a[2] - chunk_b[2];
                let d3 = chunk_a[3] - chunk_b[3];

                acc0 += d0 * d0;
                acc1 += d1 * d1;
                acc2 += d2 * d2;
                acc3 += d3 * d3;
            }

            let mut total = (acc0 + acc1) + (acc2 + acc3);
            for (&x, &y) in a_rem.iter().zip(b_rem.iter()) {
                let diff = x - y;
                total += diff * diff;
            }
            total
        }
    }
}

/// Compute squared Euclidean distance between two float slices using runtime SIMD dispatch with 4x unrolling.
#[inline]
pub fn simd_euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    let arch = Arch::new();
    arch.dispatch(EuclideanSqKernel { a, b })
}

/// SIMD sum of squares kernel: $\sum a_i^2$.
struct SumOfSquaresKernel<'a> {
    a: &'a [f32],
}

impl<'a> WithSimd for SumOfSquaresKernel<'a> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (head, tail) = S::as_simd_f32s(self.a);

        if !head.is_empty() {
            let mut acc0 = simd.splat_f32s(0.0);
            let mut acc1 = simd.splat_f32s(0.0);
            let mut acc2 = simd.splat_f32s(0.0);
            let mut acc3 = simd.splat_f32s(0.0);

            let mut i = 0;
            let len = head.len();
            while i + 4 <= len {
                if let (Some(&a0), Some(&a1), Some(&a2), Some(&a3)) = (
                    head.get(i),
                    head.get(i + 1),
                    head.get(i + 2),
                    head.get(i + 3),
                ) {
                    acc0 = simd.mul_add_f32s(a0, a0, acc0);
                    acc1 = simd.mul_add_f32s(a1, a1, acc1);
                    acc2 = simd.mul_add_f32s(a2, a2, acc2);
                    acc3 = simd.mul_add_f32s(a3, a3, acc3);
                }
                i += 4;
            }
            while i < len {
                if let Some(&a_val) = head.get(i) {
                    acc0 = simd.mul_add_f32s(a_val, a_val, acc0);
                }
                i += 1;
            }
            let acc = simd.add_f32s(simd.add_f32s(acc0, acc1), simd.add_f32s(acc2, acc3));
            let mut total = simd.reduce_sum_f32s(acc);
            for &x in tail {
                total += x * x;
            }
            total
        } else {
            let (chunks, rem) = pulp::as_arrays::<4, f32>(self.a);

            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut acc3 = 0.0f32;

            for chunk in chunks {
                acc0 += chunk[0] * chunk[0];
                acc1 += chunk[1] * chunk[1];
                acc2 += chunk[2] * chunk[2];
                acc3 += chunk[3] * chunk[3];
            }

            let mut total = (acc0 + acc1) + (acc2 + acc3);
            for &x in rem {
                total += x * x;
            }
            total
        }
    }
}

/// Compute sum of squares of a float slice using runtime SIMD dispatch with 4x unrolling.
#[inline]
pub fn simd_sum_of_squares(a: &[f32]) -> f32 {
    let arch = Arch::new();
    arch.dispatch(SumOfSquaresKernel { a })
}

/// SIMD AXPY kernel: $y \leftarrow y + \alpha x$ with 4x unrolled register operations.
struct AxpyKernel<'a> {
    alpha: f32,
    x: &'a [f32],
    y: &'a mut [f32],
}

impl<'a> WithSimd for AxpyKernel<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (head_x, tail_x) = S::as_simd_f32s(self.x);
        let (head_y, tail_y) = S::as_mut_simd_f32s(self.y);

        if head_x.len() == head_y.len() && !head_x.is_empty() {
            let alpha_simd = simd.splat_f32s(self.alpha);
            let (x_quads, x_rem) = pulp::as_arrays::<4, S::f32s>(head_x);
            let (y_quads, y_rem) = pulp::as_arrays_mut::<4, S::f32s>(head_y);

            for (qx, qy) in x_quads.iter().zip(y_quads.iter_mut()) {
                qy[0] = simd.mul_add_f32s(alpha_simd, qx[0], qy[0]);
                qy[1] = simd.mul_add_f32s(alpha_simd, qx[1], qy[1]);
                qy[2] = simd.mul_add_f32s(alpha_simd, qx[2], qy[2]);
                qy[3] = simd.mul_add_f32s(alpha_simd, qx[3], qy[3]);
            }
            for (&rx, ry) in x_rem.iter().zip(y_rem.iter_mut()) {
                *ry = simd.mul_add_f32s(alpha_simd, rx, *ry);
            }
            for (&x_elem, y_elem) in tail_x.iter().zip(tail_y.iter_mut()) {
                *y_elem += self.alpha * x_elem;
            }
        } else {
            let (x_chunks, x_rem) = pulp::as_arrays::<4, f32>(self.x);
            let (y_chunks, y_rem) = pulp::as_arrays_mut::<4, f32>(self.y);

            for (cx, cy) in x_chunks.iter().zip(y_chunks.iter_mut()) {
                cy[0] += self.alpha * cx[0];
                cy[1] += self.alpha * cx[1];
                cy[2] += self.alpha * cx[2];
                cy[3] += self.alpha * cx[3];
            }
            for (&x_elem, y_elem) in x_rem.iter().zip(y_rem.iter_mut()) {
                *y_elem += self.alpha * x_elem;
            }
        }
    }
}

/// Compute in-place AXPY $y \leftarrow y + \alpha x$ using runtime SIMD dispatch with 4x unrolling.
#[inline]
pub fn simd_axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    let len = x.len().min(y.len());
    let x = match x.get(..len) {
        Some(s) => s,
        None => return,
    };
    let y = match y.get_mut(..len) {
        Some(s) => s,
        None => return,
    };
    let arch = Arch::new();
    arch.dispatch(AxpyKernel { alpha, x, y });
}

/// Pure scalar reference implementation of dot product for verification.
#[inline]
pub fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

/// Pure scalar reference implementation of squared Euclidean distance for verification.
#[inline]
pub fn scalar_euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}

/// Pure scalar reference implementation of sum of squares for verification.
#[inline]
pub fn scalar_sum_of_squares(a: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in a {
        sum += x * x;
    }
    sum
}
