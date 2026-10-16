//! Project Aether-Manifold: Vectorized Sinkhorn-Knopp Optimal Transport
//!
//! Implements Entropy-Regularized Wasserstein Distance $\mathcal{W}_{\varepsilon}(A, B)$
//! between multi-vector bags and local ego-graphs using $O(k \cdot n \cdot m)$ vectorized
//! fused-multiply-add (FMA) matrix scaling operations, in strict compliance with
//! roadmap.md Section 2.3, AGENTS.md Rule 2.1, Rule 3.1 (Zero Hot-Path Allocations),
//! and test_suite_manifest.md Section 6 (Category V).
//!
//! Mathematical formulation:
//! $$\mathcal{W}_{\varepsilon}(A, B) = \min_{P \in U(\mu, \nu)} \left\{ \sum_{i=1}^{n} \sum_{j=1}^{m} P_{ij} d_{\mathcal{M}}^2(a_i, b_j) + \varepsilon H(P) \right\}$$
//! Where $U(\mu, \nu) = \{ P \in \mathbb{R}_{+}^{n \times m} \mid P \mathbf{1}_m = \mu, \, P^T \mathbf{1}_n = \nu \}$
//! and $H(P) = \sum_{i,j} P_{ij} (\log P_{ij} - 1)$.
//!
//! The unique optimal transport plan is:
//! $$P_{ij} = u_i K_{ij} v_j, \quad \text{where } K_{ij} = \exp\left( -\frac{C_{ij}}{\varepsilon} \right)$$
//! Solved via alternating Sinkhorn projections:
//! $$u^{(t+1)} = \frac{\mu}{K v^{(t)}}, \quad v^{(t+1)} = \frac{\nu}{K^T u^{(t+1)}}$$
//!
//! Debiased Sinkhorn Divergence:
//! $$S_{\varepsilon}(A, B) = \mathcal{W}_{\varepsilon}(A, B) - \frac{1}{2}\mathcal{W}_{\varepsilon}(A, A) - \frac{1}{2}\mathcal{W}_{\varepsilon}(B, B)$$

extern crate alloc;
use alloc::vec;

use crate::simd::{simd_axpy, simd_dot_product, simd_euclidean_distance_squared};
use aether_core::error::AetherError;

/// Threshold below which log-domain stabilization is automatically engaged
/// to eliminate floating-point underflow or overflow.
pub const LOG_DOMAIN_EPSILON_THRESHOLD: f32 = 0.02;

/// Configuration for the Sinkhorn-Knopp Optimal Transport solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SinkhornConfig {
    /// Entropic regularization parameter $\varepsilon > 0$.
    pub epsilon: f32,
    /// Maximum number of scaling iterations $k$.
    pub max_iter: usize,
    /// Marginal error convergence tolerance $\tau$ (e.g. $10^{-5}$).
    pub tolerance: f32,
}

impl Default for SinkhornConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            max_iter: 100,
            tolerance: 1e-5,
        }
    }
}

/// Result of the Sinkhorn-Knopp Optimal Transport evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SinkhornResult {
    /// Un-regularized Wasserstein transport cost $\langle P, C \rangle$.
    pub cost: f32,
    /// Entropy-regularized Wasserstein distance $\mathcal{W}_{\varepsilon}(A, B)$.
    pub regularized_cost: f32,
    /// Number of scaling iterations executed.
    pub iterations: usize,
    /// Whether the scaling converged within `tolerance`.
    pub converged: bool,
    /// L1 marginal discrepancy on source: $\|P \mathbf{1}_m - \mu\|_1$.
    pub marginal_error_u: f32,
    /// L1 marginal discrepancy on target: $\|P^T \mathbf{1}_n - \nu\|_1$.
    pub marginal_error_v: f32,
}

/// Calculate the number of `f32` elements required in the pre-allocated workspace
/// for evaluating Sinkhorn optimal transport between sets of sizes $n$ and $m$.
#[inline]
pub const fn sinkhorn_workspace_size(n: usize, m: usize) -> usize {
    // cost matrix C (n * m) + kernel K (n * m) + u (n) + v (m) + Kv (n) + KTu (m)
    2 * (n * m) + 2 * n + 2 * m
}

/// Compute Entropy-Regularized Wasserstein Distance $\mathcal{W}_{\varepsilon}(A, B)$
/// using a pre-allocated scratchpad workspace to strictly enforce AGENTS.md Rule 3.1
/// (Zero Hot-Path Heap Allocations).
///
/// Parameters:
/// - `a_bag`: Flat slice of $n$ vectors each of dimension $d$ ($n \times d$ floats).
/// - `n`: Number of vectors in bag $A$.
/// - `b_bag`: Flat slice of $m$ vectors each of dimension $d$ ($m \times d$ floats).
/// - `m`: Number of vectors in bag $B$.
/// - `dim`: Feature dimension $d$ of each vector.
/// - `config`: Entropic regularization $\varepsilon$, max iterations, and tolerance.
/// - `workspace`: Pre-allocated slice of at least `sinkhorn_workspace_size(n, m)` floats.
pub fn sinkhorn_knopp_with_workspace(
    a_bag: &[f32],
    n: usize,
    b_bag: &[f32],
    m: usize,
    dim: usize,
    config: &SinkhornConfig,
    workspace: &mut [f32],
) -> Result<SinkhornResult, AetherError> {
    if config.epsilon <= 0.0 {
        return Err(AetherError::LorentzHorizonViolation);
    }
    if n == 0 || m == 0 || dim == 0 {
        return Err(AetherError::InvalidBufferSize {
            actual: 0,
            expected: 1,
        });
    }
    if a_bag.len() != n * dim {
        return Err(AetherError::InvalidBufferSize {
            actual: a_bag.len(),
            expected: n * dim,
        });
    }
    if b_bag.len() != m * dim {
        return Err(AetherError::InvalidBufferSize {
            actual: b_bag.len(),
            expected: m * dim,
        });
    }

    let required_len = sinkhorn_workspace_size(n, m);
    if workspace.len() < required_len {
        return Err(AetherError::ScratchpadExhausted {
            requested: required_len * core::mem::size_of::<f32>(),
            available: core::mem::size_of_val(workspace),
        });
    }

    // Partition pre-allocated workspace without heap allocations:
    // [0 .. nm]: Cost matrix C
    // [nm .. 2nm]: Gibbs Kernel K / log-domain work
    // [2nm .. 2nm + n]: u (dual variable)
    // [2nm + n .. 2nm + n + m]: v (dual variable)
    // [2nm + n + m .. 2nm + 2n + m]: kv_buf
    // [2nm + 2n + m .. 2nm + 2n + 2m]: ktu_buf
    let nm = n * m;
    let (c_mat, rest) = workspace.split_at_mut(nm);
    let (k_mat, rest) = rest.split_at_mut(nm);
    let (u_vec, rest) = rest.split_at_mut(n);
    let (v_vec, rest) = rest.split_at_mut(m);
    let (kv_buf, rest) = rest.split_at_mut(n);
    let (ktu_buf, _unused) = rest.split_at_mut(m);

    // 1. Compute Cost Matrix C_ij = ||a_i - b_j||_2^2 using 4x unrolled SIMD
    let mut min_cost = f32::INFINITY;
    let mut max_cost = 0.0f32;
    for i in 0..n {
        let a_vec = match a_bag.get(i * dim..(i + 1) * dim) {
            Some(s) => s,
            None => {
                return Err(AetherError::OutOfBounds {
                    index: (i + 1) * dim,
                    limit: a_bag.len(),
                })
            }
        };
        for j in 0..m {
            let b_vec = match b_bag.get(j * dim..(j + 1) * dim) {
                Some(s) => s,
                None => {
                    return Err(AetherError::OutOfBounds {
                        index: (j + 1) * dim,
                        limit: b_bag.len(),
                    })
                }
            };
            let cost = simd_euclidean_distance_squared(a_vec, b_vec);
            if cost > max_cost {
                max_cost = cost;
            }
            if cost < min_cost {
                min_cost = cost;
            }
            if let Some(c_ref) = c_mat.get_mut(i * m + j) {
                *c_ref = cost;
            }
        }
    }

    // Check if small epsilon requires log-domain stabilization to prevent underflow
    if config.epsilon < LOG_DOMAIN_EPSILON_THRESHOLD {
        return sinkhorn_log_domain_internal(
            c_mat, k_mat, u_vec, v_vec, kv_buf, ktu_buf, n, m, config, min_cost,
        );
    }

    // 2. Standard Kernel-Scaling Sinkhorn with shift invariance:
    // K_ij = exp(-(C_ij - min_cost) / epsilon) in [exp(-spread/eps), 1.0]
    let inv_eps = 1.0 / config.epsilon;
    for (c, k) in c_mat.iter().zip(k_mat.iter_mut()) {
        *k = libm::expf(-(c - min_cost) * inv_eps);
    }

    // Target uniform marginals: mu_i = 1/n, nu_j = 1/m
    let mu_i = 1.0f32 / (n as f32);
    let nu_j = 1.0f32 / (m as f32);

    // Initialize v to target uniform marginal nu_j
    for v in v_vec.iter_mut() {
        *v = nu_j;
    }

    let mut iterations = 0;
    let mut converged = false;
    let mut err_u = 1.0f32;
    let mut err_v = 1.0f32;

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        // Step 1: Compute (K v)_i = sum_j K_ij * v_j via 4x unrolled SIMD dot product
        let mut current_err_u = 0.0f32;
        for i in 0..n {
            let k_row = match k_mat.get(i * m..(i + 1) * m) {
                Some(r) => r,
                None => {
                    return Err(AetherError::OutOfBounds {
                        index: (i + 1) * m,
                        limit: k_mat.len(),
                    })
                }
            };
            let kv = simd_dot_product(k_row, v_vec);
            if let Some(kv_ref) = kv_buf.get_mut(i) {
                *kv_ref = kv;
            }
            if iter > 0 {
                if let Some(&u_val) = u_vec.get(i) {
                    let row_sum = u_val * kv;
                    current_err_u += (row_sum - mu_i).abs();
                }
            }
        }

        // Update u_i = mu_i / (K v)_i
        for i in 0..n {
            if let (Some(u_ref), Some(&kv)) = (u_vec.get_mut(i), kv_buf.get(i)) {
                *u_ref = if kv > f32::MIN_POSITIVE {
                    mu_i / kv
                } else {
                    0.0
                };
            }
        }

        // Step 2: Compute (K^T u)_j = sum_i K_ij * u_i via 4x unrolled SIMD AXPY
        for ktu in ktu_buf.iter_mut() {
            *ktu = 0.0;
        }
        for i in 0..n {
            let u_val = match u_vec.get(i) {
                Some(&u) => u,
                None => 0.0,
            };
            let k_row = match k_mat.get(i * m..(i + 1) * m) {
                Some(r) => r,
                None => {
                    return Err(AetherError::OutOfBounds {
                        index: (i + 1) * m,
                        limit: k_mat.len(),
                    })
                }
            };
            simd_axpy(u_val, k_row, ktu_buf);
        }

        // Compute column marginal error ||P^T 1_n - nu||_1 and update v_j
        let mut current_err_v = 0.0f32;
        for (v_ref, &sum) in v_vec.iter_mut().zip(ktu_buf.iter()) {
            let col_sum = *v_ref * sum;
            current_err_v += (col_sum - nu_j).abs();
            *v_ref = if sum > f32::MIN_POSITIVE {
                nu_j / sum
            } else {
                0.0
            };
        }

        if iter > 0 {
            err_u = current_err_u;
            err_v = current_err_v;
            let eff_tol = config.tolerance + 1e-6;
            if err_u <= eff_tol && err_v <= eff_tol {
                converged = true;
                break;
            }
        } else {
            err_v = current_err_v;
        }
    }

    if !converged {
        // Evaluate the marginal error of the final (u_vec, v_vec) state
        let mut final_err_u = 0.0f32;
        for i in 0..n {
            let k_row = match k_mat.get(i * m..(i + 1) * m) {
                Some(r) => r,
                None => {
                    return Err(AetherError::OutOfBounds {
                        index: (i + 1) * m,
                        limit: k_mat.len(),
                    })
                }
            };
            let kv = simd_dot_product(k_row, v_vec);
            if let Some(&u_val) = u_vec.get(i) {
                let row_sum = u_val * kv;
                final_err_u += (row_sum - mu_i).abs();
            }
        }
        let eff_tol = config.tolerance + 1e-6;
        if final_err_u <= eff_tol {
            converged = true;
            err_u = final_err_u;
            err_v = 0.0;
        }
    }

    // 3. Compute un-regularized cost: <P, C> = sum_i u_i sum_j K_ij v_j C_ij
    // and entropy: H(P) = sum_ij P_ij * (log(P_ij) - 1)
    let mut total_cost = 0.0f32;
    let mut entropy = 0.0f32;
    for i in 0..n {
        let u_val = match u_vec.get(i) {
            Some(&u) => u,
            None => 0.0,
        };
        for j in 0..m {
            if let (Some(&k_val), Some(&v_val), Some(&c_val)) =
                (k_mat.get(i * m + j), v_vec.get(j), c_mat.get(i * m + j))
            {
                let p_ij = u_val * k_val * v_val;
                total_cost += p_ij * c_val;
                if p_ij > 1e-12 {
                    entropy += p_ij * (libm::logf(p_ij) - 1.0);
                }
            }
        }
    }
    let regularized_cost = total_cost + config.epsilon * entropy;

    Ok(SinkhornResult {
        cost: total_cost,
        regularized_cost,
        iterations,
        converged,
        marginal_error_u: err_u,
        marginal_error_v: err_v,
    })
}

/// Internal log-domain stabilized Sinkhorn solver using the Log-Sum-Exp trick
/// to prevent floating-point underflow or overflow when $\varepsilon$ is small.
#[allow(clippy::too_many_arguments)]
fn sinkhorn_log_domain_internal(
    c_mat: &[f32],
    _k_mat: &mut [f32],
    f_vec: &mut [f32],
    g_vec: &mut [f32],
    log_mu: &mut [f32],
    log_nu: &mut [f32],
    n: usize,
    m: usize,
    config: &SinkhornConfig,
    min_cost: f32,
) -> Result<SinkhornResult, AetherError> {
    let eps = config.epsilon;
    let mu_i = 1.0f32 / (n as f32);
    let nu_j = 1.0f32 / (m as f32);

    let l_mu = libm::logf(mu_i);
    let l_nu = libm::logf(nu_j);

    for x in log_mu.iter_mut() {
        *x = l_mu;
    }
    for x in log_nu.iter_mut() {
        *x = l_nu;
    }

    // Initialize potential functions f and g to 0
    for f in f_vec.iter_mut() {
        *f = 0.0;
    }
    for g in g_vec.iter_mut() {
        *g = 0.0;
    }

    let mut iterations = 0;
    let mut converged = false;
    let mut err_u = 1.0f32;
    let mut err_v = 1.0f32;

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        // Step 1: Update f_i = eps * log(mu_i) - eps * LSE_j((g_j - (C_ij - min_cost)) / eps)
        for i in 0..n {
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..m {
                if let (Some(&g), Some(&c)) = (g_vec.get(j), c_mat.get(i * m + j)) {
                    let c_shifted = c - min_cost;
                    let val = (g - c_shifted) / eps;
                    if val > max_val {
                        max_val = val;
                    }
                }
            }
            let mut sum_exp = 0.0f32;
            for j in 0..m {
                if let (Some(&g), Some(&c)) = (g_vec.get(j), c_mat.get(i * m + j)) {
                    let c_shifted = c - min_cost;
                    let val = (g - c_shifted) / eps;
                    sum_exp += libm::expf(val - max_val);
                }
            }
            let lse = max_val + libm::logf(sum_exp.max(1e-12));
            if let Some(f_ref) = f_vec.get_mut(i) {
                *f_ref = eps * l_mu - eps * lse;
            }
        }

        // Step 2: Update g_j = eps * log(nu_j) - eps * LSE_i((f_i - (C_ij - min_cost)) / eps)
        for j in 0..m {
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..n {
                if let (Some(&f), Some(&c)) = (f_vec.get(i), c_mat.get(i * m + j)) {
                    let c_shifted = c - min_cost;
                    let val = (f - c_shifted) / eps;
                    if val > max_val {
                        max_val = val;
                    }
                }
            }
            let mut sum_exp = 0.0f32;
            for i in 0..n {
                if let (Some(&f), Some(&c)) = (f_vec.get(i), c_mat.get(i * m + j)) {
                    let c_shifted = c - min_cost;
                    let val = (f - c_shifted) / eps;
                    sum_exp += libm::expf(val - max_val);
                }
            }
            let lse = max_val + libm::logf(sum_exp.max(1e-12));
            if let Some(g_ref) = g_vec.get_mut(j) {
                *g_ref = eps * l_nu - eps * lse;
            }
        }

        // Marginal check
        err_u = 0.0;
        for i in 0..n {
            let f = match f_vec.get(i) {
                Some(&v) => v,
                None => 0.0,
            };
            let mut p_row = 0.0f32;
            for j in 0..m {
                if let (Some(&g), Some(&c)) = (g_vec.get(j), c_mat.get(i * m + j)) {
                    let c_shifted = c - min_cost;
                    let exponent = (f + g - c_shifted) / eps;
                    p_row += libm::expf(exponent.clamp(-80.0, 80.0));
                }
            }
            err_u += (p_row - mu_i).abs();
        }

        err_v = 0.0;
        for j in 0..m {
            let g = match g_vec.get(j) {
                Some(&v) => v,
                None => 0.0,
            };
            let mut p_col = 0.0f32;
            for i in 0..n {
                if let (Some(&f), Some(&c)) = (f_vec.get(i), c_mat.get(i * m + j)) {
                    let c_shifted = c - min_cost;
                    let exponent = (f + g - c_shifted) / eps;
                    p_col += libm::expf(exponent.clamp(-80.0, 80.0));
                }
            }
            err_v += (p_col - nu_j).abs();
        }

        if err_u <= config.tolerance && err_v <= config.tolerance {
            converged = true;
            break;
        }
    }

    // Compute cost
    let mut total_cost = 0.0f32;
    for i in 0..n {
        let f = match f_vec.get(i) {
            Some(&v) => v,
            None => 0.0,
        };
        for j in 0..m {
            if let (Some(&g), Some(&c)) = (g_vec.get(j), c_mat.get(i * m + j)) {
                let c_shifted = c - min_cost;
                let exponent = (f + g - c_shifted) / eps;
                let p_ij = libm::expf(exponent.clamp(-80.0, 80.0));
                total_cost += p_ij * c;
            }
        }
    }

    let mut dual_sum = 0.0f32;
    for &f in f_vec.iter() {
        dual_sum += mu_i * f;
    }
    for &g in g_vec.iter() {
        dual_sum += nu_j * g;
    }
    let regularized_cost = min_cost + dual_sum;

    Ok(SinkhornResult {
        cost: total_cost,
        regularized_cost,
        iterations,
        converged,
        marginal_error_u: err_u,
        marginal_error_v: err_v,
    })
}

/// Convenience wrapper computing Entropy-Regularized Wasserstein Distance $\mathcal{W}_{\varepsilon}(A, B)$
/// allocating workspace internally.
pub fn sinkhorn_knopp(
    a_bag: &[f32],
    n: usize,
    b_bag: &[f32],
    m: usize,
    dim: usize,
    config: &SinkhornConfig,
) -> Result<SinkhornResult, AetherError> {
    let req = sinkhorn_workspace_size(n, m);
    let mut workspace = vec![0.0f32; req];
    sinkhorn_knopp_with_workspace(a_bag, n, b_bag, m, dim, config, &mut workspace)
}

/// Compute Debiased Sinkhorn Divergence $S_{\varepsilon}(A, B)$ with zero heap allocations
/// using a pre-allocated workspace of size `sinkhorn_workspace_size(n.max(m), n.max(m))`:
///
/// $$S_{\varepsilon}(A, B) = \mathcal{W}_{\varepsilon}(A, B) - \frac{1}{2}\left(\mathcal{W}_{\varepsilon}(A, A) + \mathcal{W}_{\varepsilon}(B, B)\right)$$
///
/// # Workspace Requirements
/// The caller must provide a pre-allocated `workspace` slice with length of at least:
/// $$\text{req} = \text{sinkhorn\_workspace\_size}(\max(n, m), \max(n, m)) = 2\max(n, m)^2 + 4\max(n, m)$$
///
/// Guaranteed Properties:
/// 1. Non-negativity: $S_{\varepsilon}(A, B) \ge 0$.
/// 2. Identity of indiscernibles: $S_{\varepsilon}(A, A) = 0$.
/// 3. Symmetry: $S_{\varepsilon}(A, B) = S_{\varepsilon}(B, A)$.
/// 4. Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
pub fn debiased_sinkhorn_divergence_with_workspace(
    a_bag: &[f32],
    n: usize,
    b_bag: &[f32],
    m: usize,
    dim: usize,
    config: &SinkhornConfig,
    workspace: &mut [f32],
) -> Result<f32, AetherError> {
    if a_bag.len() < n * dim {
        return Err(AetherError::InvalidBufferSize {
            actual: a_bag.len(),
            expected: n * dim,
        });
    }
    if b_bag.len() < m * dim {
        return Err(AetherError::InvalidBufferSize {
            actual: b_bag.len(),
            expected: m * dim,
        });
    }

    if core::ptr::eq(a_bag, b_bag) && n == m {
        return Ok(0.0);
    }

    let max_k = n.max(m);
    let required_len = sinkhorn_workspace_size(max_k, max_k);
    if workspace.len() < required_len {
        return Err(AetherError::ScratchpadExhausted {
            requested: required_len * core::mem::size_of::<f32>(),
            available: core::mem::size_of_val(workspace),
        });
    }

    let w_ab = sinkhorn_knopp_with_workspace(a_bag, n, b_bag, m, dim, config, workspace)?;
    let w_aa = sinkhorn_knopp_with_workspace(a_bag, n, a_bag, n, dim, config, workspace)?;
    let w_bb = sinkhorn_knopp_with_workspace(b_bag, m, b_bag, m, dim, config, workspace)?;

    let debiased = w_ab.regularized_cost - 0.5 * (w_aa.regularized_cost + w_bb.regularized_cost);
    Ok(if debiased < 0.0 { 0.0 } else { debiased })
}

/// Compute the Debiased Sinkhorn Divergence $S_{\varepsilon}(A, B)$ between two multi-vector bags:
/// $$S_{\varepsilon}(A, B) = \mathcal{W}_{\varepsilon}(A, B) - \frac{1}{2}\mathcal{W}_{\varepsilon}(A, A) - \frac{1}{2}\mathcal{W}_{\varepsilon}(B, B)$$
///
/// Guarantees:
/// 1. Non-negativity: $S_{\varepsilon}(A, B) \ge 0$.
/// 2. Identity of indiscernibles: $S_{\varepsilon}(A, A) = 0$.
/// 3. Symmetry: $S_{\varepsilon}(A, B) = S_{\varepsilon}(B, A)$.
pub fn debiased_sinkhorn_divergence(
    a_bag: &[f32],
    n: usize,
    b_bag: &[f32],
    m: usize,
    dim: usize,
    config: &SinkhornConfig,
) -> Result<f32, AetherError> {
    if core::ptr::eq(a_bag, b_bag) && n == m {
        return Ok(0.0);
    }

    let max_k = n.max(m);
    let req = sinkhorn_workspace_size(max_k, max_k);
    let mut workspace = vec![0.0f32; req];
    debiased_sinkhorn_divergence_with_workspace(a_bag, n, b_bag, m, dim, config, &mut workspace)
}
