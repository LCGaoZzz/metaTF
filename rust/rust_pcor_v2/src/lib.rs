//! rust_pcor_v2 — Spearman partial-correlation matrix, parity-targeting
//! `ppcor::pcor(t(exp), method="spearman")$estimate` as called by metaTF
//! `inferWeightExpMat(method="pcor")` (repo/R/inferGRNs.R L200-203).
//! Semantics anchor: ISSUE-FACTS/r-intermediates.md §1, transcribed verbatim.
//!
//! Pipeline:
//!   1. midrank (average ties, 1-based, ascending) of every gene's values
//!      across cells == R `rank(ties.method="average")`.
//!   2. **COVARIANCE matrix of the ranks** — cvx <- cov(x, method="spearman").
//!      This is THE round2 bug: round2 unit-normalized each row to length 1
//!      (correlation matrix) before inverting. R inverts the covariance of
//!      the rank vectors (per-gene rank variances kept!). v2 centers each
//!      rank row and keeps its scale.
//!   3. ginv-tolerance pseudo-inverse of the (symmetric PSD) rank covariance:
//!        - symmetric eigendecomposition (cyclic Jacobi); for a symmetric PSD
//!          matrix singular values == eigenvalues, so MASS::ginv's
//!          `Xsv$d > tol * Xsv$d[1]` keep-rule becomes
//!          `lambda > tol_rel * lambda_max` (STRICT >, tol_rel =
//!          sqrt(.Machine$double.eps) = 1.4901161193847656e-8).
//!        - the 1/(n-1) divisor of R's cov() is omitted: a positive scalar
//!          scales all eigenvalues uniformly, leaves the relative truncation
//!          invariant, and cancels inside cov2cor — the final partial
//!          correlations are identical (documented in MANIFEST).
//!        - when cells <= genes (always true for metaTF data) the exact rank
//!          of the covariance is <= cells-1, so the eigendecomposition is
//!          done on the DUAL Gram matrix C = Y^T Y (cells x cells) which is
//!          far smaller; eigenvectors of S = Y Y^T are v_k = Y u_k/sqrt(mu_k)
//!          with the same eigenvalues mu_k. When genes < cells the direct
//!          path (Jacobi on S = Y Y^T) is used. Both yield W (genes x kept)
//!          with Theta = pinv(S) = W W^T.
//!   4. partial correlation (ppcor's -cov2cor(icvx), diag<-1):
//!        p_ij = -Theta_ij / sqrt(Theta_ii * Theta_jj), diagonal set to 1.
//!
//! Parallelism: rayon over row blocks; thread count is a parameter
//! (default min(8, available cores)). Output rows are computed independently,
//! so results are bit-identical across thread counts.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// sqrt(.Machine$double.eps) — MASS::ginv default tolerance.
const GINV_TOL_REL: f64 = 1.4901161193847656e-8;

fn midrank_average(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut ord: Vec<u32> = (0..n as u32).collect();
    ord.sort_by(|&a, &b| x[a as usize].partial_cmp(&x[b as usize]).unwrap());
    let mut r = vec![0f64; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && x[ord[j + 1] as usize] == x[ord[i] as usize] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0; // mean of 1-based positions i+1..=j+1
        for &k in &ord[i..=j] {
            r[k as usize] = avg;
        }
        i = j + 1;
    }
    r
}

/// Cyclic Jacobi eigendecomposition of a symmetric row-major matrix
/// (a is destroyed).  Returns (eigenvalues, eigenvectors) where
/// eigvecs[k * n + j] is component j of eigenvector k (columns of V, A = V diag(λ) Vᵀ).
fn jacobi_eigen_sym(a: &mut Vec<f64>, n: usize, max_sweeps: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..max_sweeps {
        let mut off = 0f64;
        for p in 0..n.saturating_sub(1) {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                off += apq * apq;
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // A <- Jᵀ A J  (column pass then row pass)
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                // V <- V J
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
        if off < 1e-24 {
            break;
        }
    }
    let mut lam = vec![0f64; n];
    for i in 0..n {
        lam[i] = a[i * n + i];
    }
    (lam, v)
}

#[pyfunction(signature = (mat, n_threads=None, tol_rel=None))]
fn partial_corr_spearman<'py>(
    py: Python<'py>,
    mat: &Bound<'py, PyArray2<f64>>,
    n_threads: Option<usize>,
    tol_rel: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let dims = mat.dims();
    let (g, n) = (dims[0], dims[1]);
    if g < 2 {
        return Err(PyValueError::new_err("need at least 2 genes (rows)"));
    }
    if n < 3 {
        return Err(PyValueError::new_err(
            "need at least 3 cells (columns) for partial correlation",
        ));
    }
    let threads = n_threads.unwrap_or_else(|| {
        std::cmp::min(8, std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1))
    });
    let tol_rel = tol_rel.unwrap_or(GINV_TOL_REL);

    let x = unsafe { mat.as_slice() }.map_err(|_| {
        PyValueError::new_err("input must be a C-contiguous genes x cells float64 array")
    })?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| PyValueError::new_err(format!("thread pool: {e}")))?;

    // ---- 1+2: midrank + CENTER (covariance, NOT correlation) ----------------
    // Y holds centered rank vectors; each row keeps its own rank variance.
    let mut y = vec![0f64; g * n];
    let zero_var = std::sync::atomic::AtomicBool::new(false);
    pool.install(|| {
        y.par_chunks_mut(n)
            .zip(x.par_chunks(n))
            .for_each(|(yrow, xrow)| {
                let ranks = midrank_average(xrow);
                let mean = ranks.iter().sum::<f64>() / n as f64;
                let mut ss = 0f64;
                for j in 0..n {
                    let d = ranks[j] - mean;
                    yrow[j] = d;
                    ss += d * d;
                }
                if ss <= 0.0 {
                    zero_var.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            });
    });
    if zero_var.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(PyValueError::new_err(
            "constant row(s) in input: Spearman rank covariance undefined. \
             Filter zero-variance genes first (metaTF exp_cutoff does rowSums>0 upstream).",
        ));
    }

    // ---- 3: ginv-tolerance eigen truncation on the rank covariance ----------
    // W (g x r) with Theta = pinv(Y Yᵀ) = W Wᵀ  (∝ pinv(cov(ranks)) by 1/(n-1)).
    // keep rule: lambda > tol_rel * lambda_max  (MASS::ginv, STRICT >).
    let w: Vec<f64>;
    if n <= g {
        // dual Gram: C = Yᵀ Y (n x n); its nonzero eigenpairs mirror S = Y Yᵀ.
        let mut c = vec![0f64; n * n];
        pool.install(|| {
            c.par_chunks_mut(n).enumerate().for_each(|(k, crow)| {
                // row k of C: C_kl = Σ_i y_ik y_il
                for l in 0..n {
                    let mut acc = 0f64;
                    for i in 0..g {
                        acc += y[i * n + k] * y[i * n + l];
                    }
                    crow[l] = acc;
                }
            });
        });
        // symmetrize numerical noise
        for k in 0..n {
            for l in (k + 1)..n {
                let m = 0.5 * (c[k * n + l] + c[l * n + k]);
                c[k * n + l] = m;
                c[l * n + k] = m;
            }
        }
        let (mu, u) = jacobi_eigen_sym(&mut c, n, 100);
        let mu_max = mu.iter().cloned().fold(0f64, f64::max);
        if !(mu_max > 0.0) {
            return Err(PyValueError::new_err("degenerate rank covariance matrix"));
        }
        let kept: Vec<usize> = (0..n)
            .filter(|&k| mu[k] > tol_rel * mu_max && mu[k] > 0.0)
            .collect();
        let r = kept.len();
        if r == 0 {
            return Err(PyValueError::new_err("pinv tolerance removed every eigenvalue"));
        }
        // W (g x r): column c is Y u_k / mu_k, so that
        // Theta = W Wᵀ = Σ_k (Y u_k)(Y u_k)ᵀ / mu_k² = pinv(Y Yᵀ).
        let mut wbuf = vec![0f64; g * r];
        let inv_mu: Vec<f64> = kept.iter().map(|&k| 1.0 / mu[k]).collect();
        pool.install(|| {
            wbuf.par_chunks_mut(r).zip(y.par_chunks(n)).for_each(|(wrow, yrow)| {
                for (ci, &k) in kept.iter().enumerate() {
                    let mut acc = 0f64;
                    for j in 0..n {
                        acc += yrow[j] * u[j * n + k]; // eigenvector k = column k of V
                    }
                    wrow[ci] = acc * inv_mu[ci];
                }
            });
        });
        w = wbuf;
    } else {
        // direct: S = Y Yᵀ (g x g), Jacobi
        let mut a = vec![0f64; g * g];
        pool.install(|| {
            a.par_chunks_mut(g).zip(y.par_chunks(n)).for_each(|(arow, yrow)| {
                for (j, yj) in y.chunks(n).enumerate() {
                    let mut acc = 0f64;
                    for jj in 0..n {
                        acc += yrow[jj] * yj[jj];
                    }
                    arow[j] = acc;
                }
            });
        });
        for i in 0..g {
            for j in (i + 1)..g {
                let m = 0.5 * (a[i * g + j] + a[j * g + i]);
                a[i * g + j] = m;
                a[j * g + i] = m;
            }
        }
        let (lam, v) = jacobi_eigen_sym(&mut a, g, 100);
        let lam_max = lam.iter().cloned().fold(0f64, f64::max);
        let kept: Vec<usize> = (0..g)
            .filter(|&k| lam[k] > tol_rel * lam_max && lam[k] > 0.0)
            .collect();
        let r = kept.len();
        if r == 0 {
            return Err(PyValueError::new_err("pinv tolerance removed every eigenvalue"));
        }
        let mut wbuf = vec![0f64; g * r];
        let sqrt_lam: Vec<f64> = kept.iter().map(|&k| lam[k].sqrt()).collect();
        wbuf.par_chunks_mut(r).enumerate().for_each(|(i, wrow)| {
            for (ci, &k) in kept.iter().enumerate() {
                wrow[ci] = v[i * g + k] / sqrt_lam[ci]; // column k of V
            }
        });
        w = wbuf;
    }
    let r = w.len() / g;

    // ---- 4: p_ij = -Θ_ij / sqrt(Θ_ii Θ_jj), diag 1 ---------------------------
    let out_arr = PyArray2::<f64>::zeros_bound(py, [g, g], false);
    let out = unsafe { out_arr.as_slice_mut().unwrap() };

    let d: Vec<f64> = {
        let mut d = vec![0f64; g];
        pool.install(|| {
            d.par_iter_mut().zip(w.par_chunks(r)).for_each(|(dv, wrow)| {
                *dv = wrow.iter().map(|v| v * v).sum::<f64>().sqrt();
            });
        });
        d
    };

    pool.install(|| {
        out.par_chunks_mut(g).enumerate().for_each(|(i, orow)| {
            let wi = &w[i * r..(i + 1) * r];
            for j in 0..g {
                if j == i {
                    orow[j] = 1.0;
                } else {
                    let wj = &w[j * r..(j + 1) * r];
                    let mut acc = 0f64;
                    for k in 0..r {
                        acc += wi[k] * wj[k];
                    }
                    orow[j] = -acc / (d[i] * d[j]);
                }
            }
        });
    });

    Ok(out_arr)
}

#[pymodule]
pub fn rust_pcor_v2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(partial_corr_spearman, m)?)?;
    Ok(())
}
